import glob
import os
import pickle
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eda_artifact_detection import clean_eda_signal

# ===========================================================================
# 1. Dataset Helper: Load 10s WESAD EDA & Pad 40 Samples -> 64 Samples,
#    plus binary stress/baseline labels aligned from the 700 Hz chest channel.
# ===========================================================================
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
EXTRACT_DIR = os.path.join(DOWNLOADS_DIR, "WESAD_extracted")

SEED = 42
TEST_FRACTION = 0.2      # fraction of *subjects* (not windows) held out for test
LABEL_FS = 700           # WESAD 'label' array is sampled at the chest rate
PURITY_THRESHOLD = 0.8   # matches the paper's window-purity requirement

# --- Lambda (classification loss weight) grid search config ---
# Mirrors the paper's approach: grid search over these candidates using
# inner cross-validation on the TRAINING subjects only. The held-out test
# subjects are never touched during this search.
RUN_LAMBDA_SEARCH = False   # set True to re-run the search; False uses FIXED_LAMBDA below
FIXED_LAMBDA = 2.0          # used when RUN_LAMBDA_SEARCH is False (winner of the last search)
LAMBDA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]
N_INNER_FOLDS = 5        # subject-level folds within the training subjects
INNER_EPOCHS = 20        # shorter than the final fit -- just for ranking lambdas
FINAL_EPOCHS = 50        # epochs for the final model, once lambda is chosen


def _find_wesad_root(search_dir: str):
    for dirpath, _, _ in os.walk(search_dir):
        candidate_pkls = glob.glob(os.path.join(dirpath, "S*", "S*.pkl"))
        if candidate_pkls:
            return dirpath
    return None


def _find_and_prepare_wesad_root(downloads_dir=DOWNLOADS_DIR, extract_dir=EXTRACT_DIR):
    root = _find_wesad_root(downloads_dir)
    if root:
        return root
    if os.path.isdir(extract_dir):
        root = _find_wesad_root(extract_dir)
        if root:
            return root
    zips = glob.glob(os.path.join(downloads_dir, "*wesad*.zip")) or glob.glob(
        os.path.join(downloads_dir, "*.zip")
    )
    if not zips:
        raise FileNotFoundError(f"Couldn't find a WESAD folder or .zip under {downloads_dir}.")
    zip_path = zips[0]
    print(f"Extracting {zip_path} -> {extract_dir} ...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    root = _find_wesad_root(extract_dir)
    if not root:
        raise FileNotFoundError(
            f"Extracted {zip_path} to {extract_dir} but couldn't find S*/S*.pkl files."
        )
    return root


def _window_label(label_array, start_sample, raw_win_len, eda_fs, label_fs=LABEL_FS,
                   purity_threshold=PURITY_THRESHOLD):
    """
    Maps a wrist-EDA window (in wrist-sample-rate index space) to the
    corresponding slice of the 700 Hz chest label array, then returns a
    binary label (0=baseline, 1=stress) if that slice is dominated
    (>= purity_threshold) by baseline(1) or stress(2), else None.

    WESAD's chest and wrist signals are recorded on a shared timeline, so a
    wrist-sample index converts to a label-array index via the ratio of
    their sampling rates (label_fs / eda_fs).
    """
    scale = label_fs / eda_fs
    lbl_start = int(start_sample * scale)
    lbl_end = int((start_sample + raw_win_len) * scale)
    lbl_end = min(lbl_end, len(label_array))
    if lbl_end <= lbl_start:
        return None

    segment = label_array[lbl_start:lbl_end]
    values, counts = np.unique(segment, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_label = values[majority_idx]
    purity = counts[majority_idx] / len(segment)

    if majority_label not in (1, 2) or purity < purity_threshold:
        return None  # transition / amusement / meditation / impure window -- drop
    return 0 if majority_label == 1 else 1  # 0=baseline, 1=stress


def load_wesad_eda_windows(
    wesad_root: str,
    window_sec: float = 60.0,
    stride_sec: float = 30.0,
    target_len: int = 256,
    fs: int = 4,
    apply_highpass: bool = True,
    highpass_cutoff_hz: float = 0.05,
    highpass_order: int = 4,
):
    """
    Loads Empatica E4 wrist EDA (4 Hz), extracts 60s windows (240 samples,
    50% overlap -- matching the paper's windowing choice), z-score
    normalizes per subject, zero-pads to 256 samples (only 16 padding
    samples, 6.25% of the window -- much less than the old 10s/64 setup's
    37.5%), and attaches a binary stress/baseline label to each window
    using the chest label channel (dropping windows that aren't a clean
    baseline/stress segment).
    Also returns subject IDs for subject-level (patient-level) splitting.

    apply_highpass/highpass_cutoff_hz/highpass_order: passed through to
    clean_eda_signal's baseline-drift removal (Section 2.2.1). See that
    function's docstring in eda_artifact_detection.py for the tension in
    the paper's own description of this filter's effect on genuine slow
    SCL trends.
    """
    subject_paths = glob.glob(os.path.join(wesad_root, "S*", "S*.pkl"))
    if not subject_paths:
        raise FileNotFoundError(f"No subject pkl files found in {wesad_root}")

    raw_win_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    pad_amount = target_len - raw_win_len

    all_windows, all_subject_ids, all_labels, all_starts = [], [], [], []

    for path in sorted(subject_paths):
        subject_id = os.path.basename(path).replace(".pkl", "")
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eda_signal = data["signal"]["wrist"]["EDA"].flatten()
        label_array = np.asarray(data["label"]).flatten()

        eda_signal = clean_eda_signal(eda_signal, fs=fs, subject_label=subject_id,
                                       apply_highpass=apply_highpass,
                                       highpass_cutoff_hz=highpass_cutoff_hz,
                                       highpass_order=highpass_order)

        mean_val = np.mean(eda_signal)
        std_val = np.std(eda_signal) + 1e-8
        norm_eda = (eda_signal - mean_val) / std_val

        n_samples = len(norm_eda)
        n_kept = 0
        for start in range(0, n_samples - raw_win_len + 1, stride_len):
            y = _window_label(label_array, start, raw_win_len, eda_fs=fs)
            if y is None:
                continue  # not a clean baseline/stress window -- skip
            window = norm_eda[start:start + raw_win_len]
            all_windows.append(window)
            all_subject_ids.append(subject_id)
            all_labels.append(y)
            all_starts.append(start)
            n_kept += 1

        print(f"  {subject_id}: {n_samples} raw samples @ {fs} Hz -> "
              f"{n_kept} labeled baseline/stress windows")

    windows_tensor = torch.tensor(np.array(all_windows, dtype=np.float32)).unsqueeze(1)
    padded_windows = F.pad(windows_tensor, (0, pad_amount), value=0.0)
    subject_ids = np.array(all_subject_ids)
    labels = np.array(all_labels, dtype=np.float32)
    starts = np.array(all_starts, dtype=np.int64)

    n_baseline = int((labels == 0).sum())
    n_stress = int((labels == 1).sum())
    print(f"Windows created: {padded_windows.shape[0]} "
          f"(baseline: {n_baseline}, stress: {n_stress}, "
          f"ratio 1:{n_baseline / max(n_stress, 1):.2f})")
    print(f"Active samples: {raw_win_len}, Padded zeros: {pad_amount} (Total: {target_len})")

    return padded_windows, subject_ids, labels, raw_win_len, starts


def split_subjects_train_test(subject_ids, test_fraction=TEST_FRACTION, seed=SEED):
    unique_subjects = sorted(set(subject_ids.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_test = max(1, round(len(unique_subjects) * test_fraction))
    test_subjects = set(unique_subjects[:n_test])
    train_subjects = set(unique_subjects[n_test:])
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Test subjects  ({len(test_subjects)}): {sorted(test_subjects)}")
    return train_subjects, test_subjects


# ===========================================================================
# 2. 3-Stage Model Architecture (64-sample input) + classification head
# ===========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class EDAEncoder(nn.Module):
    """3-Stage Encoder: (batch, 1, 64) -> (batch, 32) latent bottleneck."""

    def __init__(self, input_length: int = 64, latent_dim: int = 32):
        super().__init__()
        channels = [1, 16, 32, 64]
        n_stages = len(channels) - 1
        self.conv_stack = nn.Sequential(
            *[ConvBlock(channels[i], channels[i + 1]) for i in range(n_stages)]
        )
        reduced_length = input_length // (2 ** n_stages)
        flat_dim = channels[-1] * reduced_length
        self.fc = nn.Sequential(nn.Linear(flat_dim, latent_dim), nn.BatchNorm1d(latent_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, final_layer=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.final_layer = final_layer
        if not final_layer:
            self.bn = nn.BatchNorm1d(out_channels)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(self.upsample(x))
        if not self.final_layer:
            x = self.act(self.bn(x))
        return x


class EDADecoder(nn.Module):
    """3-Stage Decoder: (batch, 32) latent bottleneck -> (batch, 1, 64) reconstruction."""

    def __init__(self, output_length: int = 64, latent_dim: int = 32):
        super().__init__()
        channels = [64, 32, 16, 1]
        n_stages = len(channels) - 1
        self.reduced_length = output_length // (2 ** n_stages)
        flat_dim = channels[0] * self.reduced_length
        self._start_channels = channels[0]
        self.project = nn.Sequential(nn.Linear(latent_dim, flat_dim), nn.ReLU(inplace=True))
        self.deconv_stack = nn.Sequential(
            *[DeconvBlock(channels[i], channels[i + 1], final_layer=(i == n_stages - 1))
              for i in range(n_stages)]
        )

    def forward(self, z):
        x = self.project(z).view(-1, self._start_channels, self.reduced_length)
        return self.deconv_stack(x)


class EDAClassifierHead(nn.Module):
    """Small MLP head on top of the latent bottleneck -> single stress logit."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)  # (batch,) raw logits for BCEWithLogitsLoss


class EDASupervisedAutoencoder(nn.Module):
    """
    Dual-head architecture matching Eq. 1 of the paper:
        L_total = L_reconstruction + lambda * L_classification
    Same encoder/decoder as before, plus a classification head branching
    off the same 32-dim latent bottleneck.
    """

    def __init__(self, input_length: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = EDAEncoder(input_length=input_length, latent_dim=latent_dim)
        self.decoder = EDADecoder(output_length=input_length, latent_dim=latent_dim)
        self.classifier = EDAClassifierHead(latent_dim=latent_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        logits = self.classifier(latent)
        return reconstruction, logits, latent


# ===========================================================================
# 3. Lambda grid search (subject-level inner CV, training subjects only)
# ===========================================================================
def make_subject_folds(subjects, n_folds, seed=SEED):
    """Splits a subject list into n_folds roughly-equal groups (round-robin)."""
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    folds = [[] for _ in range(n_folds)]
    for i, subj in enumerate(subjects):
        folds[i % n_folds].append(subj)
    return folds


def build_loader_for_subjects(windows, subject_ids, labels, wanted_subjects,
                               batch_size=64, shuffle=True):
    mask = np.isin(subject_ids, list(wanted_subjects))
    x = windows[torch.from_numpy(mask)]
    y = torch.from_numpy(labels[mask])
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
    return loader


def train_and_eval(train_loader, val_loader, lambda_val, raw_win_len, device,
                    epochs=INNER_EPOCHS, lr=1e-3, seed=SEED):
    """Trains a fresh model with a given lambda and returns final val MSE/accuracy."""
    torch.manual_seed(seed)
    model = EDASupervisedAutoencoder(input_length=256, latent_dim=32).to(device)
    mse_loss = nn.MSELoss()

    y_train = train_loader.dataset.tensors[1]
    n_baseline = max(int((y_train == 0).sum().item()), 1)
    n_stress = max(int((y_train == 1).sum().item()), 1)
    pos_weight = torch.tensor(n_baseline / n_stress, dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            recon, logits, _ = model(x_batch)
            recon_loss = mse_loss(recon[:, :, :raw_win_len], x_batch[:, :, :raw_win_len])
            cls_loss = bce(logits, y_batch)
            (recon_loss + lambda_val * cls_loss).backward()
            optimizer.step()

    model.eval()
    recon_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            recon, logits, _ = model(x_batch)
            recon_loss = mse_loss(recon[:, :, :raw_win_len], x_batch[:, :, :raw_win_len])
            recon_sum += recon_loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            n += x_batch.size(0)

    return recon_sum / n, correct / n


def run_lambda_grid_search(windows, subject_ids, labels, raw_win_len, train_subjects,
                            device, lambda_grid=LAMBDA_GRID, n_folds=N_INNER_FOLDS):
    """
    Subject-level k-fold grid search over lambda, entirely within the
    training subjects (test subjects are never referenced here). For each
    fold, one group of training subjects becomes the inner-validation set
    and the rest become the inner-training set; a fresh model is trained
    per (lambda, fold) and scored on that fold's held-out subjects.
    """
    folds = make_subject_folds(sorted(train_subjects), n_folds)
    results = {lam: {"val_mse": [], "val_acc": []} for lam in lambda_grid}

    for fold_idx in range(n_folds):
        inner_val_subjects = set(folds[fold_idx])
        inner_train_subjects = set(train_subjects) - inner_val_subjects
        if not inner_val_subjects or not inner_train_subjects:
            continue  # degenerate fold (too few subjects) -- skip

        inner_train_loader = build_loader_for_subjects(
            windows, subject_ids, labels, inner_train_subjects, shuffle=True
        )
        inner_val_loader = build_loader_for_subjects(
            windows, subject_ids, labels, inner_val_subjects, shuffle=False
        )

        print(f"\nFold {fold_idx + 1}/{n_folds} | "
              f"inner-train={sorted(inner_train_subjects)} | "
              f"inner-val={sorted(inner_val_subjects)}")

        for lam in lambda_grid:
            val_mse, val_acc = train_and_eval(
                inner_train_loader, inner_val_loader, lam, raw_win_len, device
            )
            results[lam]["val_mse"].append(val_mse)
            results[lam]["val_acc"].append(val_acc)
            print(f"  lambda={lam:<4} | val MSE={val_mse:.6f} | val Acc={val_acc:.4f}")

    print("\n=== Lambda grid search summary (mean over folds) ===")
    summary = []
    for lam in lambda_grid:
        if not results[lam]["val_acc"]:
            continue
        mean_acc = float(np.mean(results[lam]["val_acc"]))
        mean_mse = float(np.mean(results[lam]["val_mse"]))
        summary.append((lam, mean_acc, mean_mse))
        print(f"lambda={lam:<4} | mean val Acc={mean_acc:.4f} | mean val MSE={mean_mse:.6f}")

    if not summary:
        raise RuntimeError("Lambda grid search produced no results -- check fold sizes.")

    # Best = highest mean validation accuracy. On a tie, prefer the lower
    # lambda with the better (lower) reconstruction MSE, matching the
    # paper's stated criterion ("highest validation accuracy while
    # preserving acceptable reconstruction fidelity").
    best_lambda, best_acc, best_mse = max(summary, key=lambda t: (t[1], -t[2]))
    print(f"\nBest lambda: {best_lambda} (mean val Acc={best_acc:.4f}, mean val MSE={best_mse:.6f})")
    return best_lambda, summary


# ===========================================================================
# 4. Visualization Helper
# ===========================================================================
def plot_learning_curves(history, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    epoch_axis = range(1, epochs + 1)

    axes[0].plot(epoch_axis, history["train_recon_mse"], label="Train", color="blue", marker="o")
    axes[0].plot(epoch_axis, history["test_recon_mse"], label="Test (held-out subjects)",
                 color="orange", linestyle="--", marker="s")
    axes[0].set_title("Reconstruction MSE (real samples only)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epoch_axis, history["train_acc"], label="Train", color="blue", marker="o")
    axes[1].plot(epoch_axis, history["test_acc"], label="Test (held-out subjects)",
                 color="orange", linestyle="--", marker="s")
    axes[1].set_title("Stress vs Baseline Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_filename = "supervised_autoencoder_learning_curves.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[Plot Saved] Learning curves saved to '{plot_filename}'")
    plt.show()


# ===========================================================================
# 5. Main Execution Loop
# ===========================================================================
# ===========================================================================
# 5. Reusable final-fit training function (used by main below, and importable
#    by downstream scripts such as the hybrid feature-fusion pipeline)
# ===========================================================================
def train_supervised_autoencoder_final(train_windows, train_labels, test_windows, test_labels,
                                        raw_win_len, device, lambda_cls=2.0,
                                        epochs=FINAL_EPOCHS, lr=1e-3, seed=SEED, verbose=True):
    """
    Trains the dual-head EDASupervisedAutoencoder with a fixed lambda on
    the given train/test window tensors (already subject-split). Returns
    the trained model and a history dict of per-epoch metrics.
    """
    torch.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(train_windows, train_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_windows, test_labels), batch_size=64, shuffle=False)

    n_baseline_train = max(int((train_labels == 0).sum().item()), 1)
    n_stress_train = max(int((train_labels == 1).sum().item()), 1)
    pos_weight = torch.tensor(n_baseline_train / n_stress_train, dtype=torch.float32).to(device)
    if verbose:
        print(f"BCE pos_weight (baseline/stress ratio in train set): {pos_weight.item():.3f}")

    model = EDASupervisedAutoencoder(input_length=256, latent_dim=32).to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_recon_mse": [], "test_recon_mse": [], "train_acc": [], "test_acc": []}

    if verbose:
        print(f"\nFINAL FIT | lambda = {lambda_cls} (fixed) | {epochs} epochs")
        print("L_total = L_reconstruction + lambda * L_classification")

    for epoch in range(1, epochs + 1):
        model.train()
        recon_sum, cls_correct, n_seen = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            recon, logits, _ = model(x_batch)
            recon_loss = mse_loss(recon[:, :, :raw_win_len], x_batch[:, :, :raw_win_len])
            cls_loss = bce_loss(logits, y_batch)
            (recon_loss + lambda_cls * cls_loss).backward()
            optimizer.step()

            recon_sum += recon_loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            cls_correct += (preds == y_batch).sum().item()
            n_seen += x_batch.size(0)

        history["train_recon_mse"].append(recon_sum / n_seen)
        history["train_acc"].append(cls_correct / n_seen)

        model.eval()
        recon_sum, cls_correct, n_seen = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                recon, logits, _ = model(x_batch)
                recon_loss = mse_loss(recon[:, :, :raw_win_len], x_batch[:, :, :raw_win_len])
                recon_sum += recon_loss.item() * x_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                cls_correct += (preds == y_batch).sum().item()
                n_seen += x_batch.size(0)

        history["test_recon_mse"].append(recon_sum / n_seen)
        history["test_acc"].append(cls_correct / n_seen)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
                  f"Train MSE: {history['train_recon_mse'][-1]:.6f} | "
                  f"Train Acc: {history['train_acc'][-1]:.4f} | "
                  f"Test MSE: {history['test_recon_mse'][-1]:.6f} | "
                  f"Test Acc: {history['test_acc'][-1]:.4f}")

    return model, history


# ===========================================================================
# 6. Main Execution Loop
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wesad_root = _find_and_prepare_wesad_root()
    print(f"Using WESAD_ROOT = {wesad_root}")

    windows, subject_ids, labels, raw_win_len, starts = load_wesad_eda_windows(
        wesad_root, window_sec=60.0, stride_sec=30.0, target_len=256, fs=4
    )

    train_subjects, test_subjects = split_subjects_train_test(subject_ids)
    train_mask = np.isin(subject_ids, list(train_subjects))
    test_mask = np.isin(subject_ids, list(test_subjects))

    train_windows = windows[torch.from_numpy(train_mask)]
    test_windows = windows[torch.from_numpy(test_mask)]
    train_labels = torch.from_numpy(labels[train_mask])
    test_labels = torch.from_numpy(labels[test_mask])

    print(f"Train windows: {train_windows.shape[0]} "
          f"(baseline: {int((train_labels == 0).sum())}, stress: {int((train_labels == 1).sum())})")
    print(f"Test windows:  {test_windows.shape[0]} "
          f"(baseline: {int((test_labels == 0).sum())}, stress: {int((test_labels == 1).sum())})")

    # ---- Lambda: LOCKED at 2.0 (winner of the earlier grid search: ----
    # ---- 84% test accuracy, ~0.01 MSE). Grid search disabled. ----
    LAMBDA_CLS = 2.0
    print(f"\nUsing fixed lambda = {LAMBDA_CLS} (grid search disabled)")

    model, history = train_supervised_autoencoder_final(
        train_windows, train_labels, test_windows, test_labels,
        raw_win_len, device, lambda_cls=LAMBDA_CLS, epochs=FINAL_EPOCHS,
    )
    epochs = FINAL_EPOCHS

    print("-" * 70)
    print(f"Final: Test MSE = {history['test_recon_mse'][-1]:.6f}, "
          f"Test Accuracy = {history['test_acc'][-1]:.4f}")

    plot_learning_curves(history, epochs)