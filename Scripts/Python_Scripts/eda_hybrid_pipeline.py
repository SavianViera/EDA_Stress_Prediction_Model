"""
Hybrid Feature Fusion + Classification (Birkenmaier et al., 2026)
====================================================================
Combines:
  - 32-dim latent features from the trained supervised autoencoder
    (eda_supervised_autoencoder.py, lambda=2.0 fixed)
  - 20-dim hand-crafted statistical features (eda_statistical_features.py,
    Table 3), extracted from the SAME window boundaries via cvxEDA

into a 52-dim hybrid feature vector per window, z-scored using
TRAINING-SET-ONLY statistics (no leakage), then trains classifiers
(KNN k=7, Random Forest, SVM) and evaluates on held-out test subjects.

Subject-level split is reused throughout: the same train/test subject
partition drives the autoencoder training, the feature normalization,
and the final classifier evaluation, so no windows from a test subject
touch any part of the fitting process.
"""

import glob
import os
import pickle
import time

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from eda_artifact_detection import clean_eda_signal
from eda_statistical_features import extract_features_for_subject, normalize_features_per_subject, FEATURE_NAMES
from eda_supervised_autoencoder import (
    SEED,
    FINAL_EPOCHS,
    _find_and_prepare_wesad_root,
    load_wesad_eda_windows,
    split_subjects_train_test,
    train_supervised_autoencoder_final,
)

LAMBDA_CLS = 2.0  # locked value from the earlier grid search
N_INNER_FOLDS = 5  # paper: StratifiedKFold(k=5) for the inner grid search

# Winning combination from eda_cvxeda_search.py (54-combo grid, CV Acc 0.9106).
# Single source of truth -- passed explicitly below, not relying on
# compute_statistical_features' own internal defaults.
CVXEDA_PARAMS = dict(tau0=1.0, tau1=0.7, delta_knot=40.0, alpha=4e-4)

# Winning cutoff from eda_highpass_search.py: beat the paper's literal
# 0.05 Hz on every tracked metric simultaneously (S17, S7, and overall
# accuracy) -- see search results for the full trade-off analysis.
HIGHPASS_CUTOFF_HZ = 0.03

# Hyperparameter grids matching the paper's Table 4 exactly.
CLASSIFIER_GRIDS = {
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=SEED),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=SEED, eval_metric="logloss"),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.10],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3],
        },
    },
    "SVM": {
        "estimator": SVC(random_state=SEED, probability=True),
        "param_grid": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto"],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    },
}


def load_raw_subject_eda(pkl_path: str, fs: int = 4, verbose: bool = True,
                          apply_highpass: bool = True, highpass_cutoff_hz: float = 0.05,
                          highpass_order: int = 4) -> np.ndarray:
    """
    Loads a subject's wrist EDA trace (microsiemens) and applies the same
    artifact cleaning used by the autoencoder branch, so both halves of the
    hybrid feature vector are built from an identically cleaned signal.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    raw = np.asarray(data["signal"]["wrist"]["EDA"]).flatten().astype(np.float64)
    subject_id = os.path.basename(pkl_path).replace(".pkl", "")
    return clean_eda_signal(raw, fs=fs, subject_label=f"{subject_id} (stats branch)", verbose=verbose,
                             apply_highpass=apply_highpass, highpass_cutoff_hz=highpass_cutoff_hz,
                             highpass_order=highpass_order)


def compute_statistical_features(wesad_root: str, subject_ids: np.ndarray,
                                  starts: np.ndarray, raw_win_len: int,
                                  fs: int = 4, tau0: float = 2.0, tau1: float = 0.7,
                                  delta_knot: float = 10.0, alpha: float = 8e-4,
                                  use_adaptive_threshold: bool = True,
                                  adaptive_threshold_k: float = 1.0,
                                  apply_highpass: bool = True, highpass_cutoff_hz: float = 0.05,
                                  highpass_order: int = 4,
                                  verbose: bool = True) -> np.ndarray:
    """
    Computes the 20 hand-crafted features for every window, in the SAME
    row order as `subject_ids`/`starts` (which is the same order as the
    `windows` tensor from load_wesad_eda_windows). Processes one
    subject's raw trace at a time (cvxEDA decomposed once per subject),
    matching the original loader's per-subject, ascending-start-index
    construction order exactly.

    tau0/tau1/delta_knot/alpha: cvxEDA decomposition parameters, passed
    through to extract_features_for_subject. Defaults are neurokit2's own
    defaults (== the paper's unadapted lab-data values); override with
    values found via eda_cvxeda_search.py for a fit tuned to your actual
    sampling rate.

    use_adaptive_threshold/adaptive_threshold_k: controls SCR peak
    detection's threshold. See adaptive_peak_threshold's docstring in
    eda_statistical_features.py for the leakage-safe interpretation of the
    paper's "adaptive, condition-aware" threshold (Table 2).

    apply_highpass/highpass_cutoff_hz/highpass_order: passed through to
    load_raw_subject_eda -> clean_eda_signal's baseline-drift removal.

    verbose: set False to suppress the per-subject print (useful when this
    is called many times in a loop, e.g. during a parameter search).
    """
    subject_paths = {
        os.path.basename(p).replace(".pkl", ""): p
        for p in glob.glob(os.path.join(wesad_root, "S*", "S*.pkl"))
    }

    all_feature_blocks = []
    for subject in sorted(set(subject_ids.tolist())):
        subj_mask = subject_ids == subject
        subj_starts = starts[subj_mask]  # already ascending, matches loader's construction order

        raw_eda = load_raw_subject_eda(subject_paths[subject], verbose=verbose,
                                       apply_highpass=apply_highpass,
                                       highpass_cutoff_hz=highpass_cutoff_hz,
                                       highpass_order=highpass_order)
        feats = extract_features_for_subject(raw_eda, subj_starts, raw_win_len, fs=fs,
                                              tau0=tau0, tau1=tau1, delta_knot=delta_knot,
                                              alpha=alpha, use_adaptive_threshold=use_adaptive_threshold,
                                              adaptive_threshold_k=adaptive_threshold_k)
        all_feature_blocks.append((subj_mask, feats))
        if verbose:
            print(f"  {subject}: extracted stats for {feats.shape[0]} windows")

    # Reassemble into the original row order (same order as `subject_ids`).
    n_total = len(subject_ids)
    stat_features = np.empty((n_total, 20), dtype=np.float32)
    for subj_mask, feats in all_feature_blocks:
        stat_features[subj_mask] = feats

    return stat_features


def tune_and_evaluate(name, estimator, param_grid, X_train, y_train, train_groups,
                       X_test, y_test, n_folds=N_INNER_FOLDS, seed=SEED):
    """
    Grid search over param_grid using StratifiedGroupKFold inner CV on the
    TRAINING set only, grouped by subject -- this is essential because
    consecutive windows overlap 50% (60s window / 30s stride), so a plain
    (non-grouped) StratifiedKFold would split near-duplicate windows from
    the same subject across different inner folds, leaking information
    and inflating validation accuracy relative to true held-out
    performance. Otherwise matches the paper's Table 4 protocol.

    After the search, discards GridSearchCV's own refit estimator and
    instantiates a FRESH classifier with the winning hyperparameters, fit
    on the full training set, evaluated once on the held-out test set.
    """
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)

    n_unique_groups = len(set(train_groups.tolist() if hasattr(train_groups, "tolist") else train_groups))
    effective_folds = min(n_folds, n_unique_groups)
    if effective_folds < n_folds:
        print(f"\n{name}: only {n_unique_groups} training subjects available -- "
              f"reducing inner folds from {n_folds} to {effective_folds}")

    inner_cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, param_grid, scoring="accuracy", cv=inner_cv,
                           n_jobs=-1, refit=False)

    print(f"\n{name}: searching {n_combos} combinations x {effective_folds} folds "
          f"= {n_combos * effective_folds} fits...")
    start = time.time()
    search.fit(X_train, y_train, groups=train_groups)
    elapsed = time.time() - start

    best_params = search.best_params_
    best_cv_score = search.best_score_

    final_model = clone(estimator).set_params(**best_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on stress
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall on baseline

    print(f"  Done in {elapsed:.1f}s | Best params: {best_params}")
    print(f"  Inner-CV val Acc: {best_cv_score:.4f} | "
          f"Test: Acc={acc:.4f} Sens={sensitivity:.4f} Spec={specificity:.4f}")

    return {"name": name, "best_params": best_params, "cv_accuracy": best_cv_score,
            "accuracy": acc, "sensitivity": sensitivity, "specificity": specificity}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wesad_root = _find_and_prepare_wesad_root()
    print(f"Using WESAD_ROOT = {wesad_root}")

    # ---- 1. Load windows (same as the autoencoder pipeline) ----
    windows, subject_ids, labels, raw_win_len, starts = load_wesad_eda_windows(
        wesad_root, window_sec=60.0, stride_sec=30.0, target_len=256, fs=4,
        apply_highpass=True, highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
    )

    train_subjects, test_subjects = split_subjects_train_test(subject_ids)
    train_mask = np.isin(subject_ids, list(train_subjects))
    test_mask = np.isin(subject_ids, list(test_subjects))

    train_windows = windows[torch.from_numpy(train_mask)]
    test_windows = windows[torch.from_numpy(test_mask)]
    train_labels_t = torch.from_numpy(labels[train_mask])
    test_labels_t = torch.from_numpy(labels[test_mask])

    # ---- 2. Train the supervised autoencoder (lambda=2.0, fixed) ----
    print("\n" + "=" * 70)
    print("STAGE 1: Training supervised autoencoder (32-dim latent features)")
    print("=" * 70)
    model, history = train_supervised_autoencoder_final(
        train_windows, train_labels_t, test_windows, test_labels_t,
        raw_win_len, device, lambda_cls=LAMBDA_CLS, epochs=FINAL_EPOCHS,
    )
    print(f"Autoencoder final: Test MSE={history['test_recon_mse'][-1]:.6f}, "
          f"Test Acc (latent-only)={history['test_acc'][-1]:.4f}")

    # ---- 3. Extract 32-dim latent features for ALL windows ----
    model.eval()
    with torch.no_grad():
        all_windows_dev = windows.to(device)
        latent_features = model.encoder(all_windows_dev).cpu().numpy()  # (N, 32)
    print(f"\nLatent feature matrix: {latent_features.shape}")

    # ---- 4. Extract 20-dim hand-crafted statistical features (cvxEDA) ----
    print("\n" + "=" * 70)
    print("STAGE 2: Extracting hand-crafted statistical features (Table 3)")
    print("=" * 70)
    stat_features = compute_statistical_features(wesad_root, subject_ids, starts, raw_win_len,
                                                  fs=4, apply_highpass=True,
                                                  highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
                                                  **CVXEDA_PARAMS)
    print(f"Statistical feature matrix: {stat_features.shape}")

    # Per-subject normalization -- removes between-subject differences in
    # absolute EDA reactivity scale (see diagnose_subject.py findings: a
    # subject can have a completely normal within-subject stress>baseline
    # direction while still being misclassified because their absolute
    # amplitudes are much smaller than other subjects').
    stat_features = normalize_features_per_subject(stat_features, subject_ids)
    print("Applied per-subject normalization to statistical features")

    # ---- 5. Fuse into 52-dim hybrid vector ----
    hybrid_features = np.concatenate([stat_features, latent_features], axis=1)  # (N, 52)
    print(f"\nHybrid feature matrix: {hybrid_features.shape} "
          f"(20 statistical + 32 latent = {hybrid_features.shape[1]})")

    # ---- 6. Z-score normalize using TRAIN-set statistics only ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(hybrid_features[train_mask])
    X_test = scaler.transform(hybrid_features[test_mask])
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    print(f"\nTrain: {X_train.shape[0]} windows | Test: {X_test.shape[0]} windows "
          f"(subjects: train={sorted(train_subjects)}, test={sorted(test_subjects)})")

    train_groups = subject_ids[train_mask]  # which subject each training window belongs to

    # ---- 7. Grid search (train-only) + evaluate on held-out test subjects ----
    print("\n" + "=" * 70)
    print("STAGE 3: Hyperparameter search + classification (Table 4 grids)")
    print("=" * 70)

    results = [
        tune_and_evaluate(name, cfg["estimator"], cfg["param_grid"], X_train, y_train,
                           train_groups, X_test, y_test)
        for name, cfg in CLASSIFIER_GRIDS.items()
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best = max(results, key=lambda r: r["accuracy"])
    for r in results:
        marker = "  <-- best" if r["name"] == best["name"] else ""
        print(f"{r['name']:<15} Acc={r['accuracy']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Spec={r['specificity']:.4f}  (best params: {r['best_params']}){marker}")