"""
Helper functions shared by the notebooks in code_draft2.

- windowing of the WESAD E4 (wrist, 4 Hz) EDA signal into 1-minute windows
- per-window decomposition with eda_proj.gms_eda.gmsEDA (+ a naive running-median control)
- hand-crafted window features (raw signal / tonic / phasic / sparse driver)
- leave-one-subject-out (LOSO) evaluation, paired comparisons, nested configuration selection
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

FS = 4                 # E4 EDA sampling rate (Hz)
WIN_SEC = 60           # 1-minute windows
WIN = WIN_SEC * FS     # 240 samples

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PKL = ROOT / "Data" / "wesad_eda" / "wesad_subject_list.pkl"
CACHE = HERE / "cache"

# make the project package importable even if `pip install -e .` was not run
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_subjects(pkl_path=DATA_PKL):
    """Return dict {subject_id: (eda[µS] 4Hz, label 4Hz)} from wesad_subject_list.pkl.

    wesad_subject_list.pkl is produced by code_draft/01load_eda.ipynb from the
    `signal/wrist/EDA` (= Empatica E4) channel of each WESAD S*.pkl, with the
    700 Hz labels downsampled to 4 Hz by the majority (mode) rule.
    """
    with open(pkl_path, "rb") as f:
        lst = pickle.load(f)
    out = {}
    for df in lst:
        sid = df.columns[0].split("_")[0]
        out[sid] = (df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(int))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def make_windows(subjects, labels=(1, 2), stride_sec=10, win=WIN, fs=FS):
    """Cut 1-min windows that lie entirely inside one condition block (100 % label purity).

    Returns
    -------
    W    : (n_windows, win) raw EDA windows (µS)
    meta : DataFrame with subject, label, start index
    """
    rows, W = [], []
    stride = int(stride_sec * fs)
    for sid, (eda, lab) in subjects.items():
        for c in labels:
            idx = np.flatnonzero(lab == c)
            if len(idx) == 0:
                continue
            # split into contiguous blocks
            blocks = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for b in blocks:
                for s in range(b[0], b[-1] + 2 - win, stride):
                    W.append(eda[s:s + win])
                    rows.append((sid, c, s))
    meta = pd.DataFrame(rows, columns=["subject", "label", "start"])
    return np.asarray(W), meta


def reshape_windows(subjects, stride_sec=10, purity=1.0, labels=(1, 2), win_sec=WIN_SEC, fs=FS):
    """1-min windows with the project's reshape_function (eda_proj.features, from
    python_code/Statistical_Feature_Extraction.ipynb). It slides from the start of each recording with the
    given stride and keeps windows whose majority label covers >= `purity` of the window.
    Returns W (n, 240) and meta (subject, label, start), restricted to `labels`. reshape_function does not
    return the window start, so `start` is recovered by matching each window to the recording."""
    from eda_proj.features import reshape_function
    wide = pd.concat([pd.DataFrame({f"{sid}_EDA": e, f"{sid}_Label": l}) for sid, (e, l) in subjects.items()], axis=1)
    R = reshape_function(wide, name="EDA", window_size=win_sec, stride_size=stride_sec, hertz=fs,
                         purity_threshold=purity)
    W = R.filter(like="sample_").to_numpy(float)
    win, stride = int(win_sec * fs), int(stride_sec * fs)
    starts, pos = np.empty(len(R), int), {}
    for k, sid in enumerate(R.subject_id.values):             # windows are in time order within each subject
        eda = subjects[sid][0]
        j = pos.get(sid, 0)
        while not np.array_equal(eda[j:j + win], W[k]):
            j += stride
            assert j + win <= len(eda), "window not found in recording"
        starts[k], pos[sid] = j, j + stride
    keep = R.label.isin(labels).values
    meta = pd.DataFrame({"subject": R.subject_id.values, "label": R.label.astype(int).values, "start": starts})[keep]
    return W[keep], meta.reset_index(drop=True)


WINDOWINGS = {
    "reshape": "project reshape_function: slides from the start of each recording (default)",
    "block": "slides from the start of each condition block (eda_utils.make_windows)",
}
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def cache_for(windowing):
    """Cache folder of one windowing scheme: cache/reshape/ or cache/block/."""
    assert windowing in WINDOWINGS, windowing
    d = CACHE / windowing
    d.mkdir(parents=True, exist_ok=True)
    return d


def label_dir(windowing, label):
    """Per-label cache folder, e.g. cache/reshape/label3/. Everything cached here is computed window by
    window (windows, decompositions, features), so any set of labels is just a concatenation."""
    d = cache_for(windowing) / f"label{int(label)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(windowing, labels, merge=None):
    """Folder for results that depend on the classification task (grid screening, nested selection):
    cache/<windowing>/results/labels_1-2/, labels_1-2-3/, labels_1-2-3_3to1/ (with label merging), ..."""
    name = "labels_" + "-".join(str(int(l)) for l in labels)
    if merge:
        name += "_" + "_".join(f"{a}to{b}" for a, b in sorted(merge.items()))
    d = cache_for(windowing) / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_windows(windowing, W, meta, stride_sec=10):
    """Save windows split by label into cache/<windowing>/label<k>/ (meta: subject, label, start)."""
    for lab in sorted(meta.label.unique()):
        m = (meta.label == lab).values
        d = label_dir(windowing, lab)
        np.savez_compressed(d / f"windows_stride{stride_sec}.npz", W=W[m])
        meta.loc[m, ["subject", "label", "start"]].to_csv(d / f"windows_meta_stride{stride_sec}.csv", index=False)


def _window_order(windowing, meta):
    """Row order of the combined label sets = the order the windowing function itself produces:
    reshape_function: subject (sorted as text), then time; make_windows: subject number, label, time."""
    if windowing == "reshape":
        keys = (meta.start.values, meta.subject.values)
    else:
        keys = (meta.start.values, meta.label.values, meta.subject.str[1:].astype(int).values)
    return np.lexsort(keys)


def _label_parts(windowing, labels, stride_sec=10):
    parts = []
    for lab in labels:
        d = label_dir(windowing, lab)
        f = d / f"windows_stride{stride_sec}.npz"
        assert f.exists(), f"no windows for label {lab}: run notebook 01"
        parts.append((d, np.load(f)["W"], pd.read_csv(d / f"windows_meta_stride{stride_sec}.csv")))
    return parts


def load_windows(windowing, labels=(1, 2), stride_sec=10):
    """Windows and meta (subject, label, start) of the given labels, combined from the per-label caches."""
    parts = _label_parts(windowing, labels, stride_sec)
    W = np.concatenate([p[1] for p in parts])
    meta = pd.concat([p[2] for p in parts], ignore_index=True)
    o = _window_order(windowing, meta)
    return W[o], meta.iloc[o].reset_index(drop=True)


def per_label(windowing, labels, fn, stride_sec=10):
    """Run fn(W_label, label_dir) -> DataFrame for each label (fn does its own caching in label_dir) and
    combine the results in the same row order as load_windows."""
    parts = _label_parts(windowing, labels, stride_sec)
    X = pd.concat([fn(W_, d) for d, W_, _ in parts], ignore_index=True)
    meta = pd.concat([p[2] for p in parts], ignore_index=True)
    return X.iloc[_window_order(windowing, meta)].reset_index(drop=True)


def read_table(path):
    """Read a cached feature CSV with exact float round-trip (the default pandas parser can differ in the last
    digit, which would make results depend on whether a table was just computed or read from the cache)."""
    return pd.read_csv(path, float_precision="round_trip")


def cached_table(path, compute):
    """Read a cached feature CSV, or compute it (a DataFrame), save and read it back."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        compute().to_csv(path, index=False)
    return read_table(path)


def task_labels(meta, merge=None):
    """Class labels for the task: the WESAD labels, optionally merged (e.g. {3: 1} = amusement counted as
    non-stress together with baseline)."""
    return meta.label.replace(merge or {}).values


def make_unlabeled_windows(subjects, stride_sec=60, win=WIN, fs=FS):
    """1-min windows over each subject's WHOLE recording, ignoring labels.
    Used only for label-free per-subject normalisation statistics."""
    rows, W = [], []
    stride = int(stride_sec * fs)
    for sid, (eda, _) in subjects.items():
        for s in range(0, len(eda) - win + 1, stride):
            W.append(eda[s:s + win])
            rows.append((sid, -1, s))
    return np.asarray(W), pd.DataFrame(rows, columns=["subject", "label", "start"])


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------
def smooth_decompose_window(w, size_sec=4):
    """Simple reference decomposition (like neurokit2 'smoothmedian'):
    tonic = running median of width `size_sec`, phasic = w - tonic."""
    k = int(size_sec * FS) | 1
    pad = k // 2
    wp = np.pad(w, pad, mode="edge")
    tonic = np.array([np.median(wp[i:i + k]) for i in range(len(w))])
    return tonic, w - tonic


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _slope(x):
    t = np.arange(len(x)) / FS
    return np.polyfit(t, x, 1)[0]


def raw_features(w):
    """Statistical / morphological features of one EDA window (any scale)."""
    d = np.diff(w) * FS
    dd = np.diff(d) * FS
    q = np.percentile(w, [10, 25, 50, 75, 90])
    f = {
        "mean": w.mean(), "std": w.std(), "min": w.min(), "max": w.max(),
        "range": np.ptp(w), "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4],
        "iqr": q[3] - q[1], "skew": stats.skew(w), "kurt": stats.kurtosis(w),
        "slope": _slope(w), "first_last": w[-1] - w[0],
        "d_mean": d.mean(), "d_std": d.std(), "d_absmean": np.abs(d).mean(),
        "d_max": d.max(), "d_min": d.min(), "d_pos_sum": d[d > 0].sum() / FS,
        "d_neg_sum": d[d < 0].sum() / FS, "d_pos_frac": (d > 0).mean(),
        "dd_std": dd.std(), "dd_absmean": np.abs(dd).mean(),
        "line_len": np.abs(np.diff(w)).sum(),
    }
    # SCR-like peaks on the raw window (prominence relative to window range)
    pk, pr = sps.find_peaks(w, prominence=0.01 * max(np.ptp(w), 1e-6) + 1e-3)
    f["n_peaks"] = len(pk)
    f["peak_prom_mean"] = pr["prominences"].mean() if len(pk) else 0.0
    f["peak_prom_sum"] = pr["prominences"].sum() if len(pk) else 0.0
    # spectral power in SCR band
    fr, pxx = sps.welch(w - w.mean(), fs=FS, nperseg=min(len(w), 128))
    tot = pxx.sum() + 1e-12
    for lo, hi in [(0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 2.0)]:
        m = (fr >= lo) & (fr < hi)
        f[f"bp_{lo}_{hi}"] = pxx[m].sum() / tot
    return f


def phasic_tonic_features(tonic, phasic, driver=None, prefix="gms"):
    f = {}
    f[f"{prefix}_ton_mean"] = tonic.mean()
    f[f"{prefix}_ton_slope"] = _slope(tonic)
    f[f"{prefix}_ton_range"] = np.ptp(tonic)
    f[f"{prefix}_ton_std"] = tonic.std()
    f[f"{prefix}_ton_min"] = tonic.min()
    f[f"{prefix}_ph_mean"] = phasic.mean()
    f[f"{prefix}_ph_std"] = phasic.std()
    f[f"{prefix}_ph_max"] = phasic.max()
    f[f"{prefix}_ph_auc"] = np.abs(phasic).sum() / FS
    f[f"{prefix}_ph_frac"] = np.abs(phasic).sum() / (np.abs(tonic).sum() + 1e-9)
    pk, pr = sps.find_peaks(phasic, prominence=0.005)
    f[f"{prefix}_ph_npeaks"] = len(pk)
    f[f"{prefix}_ph_peakamp_mean"] = pr["prominences"].mean() if len(pk) else 0.0
    f[f"{prefix}_ph_peakamp_max"] = pr["prominences"].max() if len(pk) else 0.0
    if driver is not None:
        thr = 1e-3 * (driver.max() + 1e-12)
        nz = driver > max(thr, 1e-6)
        f[f"{prefix}_drv_sum"] = driver.sum()
        f[f"{prefix}_drv_max"] = driver.max()
        f[f"{prefix}_drv_nnz"] = nz.sum()
        f[f"{prefix}_drv_std"] = driver.std()
        # number of separate driver "events" (runs of non-zeros)
        f[f"{prefix}_drv_events"] = int(np.sum(np.diff(nz.astype(int)) == 1) + nz[0])
        big = driver > 0.1 * (driver.max() + 1e-12)
        f[f"{prefix}_drv_bigevents"] = int(np.sum(np.diff(big.astype(int)) == 1) + big[0])
        # temporal spread of driver mass
        if driver.sum() > 0:
            t = np.arange(len(driver))
            mu = (t * driver).sum() / driver.sum()
            f[f"{prefix}_drv_tspread"] = np.sqrt(((t - mu) ** 2 * driver).sum() / driver.sum()) / FS
        else:
            f[f"{prefix}_drv_tspread"] = 0.0
    return f


def subject_scale(subjects, kind="zscore"):
    """Label-free, per-subject normalisation constants computed from the subject's
    WHOLE recording (all conditions; no label information is used)."""
    out = {}
    for sid, (eda, _) in subjects.items():
        if kind == "zscore":
            out[sid] = (eda.mean(), eda.std())
        elif kind == "minmax":
            out[sid] = (eda.min(), np.ptp(eda))
        elif kind == "robust":
            q5, q50, q95 = np.percentile(eda, [5, 50, 95])
            out[sid] = (q50, q95 - q5)
        else:
            out[sid] = (0.0, 1.0)
    return out


def build_feature_table(W, meta, scale, D=None, T=None, P=None, sm=False):
    """Feature table for all windows.

    Every window (and its components) is first normalised with its subject's
    constants (loc, scale): x -> (x - loc)/scale for levels, x/scale for
    tonic-free quantities (phasic, driver). Features that depend on the signal
    level (mean, min, ...) therefore become comparable across subjects.
    """
    rows = []
    for i in range(len(W)):
        loc, sc = scale[meta.subject.iat[i]]
        w = (W[i] - loc) / sc
        f = {f"raw_{k}": v for k, v in raw_features(w).items()}
        if T is not None:
            f.update(phasic_tonic_features((T[i] - loc) / sc, P[i] / sc, D[i] / sc, prefix="gms"))
        if sm:
            t_, p_ = smooth_decompose_window(W[i])
            f.update(phasic_tonic_features((t_ - loc) / sc, p_ / sc, None, prefix="med"))
        rows.append(f)
    F = pd.DataFrame(rows)
    return F.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def loso_eval(model_fn, X, y, groups, return_pred=False):
    """Leave-one-subject-out CV. Returns per-subject accuracy / macro-F1 table and
    pooled accuracy, macro-F1, balanced accuracy."""
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    X = np.asarray(X, float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    pred = np.empty_like(y)
    per = []
    for g in pd.unique(groups):
        te = groups == g
        m = model_fn()
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
        per.append((g, accuracy_score(y[te], pred[te]),
                    f1_score(y[te], pred[te], average="macro")))
    per = pd.DataFrame(per, columns=["subject", "acc", "f1"])
    summ = dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"),
                bacc=balanced_accuracy_score(y, pred),
                subj_acc_mean=per.acc.mean(), subj_acc_std=per.acc.std())
    return (summ, per, pred) if return_pred else (summ, per)


# ---------------------------------------------------------------------------
# Goal-2 comparison helper
# ---------------------------------------------------------------------------
def default_models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    return {
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(C=1)),
        "RandomForest": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3, random_state=0),
    }


def compare_feature_sets(sets, y, groups, models=None, baseline="raw"):
    """LOSO for every {name: feature DataFrame} x model. Returns (results table, paired table vs baseline)."""
    from scipy.stats import wilcoxon
    models = models or default_models()
    rows, per = [], {}
    for fs, F in sets.items():
        for m, fn in models.items():
            s, p = loso_eval(fn, F.values, y, groups)
            rows.append(dict(features=fs, model=m, **s))
            per[(fs, m)] = p.set_index("subject").acc
    res = pd.DataFrame(rows)
    prow = []
    for fs in sets:
        if fs == baseline:
            continue
        for m in models:
            a, b = per[(fs, m)], per[(baseline, m)]
            d = (a - b).values
            pv = wilcoxon(a, b).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
            prow.append(dict(features=fs, model=m, delta=d.mean(), wins=int((d > 1e-9).sum()),
                             losses=int((d < -1e-9).sum()), p=pv))
    return res, pd.DataFrame(prow)


# ---------------------------------------------------------------------------
# src/eda_proj/gms_eda.gmsEDA : per-window decomposition with tunable
# num_of_chops / overlap / lam_scale / force_nn, NO per-subject normalisation
# ---------------------------------------------------------------------------
def gmsEDA_window(w, num_of_chops=5, overlap=0.8, lam_scale=3, force_nn=0):
    """eda_proj.gms_eda.gmsEDA on one 1-min window -> (driver, tonic, phasic=w-tonic)."""
    from eda_proj.gms_eda import gmsEDA
    w = np.asarray(w, float)
    X, B = gmsEDA(w.reshape(-1, 1), FS, num_of_chops, overlap, force_nn=force_nn, lam_scale=lam_scale)
    X, B = X.ravel(), B.ravel()
    return X, B, w - B


def gms_signed_features(w, driver, tonic, phasic, prefix="gms"):
    """Decomposition features in absolute units (µS) plus scale-free ratios.
    The driver of gmsEDA may be signed (force_nn=0): positive and negative parts are summarised separately."""
    pos = np.maximum(driver, 0)
    neg = np.maximum(-driver, 0)
    f = phasic_tonic_features(tonic, phasic, pos, prefix=prefix)
    lvl = abs(w.mean()) + 1e-6
    f[f"{prefix}_drv_neg_sum"] = neg.sum()
    f[f"{prefix}_drv_neg_nnz"] = int((neg > 1e-6).sum())
    # scale-free versions (independent of the subject's absolute skin-conductance level)
    f[f"{prefix}_rel_ton_range"] = np.ptp(tonic) / lvl
    f[f"{prefix}_rel_ton_slope"] = _slope(tonic) / lvl
    f[f"{prefix}_rel_ph_std"] = phasic.std() / lvl
    f[f"{prefix}_rel_ph_auc"] = np.abs(phasic).sum() / FS / lvl
    f[f"{prefix}_rel_drv_sum"] = pos.sum() / lvl
    f[f"{prefix}_rel_drv_max"] = pos.max() / lvl
    return f


def raw_features_abs(w):
    """Raw-signal features in µS plus the same scale-free ratios (no per-subject normalisation)."""
    f = raw_features(w)
    lvl = abs(w.mean()) + 1e-6
    for k in ["std", "range", "iqr", "slope", "first_last", "d_std", "d_absmean", "d_max", "d_min",
              "d_pos_sum", "d_neg_sum", "line_len", "peak_prom_sum", "peak_prom_mean"]:
        f[f"rel_{k}"] = f[k] / lvl
    return f


def _gms_cfg_worker(args):
    w, cfg = args
    D, T, P = gmsEDA_window(w, **cfg)
    return gms_signed_features(w, D, T, P)


def gms_config_features(W, cfg, n_jobs=None, cache_dir=None):
    """Decompose every window with one gmsEDA configuration and return its feature table.
    Only the features (small) are cached: cache/gmsfeat_c{chops}_o{overlap}_l{lam}_nn{force_nn}.csv"""
    from multiprocessing import Pool
    cache_dir = Path(cache_dir or CACHE)
    tag = f"c{cfg['num_of_chops']}_o{cfg['overlap']:g}_l{cfg['lam_scale']:g}_nn{cfg.get('force_nn', 0)}"
    f = cache_dir / f"gmsfeat_{tag}.csv"
    if f.exists():
        return read_table(f)
    import glob
    for pack in sorted(glob.glob(str(cache_dir / "gms_grid_features*.npz"))):   # packed grid configurations
        z = np.load(pack, allow_pickle=False)
        if f"{tag}__X" in z.files:
            return pd.DataFrame(z[f"{tag}__X"].astype(float), columns=list(z["columns"]))
    with Pool(n_jobs or os.cpu_count()) as p:
        rows = p.map(_gms_cfg_worker, [(w, cfg) for w in W], chunksize=32)
    F = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    F.to_csv(f, index=False)
    return F


def nested_select_loso(candidates, y, groups, select_fn, final_fns, inner_splits=5, n_jobs=2):
    """Nested LOSO with the feature set (= decomposition configuration) chosen INSIDE each outer fold.

    candidates : {name: feature DataFrame}  (include the no-decomposition option to let it win)
    select_fn  : model used for the inner selection (inner GroupKFold over the training subjects)
    final_fns  : {model name: model factory} refit on the chosen set and tested on the held-out subject
    Returns (summary table, chosen config per outer fold).
    """
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    y, groups = np.asarray(y), np.asarray(groups)
    names = list(candidates)
    mats = {k: candidates[k].values for k in names}

    def inner_score(name, tr):
        X, yt, gt = mats[name][tr], y[tr], groups[tr]
        acc = []
        for a, b in GroupKFold(inner_splits).split(X, yt, gt):
            acc.append(accuracy_score(yt[b], select_fn().fit(X[a], yt[a]).predict(X[b])))
        return np.mean(acc)

    preds = {m: np.empty_like(y) for m in final_fns}
    chosen = []
    for g in pd.unique(groups):
        te = groups == g
        tr = ~te
        sc = Parallel(n_jobs=n_jobs)(delayed(inner_score)(n, tr) for n in names)
        best = names[int(np.argmax(sc))]
        chosen.append((g, best, max(sc)))
        X = mats[best]
        for m, fn in final_fns.items():
            preds[m][te] = fn().fit(X[tr], y[tr]).predict(X[te])
    summ = pd.DataFrame({m: dict(acc=accuracy_score(y, p), f1=f1_score(y, p, average="macro"))
                         for m, p in preds.items()}).T
    return summ, pd.DataFrame(chosen, columns=["test_subject", "chosen", "inner_acc"])


def pack_grid_features(cache_dir=None, remove_csv=True, n_parts=3):
    """Pack gmsfeat_*.csv (one per gmsEDA configuration) together with any configurations already packed into
    gms_grid_features_part*.npz (fewer, smaller files). The CSVs are removed afterwards unless remove_csv=False.
    Returns the configuration tags."""
    import glob
    cache_dir = Path(cache_dir or CACHE)
    arrs, cols = {}, None
    for pack in sorted(glob.glob(str(cache_dir / "gms_grid_features_part*.npz"))):
        z = np.load(pack, allow_pickle=False)
        cols = list(z["columns"])
        arrs.update({k: z[k] for k in z.files if k.endswith("__X")})
    csvs = sorted(glob.glob(str(cache_dir / "gmsfeat_*.csv")))
    for f in csvs:
        F = read_table(f)
        cols = list(F.columns) if cols is None else cols
        assert list(F.columns) == cols
        arrs[Path(f).stem[len("gmsfeat_"):] + "__X"] = F.values
    keys = sorted(arrs)                                # split in parts < 30 MB each
    for i in range(n_parts):
        part = {k: arrs[k] for k in keys[i::n_parts]}
        np.savez_compressed(cache_dir / f"gms_grid_features_part{i + 1}.npz", columns=np.array(cols), **part)
    if remove_csv:
        for f in csvs:
            os.remove(f)
    return [k[:-3] for k in keys]


def gms_grid():
    """The 112 gmsEDA configurations of notebook 02: the full 5 chops x 4 overlaps x 5 lam_scale grid (force_nn=0)
    plus 12 refinements around the best region (more chops, overlap 0.95, lam 2, and force_nn=1)."""
    import itertools, re
    tags = [f"c{c}_o{o:g}_l{l:g}_nn0" for c, o, l in
            itertools.product([2, 3, 5, 10, 20], [0, .5, .8, .9], [0.3, 0.5, 1, 3, 10])]
    tags += ["c20_o0.8_l2_nn0", "c20_o0.95_l1_nn0", "c20_o0.95_l2_nn0", "c20_o0.9_l1_nn1", "c20_o0.9_l2_nn0",
             "c30_o0.8_l1_nn0", "c30_o0.9_l1_nn0", "c30_o0.9_l2_nn0", "c40_o0.8_l1_nn0", "c40_o0.8_l2_nn0",
             "c40_o0.9_l1_nn0", "c60_o0.8_l1_nn0"]
    rows = []
    for t in sorted(tags):
        m = re.match(r"c(\d+)_o([\d.]+)_l([\d.]+)_nn(\d)", t)
        rows.append(dict(tag=t, num_of_chops=int(m[1]), overlap=float(m[2]), lam_scale=float(m[3]), force_nn=int(m[4])))
    return pd.DataFrame(rows)


def grid_tags(cache_dir=None):
    """All gmsEDA configurations available in the cache, parsed into a DataFrame."""
    import glob, re
    cache_dir = Path(cache_dir or CACHE)
    tags = {Path(f).stem[len("gmsfeat_"):] for f in glob.glob(str(cache_dir / "gmsfeat_*.csv"))}
    for pack in glob.glob(str(cache_dir / "gms_grid_features*.npz")):
        tags |= {k[:-3] for k in np.load(pack).files if k.endswith("__X")}
    rows = []
    for t in sorted(tags):
        m = re.match(r"c(\d+)_o([\d.]+)_l([\d.]+)_nn(\d)", t)
        rows.append(dict(tag=t, num_of_chops=int(m[1]), overlap=float(m[2]), lam_scale=float(m[3]), force_nn=int(m[4])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NeuroKit2 eda_phasic re-implementation (used only when neurokit2 is not installed)
# ---------------------------------------------------------------------------
def nk_eda_phasic(signal, sampling_rate=FS, method="highpass"):
    """Line-for-line copy of neurokit2.eda_phasic for method='highpass' (NeuroKit2's default) and 'smoothmedian'.

    highpass:     signal_filter(..., method="butter", order=2) -> scipy butter(2, 0.05, sos) + sosfiltfilt;
                  phasic = high-pass at 0.05 Hz, tonic = low-pass at 0.05 Hz.
    smoothmedian: tonic = scipy.signal.medfilt(signal, 4*fs, made odd), phasic = signal - tonic.
    Returns (tonic, phasic).
    """
    x = np.asarray(signal, float)
    if method in ("highpass", "neurokit", "biopac", "acqknowledge"):
        hp = sps.butter(2, 0.05, btype="highpass", output="sos", fs=sampling_rate)
        lp = sps.butter(2, 0.05, btype="lowpass", output="sos", fs=sampling_rate)
        return sps.sosfiltfilt(lp, x), sps.sosfiltfilt(hp, x)
    if method in ("smoothmedian", "median"):
        size = int(4 * sampling_rate)
        if size % 2 == 0:
            size += 1
        tonic = sps.medfilt(x, kernel_size=size)
        return tonic, x - tonic
    raise ValueError(method)


# ---------------------------------------------------------------------------
# sparsEDA (Hernando-Gallego et al., 2018) — copy of neurokit2's _eda_phasic_sparsEDA
# (neurokit2/eda/eda_phasic.py, MIT licence; credits there to github.com/fhernandogallego/sparsEDA and
# github.com/yskong224/SparsEDA-python). Only change: the sparse driver, which neurokit2 computes but
# discards, is also returned (resampled to the input rate) so it can be used for features like gmsEDA's.
# ---------------------------------------------------------------------------
def _nk_signal_resample(signal, desired_length=(), sampling_rate=None, desired_sampling_rate=None):
    """neurokit2.signal_resample with its default method='interpolation' (scipy.ndimage.zoom)."""
    import scipy.ndimage
    if desired_length == ():  # noqa: F632
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))
    if len(signal) == desired_length:
        return signal
    return scipy.ndimage.zoom(signal, desired_length / len(signal))


def sparsEDA(eda_signal, sampling_rate=FS, epsilon=0.0001, Kmax=40, Nmin=5 / 4, rho=0.025, return_driver=True):
    """Returns (SCL = tonic, SCR = phasic[, driver]) exactly as neurokit2.eda_phasic(method='sparsEDA'),
    plus the post-processed sparse driver."""
    import scipy.linalg
    eda_signal = np.asarray(eda_signal, float)
    dmin = Nmin * sampling_rate
    original_length = len(eda_signal)
    if np.sum(np.isnan(eda_signal)) > 0:
        raise AssertionError("Signal contains NaN")

    eda_signal = _nk_signal_resample(eda_signal, sampling_rate=sampling_rate, desired_sampling_rate=8)
    new_sr = 8

    signalAdd = np.zeros(len(eda_signal) + (20 * new_sr) + (60 * new_sr))
    signalAdd[0: 20 * new_sr] = eda_signal[0]
    signalAdd[20 * new_sr: 20 * new_sr + len(eda_signal)] = eda_signal
    signalAdd[20 * new_sr + len(eda_signal):] = eda_signal[-1]

    Nss = len(eda_signal)
    Ns = len(signalAdd)
    b0 = 0
    pointerS = 20 * new_sr
    pointerE = pointerS + Nss

    durationR = 70
    Lreg = int(20 * new_sr * 3)
    L = 10 * new_sr
    N = durationR * new_sr
    T = 6

    Rzeros = np.zeros([N + L, Lreg * 5])
    srF = new_sr * np.array([0.5, 0.75, 1, 1.25, 1.5])
    for j in range(0, len(srF)):
        t_rf = np.arange(0, 10 + 1e-10, 1 / srF[j])
        taus = np.array([0.5, 2, 1])
        rf_biexp = np.exp(-t_rf / taus[1]) - np.exp(-t_rf / taus[0])
        rf_est = taus[2] * rf_biexp
        rf_est = rf_est / np.sqrt(np.sum(rf_est ** 2))
        rf_est_zeropad = np.zeros(len(rf_est) + (N - len(rf_est)))
        rf_est_zeropad[: len(rf_est)] = rf_est
        Rzeros[0:N, j * Lreg: (j + 1) * Lreg] = scipy.linalg.toeplitz(rf_est_zeropad, np.zeros(Lreg))

    R0 = Rzeros[0:N, 0: 5 * Lreg]
    R = np.zeros([N, T + Lreg * 5])
    R[0:N, T:] = R0
    R[0:Lreg, 0] = np.linspace(0, 1, Lreg)
    R[0:Lreg, 1] = -np.linspace(0, 1, Lreg)
    R[int(Lreg / 3): Lreg, 2] = np.linspace(0, 2 / 3, int((2 * Lreg) / 3))
    R[int(Lreg / 3): Lreg, 3] = -np.linspace(0, 2 / 3, int((2 * Lreg) / 3))
    R[int(2 * Lreg / 3): Lreg, 4] = np.linspace(0, 1 / 3, int(Lreg / 3))
    R[int(2 * Lreg / 3): Lreg, 5] = -np.linspace(0, 1 / 3, int(Lreg / 3))
    Cte = np.sum(R[:, 0] ** 2)
    R[:, 0:6] = R[:, 0:6] / np.sqrt(Cte)

    cutS = 0
    cutE = N
    slcAux = np.zeros(Ns)
    driverAux = np.zeros(Ns)
    resAux = np.zeros(Ns)
    while cutE < Ns:
        signalCut = signalAdd[cutS:cutE]
        if b0 == 0:
            b0 = signalCut[0]
        signalCutIn = signalCut - b0
        beta, _, _, _, _, _ = _sparsEDA_lasso(R, signalCutIn, new_sr, Kmax, epsilon)
        signalEst = (np.matmul(R, beta) + b0).reshape(-1)
        remAout = (signalCut - signalEst) ** 2
        res2 = np.sum(remAout[20 * new_sr: 40 * new_sr])
        res3 = np.sum(remAout[40 * new_sr: 60 * new_sr])
        jump = 1
        if res2 < 1:
            jump = 2
            if res3 < 1:
                jump = 3
        SCL = np.matmul(R[:, 0:6], beta[0:6, :]) + b0
        SCRline = beta[6:, :]
        SCRaux = np.zeros([Lreg, 5])
        SCRaux[:] = SCRline.reshape([5, Lreg]).transpose()
        driver = SCRaux.sum(axis=1)
        b0 = np.matmul(R[jump * 20 * new_sr - 1, 0:6], beta[0:6, :]) + b0
        driverAux[cutS: cutS + (jump * 20 * new_sr)] = driver[0: jump * new_sr * 20]
        slcAux[cutS: cutS + (jump * 20 * new_sr)] = SCL[0: jump * new_sr * 20].reshape(-1)
        resAux[cutS: cutS + (jump * 20 * new_sr)] = remAout[0: jump * new_sr * 20]
        cutS = cutS + jump * 20 * new_sr
        cutE = cutS + N

    SCRaux = driverAux[pointerS:pointerE]
    SCL = slcAux[pointerS:pointerE]

    # post-processing of the driver (kept from neurokit2; there the result is not returned)
    driver = np.zeros(len(SCRaux))
    ind = np.argwhere(SCRaux > 0).reshape(-1)
    if len(ind) > 0:
        scr_temp = SCRaux[ind]
        ind2 = np.argsort(scr_temp)[::-1]
        scr_ord = scr_temp[ind2]
        scr_fin = [scr_ord[0]]
        ind_fin = [ind[ind2[0]]]
        for i in range(1, len(ind2)):
            if np.all(np.abs(ind[ind2[i]] - ind_fin) >= dmin):
                scr_fin.append(scr_ord[i])
                ind_fin.append(ind[ind2[i]])
        driver[np.array(ind_fin)] = np.array(scr_fin)
        driver[driver < rho * scr_fin[0]] = 0

    SCR = eda_signal - SCL
    SCR = _nk_signal_resample(SCR, desired_length=original_length)
    SCL = _nk_signal_resample(SCL, desired_length=original_length)
    if not return_driver:
        return SCL, SCR
    # driver: 8 Hz impulses -> input rate by summing the impulses that fall into each input sample
    pos = np.minimum((np.arange(len(driver)) * original_length / len(driver)).astype(int), original_length - 1)
    drv = np.zeros(original_length)
    np.add.at(drv, pos, driver)
    return SCL, SCR, drv


def _sparsEDA_lasso(R, s, sampling_rate, maxIters, epsilon):
    """neurokit2 eda_phasic.lasso (non-negative LARS used by sparsEDA), copied verbatim."""
    import scipy.linalg
    N = len(s)
    W = R.shape[1]
    OptTol = -10
    solFreq = 0
    resStop2 = 0.0005
    lmbdaStop = 0
    zeroTol = 1e-5
    x = np.zeros(W)
    x_old = np.zeros(W)
    iter = 0
    c = np.matmul(R.transpose(), s.reshape(-1, 1)).reshape(-1)
    lmbda = np.max(c)
    if lmbda < 0:
        raise Exception("y is not expressible as a non-negative linear combination of the columns of X")
    newIndices = np.argwhere(np.abs(c - lmbda) < zeroTol).flatten()
    collinearIndices = []
    beta = []
    duals = []
    res = s
    R_I = []
    activeSet = []
    for j in range(0, len(newIndices)):
        iter = iter + 1
        R_I, flag = _sparsEDA_updateChol(R_I, N, W, R, 1, activeSet, newIndices[j], zeroTol)
        activeSet.append(newIndices[j])
    activationHist = activeSet.copy()

    done = 0
    while done == 0:
        if len(activationHist) == 4:
            lmbda = np.max(c)
            newIndices = np.argwhere(np.abs(c - lmbda) < zeroTol).flatten()
            activeSet = []
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = _sparsEDA_updateChol(R_I, N, W, R, 1, activeSet, newIndices[j], zeroTol)
                activeSet.append(newIndices[j])
            [activationHist.append(ele) for ele in activeSet]
        else:
            lmbda = c[activeSet[0]]

        dx = np.zeros(W)
        if len(np.array([R_I]).flatten()) == 1:
            z = scipy.linalg.solve(R_I.reshape([-1, 1]), np.sign(c[np.array(activeSet).flatten()].reshape(-1, 1)),
                                   transposed=True, lower=False)
        else:
            z = scipy.linalg.solve(R_I, np.sign(c[np.array(activeSet).flatten()].reshape(-1, 1)),
                                   transposed=True, lower=False)
        if len(np.array([R_I]).flatten()) == 1:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(R_I.reshape([-1, 1]), z, transposed=False, lower=False)
        else:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(R_I, z, transposed=False, lower=False).flatten()

        v = np.matmul(R[:, np.array(activeSet).flatten()], dx[np.array(activeSet).flatten()].reshape(-1, 1))
        ATv = np.matmul(R.transpose(), v).flatten()
        gammaI = np.inf
        removeIndices = []
        inactiveSet = np.arange(0, W)
        if len(np.array(activeSet).flatten()) > 0:
            inactiveSet[np.array(activeSet).flatten()] = -1
        if len(np.array(collinearIndices).flatten()) > 0:
            inactiveSet[np.array(collinearIndices).flatten()] = -1
        inactiveSet = np.argwhere(inactiveSet >= 0).flatten()
        if len(inactiveSet) == 0:
            gammaIc = 1
            newIndices = []
        else:
            epsilon = 1e-12
            gammaArr = (lmbda - c[inactiveSet]) / (1 - ATv[inactiveSet] + epsilon)
            gammaArr[gammaArr < zeroTol] = np.inf
            gammaIc = np.min(gammaArr)
            newIndices = inactiveSet[(np.abs(gammaArr - gammaIc) < zeroTol)]
        gammaMin = min(gammaIc, gammaI)
        x = x + gammaMin * dx
        res = res - gammaMin * v.flatten()
        c = c - gammaMin * ATv
        if (((lmbda - gammaMin) < OptTol) or ((lmbdaStop > 0) and (lmbda <= lmbdaStop))
                or ((epsilon > 0) and (np.linalg.norm(res) <= epsilon))):
            newIndices = []
            removeIndices = []
            done = 1
        if np.linalg.norm(res[0: sampling_rate * 20]) <= resStop2:
            done = 1
        if gammaIc <= gammaI and len(newIndices) > 0:
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = _sparsEDA_updateChol(R_I, N, W, R, 1, np.array(activeSet).flatten(), newIndices[j], zeroTol)
                if flag:
                    collinearIndices.append(newIndices[j])
                else:
                    activeSet.append(newIndices[j])
                    activationHist.append(newIndices[j])
        if iter >= maxIters:
            done = 1
        if len(np.argwhere(x < 0).flatten()) > 0:
            x = x_old.copy()
            done = 1
        else:
            x_old = x.copy()
        if done or ((solFreq > 0) and not (iter % solFreq)):
            beta.append(x)
            duals.append(v)
    return np.array(beta).reshape(-1, 1), iter, activationHist, duals, lmbda, res


def _sparsEDA_updateChol(R_I, n, N, R, explicitA, activeSet, newIndex, zeroTol):
    """neurokit2 eda_phasic.updateChol (explicitA=1 branch), copied verbatim."""
    import scipy.linalg
    flag = 0
    newVec = R[:, newIndex]
    if len(activeSet) == 0:
        R_I0 = np.sqrt(np.sum(newVec ** 2))
    else:
        if len(np.array([R_I]).flatten()) == 1:
            p = scipy.linalg.solve(np.array(R_I).reshape(-1, 1), np.matmul(R[:, activeSet].transpose(), R[:, newIndex]),
                                   transposed=True, lower=False)
        else:
            p = scipy.linalg.solve(R_I, np.matmul(R[:, activeSet].transpose(), R[:, newIndex]),
                                   transposed=True, lower=False)
        q = np.sum(newVec ** 2) - np.sum(p ** 2)
        if q <= zeroTol:
            flag = 1
            R_I0 = R_I.copy()
        else:
            if len(np.array([R_I]).shape) == 1:
                R_I = np.array([R_I]).reshape(-1, 1)
            R_I0 = np.zeros([np.array(R_I).shape[0] + 1, R_I.shape[1] + 1])
            R_I0[0: R_I.shape[0], 0: R_I.shape[1]] = R_I
            R_I0[0: R_I.shape[0], -1] = p
            R_I0[-1, -1] = np.sqrt(q)
    return R_I0, flag


def _sparsEDA_worker(w):
    return sparsEDA(w)


def sparsEDA_windows(W, n_jobs=None):
    """sparsEDA on every 1-min window (parallel). Returns tonic, phasic, driver arrays."""
    from multiprocessing import Pool
    with Pool(n_jobs or os.cpu_count()) as p:
        res = p.map(_sparsEDA_worker, list(W), chunksize=16)
    return tuple(np.stack([r[j] for r in res]) for j in range(3))


# ---------------------------------------------------------------------------
# Feature set of python_code/Statistical_Feature_Extraction.ipynb, used through
# eda_proj.features.sig2features (moved there unchanged from that notebook)
# ---------------------------------------------------------------------------
def windows_to_frame(A, meta=None):
    """Windows (n, 240) -> the DataFrame layout reshape_function returns (subject_id, label, sample_0..)."""
    df = pd.DataFrame(np.asarray(A, float), columns=[f"sample_{i}" for i in range(np.shape(A)[1])])
    df.insert(0, "label", meta.label.values if meta is not None else 0)
    df.insert(0, "subject_id", meta.subject.values if meta is not None else "")
    return df


def _sig2features_worker(args):
    import warnings
    warnings.simplefilter("ignore")
    from eda_proj.features import sig2features
    A, name = args
    return sig2features(windows_to_frame(A), name).drop(columns=["label", "subject_id"])


def sig2features_parallel(A, name, n_jobs=None, chunk=100):
    """eda_proj.features.sig2features(windows, name), run on chunks of rows in parallel."""
    from multiprocessing import Pool
    A = np.asarray(A, float)
    parts = [(A[i:i + chunk], name) for i in range(0, len(A), chunk)]
    with Pool(n_jobs or os.cpu_count()) as p:
        out = p.map(_sig2features_worker, parts)
    return pd.concat(out, ignore_index=True)


def original_feature_table(W, T=None, P=None, D=None, n_jobs=None, cache=None):
    """Features of Statistical_Feature_Extraction.ipynb, computed with its own sig2features():
    eda_* (12, raw window) + phasic_* (12 descriptives + 12 SCR/spectral via eda_feature_extraction)
    + tonic_* (12) -> the 48 columns of feature_dataframe.csv. With T=P=None only the 12 eda_* columns;
    with D given also 12 driver_* descriptives (an extension, not in the original notebook)."""
    if cache is not None and Path(cache).exists():
        return read_table(cache)
    parts = [sig2features_parallel(W, "eda", n_jobs)]
    if P is not None:
        parts += [sig2features_parallel(P, "phasic", n_jobs), sig2features_parallel(T, "tonic", n_jobs)]
    if D is not None:
        parts.append(sig2features_parallel(D, "driver", n_jobs))
    F = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        F.to_csv(cache, index=False)
    return F


def models_with_imputer():
    """default_models() with median imputation (fitted inside each training fold), needed because the
    SCR features are NaN for windows without detected peaks."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    return {k: (lambda f=f: make_pipeline(SimpleImputer(strategy="median"), f())) for k, f in default_models().items()}


def _gms_decomp_worker(args):
    w, cfg = args
    return gmsEDA_window(w, **cfg)


def gmsEDA_decompose_windows(W, cfg, n_jobs=None):
    """gmsEDA on every window for one configuration -> (driver, tonic, phasic) arrays."""
    from multiprocessing import Pool
    with Pool(n_jobs or os.cpu_count()) as p:
        res = p.map(_gms_decomp_worker, [(w, cfg) for w in W], chunksize=32)
    return tuple(np.stack([r[j] for r in res]) for j in range(3))


# ---------------------------------------------------------------------------
# Per-label cached feature tables (used by notebooks 02-05)
# ---------------------------------------------------------------------------
def nk_tonic_phasic(W, method):
    """NeuroKit2 eda_phasic (real neurokit2 if installed, otherwise the identical copy) on every window."""
    try:
        import neurokit2 as nk
        out = [nk.eda_phasic(w, sampling_rate=FS, method=method).values.T for w in W]
    except ImportError:
        out = [np.vstack(nk_eda_phasic(w, method=method)) for w in W]
    T, P = map(np.array, zip(*out))
    return T, P


def decomp_feature_table(W, method):
    """The gms_signed_features of notebooks 02-03 applied to another decomposition of each window:
    method = 'nk_highpass', 'nk_median' (NeuroKit2 smoothmedian; no driver, so driver features are dropped)
    or 'sparsEDA' (with its sparse driver)."""
    if method == "sparsEDA":
        T, P, D = sparsEDA_windows(W)
        rows = [gms_signed_features(w, d, t, p, prefix="sparsEDA") for w, t, p, d in zip(W, T, P, D)]
    else:
        T, P = nk_tonic_phasic(W, {"nk_highpass": "highpass", "nk_median": "smoothmedian"}[method])
        rows = [{k: v for k, v in gms_signed_features(w, np.zeros_like(w), t, p, prefix=method).items() if "drv" not in k}
                for w, t, p in zip(W, T, P)]
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def own_features(windowing, labels, kind):
    """Feature tables of notebooks 02-03 for the given labels, cached per label in cache/<windowing>/label<k>/:
    kind = 'raw' (33 + 14 rel_ columns), 'med' (running-median control), 'nk_highpass', 'nk_median', 'sparsEDA',
    or a gmsEDA configuration tag such as 'c20_o0.9_l1_nn0'."""
    import re

    def fn(W_, d):
        if kind == "raw":
            return cached_table(d / "rawfeat_abs.csv", lambda: pd.DataFrame([raw_features_abs(w) for w in W_])
                                .replace([np.inf, -np.inf], np.nan).fillna(0))
        if kind == "med":
            def med():
                rows = []
                for w in W_:
                    ton, ph = smooth_decompose_window(w)
                    rows.append(gms_signed_features(w, np.zeros_like(w), ton, ph, prefix="med"))
                return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
            return cached_table(d / "medfeat_abs.csv", med)
        if kind in ("nk_highpass", "nk_median", "sparsEDA"):
            return cached_table(d / f"decompfeat_{kind}.csv", lambda: decomp_feature_table(W_, kind))
        m = re.fullmatch(r"c(\d+)_o([\d.]+)_l([\d.]+)_nn(\d)", kind)
        assert m, kind
        cfg = dict(num_of_chops=int(m[1]), overlap=float(m[2]), lam_scale=float(m[3]), force_nn=int(m[4]))
        return gms_config_features(W_, cfg, cache_dir=d)
    return per_label(windowing, labels, fn)


def orig_features(windowing, labels, kind, n_jobs=None):
    """Original feature set (Statistical_Feature_Extraction.ipynb, via eda_proj.features.sig2features) for the
    given labels, cached per label in cache/<windowing>/label<k>/origfeat/<kind>.csv. kind = 'raw',
    'nk_highpass', 'nk_smoothmedian', 'sparsEDA', 'sparsEDA_drv', 'gms_c20_o0.9_l1' or 'gms_c20_o0.9_l1_drv'
    (driver descriptives added). A decomposition is computed only when its table is missing; the tables with and
    without driver are then written together."""
    import re

    def fn(W_, d):
        fd = d / "origfeat"
        f = fd / f"{kind}.csv"
        if f.exists():
            return read_table(f)
        base = kind[:-4] if kind.endswith("_drv") else kind
        if base == "raw":
            return original_feature_table(W_, n_jobs=n_jobs, cache=f)
        if base.startswith("nk_"):
            T, P = nk_tonic_phasic(W_, base[3:])
            return original_feature_table(W_, T, P, n_jobs=n_jobs, cache=f)
        if base == "sparsEDA":
            T, P, D = sparsEDA_windows(W_)
        else:
            m = re.fullmatch(r"gms_c(\d+)_o([\d.]+)_l([\d.]+)", base)
            assert m, kind
            D, T, P = gmsEDA_decompose_windows(W_, dict(num_of_chops=int(m[1]), overlap=float(m[2]),
                                                        lam_scale=float(m[3]), force_nn=0))
        original_feature_table(W_, T, P, n_jobs=n_jobs, cache=fd / f"{base}.csv")
        original_feature_table(W_, T, P, D, n_jobs=n_jobs, cache=fd / f"{base}_drv.csv")
        return read_table(f)
    return per_label(windowing, labels, fn)


def compare_feature_sets_cached(sets, y, groups, cache_prefix, models=None, baseline="raw"):
    """compare_feature_sets with its two result tables cached as <cache_prefix>_res.csv / _paired.csv
    (recomputed when the feature-set names differ)."""
    cache_prefix = Path(cache_prefix)
    fr, fp = Path(f"{cache_prefix}_res.csv"), Path(f"{cache_prefix}_paired.csv")
    if fr.exists() and fp.exists():
        res, ptab = pd.read_csv(fr), pd.read_csv(fp)
        if list(res.features.unique()) == list(sets):
            return res, ptab
    res, ptab = compare_feature_sets(sets, y, groups, models=models, baseline=baseline)
    fr.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(fr, index=False); ptab.to_csv(fp, index=False)
    return res, ptab
