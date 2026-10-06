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
        return pd.read_csv(f)
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


def pack_grid_features(cache_dir=None):
    """Pack all cache/gmsfeat_*.csv into cache/gms_grid_features_part*.npz and return the config tags."""
    import glob
    cache_dir = Path(cache_dir or CACHE)
    arrs, cols = {}, None
    for f in sorted(glob.glob(str(cache_dir / "gmsfeat_*.csv"))):
        F = pd.read_csv(f)
        cols = list(F.columns) if cols is None else cols
        assert list(F.columns) == cols
        arrs[Path(f).stem[len("gmsfeat_"):] + "__X"] = F.values
    keys = sorted(arrs)                                # split in parts < 30 MB each
    n_parts = 3
    for i in range(n_parts):
        part = {k: arrs[k] for k in keys[i::n_parts]}
        np.savez_compressed(cache_dir / f"gms_grid_features_part{i + 1}.npz", columns=np.array(cols), **part)
    return [k[:-3] for k in arrs]


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
