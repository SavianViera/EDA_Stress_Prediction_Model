"""
Hand-Crafted Statistical Feature Extraction (Table 3, Birkenmaier et al., 2026)
=================================================================================
Extracts the paper's 20 statistical EDA features per window, to be fused
later with the 32-dim supervised-autoencoder latent vector into a 52-dim
hybrid feature vector.

Pipeline:
  1. Decompose each subject's RAW (non-z-scored) wrist EDA trace into tonic
     (SCL) and phasic (SCR) components via cvxEDA (through neurokit2).
  2. Slice the SAME window boundaries used for the autoencoder (same start
     indices, same 10s window / 2s stride / baseline-stress purity filter)
     out of the decomposed tonic/phasic traces.
  3. Compute 20 features per window from those slices.

Deviations from the paper, stated explicitly:
  - The paper's chest sensor pipeline tuned cvxEDA's internal parameters
    (tau0, tau1, alpha, knot spacing) for a 20 Hz downsampled signal; here
    we use neurokit2's default cvxEDA parameters on 4 Hz wrist data, since
    the paper's specific adapted values were derived for a different
    sensor/sampling-rate combination. If features look off, this is the
    first place to tune.
  - `signal_snr` is approximated as the ratio of a smoothed-signal power to
    residual (raw - smoothed) power, since we don't have a raw high-rate
    reference signal for the E4 (unlike the paper's 700 Hz chest sensor,
    which could be compared against its own downsampled version).
  - `movement_ratio` is approximated via a MAD-based outlier rate on the
    first-difference signal, rather than the paper's full four-category
    (step/needle/flatline/wall) artifact detector, which was designed for
    the chest sensor and hasn't been rebuilt here for the wrist channel.
  - Peak detection thresholds follow the paper's Table 3 caption (0.05 uS
    height) combined with the 1s minimum peak separation mentioned in
    Section 2.5.1.
"""

import numpy as np
from scipy.signal import find_peaks, welch

from neurokit2.eda.eda_phasic import _eda_phasic_cvxeda

# np.trapz was removed in numpy 2.0+ (renamed to np.trapezoid). Support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

FEATURE_NAMES = [
    # SCR (phasic) group
    "scr_mean", "scr_std", "scr_max", "scr_min",
    "scr_peak_count", "scr_peak_mean_amp", "scr_peak_max_amp",
    "scr_peaks_per_min", "scr_avg_rise_time", "scr_auc",
    # SCL (tonic) group
    "scl_mean", "scl_std", "scl_max", "scl_min",
    "scl_slope", "scl_range", "scl_variation",
    # Quality
    "signal_snr", "movement_ratio",
    # Spectral
    "eda_spectral_power_low",
]
assert len(FEATURE_NAMES) == 20


def decompose_subject_eda(raw_eda: np.ndarray, fs: int = 4, tau0: float = 2.0,
                           tau1: float = 0.7, delta_knot: float = 10.0,
                           alpha: float = 8e-4):
    """
    Decomposes one subject's full raw (non-z-scored) EDA trace into tonic
    (SCL) and phasic (SCR) components via cvxEDA. Run once per subject on
    the full trace -- NOT per-window -- since cvxEDA needs enough context
    to separate slow tonic drift from fast phasic bursts.

    tau0/tau1/delta_knot/alpha default to the underlying cvxEDA function's
    own defaults, which are the paper's *lab-data, pre-adaptation* values
    (Table 2's "Original" column) -- i.e. NOT adapted for any
    downsampled/lower-rate signal. Override these with values found via
    eda_cvxeda_search.py for a fit tuned to your actual sampling rate.

    NOTE: calls neurokit2's internal _eda_phasic_cvxeda directly rather
    than the public nk.eda_phasic(method="cvxeda", ...) wrapper. Confirmed
    via direct testing that nk.eda_phasic's dispatch branch for the cvxeda
    method does NOT forward **kwargs to the underlying function (unlike
    its other methods, e.g. "median", which do) -- so tau0/tau1/delta_knot
    /alpha passed through the public API are silently ignored and the
    decomposition always runs with hardcoded defaults regardless of what's
    requested. This bypasses that bug entirely.

    Returns:
        tonic: np.ndarray, same length as raw_eda
        phasic: np.ndarray, same length as raw_eda
    """
    tonic, phasic = _eda_phasic_cvxeda(np.asarray(raw_eda, dtype=np.float64),
                                        sampling_rate=fs, tau0=tau0, tau1=tau1,
                                        delta_knot=delta_knot, alpha=alpha)
    return np.asarray(tonic), np.asarray(phasic)


def _snr_estimate(window: np.ndarray, smooth_kernel: int = 3) -> float:
    """
    Approximate SNR (dB): power of a smoothed version of the window vs.
    power of the (raw - smoothed) residual. See module docstring for why
    this differs from the paper's approach.
    """
    if len(window) < smooth_kernel:
        return 0.0
    kernel = np.ones(smooth_kernel) / smooth_kernel
    smoothed = np.convolve(window, kernel, mode="same")
    residual = window - smoothed
    signal_power = np.mean(smoothed ** 2)
    noise_power = np.mean(residual ** 2) + 1e-12
    return float(10 * np.log10(signal_power / noise_power + 1e-12))


def _movement_ratio_estimate(window: np.ndarray, mad_factor: float = 5.0) -> float:
    """
    Approximate movement-artifact ratio: fraction of first-difference
    samples whose magnitude exceeds mad_factor * MAD(first differences).
    See module docstring for how this differs from the paper's full
    four-category artifact detector.
    """
    if len(window) < 2:
        return 0.0
    diffs = np.diff(window)
    mad = np.median(np.abs(diffs - np.median(diffs))) + 1e-12
    threshold = mad_factor * mad
    return float(np.mean(np.abs(diffs) > threshold))


def adaptive_peak_threshold(scr_window: np.ndarray, k: float = 1.0,
                             min_threshold: float = 0.01) -> float:
    """
    Adapts Section 2.4's "adaptive, condition-aware" peak threshold
    (Table 2's fifth adapted parameter: fixed 0.05 uS -> adaptive).

    IMPORTANT deviation from a literal reading of the paper: the paper
    states this threshold is "conditioned on the experimental label
    (stress vs. non-stress)" -- taken literally, that means using the
    ground-truth label to help compute a feature that then predicts that
    same label, which is leakage (and leakage that would corrupt even
    held-out LOSO test windows, since a real deployment has no ground
    truth to condition on). This implementation instead computes the
    threshold from ONLY the window's own phasic signal (rolling std within
    the window, matching the paper's other stated criterion), with no
    label lookup at all. Since stress windows are physiologically expected
    to have higher phasic variability than baseline windows anyway (the
    paper's own numbers: stress amplitude 0.691+/-0.697 uS vs. baseline
    0.360+/-0.289 uS), this naturally produces a threshold that ends up
    higher for stress-like windows and lower for baseline-like windows --
    the same qualitative outcome the paper describes -- without ever using
    the ground truth to get there.

    Returns a single float threshold (in the phasic signal's units) for
    this window's peak detection.
    """
    rolling_std = float(np.std(scr_window))
    threshold = k * rolling_std
    return max(threshold, min_threshold)


def extract_window_features(scl_window: np.ndarray, scr_window: np.ndarray,
                             fs: int = 4, peak_height: float = 0.05,
                             min_peak_distance_sec: float = 1.0,
                             use_adaptive_threshold: bool = True,
                             adaptive_threshold_k: float = 1.0) -> np.ndarray:
    """
    Computes the 20 hand-crafted features (Table 3) for one window, given
    the ALREADY-DECOMPOSED tonic (scl_window) and phasic (scr_window)
    slices for that window (same length, raw physiological units).

    use_adaptive_threshold: if True (default), peak detection uses
    adaptive_peak_threshold(scr_window, k=adaptive_threshold_k) instead of
    the fixed `peak_height` value -- see that function's docstring for the
    leakage-safe interpretation used here. Set False to fall back to the
    fixed threshold (e.g. for direct comparison against earlier results).

    Returns a (20,) float32 array in the order given by FEATURE_NAMES.
    """
    scl = np.asarray(scl_window, dtype=np.float64)
    scr = np.asarray(scr_window, dtype=np.float64)
    n = len(scr)
    dt = 1.0 / fs

    # --- SCR (phasic) features ---
    scr_mean = float(np.mean(scr))
    scr_std = float(np.std(scr))
    scr_max = float(np.max(scr))
    scr_min = float(np.min(scr))

    min_distance = max(1, int(min_peak_distance_sec * fs))
    effective_peak_height = (
        adaptive_peak_threshold(scr, k=adaptive_threshold_k) if use_adaptive_threshold
        else peak_height
    )
    peak_idx, _ = find_peaks(scr, height=effective_peak_height, distance=min_distance)

    if len(peak_idx) > 0:
        peak_amps = scr[peak_idx]
        scr_peak_count = float(len(peak_idx))
        scr_peak_mean_amp = float(np.mean(peak_amps))
        scr_peak_max_amp = float(np.max(peak_amps))
        window_minutes = (n * dt) / 60.0
        scr_peaks_per_min = float(len(peak_idx) / window_minutes) if window_minutes > 0 else 0.0

        rise_times = []
        for p in peak_idx:
            # Search backward from the peak for the local minimum (onset),
            # bounded by the previous peak or the start of the window.
            search_start = 0
            prior_peaks = peak_idx[peak_idx < p]
            if len(prior_peaks) > 0:
                search_start = prior_peaks[-1]
            segment = scr[search_start:p + 1]
            if len(segment) > 1:
                onset_offset = int(np.argmin(segment))
                onset_idx = search_start + onset_offset
                rise_times.append((p - onset_idx) * dt)
        scr_avg_rise_time = float(np.mean(rise_times)) if rise_times else 0.0
    else:
        scr_peak_count = 0.0
        scr_peak_mean_amp = 0.0
        scr_peak_max_amp = 0.0
        scr_peaks_per_min = 0.0
        scr_avg_rise_time = 0.0

    scr_auc = float(_trapz(np.clip(scr, 0, None), dx=dt))

    # --- SCL (tonic) features ---
    scl_mean = float(np.mean(scl))
    scl_std = float(np.std(scl))
    scl_max = float(np.max(scl))
    scl_min = float(np.min(scl))
    if n > 1:
        time_axis = np.arange(n) * dt
        scl_slope = float(np.polyfit(time_axis, scl, 1)[0])
    else:
        scl_slope = 0.0
    scl_range = float(scl_max - scl_min)
    scl_variation = float(np.std(np.diff(scl))) if n > 1 else 0.0

    # --- Quality features ---
    signal_snr = _snr_estimate(scr + scl)
    movement_ratio = _movement_ratio_estimate(scr + scl)

    # --- Spectral feature ---
    nperseg = min(n, 64)  # longer windows (60s) now allow finer PSD resolution
    if n >= 4:
        freqs, psd = welch(scr + scl, fs=fs, nperseg=nperseg)
        band_mask = freqs <= 0.5
        eda_spectral_power_low = float(_trapz(psd[band_mask], freqs[band_mask])) if band_mask.any() else 0.0
    else:
        eda_spectral_power_low = 0.0

    features = np.array([
        scr_mean, scr_std, scr_max, scr_min,
        scr_peak_count, scr_peak_mean_amp, scr_peak_max_amp,
        scr_peaks_per_min, scr_avg_rise_time, scr_auc,
        scl_mean, scl_std, scl_max, scl_min,
        scl_slope, scl_range, scl_variation,
        signal_snr, movement_ratio,
        eda_spectral_power_low,
    ], dtype=np.float32)

    return features


def extract_features_for_subject(raw_eda: np.ndarray, window_starts, raw_win_len: int,
                                  fs: int = 4, tau0: float = 2.0, tau1: float = 0.7,
                                  delta_knot: float = 10.0, alpha: float = 8e-4,
                                  use_adaptive_threshold: bool = True,
                                  adaptive_threshold_k: float = 1.0) -> np.ndarray:
    """
    Convenience wrapper: decomposes one subject's full raw EDA trace once,
    then extracts the 20 features for each of that subject's windows.

    Args:
        raw_eda: full raw (non-z-scored) EDA trace for one subject.
        window_starts: iterable of window start sample indices (same
            indices used to build the autoencoder's windows for this
            subject, i.e. same `start` values from the range(...) loop in
            the windowing function).
        raw_win_len: window length in samples (e.g. 40 for 10s @ 4Hz).
        fs: sampling rate in Hz.
        tau0/tau1/delta_knot/alpha: cvxEDA decomposition parameters, passed
            through to decompose_subject_eda. See that function's docstring.
        use_adaptive_threshold/adaptive_threshold_k: passed through to
            extract_window_features -- see adaptive_peak_threshold's
            docstring for the leakage-safe interpretation used here.

    Returns:
        (num_windows, 20) float32 array.
    """
    tonic, phasic = decompose_subject_eda(raw_eda, fs=fs, tau0=tau0, tau1=tau1,
                                           delta_knot=delta_knot, alpha=alpha)
    all_features = []
    for start in window_starts:
        scl_window = tonic[start:start + raw_win_len]
        scr_window = phasic[start:start + raw_win_len]
        all_features.append(extract_window_features(
            scl_window, scr_window, fs=fs,
            use_adaptive_threshold=use_adaptive_threshold,
            adaptive_threshold_k=adaptive_threshold_k,
        ))
    return np.stack(all_features) if all_features else np.empty((0, 20), dtype=np.float32)


def normalize_features_per_subject(stat_features: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """
    Z-scores each subject's own statistical-feature windows independently
    (per feature column), removing between-subject differences in absolute
    EDA reactivity scale before pooling across subjects for a shared
    classifier.

    This matters more for wrist EDA than the paper's primary chest channel:
    wrist sensors show substantially larger between-subject variability in
    absolute SCR amplitude (Hosseini et al., cited in this paper as [3,4],
    applied per-subject normalization for exactly this reason on this same
    wrist channel). Without it, a subject with naturally lower absolute
    reactivity can have their genuine STRESS windows sit, in globally-scaled
    terms, below another subject's calm BASELINE windows -- producing
    systematically wrong predictions for that subject even though their own
    within-subject stress/baseline direction is completely normal.

    Safe to apply globally, before any train/test split: this only uses
    each subject's own data, exactly like the existing per-subject z-scoring
    of the raw EDA signal feeding the autoencoder -- no cross-subject
    information is used, so there's no leakage risk.
    """
    normalized = np.empty_like(stat_features, dtype=np.float32)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        block = stat_features[mask]
        mean = block.mean(axis=0)
        std = block.std(axis=0) + 1e-8
        normalized[mask] = (block - mean) / std
    return normalized


if __name__ == "__main__":
    # Quick self-test on a synthetic subject trace
    np.random.seed(0)
    fs = 4
    n = 4800  # 20 min
    t = np.linspace(0, 1200, n)
    raw = 2.0 + 0.3 * np.sin(t / 50) + np.random.randn(n) * 0.05
    raw[2000:2010] += np.linspace(0, 1.0, 10)  # inject a fake SCR-like bump

    raw_win_len = 40   # 10s @ 4Hz
    stride_len = 8     # 2s @ 4Hz
    starts = list(range(0, n - raw_win_len + 1, stride_len))

    feats = extract_features_for_subject(raw, starts, raw_win_len, fs=fs)
    print(f"Feature matrix shape: {feats.shape}  (expected: ({len(starts)}, 20))")
    print("Feature names:", FEATURE_NAMES)
    print("First window's features:")
    for name, val in zip(FEATURE_NAMES, feats[0]):
        print(f"  {name:<24} {val:.6f}")
    print(f"\nAny NaNs? {np.isnan(feats).any()}")