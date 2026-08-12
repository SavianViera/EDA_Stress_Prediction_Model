"""
Two-Stage Artifact Detection & Correction (Section 2.3, Birkenmaier et al., 2026)
====================================================================================
Adapts the paper's MAD-based, four-category artifact detector + physiological
SCR-morphology validation + linear-interpolation correction pipeline to 4 Hz
wrist EDA. Runs ONCE per subject on the full raw trace (not per-window),
matching the paper's approach ("MAD was computed once per subject over the
full recording length").

Deviations from the paper, stated explicitly:
  - The paper's "1-2 sample" needle/wall durations were defined at 700 Hz
    (~1.4-2.9 ms). At 4 Hz, 1-2 samples is 250-500 ms -- a much coarser
    definition of "brief." The category logic is preserved, but expect it
    to catch a broader class of anomalies than the original did.
  - Several implementation details aren't published (e.g., the exact
    rolling-window length for local MAD in needle detection, or the local
    window used to check "top two consecutive samples" for wall
    artifacts). Reasonable, documented choices are made below; tune the
    constants at the top if your data suggests otherwise.
  - Rise/recovery-time bounds for physiological validation (0.3-5.0s /
    1.0-20.0s) are taken directly from the paper's Section 2.4, since
    those are stated as biological constants independent of sensor/rate.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# --- High-pass baseline-drift filter (Section 2.2.1) ---
HIGHPASS_CUTOFF_HZ = 0.05  # paper: 0.05 Hz = 20s longest plausible autonomic-response period
HIGHPASS_ORDER = 4         # paper: 4th-order Butterworth
APPLY_HIGHPASS_DEFAULT = True

# --- Detection thresholds (paper's stated values where given) ---
STEP_FACTOR = 12.0                  # paper: step factor = 12.0
NEEDLE_FACTOR = 10.0                # paper: needle factor = 10.0
FLATLINE_STD_RATIO = 0.001          # paper: rolling std < 0.001 * overall std
FLATLINE_WINDOW_SEC = 5.0           # not specified by paper; reasonable default
NEEDLE_LOCAL_WINDOW_SEC = 2.5       # local window for needle's local MAD
STEP_SUSTAIN_SEC = 2.0              # window used to confirm a jump is "sustained"
WALL_LOCAL_WINDOW_SAMPLES = 5       # local window for "top-2-consecutive-samples" check
WALL_CONCENTRATION_THRESHOLD = 0.9  # paper: >90% of amplitude change in top-2 samples

# --- Physiological validation (Section 2.4's widened SCR bounds) ---
RISE_TIME_RANGE_SEC = (0.3, 5.0)
RECOVERY_TIME_RANGE_SEC = (1.0, 20.0)
VALIDATION_CONTEXT_BEFORE_SEC = 2.0
VALIDATION_CONTEXT_AFTER_SEC = 8.0

# --- Correction ---
MAX_INTERP_DURATION_SEC = 3.5       # paper: segments >= this are left uncorrected


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def remove_baseline_drift(raw_eda: np.ndarray, fs: int, cutoff_hz: float = HIGHPASS_CUTOFF_HZ,
                           order: int = HIGHPASS_ORDER) -> np.ndarray:
    """
    High-pass Butterworth filter (Section 2.2.1) removing slow,
    non-physiological baseline drift (gradual temperature change,
    electrode-skin contact pressure drift, sensor warm-up) below cutoff_hz.

    TENSION IN THE PAPER'S OWN DESCRIPTION, stated explicitly: elsewhere,
    the paper describes genuine tonic SCL as varying "over minutes to
    hours" -- far slower than this filter's 20s-period (0.05 Hz) cutoff.
    Applying this filter therefore also strips genuine slow SCL trends,
    not just non-physiological drift, before the signal ever reaches
    cvxEDA decomposition (which has its own explicit smooth-tonic model
    via the delta_knot spline). This is implemented for direct fidelity to
    the paper's literal description and matches their own Figure 1 (raw
    ~4.6-6.0 uS -> "cleaned" ~-0.2 to 0.4, confirming they do mean a full
    drift-removed, near-zero-centered signal) -- the tension is flagged,
    not resolved. Toggle off via apply_highpass=False in clean_eda_signal
    if you'd rather let cvxEDA's own tonic model handle the full
    slow-timescale content unfiltered.

    Uses second-order-sections (sos) form + zero-phase filtering
    (sosfiltfilt), which is the numerically stable choice at this filter's
    very low cutoff-to-Nyquist ratio (0.05 Hz cutoff vs. 2 Hz Nyquist at
    4 Hz sampling -- a normalized cutoff of 0.025, right at the edge of
    where naive transfer-function ('ba') filter designs can become
    numerically unstable).
    """
    raw_eda = np.asarray(raw_eda, dtype=np.float64)
    min_len = 3 * order * 3  # conservative margin above sosfiltfilt's own padlen requirement
    if len(raw_eda) < min_len:
        return raw_eda.copy()  # too short to filter safely -- return unchanged
    sos = butter(order, cutoff_hz, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, raw_eda)


def mask_to_segments(mask: np.ndarray):
    """Converts a boolean mask into a list of (start, end) index tuples (end exclusive)."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask)))
    return segments


# ===========================================================================
# Stage 1: Conservative artifact detection (4 categories)
# ===========================================================================
def detect_flatline(signal: np.ndarray, fs: int, window_sec: float = FLATLINE_WINDOW_SEC,
                     std_ratio: float = FLATLINE_STD_RATIO) -> np.ndarray:
    """Flags samples where local rolling std is near-zero relative to the global std."""
    window = max(2, int(round(window_sec * fs)))
    s = pd.Series(signal)
    rolling_std = s.rolling(window=window, center=True, min_periods=2).std()
    rolling_std = rolling_std.bfill().ffill()
    global_std = float(s.std()) + 1e-12
    return (rolling_std / global_std < std_ratio).to_numpy()


def detect_needle(signal: np.ndarray, fs: int, factor: float = NEEDLE_FACTOR,
                   local_window_sec: float = NEEDLE_LOCAL_WINDOW_SEC) -> np.ndarray:
    """
    Flags brief 1-2 sample spikes: local outliers (relative to a local
    rolling MAD) that revert close to baseline within ~2 samples -- the
    reversion check is what distinguishes a "needle" from a "step".
    """
    window = max(3, int(round(local_window_sec * fs)))
    s = pd.Series(signal)
    rolling_median = s.rolling(window=window, center=True, min_periods=3).median().bfill().ffill()
    abs_dev = (s - rolling_median).abs()
    rolling_mad = abs_dev.rolling(window=window, center=True, min_periods=3).median()
    fallback_mad = abs_dev.median() + 1e-9
    rolling_mad = rolling_mad.replace(0, np.nan).fillna(fallback_mad).bfill().ffill()

    outlier_score = abs_dev / (rolling_mad + 1e-9)
    candidates = np.where((outlier_score > factor).to_numpy())[0]

    med = rolling_median.to_numpy()
    confirmed = np.zeros(len(signal), dtype=bool)
    for i in candidates:
        j = min(i + 2, len(signal) - 1)
        deviation_at_i = abs(signal[i] - med[i])
        deviation_at_j = abs(signal[j] - med[i])
        if deviation_at_i > 0 and deviation_at_j < 0.5 * deviation_at_i:
            confirmed[i] = True
    return confirmed


def detect_step(signal: np.ndarray, fs: int, factor: float = STEP_FACTOR,
                 sustain_sec: float = STEP_SUSTAIN_SEC) -> np.ndarray:
    """
    Flags large, abrupt, SUSTAINED shifts: a jump exceeding factor*MAD(diff)
    where the post-jump level stays shifted over the sustain window (as
    opposed to a needle, which reverts almost immediately).
    """
    diffs = np.diff(signal, prepend=signal[0])
    mad_diff = _mad(diffs) + 1e-12
    candidates = np.where(np.abs(diffs) > factor * mad_diff)[0]

    sustain_len = max(1, int(round(sustain_sec * fs)))
    flags = np.zeros(len(signal), dtype=bool)
    for i in candidates:
        pre = signal[max(0, i - sustain_len):i]
        post = signal[i:min(len(signal), i + sustain_len)]
        if len(pre) == 0 or len(post) == 0:
            continue
        if abs(np.mean(post) - np.mean(pre)) > 0.5 * factor * mad_diff:
            flags[i:min(len(signal), i + sustain_len)] = True
    return flags


def detect_wall(signal: np.ndarray, fs: int, local_window: int = WALL_LOCAL_WINDOW_SAMPLES,
                 concentration_threshold: float = WALL_CONCENTRATION_THRESHOLD) -> np.ndarray:
    """
    Flags extremely abrupt jumps confined to 1-2 samples: within a small
    local window, checks whether the single largest consecutive-sample
    difference accounts for > concentration_threshold of the window's
    total amplitude range.
    """
    n = len(signal)
    flags = np.zeros(n, dtype=bool)
    half = local_window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = signal[lo:hi]
        if len(seg) < 3:
            continue
        seg_range = np.max(seg) - np.min(seg)
        if seg_range < 1e-9:
            continue
        seg_diffs = np.abs(np.diff(seg))
        if len(seg_diffs) == 0:
            continue
        top_diff = np.max(seg_diffs)
        if top_diff / seg_range > concentration_threshold:
            flags[i] = True
    return flags


# ===========================================================================
# Stage 2: Physiological validation
# ===========================================================================
def validate_segment(signal: np.ndarray, fs: int, seg_start: int, seg_end: int,
                      context_before_sec: float = VALIDATION_CONTEXT_BEFORE_SEC,
                      context_after_sec: float = VALIDATION_CONTEXT_AFTER_SEC,
                      rise_range=RISE_TIME_RANGE_SEC,
                      recovery_range=RECOVERY_TIME_RANGE_SEC) -> bool:
    """
    Checks a flagged segment's surrounding context against genuine SCR
    morphology (rise time, recovery time). Returns True if it should be
    RECLASSIFIED as a valid physiological response (not corrected).
    """
    before = max(0, seg_start - int(context_before_sec * fs))
    after = min(len(signal), seg_end + int(context_after_sec * fs))
    context = signal[before:after]
    if len(context) < 3:
        return False

    peak_rel = int(np.argmax(context))
    onset_search = context[:peak_rel + 1]
    if len(onset_search) == 0:
        return False
    onset_rel = int(np.argmin(onset_search))
    rise_time = (peak_rel - onset_rel) / fs

    baseline = context[onset_rel]
    peak_val = context[peak_rel]
    amplitude = peak_val - baseline
    if amplitude <= 0:
        return False

    recovery_target = baseline + 0.1 * amplitude
    recovery_rel = None
    for k in range(peak_rel, len(context)):
        if context[k] <= recovery_target:
            recovery_rel = k
            break
    if recovery_rel is None:
        recovery_time = (len(context) - 1 - peak_rel) / fs
    else:
        recovery_time = (recovery_rel - peak_rel) / fs

    return (rise_range[0] <= rise_time <= rise_range[1]) and \
           (recovery_range[0] <= recovery_time <= recovery_range[1])


# ===========================================================================
# Stage 3: Correction (linear interpolation, short segments only)
# ===========================================================================
def correct_signal(signal: np.ndarray, confirmed_segments, fs: int,
                    max_duration_sec: float = MAX_INTERP_DURATION_SEC):
    """Linearly interpolates confirmed artifact segments shorter than max_duration_sec."""
    corrected = signal.copy()
    uncorrected_count = 0
    for start, end in confirmed_segments:
        duration = (end - start) / fs
        if duration >= max_duration_sec:
            uncorrected_count += 1
            continue  # too long to safely interpolate -- leave as-is

        left_idx = max(0, start - 1)
        right_idx = min(len(signal) - 1, end)
        left_val, right_val = signal[left_idx], signal[right_idx]
        n_pts = end - start
        if n_pts <= 0:
            continue
        interp_vals = np.linspace(left_val, right_val, n_pts + 2)[1:-1]
        corrected[start:end] = interp_vals

    return corrected, uncorrected_count


# ===========================================================================
# Top-level entry point
# ===========================================================================
def clean_eda_signal(raw_eda: np.ndarray, fs: int = 4, verbose: bool = True,
                      subject_label: str = "", apply_highpass: bool = APPLY_HIGHPASS_DEFAULT,
                      highpass_cutoff_hz: float = HIGHPASS_CUTOFF_HZ,
                      highpass_order: int = HIGHPASS_ORDER) -> np.ndarray:
    """
    Runs the full pipeline on one subject's full raw EDA trace: high-pass
    drift removal (Section 2.2.1, see remove_baseline_drift's docstring for
    an important tension in the paper's own description), then the
    two-stage detection + validation + correction (Section 2.3). Call ONCE
    per subject, before any windowing or z-scoring.

    Set apply_highpass=False to skip drift removal entirely (e.g. for
    direct before/after comparison against earlier results that didn't
    have this step).
    """
    raw_eda = np.asarray(raw_eda, dtype=np.float64)

    if apply_highpass:
        raw_eda = remove_baseline_drift(raw_eda, fs=fs, cutoff_hz=highpass_cutoff_hz,
                                         order=highpass_order)

    step_mask = detect_step(raw_eda, fs)
    needle_mask = detect_needle(raw_eda, fs)
    flat_mask = detect_flatline(raw_eda, fs)
    wall_mask = detect_wall(raw_eda, fs)
    combined_mask = step_mask | needle_mask | flat_mask | wall_mask

    candidate_segments = mask_to_segments(combined_mask)

    confirmed_segments = []
    reclassified = 0
    for start, end in candidate_segments:
        if validate_segment(raw_eda, fs, start, end):
            reclassified += 1
        else:
            confirmed_segments.append((start, end))

    corrected, uncorrected_count = correct_signal(raw_eda, confirmed_segments, fs)

    if verbose:
        n_cand = len(candidate_segments)
        n_confirmed = len(confirmed_segments)
        n_corrected = n_confirmed - uncorrected_count
        pct_reclassified = 100 * reclassified / max(n_cand, 1)
        prefix = f"  {subject_label}: " if subject_label else "  "
        print(f"{prefix}{n_cand} candidate artifacts -> "
              f"{reclassified} reclassified as genuine SCR ({pct_reclassified:.1f}%), "
              f"{n_corrected} corrected via interpolation, "
              f"{uncorrected_count} left uncorrected (>= {MAX_INTERP_DURATION_SEC}s)")

    return corrected


if __name__ == "__main__":
    # Self-test on a synthetic trace with injected artifacts of each type
    np.random.seed(0)
    fs = 4
    n = 2400  # 10 minutes @ 4Hz
    t = np.linspace(0, 600, n)
    signal = 2.0 + 0.1 * np.sin(t / 30) + np.random.randn(n) * 0.01

    # Inject a step artifact (sustained shift)
    signal[500:530] += 1.5
    # Inject a needle (single-sample spike that reverts)
    signal[900] += 3.0
    # Inject a flatline (sensor disconnect)
    signal[1200:1240] = signal[1200]
    # Inject a genuine-looking SCR (should be reclassified as valid, not corrected)
    scr_region = np.arange(1600, 1620)
    signal[scr_region] += np.concatenate([
        np.linspace(0, 0.8, 8), np.linspace(0.8, 0.1, 12)
    ])

    print("Running artifact detection self-test...")
    cleaned = clean_eda_signal(signal, fs=fs, subject_label="TEST")
    print(f"\nInput length: {len(signal)} | Output length: {len(cleaned)}")
    print(f"Max abs difference (raw vs cleaned): {np.max(np.abs(signal - cleaned)):.4f}")
    print(f"Samples changed by correction: {np.sum(np.abs(signal - cleaned) > 1e-6)}")