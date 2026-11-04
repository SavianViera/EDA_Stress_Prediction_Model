"""
Window and feature functions of the project, moved here unchanged from
python_code/Statistical_Feature_Extraction.ipynb so that every notebook can import them:

    from eda_proj.features import reshape_function, sig2features, eda_feature_extraction

Two compatibility edits in eda_feature_extraction, both of which otherwise make `eda_symp_power`
silently NaN (the error is caught by its try/except):
  * `np.trapz` (removed in NumPy 2.x) is called through `_trapz`, which falls back to `np.trapezoid`;
  * the row is passed to `signal.welch` as a NumPy array, because recent SciPy versions fail on a
    pandas Series there.
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

fs = 4

_trapz = getattr(np, "trapz", None) or np.trapezoid


#Now we make the reshape function
#4 parameters, your dataframe, your window size or how many hz of rows should be in each row, purity for how much each row can tolerate label confounsion, and stride size for how much we should overlap
#By default this is every 10 seconds of data
def reshape_function(df, name = 'EDA', window_size = 60, stride_size = 15, hertz=8, purity_threshold=0.8):
    window_row = window_size*hertz
    stride_row = stride_size*hertz
    #Get the subject names
    subjects = sorted(
        list(set(col.split("_")[0] for col in df.columns if "_" in col))
    )
    #Store the chunks of data while we're looping across the time span
    all_window = []
    for i in subjects:
        signal_col = f'{i}_{name}'
        label_col = f'{i}_Label'
        #Kill NAs in individual sequence
        individual_df = df[[signal_col,label_col]].dropna()
        total_rows = len(individual_df)
        #For each subject we take a chunk of their data by our determined window size then stack onto a list (in this case "all window")
        #Chunk is determined through standard range check then we check the purity threshold if we keep the chunk, based on birken's paper design
        for j in range(0,total_rows-window_row+1, stride_row):
            end = j + window_row
            signal_window = individual_df[signal_col].iloc[j:end].values
            label_window = individual_df[label_col].iloc[j:end].values

            # Determine the majority label and calculate if there exists a supermajority in the chunk to determine if kept
            value_counts = pd.Series(label_window).value_counts()
            majority_label = value_counts.index[0]
            majority_count = value_counts.iloc[0]

            purity = majority_count / window_row

            # Retain only windows meeting or exceeding the requested purity ratio
            if purity >= purity_threshold:
                row_data = [i, majority_label] + list(signal_window)
                all_window.append(row_data)

    feature_cols = [f"sample_{i}" for i in range(window_row)]
    columns = ["subject_id", "label"] + feature_cols

    return pd.DataFrame(all_window, columns=columns)


def eda_feature_extraction(row, sampling_rate=fs):
    feature_names = [
        "phasic_peak_count",
        "phasic_peak_rate",
        "phasic_mean_amplitude",
        "phasic_max_amplitude",
        "phasic_sum_amplitude",
        "phasic_mean_rise_time",
        "phasic_std_rise_time",
        "phasic_mean_recovery_time",
        "phasic_mean_onset_time",
        "phasic_mean_offset_time",
        "phasic_shape_symmetry",  # New: Ratio of rise to recovery
        "eda_symp_power"          # New: Spectral power band
    ]

    try:
        # 1. Frequency Domain Extraction (Spectral Power)
        # Calculates power in the typical EDA sympathetic band (0.045 - 0.25 Hz)
        try:
            # Adjust window segment size for Welch's method based on row length
            nperseg = min(len(row), 256) 
            freqs, psd = signal.welch(np.asarray(row, dtype=float), fs=sampling_rate, nperseg=nperseg)
            symp_band = (freqs >= 0.045) & (freqs <= 0.25)
            eda_symp_power = _trapz(psd[symp_band], freqs[symp_band])
        except Exception:
            eda_symp_power = np.nan

        # 2. Time Domain & Phasic Extraction
        signals, info = nk.eda_peaks(row, sampling_rate=sampling_rate)

        peaks = info["SCR_Peaks"]
        onsets = info["SCR_Onsets"]
        recovery = info["SCR_Recovery"]
        amplitudes = info["SCR_Amplitude"]

        peak_count = len(peaks)
        duration_minutes = (len(row) / sampling_rate) / 60

        if peak_count == 0:
            return pd.Series(
                [0, 0.0] + [np.nan] * 8 + [np.nan, eda_symp_power],
                index=feature_names,
            )

        # Convert sample indices to time durations (in seconds)
        rise_times = (peaks - onsets) / sampling_rate
        recovery_times = (recovery - peaks) / sampling_rate

        mean_rise = np.nanmean(rise_times)
        mean_recovery = np.nanmean(recovery_times)

        # 3. Phasic Shape Morphometrics
        # Ratio of rise time to recovery time (measures morphological asymmetry)
        shape_symmetry = mean_rise / mean_recovery if mean_recovery > 0 else np.nan

        onset_times = onsets / sampling_rate
        offset_times = recovery / sampling_rate 
        peak_rate = peak_count / duration_minutes

        return pd.Series(
            [
                peak_count,
                peak_rate,
                np.nanmean(amplitudes),
                np.nanmax(amplitudes),
                np.nansum(amplitudes),
                mean_rise,
                np.nanstd(rise_times),
                mean_recovery,
                np.nanmean(onset_times),
                np.nanmean(offset_times),
                shape_symmetry,
                eda_symp_power
            ],
            index=feature_names,
        )

    except Exception:
        return pd.Series([np.nan] * len(feature_names), index=feature_names)


# intend to apply this function to phasic reshaped
def rise_time_std(row, sampling_rate = fs):
    try:
        #Uses neurokits peak detection to get the onsets then convert it to a time scale and calculate the standard deviation
        signals, info = nk.eda_peaks(row, sampling_rate=sampling_rate)
        # Calculate peak count
        peaks = info["SCR_Peaks"]
        peak_count = len(peaks)
        rise_times = (info["SCR_Peaks"] - info["SCR_Onsets"]) / sampling_rate
        rise_time_std = np.nanstd(rise_times)
        return pd.Series(
                    [peak_count, rise_time_std],
                    index=["phasic_peak_count", "phasic_std_rise_time"],
                )

    #There is a lot of rows with no peaks at all, need to decide how to handle the NAs where the std depreciates thereof
    except Exception:
        return np.nan


def sig2features(sig, sig_name):
    """
    sig is a signal, like tonic_reshaped
    sig_name is a string, like "tonic"
    #feature_set is a list, like [min, max]
    """
    # only want the signals, no "subject_id" and "label" columns
    sample_cols = [col for col in sig.columns if col.startswith('sample_')]
    sig_descriptives = pd.DataFrame()
    # Bring Labels into descriptives to the database
    sig_descriptives['label'] = sig['label']
    # Bring subject ID over to descriptives
    if 'subject_id' in sig.columns:
        sig_descriptives['subject_id'] = sig['subject_id']
    # Descriptives
    sig_descriptives[f"{sig_name}_mean"] = sig[sample_cols].mean(axis=1)
    sig_descriptives[f"{sig_name}_max"]  = sig[sample_cols].max(axis=1)
    sig_descriptives[f"{sig_name}_std"]  = sig[sample_cols].std(axis=1)
    sig_descriptives[f"{sig_name}_skew"]  = sig[sample_cols].skew(axis=1)
    sig_descriptives[f"{sig_name}_norm1"]  = sig[sample_cols].abs().sum(axis=1)
    sig_descriptives[f"{sig_name}_median"] = sig[sample_cols].median(axis=1)
    sig_descriptives[f"{sig_name}_min"] = sig[sample_cols].min(axis=1)
    sig_descriptives[f"{sig_name}_range"] = sig[sample_cols].max(axis=1) - sig[sample_cols].min(axis=1)
    sig_descriptives[f"{sig_name}_mean_slope"] = (sig[sample_cols].diff(axis=1).mean(axis=1))
    sig_descriptives[f"{sig_name}_kurtosis"] = sig[sample_cols].kurtosis(axis=1)
    sig_descriptives[f"{sig_name}_sampen"] = sig[sample_cols].apply(
    lambda row: nk.entropy_sample(row.dropna())[0] if not row.dropna().empty else np.nan, 
    axis=1)
    sig_descriptives[f"{sig_name}_apen"] = sig[sample_cols].apply(
    lambda row: nk.entropy_approximate(row.dropna())[0] if not row.dropna().empty else np.nan, 
    axis=1)



    if sig_name == 'phasic':
        # Define the full list of features returned by our function
        phasic_cols = [
            "phasic_peak_count",
            "phasic_peak_rate",
            "phasic_mean_amplitude",
            "phasic_max_amplitude",
            "phasic_sum_amplitude",
            "phasic_mean_rise_time",
            "phasic_std_rise_time",
            "phasic_mean_recovery_time",
            "phasic_mean_onset_time",
            "phasic_mean_offset_time",
            "phasic_shape_symmetry",
            "eda_symp_power"
        ]
        
        # Assign the apply output to all 12 columns
        sig_descriptives[phasic_cols] = sig[sample_cols].apply(eda_feature_extraction, axis=1)
        
    return sig_descriptives
