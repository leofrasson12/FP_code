"""
preprocess.py
=============
Signal‐processing utilities:
– Artifact removal
– Zero‐phase smoothing
– Downsampling
– ΔF/F computation
– Z‐score & robust Z‐score
"""

import numpy as np
from scipy.signal import filtfilt
from scipy.stats import zscore, median_abs_deviation


def remove_artifact(time, signal, cutoff):
    idx = np.searchsorted(time, cutoff)
    return time[idx:], signal[idx:]


def smooth_signal(signal, window_len):
    if window_len < 1:
        raise ValueError("window_len must be >= 1")
    kernel = np.ones(window_len) / window_len
    return filtfilt(kernel, [1], signal)


def downsample(signal, factor):
    if factor <= 1:
        return signal.copy()
    n = (len(signal) // factor) * factor
    return signal[:n].reshape(-1, factor).mean(axis=1)


def compute_dff(sig, iso):
    b, a = np.polyfit(iso, sig, 1)
    fitted = b * iso + a
    return (sig - fitted) / fitted


def compute_zscores(dff):
    z_std = zscore(dff)
    med = np.median(dff)
    mad = median_abs_deviation(dff, scale="normal")
    z_robust = (dff - med) / mad
    return z_std, z_robust