"""
detect.py
=========
Peak and bout detection routines for fiber photometry.
"""

import numpy as np
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks


def detect_peaks(snips, baseline_mad, prominence_mad, collapse_window, fs):
    """
    Detect fluorescence transients (peaks) in each peri-event snip.

    Parameters
    ----------
    snips : np.ndarray, shape (n_trials, n_samples)
    baseline_mad : float
        Multiplier for baseline threshold: median + baseline_mad * MAD.
    prominence_mad : float
        Prominence threshold as multiple of MAD.
    collapse_window : float
        Minimum inter-peak interval (s) to collapse nearby detections.
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    peak_indices : list of np.ndarray
        For each trial, the sample indices of detected peaks.
    """
    peak_indices = []
    for trial in snips:
        # robust baseline threshold
        med = np.median(trial)
        mad = median_abs_deviation(trial, scale="normal")
        height_thr = med + baseline_mad * mad
        prom_thr = prominence_mad * mad

        # initial peak detection
        peaks, _ = find_peaks(trial, height=height_thr, prominence=prom_thr)

        # collapse peaks within collapse_window
        if collapse_window > 0:
            min_sep = int(collapse_window * fs)
            filtered = []
            last = -min_sep
            for p in peaks:
                if p - last > min_sep:
                    filtered.append(p)
                    last = p
            peaks = np.array(filtered)

        peak_indices.append(peaks)
    return peak_indices


def group_bouts(event_times, ili, min_events):
    """
    Group discrete events (e.g. lick times) into bouts based on ILI.

    Parameters
    ----------
    event_times : array-like
        Sorted event times (s).
    ili : float
        Maximum interval (s) allowed within a bout.
    min_events : int
        Minimum number of events to qualify as a bout.

    Returns
    -------
    bouts : list of np.ndarray
        Each array contains the times of events in a bout.
    """
    bouts = []
    if len(event_times) == 0:
        return bouts

    current = [event_times[0]]
    for t in event_times[1:]:
        if t - current[-1] <= ili:
            current.append(t)
        else:
            if len(current) >= min_events:
                bouts.append(np.array(current))
            current = [t]
    if len(current) >= min_events:
        bouts.append(np.array(current))
    return bouts
