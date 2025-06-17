import numpy as np
from fiber_analysis.detect import detect_peaks, group_bouts

def test_detect_peaks_single_peak():
    # One trial with a clear spike at sample 50
    trial = np.zeros(100)
    trial[50] = 10.0
    snips = np.expand_dims(trial, 0)  # shape (1, 100)
    peaks = detect_peaks(
        snips,
        baseline_mad=1.0,
        prominence_mad=1.0,
        collapse_window=0.1,
        fs=100.0
    )
    # Should detect exactly one peak at index 50
    assert isinstance(peaks, list) and len(peaks) == 1
    assert peaks[0].tolist() == [50]

def test_group_bouts_basic():
    # Events at 0, 0.5, 3.0, 3.4, 3.8, with ILI=1.0, min_events=2
    events = np.array([0.0, 0.5, 3.0, 3.4, 3.8])
    bouts = group_bouts(events, ili=1.0, min_events=2)
    # Expect two bouts: [0,0.5] and [3.0,3.4,3.8]
    assert len(bouts) == 2
    assert np.allclose(bouts[0], [0.0, 0.5])
    assert np.allclose(bouts[1], [3.0, 3.4, 3.8])
