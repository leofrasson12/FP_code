import numpy as np
import pytest
from fiber_analysis.peth import get_snips, optimal_bin_width, fit_distributions

def test_get_snips_shape_and_values():
    # Create a time vector 0–10 s sampled at 1 Hz
    time = np.arange(0, 11, 1.0)
    # Signal is just time (for simplicity)
    sig = time.copy()
    # Define two events at t=2s and t=8s
    events = np.array([2.0, 8.0])
    pre, post, fs = 1.0, 2.0, 1.0
    snips = get_snips(time, sig, events, pre, post, fs)
    # We expect 2 trials, each of length (pre+post)*fs = 3 samples
    assert snips.shape == (2, 3)
    # For event at t=2: indices [1,2,3] → times [1,2,3]
    assert np.allclose(snips[0], [1.0, 2.0, 3.0])
    # For event at t=8: indices [7,8,9] → times [7,8,9]
    assert np.allclose(snips[1], [7.0, 8.0, 9.0])

def test_optimal_bin_width_basic():
    # Two trials, each with one event at t=0
    trials = [np.array([0.0]), np.array([0.0])]
    pre, post, fs = 1.0, 1.0, 10.0
    # Compute candidate bins between 2/fs=0.2 and (pre+post)/4=0.5
    delta_opt, delta_dp, deltas, aic = optimal_bin_width(
        trials, pre, post, fs,
        n_deltas=10, delta_min=0.2, delta_max=0.5, smooth_win=3
    )
    # Should return a delta within the requested range
    assert 0.2 <= delta_opt <= 0.5
    assert 0.2 <= delta_dp <= 0.5
    # Lengths
    assert deltas.shape == (10,)
    assert aic.shape == (10,)

def test_fit_distributions_normal():
    # Generate data from a normal distribution
    rng = np.random.RandomState(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(20, 50))  # 20 trials × 50 samples
    best_name, best_params, aic_dict = fit_distributions(data)
    # The normal distribution should have the lowest AIC
    assert best_name == "normal"
    # Check that AIC dict contains all candidates
    for dist in ("normal", "expon", "lognorm", "gamma"):
        assert dist in aic_dict
