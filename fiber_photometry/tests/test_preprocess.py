import numpy as np
import pytest
from fiber_analysis.preprocess import smooth_signal, downsample, compute_dff

def test_smooth_signal_constant():
    sig = np.ones(100)
    sm = smooth_signal(sig, window_len=5)
    assert np.allclose(sm, 1.0)

def test_downsample():
    sig = np.arange(10)
    ds = downsample(sig, factor=2)
    # (0+1)/2=0.5, (2+3)/2=2.5, ...
    assert ds.shape[0] == 5
    assert np.allclose(ds, np.array([0.5, 2.5, 4.5, 6.5, 8.5]))

def test_compute_dff_zero():
    iso = np.linspace(1, 2, 10)
    sig = 2 * iso + 1
    dff = compute_dff(sig, iso)
    # since sig == fit, dff == 0
    assert np.allclose(dff, 0.0)
