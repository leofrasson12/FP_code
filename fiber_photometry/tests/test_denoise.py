import numpy as np
from fiber_analysis.denoise import mise_optimal_denoise

def test_mise_optimal_denoise_constant():
    # Constant snips should remain unchanged after denoising
    snips = np.ones((5, 100)) * 3.14  # 5 trials, flat value 3.14
    den = mise_optimal_denoise(snips, energy_cutoff=0.9)
    assert den.shape == snips.shape
    # All values should still be ≈ 3.14
    assert np.allclose(den, 3.14, atol=1e-6)

def test_mise_optimal_denoise_noise_reduction():
    # Create snips with a single sinusoidal component plus noise
    rng = np.random.RandomState(0)
    t = np.linspace(0, 1, 100)
    clean = np.sin(2*np.pi*5*t)
    snips = np.vstack([clean + 0.5*rng.randn(100) for _ in range(10)])
    den = mise_optimal_denoise(snips, energy_cutoff=0.9)
    # Denoised variance should be lower than original
    orig_var = np.var(snips)
    den_var = np.var(den)
    assert den_var < orig_var
