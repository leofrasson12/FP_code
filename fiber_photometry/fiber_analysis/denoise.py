"""
denoise.py
==========
Dimension-reduction denoising via PCA-based projection.
Implements an energy-cutoff approach: keep enough PCs to capture 
a specified fraction of total variance (energy).
"""

import numpy as np


def mise_optimal_denoise(snips, energy_cutoff):
    """
    Denoise peri‐event snips by projecting onto the top principal components.

    Parameters
    ----------
    snips : np.ndarray, shape (n_trials, n_samples)
        Peri‐event ΔF/F traces for each trial.
    energy_cutoff : float
        Fraction of total variance to retain (e.g., 0.9 for 90%).

    Returns
    -------
    denoised : np.ndarray, shape (n_trials, n_samples)
        Reconstructed snips using only the top PCs.
    """
    # Convert and center
    X = np.array(snips, dtype=float)
    mean_trace = X.mean(axis=0, keepdims=True)
    Xc = X - mean_trace  # center each column (time point)

    # Compute covariance matrix of time points
    #   shape (n_samples, n_samples)
    cov = np.cov(Xc, rowvar=False, bias=True)

    # Eigen‐decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Determine how many components to keep
    cum_energy = np.cumsum(eigvals) / np.sum(eigvals)
    n_comp = np.searchsorted(cum_energy, energy_cutoff) + 1

    # Projection matrix onto top n_comp PCs
    V = eigvecs[:, :n_comp]  # shape (n_samples, n_comp)

    # Project and reconstruct
    Xd = Xc @ V @ V.T  # back to original space
    denoised = Xd + mean_trace

    return denoised
