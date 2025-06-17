"""
peth.py
=======
Peri‐event snip extraction, bin‐width optimization, and distribution fitting.
"""

import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d


def get_snips(time, signal, events, pre, post, fs):
    """
    Extract peri‐event snips of shape (n_events, n_samples).
    """
    n_samples = int((pre + post) * fs)
    snips = []
    for t_evt in events:
        idx = np.searchsorted(time, t_evt)
        start = idx - int(pre * fs)
        end = start + n_samples
        if start >= 0 and end <= len(signal):
            snips.append(signal[start:end])
    return np.vstack(snips) if snips else np.empty((0, n_samples))


def optimal_bin_width(trials, pre, post, fs,
                      n_deltas=50, delta_min=None, delta_max=None,
                      smooth_win=10):
    """
    Compute AIC‐based optimal and deflection‐point bin widths for peri‐event histograms.

    Parameters
    ----------
    trials : array_like of shape (n_trials, n_samples)
        Each row is a series of event times (in seconds) for one trial.
    pre, post : float
        Window before/after event (s).
    fs : float
        Sampling rate (Hz).
    n_deltas : int
        Number of candidate bin widths to consider (log‐spaced).
    delta_min, delta_max : float or None
        Min/max bin widths (s). If None, defaults to [2/fs, (pre+post)/4].
    smooth_win : int
        Window (in index units) for moving‐average smoothing of the raw AIC curve.

    Returns
    -------
    delta_opt : float
        Bin width (s) at minimum AIC.
    delta_dp : float
        Deflection point: first bin width within 5% of min AIC.
    deltas, aic_smooth : np.ndarray
        Candidate bin widths (s) and the smoothed AIC values.
    """
    # defaults
    if delta_min is None:
        delta_min = 2.0 / fs
    if delta_max is None:
        delta_max = (pre + post) / 4.0

    # candidate binsizes
    deltas = np.logspace(np.log10(delta_min),
                         np.log10(delta_max),
                         n_deltas)

    # fine‐grained baseline rate
    dt = 1.0 / fs
    edges_fine = np.arange(-pre, post + dt/2, dt)
    # count histograms for each trial
    counts_fine = np.vstack([
        np.histogram(tr, bins=edges_fine)[0]
        for tr in trials
    ])
    # mean rate λ1(t) over dt
    lam1 = counts_fine.mean(axis=0) / dt
    centers_fine = edges_fine[:-1] + dt/2
    T = len(lam1)

    aic_raw = []
    for Δ in deltas:
        edges = np.arange(-pre, post + Δ/2, Δ)
        counts = np.vstack([
            np.histogram(tr, bins=edges)[0]
            for tr in trials
        ])
        lam = counts.mean(axis=0) / Δ
        centers = edges[:-1] + Δ/2
        # interpolate lam1 to these centers
        interp = interp1d(centers_fine, lam1, kind="linear", bounds_error=False, fill_value="extrapolate")
        lam1_i = interp(centers)
        SSE = np.sum((lam - lam1_i) ** 2)
        P = T / Δ
        aic = T * np.log(SSE / T) + 2.0 * P
        aic_raw.append(aic)
    aic_raw = np.array(aic_raw)

    # smooth AIC curve
    kernel = np.ones(smooth_win) / smooth_win
    aic_smooth = np.convolve(aic_raw, kernel, mode="same")

    # optimal binsize = argmin
    idx_opt = np.argmin(aic_smooth)
    delta_opt = deltas[idx_opt]

    # deflection point: first Δ where AIC_smooth < AIC_min * 1.05
    aic_min = aic_smooth[idx_opt]
    # find first index where smoothed AIC is within 5% of the minimum
    candidates = np.where(aic_smooth <= aic_min * 1.05)[0]
    delta_dp = deltas[candidates[0]] if candidates.size else delta_opt

    return delta_opt, delta_dp, deltas, aic_smooth


def fit_distributions(snips):
    """
    Fit several distributions to all peri‐event data and pick best by AIC.

    Parameters
    ----------
    snips : array_like, shape (n_trials, n_samples)
        ΔF/F or other peri‐event values.

    Returns
    -------
    best_name : str
        Name of the best‐fitting distribution.
    best_params : tuple
        Parameters for the best fit.
    aic_dict : dict
        AIC value for each distribution.
    """
    # pool all data points
    data = snips.ravel()
    n = data.size

    # candidate distributions
    cands = {
        "normal": stats.norm,
        "expon": stats.expon,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma
    }

    aic_dict = {}
    fits = {}

    for name, dist in cands.items():
        # fit params (shape, loc, scale) – dist.fit returns len(params)
        params = dist.fit(data)
        # compute log-likelihood
        logpdf = dist.logpdf(data, *params)
        logL = np.sum(logpdf)
        k = len(params)
        # AIC = -2 lnL + 2k
        aic = -2 * logL + 2 * k
        aic_dict[name] = aic
        fits[name] = params

    # select best
    best_name = min(aic_dict, key=aic_dict.get)
    best_params = fits[best_name]

    return best_name, best_params, aic_dict
