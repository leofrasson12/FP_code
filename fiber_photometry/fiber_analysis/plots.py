"""
plots.py
========
Visualization routines for fiber photometry analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_raw_traces(time, sig, iso):
    fig, ax = plt.subplots()
    ax.plot(time, sig, label="GCaMP")
    ax.plot(time, iso, label="Isosbestic")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage")
    ax.legend()
    return fig


def plot_snips_heatmap(snips, pre, post, fs, title=None):
    n_samples = snips.shape[1]
    tvec = np.linspace(-pre, post, n_samples)
    fig, ax = plt.subplots()
    im = ax.imshow(
        snips,
        aspect="auto",
        extent=[tvec[0], tvec[-1], 0, snips.shape[0]],
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="ΔF/F")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")
    if title:
        ax.set_title(title)
    return fig
