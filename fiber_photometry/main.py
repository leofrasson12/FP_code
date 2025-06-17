#!/usr/bin/env python
"""
main.py

Entry point for the fiber photometry analysis pipeline.
"""

import argparse
import yaml
import os
from pathlib import Path

from fiber_analysis.io import (
    load_tdt_block, get_stream_data, get_event_times
)
from fiber_analysis.preprocess import (
    remove_artifact, smooth_signal, downsample, compute_dff
)
from fiber_analysis.peth import (
    get_snips, optimal_bin_width
)
from fiber_analysis.denoise import mise_optimal_denoise
from fiber_analysis.detect import detect_peaks, group_bouts
from fiber_analysis.plots import (
    plot_raw_traces, plot_snips_heatmap
)


def run_analysis(config, block_override=None):
    # 1) Block path
    block_path = block_override or config["BLOCKPATH"]
    block = load_tdt_block(block_path)
    print(f"Loaded TDT block: {block_path}")

    # 2) Streams
    t_sig, sig_raw, fs_sig = get_stream_data(block, config["STREAM_SIG"])
    t_iso, iso_raw, fs_iso = get_stream_data(block, config["STREAM_ISO"])
    print(f"  • Raw samples: GCaMP={len(sig_raw)} @ {fs_sig}Hz, ISO={len(iso_raw)} @ {fs_iso}Hz")

    # 3) Events
    lever_times = get_event_times(block, config["EPOC_LEVER"])
    lick_times  = get_event_times(block, config["EPOC_LICK"])
    print(f"  • Events: Lever={len(lever_times)}   Licks={len(lick_times)}")

    # 4) Preprocess
    #    a) Trim artifact
    t_sig, sig = remove_artifact(t_sig, sig_raw, config["ARTIFACT_CUTOFF"])
    t_iso, iso = remove_artifact(t_iso, iso_raw, config["ARTIFACT_CUTOFF"])
    #    b) Smooth
    sig = smooth_signal(sig, config["SMOOTH_WINDOW"])
    iso = smooth_signal(iso, config["SMOOTH_WINDOW"])
    #    c) Downsample
    sig_ds = downsample(sig, config["DOWNSAMPLE_FACTOR"])
    iso_ds = downsample(iso, config["DOWNSAMPLE_FACTOR"])
    fs_ds = fs_sig / config["DOWNSAMPLE_FACTOR"]
    #    d) ΔF/F
    dff = compute_dff(sig_ds, iso_ds)

    # 5) Peri‐event snips
    pre, post = config["PRE_WINDOW"], config["POST_WINDOW"]
    lever_snips = get_snips(t_sig, dff, lever_times, pre, post, fs_ds)
    lick_snips  = get_snips(t_sig, dff, lick_times, pre, post, fs_ds)
    print(f"  • Snips: Lever={lever_snips.shape}   Lick={lick_snips.shape}")

    # 6) Bin‐width selection (on lever)
    delta_opt, delta_dp, deltas, aic = optimal_bin_width(
        lever_times.reshape(-1,1),  # event arrays for PETH
        pre, post, fs_ds,
        n_deltas=config["AIC_N_BINS"],
        delta_min=config.get("AIC_MIN"),
        delta_max=config.get("AIC_MAX"),
    )
    print(f"  • Optimal bin: {delta_opt:.3f}s   Deflection‐point: {delta_dp:.3f}s")

    # 7) Denoise lever snips
    lever_dn = mise_optimal_denoise(lever_snips, config["SIGNAL_ENERGY_CUTOFF"])

    # 8) Peak detection (on lever)
    peaks = detect_peaks(
        lever_dn,
        baseline_mad=config["PEAK_BASELINE_MAD"],
        prominence_mad=config["PEAK_PROMINENCE_MAD"],
        collapse_window=config["PEAK_COLLAPSE_WINDOW"],
        fs=fs_ds
    )
    print(f"  • Detected peaks on {len(peaks)} trials.")

    # 9) Bout grouping (on raw lick times)
    bouts = group_bouts(lick_times, config["BOUT_ILI"], config["BOUT_MIN_LICKS"])
    print(f"  • Identified {len(bouts)} lick bouts.")

    # 10) Prepare output directory
    outdir = Path(config.get("OUTPUT_DIR", "results"))
    outdir.mkdir(exist_ok=True)

    # 11) Plot & save figures
    #    a) Raw traces
    fig1 = plot_raw_traces(t_sig, sig_ds, iso_ds)
    fig1.savefig(outdir / "raw_traces.svg")

    #    b) Heatmaps
    fig2 = plot_snips_heatmap(lever_snips, pre, post, fs_ds, title="Lever Peri-Event")
    fig2.savefig(outdir / "lever_heatmap.svg")
    fig3 = plot_snips_heatmap(lick_snips, pre, post, fs_ds, title="Lick Peri-Event")
    fig3.savefig(outdir / "lick_heatmap.svg")

    print(f"Saved figures to {outdir.absolute()}")


def cli():
    parser = argparse.ArgumentParser(description="Fiber photometry pipeline")
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Path to YAML config")
    parser.add_argument("-b", "--block", help="Override block path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_analysis(config, block_override=args.block)


if __name__ == "__main__":
    cli()
