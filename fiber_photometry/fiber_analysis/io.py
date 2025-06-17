"""
io.py
=====
TDT data loading and extraction utilities for fiber photometry analysis.
"""

import os
import numpy as np
import tdt


def load_tdt_block(block_path):
    """
    Load a TDT block.
    """
    if not os.path.isdir(block_path):
        raise FileNotFoundError(f"TDT block not found: {block_path}")
    return tdt.read_block(block_path)


def get_stream_data(block, stream_name):
    """
    Extract timestamps, raw data, and sampling rate for a given stream.
    """
    try:
        stream = block.streams[stream_name]
    except KeyError:
        raise KeyError(f"Stream '{stream_name}' not found in block.streams.")

    data = stream.data.flatten()
    fs = stream.fs
    start_time = getattr(stream, "start_time", 0)
    if isinstance(start_time, (list, np.ndarray)):
        start_time = start_time[0]
    timestamps = start_time + np.arange(len(data)) / fs
    return timestamps, data, fs


def get_event_times(block, epoc_name):
    """
    Extract event onset times from an epoc store.
    """
    try:
        epoc = block.epocs[epoc_name]
    except KeyError:
        raise KeyError(f"Epoc '{epoc_name}' not found in block.epocs.")
    return np.array(epoc.onset)


def inspect_block(block):
    """
    Print available streams and epocs in the block.
    """
    print("Available streams:", list(block.streams.keys()))
    print("Available epocs: ", list(block.epocs.keys()))
