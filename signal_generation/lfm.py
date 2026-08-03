"""
=============================================================
lfm.py

Linear Frequency Modulated (LFM) signal generator.

One LFM pulse is generated for each PRI and then propagated
to all hydrophone channels.

Author : ChatGPT
=============================================================
"""

import numpy as np
from scipy.signal import chirp

from config import *
from filters import normalize

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================================
# Signal Power
# ==========================================================

def db_to_linear(db):
    """
    Convert dB amplitude to linear scale.
    """
    return 10.0 ** (db / 20.0)


# ==========================================================
# Random Chirp Parameters
# ==========================================================

def random_chirp_parameters():
    """
    Generate random parameters for one LFM pulse.

    Returns
    -------
    dict
    """

    duration = rng.uniform(
        MIN_CHIRP_DURATION,
        MAX_CHIRP_DURATION
    )

    samples = int(duration * FS)

    direction = (
        "up"
        if rng.random() < UPCHIRP_PROBABILITY
        else "down"
    )

    bandwidth = rng.uniform(
        1200,
        HIGH_FREQ - LOW_FREQ
    )

    f_start = rng.uniform(
        LOW_FREQ,
        HIGH_FREQ - bandwidth
    )

    f_end = f_start + bandwidth

    if direction == "down":
        f_start, f_end = f_end, f_start

    signal_db = rng.uniform(
        MIN_SIGNAL_DB,
        MAX_SIGNAL_DB
    )

    phase = rng.uniform(
        0,
        2 * np.pi
    )

    return {

        "duration": duration,

        "samples": samples,

        "direction": direction,

        "f0": f_start,

        "f1": f_end,

        "power_db": signal_db,

        "phase": phase

    }


# ==========================================================
# Generate LFM
# ==========================================================

def generate_lfm(params):
    """
    Generate one LFM waveform.

    Parameters
    ----------
    params : dict

    Returns
    -------
    ndarray
    """

    N = params["samples"]

    t = np.arange(N) / FS

    waveform = chirp(

        t,

        f0=params["f0"],

        f1=params["f1"],

        t1=t[-1],

        method="linear",

        phi=np.degrees(params["phase"])

    )

    # Smooth edges
    window = np.hanning(N)

    waveform *= window

    waveform = normalize(waveform)

    waveform *= db_to_linear(
        params["power_db"]
    )

    return waveform.astype(FILE_DTYPE)


# ==========================================================
# Insert Signal
# ==========================================================

def insert_signal(signal):

    """
    Insert the chirp into a
    SIGNAL_SAMPLES-length recording.

    Returns
    -------
    recording
    insertion_index
    """

    recording = np.zeros(
        SIGNAL_SAMPLES,
        dtype=FILE_DTYPE
    )

    length = len(signal)

    if length >= SIGNAL_SAMPLES:

        recording[:] = signal[:SIGNAL_SAMPLES]

        return recording, 0

    start = rng.integers(

        0,

        SIGNAL_SAMPLES - length

    )

    recording[
        start:
        start + length
    ] = signal

    return recording, start


# ==========================================================
# Build One Event
# ==========================================================

def create_event():
    """
    Create one transmitted pulse.

    Returns
    -------
    recording
    metadata
    """

    params = random_chirp_parameters()

    pulse = generate_lfm(params)

    recording, position = insert_signal(
        pulse
    )

    params["position"] = position

    return recording, params
