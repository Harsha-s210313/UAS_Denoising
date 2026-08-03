"""
=============================================================
propagation.py

Hydrophone propagation simulator.

This module takes ONE transmitted LFM pulse and produces
100 channel-specific received versions.

Effects included

    • Propagation delay
    • Attenuation
    • Doppler
    • Multipath
    • Phase offset
    • Gain mismatch

Author : ChatGPT
=============================================================
"""

import numpy as np
from scipy.signal import resample

from config import *

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================================
# Utility
# ==========================================================

def rms(x):
    return np.sqrt(np.mean(x ** 2))


# ==========================================================
# Delay
# ==========================================================

def apply_delay(signal, delay):
    """
    Delay a signal by integer samples.
    """

    if delay <= 0:
        return signal.copy()

    out = np.zeros_like(signal)

    if delay >= len(signal):
        return out

    out[delay:] = signal[:-delay]

    return out


# ==========================================================
# Attenuation
# ==========================================================

def apply_attenuation(signal):

    attenuation = rng.uniform(
        MIN_ATTENUATION,
        MAX_ATTENUATION
    )

    return signal * attenuation, attenuation


# ==========================================================
# Phase Offset
# ==========================================================

def apply_phase(signal):

    if not ENABLE_PHASE_OFFSET:
        return signal, 0.0

    phase = rng.uniform(
        -MAX_PHASE_OFFSET,
        MAX_PHASE_OFFSET
    )

    analytic = np.fft.fft(signal)

    analytic *= np.exp(1j * phase)

    shifted = np.real(
        np.fft.ifft(analytic)
    )

    return shifted.astype(FILE_DTYPE), phase


# ==========================================================
# Gain Mismatch
# ==========================================================

def apply_gain(signal):

    if not ENABLE_GAIN_MISMATCH:
        return signal, 0.0

    gain_db = rng.uniform(
        -GAIN_MISMATCH_DB,
        GAIN_MISMATCH_DB
    )

    gain = 10 ** (gain_db / 20)

    return signal * gain, gain_db


# ==========================================================
# Doppler
# ==========================================================

def apply_doppler(signal):
    """
    Small Doppler by resampling.
    """

    doppler = rng.uniform(
        MIN_DOPPLER,
        MAX_DOPPLER
    )

    factor = 1 + doppler

    new_length = max(
        10,
        int(len(signal) / factor)
    )

    stretched = resample(
        signal,
        new_length
    )

    out = np.zeros_like(signal)

    n = min(
        len(signal),
        len(stretched)
    )

    out[:n] = stretched[:n]

    return out.astype(FILE_DTYPE), doppler


# ==========================================================
# Multipath
# ==========================================================

def apply_multipath(signal):

    if not ENABLE_MULTIPATH:

        return signal, []

    received = signal.copy()

    metadata = []

    echoes = rng.integers(
        MIN_ECHOES,
        MAX_ECHOES + 1
    )

    for _ in range(echoes):

        delay = rng.integers(
            5,
            MAX_ECHO_DELAY
        )

        gain = rng.uniform(
            MIN_ECHO_GAIN,
            MAX_ECHO_GAIN
        )

        echo = apply_delay(
            signal,
            delay
        )

        received += gain * echo

        metadata.append({

            "delay": int(delay),

            "gain": float(gain)

        })

    return received.astype(FILE_DTYPE), metadata


# ==========================================================
# Noise Window
# ==========================================================

def extract_noise_window(noise):

    if WINDOW_RANDOM:

        start = rng.integers(
            0,
            NOISE_SAMPLES - SIGNAL_SAMPLES
        )

    else:

        start = 0

    end = start + SIGNAL_SAMPLES

    return noise[start:end].copy(), start


# ==========================================================
# Main Channel Simulation
# ==========================================================

def propagate_channel(
        transmitted_signal,
        full_noise):
    """
    Simulate one hydrophone channel.

    Parameters
    ----------
    transmitted_signal

        SIGNAL_SAMPLES samples

    full_noise

        NOISE_SAMPLES samples

    Returns
    -------
    mixture

    metadata
    """

    metadata = {}

    #######################################################
    # Delay
    #######################################################

    delay = rng.integers(
        0,
        MAX_DELAY_SAMPLES
    )

    signal = apply_delay(
        transmitted_signal,
        delay
    )

    metadata["delay"] = int(delay)

    #######################################################
    # Doppler
    #######################################################

    signal, doppler = apply_doppler(
        signal
    )

    metadata["doppler"] = float(doppler)

    #######################################################
    # Attenuation
    #######################################################

    signal, attenuation = apply_attenuation(
        signal
    )

    metadata["attenuation"] = float(
        attenuation
    )

    #######################################################
    # Multipath
    #######################################################

    signal, paths = apply_multipath(
        signal
    )

    metadata["multipath"] = paths

    #######################################################
    # Phase
    #######################################################

    signal, phase = apply_phase(
        signal
    )

    metadata["phase"] = float(phase)

    #######################################################
    # Gain mismatch
    #######################################################

    signal, gain_db = apply_gain(
        signal
    )

    metadata["gain_db"] = float(
        gain_db
    )

    #######################################################
    # Extract Noise Window
    #######################################################

    noise, start = extract_noise_window(
        full_noise
    )

    metadata["noise_window"] = int(start)

    #######################################################
    # Mix
    #######################################################

    mixture = signal + noise

    #######################################################
    # Prevent clipping
    #######################################################

    peak = np.max(
        np.abs(mixture)
    )

    if peak > 1:

        mixture /= peak

    return (
        mixture.astype(FILE_DTYPE),
        metadata
    )
