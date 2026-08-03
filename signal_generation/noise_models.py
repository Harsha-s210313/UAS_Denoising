"""
=============================================================
noise_models.py

Underwater ambient noise generation.

Each PRI shares one environment profile, while every channel
receives an independent realization of that environment.

Author : ChatGPT
=============================================================
"""

import numpy as np

from scipy.signal import chirp

from config import *

from filters import (
    bandpass_filter,
    normalize
)

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================================
# ENVIRONMENT
# ==========================================================

def create_environment():
    """
    Create one underwater environment.

    Returns
    -------
    dict
    """

    env = {

        "sea_state":
            rng.integers(
                SEA_STATE_MIN,
                SEA_STATE_MAX + 1
            ),

        "shipping":
            rng.uniform(
                SHIPPING_MIN_LEVEL,
                SHIPPING_MAX_LEVEL
            ),

        "rain":
            rng.uniform(
                RAIN_MIN_LEVEL,
                RAIN_MAX_LEVEL
            ),

        "thermal":
            rng.uniform(
                0.2,
                1.0
            ),

        "biological":
            rng.uniform(
                0.0,
                1.0
            )

    }

    return env


# ==========================================================
# COLORED NOISE
# ==========================================================

def colored_noise(beta, samples):
    """
    Generate 1/f^beta noise using FFT shaping.
    """

    white = rng.standard_normal(samples)

    spectrum = np.fft.rfft(white)

    freqs = np.fft.rfftfreq(
        samples,
        1 / FS
    )

    freqs[0] = 1

    shaping = 1 / (freqs ** (beta / 2))

    spectrum *= shaping

    noise = np.fft.irfft(
        spectrum,
        n=samples
    )

    return normalize(noise)


# ==========================================================
# SEA STATE
# ==========================================================

def sea_state_noise(samples,
                    sea_state):
    """
    Sea state noise.

    Higher sea state

    -> louder

    -> flatter spectrum
    """

    beta = np.interp(
        sea_state,
        [0, 6],
        [2.0, 0.7]
    )

    noise = colored_noise(
        beta,
        samples
    )

    gain = np.interp(
        sea_state,
        [0, 6],
        [0.3, 1.2]
    )

    noise *= gain

    return noise


# ==========================================================
# SHIPPING
# ==========================================================

def shipping_noise(samples,
                   level):
    """
    Broadband machinery
    + tonal harmonics.
    """

    t = np.arange(samples) / FS

    noise = colored_noise(
        1.6,
        samples
    )

    base = rng.uniform(
        40,
        120
    )

    tonal = np.zeros(samples)

    harmonics = rng.integers(4, 9)

    for h in range(1, harmonics):

        tonal += np.sin(
            2 * np.pi *
            base *
            h *
            t
        ) / h

    shipping = (
        0.8 * noise +
        0.2 * tonal
    )

    shipping *= level

    return normalize(shipping)


# ==========================================================
# RAIN
# ==========================================================

def rain_noise(samples,
               intensity):
    """
    Broadband rain noise with random bursts.
    """

    rain = rng.standard_normal(samples)

    bursts = rng.integers(20, 80)

    for _ in range(bursts):

        idx = rng.integers(
            0,
            samples - 30
        )

        width = rng.integers(
            5,
            25
        )

        rain[
            idx:
            idx + width
        ] += rng.uniform(
            2,
            6
        )

    rain = normalize(rain)

    return rain * intensity


# ==========================================================
# THERMAL
# ==========================================================

def thermal_noise(samples,
                  level):
    """
    Nearly white noise.
    """

    thermal = rng.standard_normal(samples)

    thermal = normalize(thermal)

    return thermal * level


# ==========================================================
# BIOLOGICAL
# ==========================================================

def biological_noise(samples,
                     activity):
    """
    Shrimp clicks and
    dolphin-like pulses.
    """

    signal = np.zeros(samples)

    clicks = int(
        np.interp(
            activity,
            [0, 1],
            [5, 80]
        )
    )

    for _ in range(clicks):

        start = rng.integers(
            0,
            samples - 50
        )

        width = rng.integers(
            5,
            40
        )

        pulse = np.hanning(width)

        signal[
            start:
            start + width
        ] += pulse

    return normalize(signal)


# ==========================================================
# MIX
# ==========================================================

def ambient_noise(environment,
                  samples=NOISE_SAMPLES):
    """
    Generate ambient underwater noise.
    """

    sea = sea_state_noise(
        samples,
        environment["sea_state"]
    )

    shipping = shipping_noise(
        samples,
        environment["shipping"]
    )

    rain = rain_noise(
        samples,
        environment["rain"]
    )

    thermal = thermal_noise(
        samples,
        environment["thermal"]
    )

    bio = biological_noise(
        samples,
        environment["biological"]
    )

    mixture = (

        0.45 * sea +

        0.25 * shipping +

        0.15 * rain +

        0.10 * thermal +

        0.05 * bio

    )

    mixture = bandpass_filter(
        mixture,
        LOW_FREQ,
        HIGH_FREQ
    )

    mixture = normalize(mixture)

    return mixture.astype(FILE_DTYPE)


# ==========================================================
# CHANNEL NOISE
# ==========================================================

def generate_channel_noise(environment):
    """
    Generate one channel of ambient noise.
    """

    noise = ambient_noise(
        environment,
        NOISE_SAMPLES
    )

    gain = rng.uniform(
        0.9,
        1.1
    )

    noise *= gain

    if ENABLE_DC_OFFSET:

        noise += rng.uniform(
            -MAX_DC_OFFSET,
            MAX_DC_OFFSET
        )

    return noise.astype(FILE_DTYPE)
