"""
=============================================================
filters.py

Digital filtering utilities used throughout the dataset
generator.

Author : ChatGPT
=============================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt

from config import (
    FS,
    FILTER_ORDER
)


# ==========================================================
# Butterworth Filter Design
# ==========================================================

def butter_lowpass(cutoff, fs=FS, order=FILTER_ORDER):
    """
    Design a Butterworth low-pass filter.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.

    fs : int
        Sampling frequency.

    order : int
        Filter order.

    Returns
    -------
    b, a
        Filter coefficients.
    """

    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype="low"
    )

    return b, a


# ==========================================================

def butter_highpass(cutoff, fs=FS, order=FILTER_ORDER):
    """
    Design a Butterworth high-pass filter.
    """

    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype="high"
    )

    return b, a


# ==========================================================

def butter_bandpass(lowcut,
                    highcut,
                    fs=FS,
                    order=FILTER_ORDER):
    """
    Design a Butterworth band-pass filter.
    """

    nyquist = fs * 0.5

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


# ==========================================================
# Filtering Functions
# ==========================================================

def lowpass_filter(signal,
                   cutoff,
                   fs=FS,
                   order=FILTER_ORDER):
    """
    Apply low-pass filter.
    """

    b, a = butter_lowpass(
        cutoff,
        fs,
        order
    )

    return filtfilt(b, a, signal)


# ==========================================================

def highpass_filter(signal,
                    cutoff,
                    fs=FS,
                    order=FILTER_ORDER):
    """
    Apply high-pass filter.
    """

    b, a = butter_highpass(
        cutoff,
        fs,
        order
    )

    return filtfilt(b, a, signal)


# ==========================================================

def bandpass_filter(signal,
                    lowcut,
                    highcut,
                    fs=FS,
                    order=FILTER_ORDER):
    """
    Apply band-pass filter.
    """

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs,
        order
    )

    return filtfilt(b, a, signal)


# ==========================================================
# Utility
# ==========================================================

def normalize(signal):
    """
    Normalize to unit RMS.

    Parameters
    ----------
    signal : ndarray

    Returns
    -------
    ndarray
    """

    rms = np.sqrt(np.mean(signal ** 2))

    if rms < 1e-12:
        return signal

    return signal / rms


# ==========================================================

def rms(signal):
    """
    Compute RMS value.
    """

    return np.sqrt(np.mean(signal ** 2))
