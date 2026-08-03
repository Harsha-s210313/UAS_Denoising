"""
=============================================================
config.py

Configuration file for the Underwater Acoustic Signal Dataset
Generator.

Modify only this file to change dataset properties.

Author : ChatGPT
=============================================================
"""

import numpy as np

# ============================================================
# RANDOM SEED
# ============================================================

# Set to None for a different dataset every run.
# Set to an integer for reproducible datasets.

RANDOM_SEED = 42


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = "Project"


# ============================================================
# DATASET STRUCTURE
# ============================================================

NUM_NOISE_PRI = 10          # PRI_01 ... PRI_10
NUM_SIGNAL_PRI = 10         # PRI_11 ... PRI_20

CHANNELS_PER_PRI = 100


# ============================================================
# SIGNAL LENGTHS
# ============================================================

NOISE_SAMPLES = 32154

SIGNAL_SAMPLES = 14113


# ============================================================
# SAMPLING
# ============================================================

FS = 96000                  # Hz


# ============================================================
# SIGNAL BAND
# ============================================================

LOW_FREQ = 21000            # Hz
HIGH_FREQ = 23000           # Hz


# ============================================================
# LFM PARAMETERS
# ============================================================

# Chirp duration limits

MIN_CHIRP_DURATION = 0.05   # seconds
MAX_CHIRP_DURATION = 0.12   # seconds

# Signal Power

MIN_SIGNAL_DB = -40
MAX_SIGNAL_DB = 6

# Up-chirp probability

UPCHIRP_PROBABILITY = 0.5


# ============================================================
# PROPAGATION
# ============================================================

# Maximum channel delay

MAX_DELAY_SAMPLES = 120

# Attenuation range

MIN_ATTENUATION = 0.35
MAX_ATTENUATION = 1.15

# Phase offset

MAX_PHASE_OFFSET = np.pi

# Doppler

MIN_DOPPLER = -0.03
MAX_DOPPLER = 0.03


# ============================================================
# MULTIPATH
# ============================================================

ENABLE_MULTIPATH = True

MIN_ECHOES = 2
MAX_ECHOES = 5

MAX_ECHO_DELAY = 300

MIN_ECHO_GAIN = 0.15
MAX_ECHO_GAIN = 0.70


# ============================================================
# CHANNEL VARIATIONS
# ============================================================

ENABLE_GAIN_MISMATCH = True

GAIN_MISMATCH_DB = 1.0

ENABLE_DC_OFFSET = True

MAX_DC_OFFSET = 0.005

ENABLE_PHASE_OFFSET = True


# ============================================================
# NOISE MODEL
# ============================================================

NOISE_TYPES = [

    "sea",

    "shipping",

    "rain",

    "thermal",

    "biological",

    "mixed"

]


# ============================================================
# SEA STATE
# ============================================================

SEA_STATE_MIN = 0
SEA_STATE_MAX = 6


# ============================================================
# SHIPPING NOISE
# ============================================================

SHIPPING_MIN_LEVEL = 0.2
SHIPPING_MAX_LEVEL = 1.0


# ============================================================
# RAIN
# ============================================================

RAIN_MIN_LEVEL = 0.1
RAIN_MAX_LEVEL = 1.0


# ============================================================
# BIOLOGICAL
# ============================================================

BIO_CLICK_RATE_MIN = 1
BIO_CLICK_RATE_MAX = 15


# ============================================================
# WINDOW EXTRACTION
# ============================================================

# A random SIGNAL_SAMPLES-length window will be extracted
# from the generated noise recording.

WINDOW_RANDOM = True


# ============================================================
# FILE FORMAT
# ============================================================

FILE_DTYPE = np.float32


# ============================================================
# FILTER
# ============================================================

FILTER_ORDER = 6


# ============================================================
# PROGRESS
# ============================================================

SHOW_PROGRESS = True


# ============================================================
# DEBUG
# ============================================================

VERBOSE = False
