import os
import numpy as np
from tqdm import tqdm

import config
from noise_models import create_environment, ambient_noise, generate_channel_noise
from lfm import create_event
from propagation import propagate_channel


def ensure_directory(path: str) -> None:
    """Create directory hierarchy if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_binary_dat(filepath: str, data: np.ndarray) -> None:
    """Save 1D float32 numpy array to binary .dat file."""
    data_float32 = data.astype(config.FILE_DTYPE)
    # Call tofile() directly on the numpy array
    data_float32.tofile(filepath)


def generate_dataset() -> dict:
    """
    Executes dataset generation using the underlying project modules.
    Returns metadata and execution statistics.
    """
    # Root paths
    noise_base_dir = os.path.join(config.OUTPUT_ROOT, "Noise")
    signal_base_dir = os.path.join(config.OUTPUT_ROOT, "Signal")

    ensure_directory(noise_base_dir)
    ensure_directory(signal_base_dir)

    total_noise_files = 0
    total_signal_files = 0

    print("Generating Noise PRIs...")
    for pri_idx in tqdm(range(1, config.NUM_NOISE_PRI + 1), desc="Noise PRIs", disable=not config.SHOW_PROGRESS):
        pri_dir = os.path.join(noise_base_dir, f"PRI_{pri_idx:02d}")
        ensure_directory(pri_dir)

        # One underwater environment per PRI
        env = create_environment()

        # Generate 100 channels for this PRI
        for ch_idx in range(1, config.CHANNELS_PER_PRI + 1):
            channel_noise = generate_channel_noise(env)
            file_path = os.path.join(pri_dir, f"channel{ch_idx:03d}.dat")
            save_binary_dat(file_path, channel_noise)
            total_noise_files += 1

    print("Generating Signal PRIs...")
    for pri_idx in tqdm(range(1, config.NUM_SIGNAL_PRI + 1), desc="Signal PRIs", disable=not config.SHOW_PROGRESS):
        actual_pri_num = config.NUM_NOISE_PRI + pri_idx
        pri_dir = os.path.join(signal_base_dir, f"PRI_{actual_pri_num:02d}")
        ensure_directory(pri_dir)

        # 1. Environment and base ambient ocean noise recording (32,154 samples)
        env = create_environment()
        full_pri_noise = ambient_noise(env, samples=config.NOISE_SAMPLES)

        # 2. One LFM pulse event per PRI (14,113 samples)
        lfm_signal, _ = create_event()

        # 3. Propagate transmitted LFM pulse & extract identical noise window across 100 channels
        for ch_idx in range(1, config.CHANNELS_PER_PRI + 1):
            received_mixture, _ = propagate_channel(
                transmitted_signal=lfm_signal,
                full_noise=full_pri_noise
            )
            file_path = os.path.join(pri_dir, f"channel{ch_idx:03d}.dat")
            save_binary_dat(file_path, received_mixture)
            total_signal_files += 1

    return {
        "num_noise_pris": config.NUM_NOISE_PRI,
        "num_signal_pris": config.NUM_SIGNAL_PRI,
        "channels_per_pri": config.CHANNELS_PER_PRI,
        "total_noise_files": total_noise_files,
        "total_signal_files": total_signal_files,
        "noise_length": config.NOISE_SAMPLES,
        "signal_length": config.SIGNAL_SAMPLES,
        "sampling_rate": config.FS,
        "freq_range": (config.LOW_FREQ, config.HIGH_FREQ),
    }


def print_summary(stats: dict) -> None:
    """Print dataset summary statistics."""
    print("\n" + "=" * 55)
    print("        DATASET GENERATION COMPLETE - SUMMARY        ")
    print("=" * 55)
    print(f"Noise PRIs Generated   : {stats['num_noise_pris']}")
    print(f"Signal PRIs Generated  : {stats['num_signal_pris']}")
    print(f"Channels per PRI       : {stats['channels_per_pri']}")
    print(f"Total Noise Files      : {stats['total_noise_files']}")
    print(f"Total Signal Files     : {stats['total_signal_files']}")
    print(f"Noise File Length      : {stats['noise_length']} samples")
    print(f"Signal File Length     : {stats['signal_length']} samples")
    print(f"Sampling Frequency     : {stats['sampling_rate']} Hz")
    print(f"Frequency Band         : {stats['freq_range'][0]} Hz - {stats['freq_range'][1]} Hz")
    print("=" * 55 + "\n")
