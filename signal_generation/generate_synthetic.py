import os
import numpy as np
from tqdm import tqdm

import config
from lfm import create_event
from noise_models import create_environment, ambient_noise, generate_channel_noise


def ensure_directory(path: str) -> None:
    """Create directory hierarchy if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_binary_dat(filepath: str, data: np.ndarray) -> None:
    """Save 1D float32 numpy array to binary .dat file."""
    data_float32 = data.astype(config.FILE_DTYPE)
    data_float32.tofile(filepath)


def generate_synthetic_dataset(num_samples: int = 100) -> None:
    """
    Generates paired clean and noisy signals for U-Net pretraining.
    Outputs to Synthetic_Data/clean and Synthetic_Data/noisy folders.
    """
    synthetic_base_dir = "Synthetic_Data"
    clean_dir = os.path.join(synthetic_base_dir, "clean")
    noisy_dir = os.path.join(synthetic_base_dir, "noisy")

    ensure_directory(clean_dir)
    ensure_directory(noisy_dir)

    print(f"Generating {num_samples} Synthetic Clean and Noisy Pairs...")

    for i in tqdm(range(1, num_samples + 1), desc="Synthetic Pairs"):
        # 1. Generate clean LFM pulse (14,113 samples)
        clean_signal, _ = create_event()

        # 2. Generate ocean noise profile matching the LFM pulse duration
        env = create_environment()
        noise_profile = generate_channel_noise(env)

        # Truncate/pad noise profile to strictly match clean signal length
        if len(noise_profile) > len(clean_signal):
            noise_profile = noise_profile[: len(clean_signal)]
        elif len(noise_profile) < len(clean_signal):
            noise_profile = np.pad(
                noise_profile, (0, len(clean_signal) - len(noise_profile)), mode="constant"
            )

        # 3. Create noisy signal mixture
        noisy_signal = clean_signal + noise_profile

        # 4. Save paired binary files
        file_name = f"sample_{i:03d}.dat"
        save_binary_dat(os.path.join(clean_dir, file_name), clean_signal)
        save_binary_dat(os.path.join(noisy_dir, file_name), noisy_signal)

    print("\n=======================================================")
    print("      SYNTHETIC DATASET GENERATION COMPLETE            ")
    print("=======================================================")
    print(f"Clean Files Generated  : {num_samples} -> {clean_dir}")
    print(f"Noisy Files Generated  : {num_samples} -> {noisy_dir}")
    print(f"Signal Sample Count    : {config.SIGNAL_SAMPLES} samples")
    print("=======================================================\n")


if __name__ == "__main__":
    generate_synthetic_dataset(num_samples=100)
