import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from detector import Configurable1DCNN
from unet import Configurable1DUNet
from utils import load_dat_file, set_seed

TEST_DIR = "Test_Generation"
NUM_TEST_SAMPLES = 10


def generate_synthetic_test_data(
    output_dir, num_samples=10, signal_length=14113
):
    """Generates a mixture of random noise and signal+noise files for testing."""
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"\n--- Generating {num_samples} Random Test Signals in '{output_dir}' ---"
    )

    np.random.seed(123)  # Fixed generator seed for test creation

    for i in range(num_samples):
        # Alternate between noise-only and signal+noise
        has_signal = i % 2 == 1

        # Base ambient noise
        noise = np.random.normal(0, 0.2, signal_length)

        if has_signal:
            # Generate synthetic pulsed radar waveform inside noise
            t = np.linspace(0, 1, signal_length)
            pulse_start = np.random.randint(2000, 4000)
            pulse_width = np.random.randint(3000, 5000)

            carrier = np.sin(2 * np.pi * 50 * t)  # Carrier wave
            envelope = np.zeros(signal_length)
            envelope[pulse_start : pulse_start + pulse_width] = 1.0

            signal = (carrier * envelope * 1.5) + noise
            label_str = "SIGNAL_ADDED"
        else:
            signal = noise
            label_str = "NOISE_ONLY"

        file_path = os.path.join(
            output_dir, f"test_sample_{i+1:02d}_{label_str}.dat"
        )
        signal.astype(np.float32).tofile(file_path)
        print(f"  Created: {file_path}")


def calculate_signal_power(signal):
    """Calculates Average Signal Power: P = mean(signal^2)."""
    return np.mean(np.square(signal))


def estimate_snr_db(signal, noise_sample_len=1000):
    """Estimates SNR (dB) using the initial segment as noise floor reference."""
    noise_floor = signal[:noise_sample_len]
    noise_power = (
        np.mean(np.square(noise_floor)) + 1e-12
    )  # Epsilon prevents div by zero

    signal_peak_power = np.max(np.square(signal))

    snr_db = 10 * np.log10(signal_peak_power / noise_power)
    return snr_db, noise_power


def run_pipeline_on_generated_data():
    set_seed(Config.SEED)

    # 1. Generate Test Files
    generate_synthetic_test_data(
        TEST_DIR,
        num_samples=NUM_TEST_SAMPLES,
        signal_length=Config.SIGNAL_LENGTH,
    )

    # 2. Resolve Checkpoints
    detector_path = os.path.join(Config.MODEL_DIR, "detector_best.pth")
    if not os.path.exists(detector_path):
        detector_path = os.path.join(Config.MODEL_DIR, "detector.pth")

    denoiser_path = os.path.join(Config.MODEL_DIR, "unet_finetuned.pth")
    if not os.path.exists(denoiser_path):
        denoiser_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")

    if not os.path.exists(detector_path) or not os.path.exists(denoiser_path):
        print("\n[Error] Model weights missing!")
        print(f"  Detector Path: {detector_path}")
        print(f"  Denoiser Path: {denoiser_path}")
        return

    print("\n--- Starting Pipeline Evaluation on Generated Folder ---")

    # 3. Load Stage 1 & Stage 2 Models
    detector = Configurable1DCNN(Config.DETECTOR_CONFIG).to(Config.DEVICE)
    detector.load_state_dict(
        torch.load(detector_path, map_location=Config.DEVICE)
    )
    detector.eval()

    denoiser = Configurable1DUNet(Config.UNET_CONFIG).to(Config.DEVICE)
    denoiser.load_state_dict(
        torch.load(denoiser_path, map_location=Config.DEVICE)
    )
    denoiser.eval()

    # 4. Gather Generated Files
    test_files = sorted(
        [
            os.path.join(TEST_DIR, f)
            for f in os.listdir(TEST_DIR)
            if f.endswith(".dat")
        ]
    )

    # 5. Process Pipeline
    with torch.no_grad():
        for idx, filepath in enumerate(test_files):
            raw_data = load_dat_file(filepath, Config.SIGNAL_LENGTH)

            # Standardize for detector pass
            std = np.std(raw_data)
            mean = np.mean(raw_data)
            norm_data = (
                (raw_data - mean) / std if std > 1e-5 else raw_data - mean
            )

            tensor_input = (
                torch.from_numpy(norm_data)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(Config.DEVICE)
            )

            # Stage 1: Detection Classification
            prob = detector(tensor_input).item()
            is_detected = prob >= Config.DETECTOR_THRESHOLD

            filename = os.path.basename(filepath)
            print(f"\n==========================================")
            print(f"Sample {idx+1}/{len(test_files)}: {filename}")
            print(f"  --> Detection Score: {prob:.4f}")
            print(
                f"  --> Classification: {'[SIGNAL DETECTED]' if is_detected else '[NOISE ONLY]'}"
            )

            # Stage 2: Denoise and Plot if Signal is Detected
            if is_detected:
                tensor_raw = (
                    torch.from_numpy(raw_data)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(Config.DEVICE)
                )
                denoised_tensor = denoiser(tensor_raw)
                denoised_signal = denoised_tensor.cpu().squeeze().numpy()

                # Calculate Signal Powers & SNR
                raw_power = calculate_signal_power(raw_data)
                denoised_power = calculate_signal_power(denoised_signal)

                raw_snr_db, _ = estimate_snr_db(raw_data)
                denoised_snr_db, _ = estimate_snr_db(denoised_signal)
                snr_improvement_db = denoised_snr_db - raw_snr_db

                print(f"  --> Raw Signal Power:      {raw_power:.6f}")
                print(f"  --> Denoised Signal Power: {denoised_power:.6f}")
                print(f"  --> Raw Estimated SNR:     {raw_snr_db:.2f} dB")
                print(f"  --> Denoised Est. SNR:     {denoised_snr_db:.2f} dB")
                print(
                    f"  --> SNR Improvement:       +{snr_improvement_db:.2f} dB"
                )

                # Plot Output
                plt.figure(figsize=(12, 6))

                plt.subplot(2, 1, 1)
                plt.plot(
                    raw_data, label="Raw Input Signal", color="orange", alpha=0.8
                )
                plt.title(
                    f"Sample {idx+1}: {filename}\nRaw Power: {raw_power:.6f} | Est. SNR: {raw_snr_db:.2f} dB"
                )
                plt.grid(True)
                plt.legend()

                plt.subplot(2, 1, 2)
                plt.plot(
                    denoised_signal, label="U-Net Denoised Output", color="blue"
                )
                plt.title(
                    f"Denoised Output\nDenoised Power: {denoised_power:.6f} | Est. SNR: {denoised_snr_db:.2f} dB | Gain: +{snr_improvement_db:.2f} dB"
                )
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.show()
            else:
                print("  --> Skipping Stage 2 Denoising (No Signal Present)")


if __name__ == "__main__":
    run_pipeline_on_generated_data()
