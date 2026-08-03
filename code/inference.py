import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from detector import Configurable1DCNN
from unet import Configurable1DUNet
from utils import load_dat_file, set_seed

def calculate_signal_power(signal):
    """Calculates Average Signal Power: P = mean(signal^2)."""
    return np.mean(np.square(signal))

def estimate_snr_db(signal, noise_sample_len=1000):
    """
    Estimates SNR (dB) using the initial silent region as noise floor reference.
    SNR = 10 * log10( Peak Signal Power / Noise Floor Power )
    """
    noise_floor = signal[:noise_sample_len]
    noise_power = np.mean(np.square(noise_floor)) + 1e-12  # Epsilon to prevent div by zero
    
    signal_peak_power = np.max(np.square(signal))
    
    snr_db = 10 * np.log10(signal_peak_power / noise_power)
    return snr_db, noise_power

def run_inference():
    set_seed(Config.SEED)

    # 1. Resolve Checkpoint Paths with Fallbacks
    detector_path = os.path.join(Config.MODEL_DIR, "detector_best.pth")
    if not os.path.exists(detector_path):
        detector_path = os.path.join(Config.MODEL_DIR, "detector.pth")

    denoiser_path = os.path.join(Config.MODEL_DIR, "unet_finetuned.pth")
    if not os.path.exists(denoiser_path):
        denoiser_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")

    # Verify both models exist
    if not os.path.exists(detector_path) or not os.path.exists(denoiser_path):
        print("\n[Error] Model weights missing!")
        print(f"  Detector Path Checked: {detector_path} (Exists: {os.path.exists(detector_path)})")
        print(f"  Denoiser Path Checked: {denoiser_path} (Exists: {os.path.exists(denoiser_path)})")
        print("Please verify your models directory or rerun detector and denoiser training.\n")
        return

    print("--- Starting Two-Stage Inference Pipeline ---")
    print(f"Loading Detector Checkpoint: {detector_path}")
    print(f"Loading Denoiser Checkpoint: {denoiser_path}")

    # 2. Load Stage 1: Detector
    detector = Configurable1DCNN(Config.DETECTOR_CONFIG).to(Config.DEVICE)
    detector.load_state_dict(torch.load(detector_path, map_location=Config.DEVICE))
    detector.eval()

    # 3. Load Stage 2: U-Net Denoiser
    denoiser = Configurable1DUNet(Config.UNET_CONFIG).to(Config.DEVICE)
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=Config.DEVICE))
    denoiser.eval()

    # 4. Find Sample Files for Testing
    signal_dir = os.path.join(Config.TRIAL_DATA_DIR, "Signal")
    sample_files = []
    
    if os.path.exists(signal_dir):
        for root, _, files in os.walk(signal_dir):
            for f in files:
                if f.endswith(".dat"):
                    sample_files.append(os.path.join(root, f))
                    if len(sample_files) >= 3:  # Pick first 3 samples
                        break

    if not sample_files:
        print(f"Error: No test .dat files found under {signal_dir}")
        return

    # 5. Execute Pipeline
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for idx, filepath in enumerate(sample_files):
            raw_data = load_dat_file(filepath, Config.SIGNAL_LENGTH)
            
            # Standardize for detector pass
            std = np.std(raw_data)
            mean = np.mean(raw_data)
            norm_data = (raw_data - mean) / std if std > 1e-5 else raw_data - mean

            tensor_input = torch.from_numpy(norm_data).float().unsqueeze(0).unsqueeze(0).to(Config.DEVICE)

            # Stage 1: Detection
            prob = detector(tensor_input).item()
            is_detected = prob >= Config.DETECTOR_THRESHOLD

            print(f"\nSample {idx+1}: {os.path.basename(filepath)}")
            print(f"  --> Detection Probability: {prob:.4f}")
            print(f"  --> Status: {'SIGNAL DETECTED' if is_detected else 'NOISE ONLY'}")

            # Stage 2: Denoising (if detected)
            if is_detected:
                tensor_raw = torch.from_numpy(raw_data).float().unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
                denoised_tensor = denoiser(tensor_raw)
                denoised_signal = denoised_tensor.cpu().squeeze().numpy()

                # Calculate Powers & SNR
                raw_power = calculate_signal_power(raw_data)
                denoised_power = calculate_signal_power(denoised_signal)
                
                raw_snr_db, raw_noise_pwr = estimate_snr_db(raw_data)
                denoised_snr_db, denoised_noise_pwr = estimate_snr_db(denoised_signal)
                snr_improvement_db = denoised_snr_db - raw_snr_db

                print(f"  --> Raw Signal Power:      {raw_power:.6f}")
                print(f"  --> Denoised Signal Power: {denoised_power:.6f}")
                print(f"  --> Raw Estimated SNR:     {raw_snr_db:.2f} dB")
                print(f"  --> Denoised Est. SNR:     {denoised_snr_db:.2f} dB")
                print(f"  --> SNR Improvement:       +{snr_improvement_db:.2f} dB")

                # Plot Results
                plt.figure(figsize=(12, 6))
                
                plt.subplot(2, 1, 1)
                plt.plot(raw_data, label="Raw Input Signal", color="orange", alpha=0.8)
                plt.title(f"Sample {idx+1} - Raw Signal (Prob: {prob:.2f} | Power: {raw_power:.6f} | SNR: {raw_snr_db:.2f} dB)")
                plt.grid(True)
                plt.legend()

                plt.subplot(2, 1, 2)
                plt.plot(denoised_signal, label="U-Net Denoised Output", color="blue")
                plt.title(f"Sample {idx+1} - Denoised Signal (Power: {denoised_power:.6f} | SNR: {denoised_snr_db:.2f} dB | Gain: +{snr_improvement_db:.2f} dB)")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                
                # Save copy to disk
                save_plot_path = os.path.join(Config.OUTPUT_DIR, f"inference_sample_{idx+1}.png")
                plt.savefig(save_plot_path)
                print(f"  --> Saved pipeline plot to: {save_plot_path}")
                
                # Display interactive plot window on screen
                plt.show()

if __name__ == "__main__":
    run_inference()
