import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from unet import Configurable1DUNet
from utils import load_dat_file, save_dat_file, enable_scroll_zoom


def run_inference(input_filepath, output_filepath=None):
    """Runs complete Stage 2 U-Net denoising inference on a target .dat file."""
    device = Config.DEVICE
    
    # 1. Load Model
    model = Configurable1DUNet(Config.UNET_CONFIG).to(device)
    model_path = os.path.join(Config.MODEL_DIR, "unet_finetuned.pth")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from: {model_path}")
    else:
        raise FileNotFoundError(f"No trained model checkpoint found in {Config.MODEL_DIR}")

    model.eval()

    # 2. Load & Prepare Input
    raw_signal = load_dat_file(input_filepath, Config.SIGNAL_LENGTH, Config.IS_COMPLEX_SIGNAL)
    input_tensor = torch.from_numpy(raw_signal).unsqueeze(0).unsqueeze(0).float().to(device)

    # 3. Predict / Denoise
    with torch.no_grad():
        denoised_tensor = model(input_tensor)

    denoised_signal = denoised_tensor.squeeze().cpu().numpy()

    # 4. Compute SNR Improvement
    noise_region_raw = raw_signal[:1000]
    noise_region_clean = denoised_signal[:1000]
    
    raw_power = np.mean(raw_signal**2)
    denoised_power = np.mean(denoised_signal**2)
    noise_floor_reduction = 10 * np.log10((np.var(noise_region_raw) + 1e-8) / (np.var(noise_region_clean) + 1e-8))

    print("\n--- Pipeline Performance Summary ---")
    print(f"Input File: {input_filepath}")
    print(f"Raw Signal Power: {raw_power:.6f}")
    print(f"Denoised Signal Power: {denoised_power:.6f}")
    print(f"Estimated Noise Floor Attenuation: {noise_floor_reduction:.2f} dB")

    # 5. Save Output
    if output_filepath:
        save_dat_file(output_filepath, denoised_signal)
        print(f"Saved denoised signal to: {output_filepath}")

    # 6. Plot Results
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_signal, color='gray', alpha=0.5, label='Raw Input Signal')
    ax.plot(denoised_signal, color='blue', linewidth=1.2, label='U-Net Denoised Output')
    ax.set_title("Stage 2 Acoustic Denoising (Scroll to Zoom)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    enable_scroll_zoom(ax)
    plt.tight_layout()
    plt.show()

    return denoised_signal


if __name__ == "__main__":
    sample_file = os.path.join(Config.TRIAL_DATA_DIR, "sample_0.dat")
    out_file = os.path.join(Config.OUTPUT_DIR, "sample_0_clean.dat")
    run_inference(sample_file, out_file)
