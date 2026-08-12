import numpy as np
import matplotlib.pyplot as plt
from utils import load_dat_file, enable_scroll_zoom
from config import Config

def debug_and_plot_signal(filepath):
    # Load raw signal as real and complex to compare
    raw_bytes = np.fromfile(filepath, dtype=np.float32)
    signal = load_dat_file(filepath, expected_length=Config.SIGNAL_LENGTH, is_complex=True)
    
    # Compute power metrics
    signal_power = np.mean(signal**2)
    peak_amplitude = np.max(np.abs(signal))
    
    print(f"File Path: {filepath}")
    print(f"Total Bytes Read (float32 count): {len(raw_bytes)}")
    print(f"Processed Signal Array Shape: {signal.shape}")
    print(f"Calculated Signal Power: {signal_power:.8f}")
    print(f"Peak Signal Amplitude: {peak_amplitude:.8f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal, color='crimson', linewidth=1.2, label='Signal Magnitude Envelope')
    ax.set_title("Acoustic Signal Waveform (Interactive Scroll-Zoom Enabled)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    enable_scroll_zoom(ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test on a file from your directory
    test_file = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project\sample_0.dat"
    debug_and_plot_signal(test_file)
