import os
import glob
import torch
from torch.utils.data import Dataset
from config import Config
from utils import load_dat_file


class SignalDetectionDataset(Dataset):
    """Dataset for Stage 1 CNN Binary Classifier."""
    def __init__(self, data_dir, signal_length=Config.SIGNAL_LENGTH, is_complex=Config.IS_COMPLEX_SIGNAL):
        self.signal_length = signal_length
        self.is_complex = is_complex
        self.file_list = []
        self.labels = []

        # Signal present (Label = 1)
        signal_files = glob.glob(os.path.join(data_dir, "signal", "*.dat"))
        for f in signal_files:
            self.file_list.append(f)
            self.labels.append(1.0)

        # Noise only (Label = 0)
        noise_files = glob.glob(os.path.join(data_dir, "noise", "*.dat"))
        for f in noise_files:
            self.file_list.append(f)
            self.labels.append(0.0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = load_dat_file(filepath, self.signal_length, self.is_complex)
        
        # Convert to Tensor with Channel Dim [1, L]
        x = torch.from_numpy(data).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class UNetPretrainDataset(Dataset):
    """Dataset for Stage 2 Pre-training (Synthetic Noisy-Clean Pairs)."""
    def __init__(self, synthetic_dir, signal_length=Config.SIGNAL_LENGTH, is_complex=Config.IS_COMPLEX_SIGNAL):
        self.signal_length = signal_length
        self.is_complex = is_complex
        self.noisy_files = sorted(glob.glob(os.path.join(synthetic_dir, "noisy", "*.dat")))
        self.clean_files = sorted(glob.glob(os.path.join(synthetic_dir, "clean", "*.dat")))

    def __len__(self):
        return min(len(self.noisy_files), len(self.clean_files))

    def __getitem__(self, idx):
        noisy_data = load_dat_file(self.noisy_files[idx], self.signal_length, self.is_complex)
        clean_data = load_dat_file(self.clean_files[idx], self.signal_length, self.is_complex)

        noisy_tensor = torch.from_numpy(noisy_data).unsqueeze(0).float()
        clean_tensor = torch.from_numpy(clean_data).unsqueeze(0).float()
        return noisy_tensor, clean_tensor


class NoiseAdaptationUNetDataset(Dataset):
    """Dataset for Stage 2 Fine-tuning on Unlabeled Real Trial Data."""
    def __init__(self, trial_dir, signal_length=Config.SIGNAL_LENGTH, is_complex=Config.IS_COMPLEX_SIGNAL, noise_window_len=1000):
        self.signal_length = signal_length
        self.is_complex = is_complex
        self.noise_window_len = noise_window_len
        self.file_list = glob.glob(os.path.join(trial_dir, "**", "*.dat"), recursive=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = load_dat_file(filepath, self.signal_length, self.is_complex)

        noisy_tensor = torch.from_numpy(data).unsqueeze(0).float()
        # Extract ambient noise profile baseline from leading edge
        noise_profile = noisy_tensor[:, :self.noise_window_len]
        return noisy_tensor, noise_profile
