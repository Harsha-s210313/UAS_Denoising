import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_dat_file


def safe_zscore_sample(arr, eps=1e-8):
    """
    Standardize sequence using Z-score (mean=0, std=1).
    Preserves structural signal variance without stretching noise floor up to full scale.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / (std + eps)).astype(np.float32)


class TrialDetectorDataset(Dataset):
    """Dataset class for Stage 1 1D-CNN Detector."""

    def __init__(self, data_dir, signal_length):
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.file_list = []
        self.labels = []

        # Load Signal Files (Label = 1.0)
        signal_dir = os.path.join(data_dir, "Signal")
        if os.path.exists(signal_dir):
            for root, _, files in os.walk(signal_dir):
                for f in files:
                    if f.endswith(".dat"):
                        self.file_list.append(os.path.join(root, f))
                        self.labels.append(1.0)

        # Load Noise Files (Label = 0.0)
        noise_dir = os.path.join(data_dir, "Noise")
        if os.path.exists(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for f in files:
                    if f.endswith(".dat"):
                        self.file_list.append(os.path.join(root, f))
                        self.labels.append(0.0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        label = self.labels[idx]
        data = load_dat_file(filepath, self.signal_length)

        # Standardize features using Z-score to preserve relative noise/signal power
        norm_data = safe_zscore_sample(data)

        tensor_data = torch.from_numpy(norm_data).float().unsqueeze(0)  # Shape: (1, L)
        tensor_label = torch.tensor(label, dtype=torch.float32)
        return tensor_data, tensor_label


class SyntheticUNetDataset(Dataset):
    """
    Dataset class for Stage 2 U-Net Synthetic Pre-training.

    Expects root_dir structure:
      - root_dir/
          - clean/ (*.dat)
          - noisy/ (*.dat)
    """

    def __init__(self, data_dir, signal_length):
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")

        self.samples = []

        if os.path.exists(self.clean_dir) and os.path.exists(self.noisy_dir):
            noisy_files = sorted(
                [f for f in os.listdir(self.noisy_dir) if f.endswith(".dat")]
            )
            for file_name in noisy_files:
                clean_path = os.path.join(self.clean_dir, file_name)
                noisy_path = os.path.join(self.noisy_dir, file_name)

                if os.path.exists(clean_path):
                    self.samples.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.samples[idx]

        noisy_data = load_dat_file(noisy_path, self.signal_length)
        clean_data = load_dat_file(clean_path, self.signal_length)

        # Standardize signals to prevent exploding loss in U-Net pretraining
        noisy_norm = safe_zscore_sample(noisy_data)
        clean_norm = safe_zscore_sample(clean_data)

        tensor_noisy = torch.from_numpy(noisy_norm).float().unsqueeze(0)
        tensor_clean = torch.from_numpy(clean_norm).float().unsqueeze(0)

        return tensor_noisy, tensor_clean


class NoiseAdaptationUNetDataset(Dataset):
    """Dataset class for Stage 2 U-Net Fine-tuning on trial noise adaptation."""

    def __init__(self, data_dir, signal_length, noise_sample_len=1000):
        self.data_dir = data_dir
        self.signal_length = signal_length
        self.noise_sample_len = noise_sample_len
        self.file_list = []

        # Load trial signal files for fine-tuning
        signal_dir = os.path.join(data_dir, "Signal")
        if os.path.exists(signal_dir):
            for root, _, files in os.walk(signal_dir):
                for f in files:
                    if f.endswith(".dat"):
                        self.file_list.append(os.path.join(root, f))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = load_dat_file(filepath, self.signal_length)
        norm_data = safe_zscore_sample(data)

        # Safely extract initial quiet segment as noise profile context with zero-padding guard
        actual_noise_len = min(len(norm_data), self.noise_sample_len)
        noise_profile = np.zeros(self.noise_sample_len, dtype=np.float32)
        noise_profile[:actual_noise_len] = norm_data[:actual_noise_len]

        tensor_noisy = torch.from_numpy(norm_data).float().unsqueeze(0)  # Shape: (1, L)
        tensor_noise_profile = torch.from_numpy(noise_profile).float().unsqueeze(0)  # Shape: (1, noise_len)

        return tensor_noisy, tensor_noise_profile
