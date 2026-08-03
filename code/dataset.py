import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_dat_file


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

        # Standardize features safely
        std = np.std(data)
        mean = np.mean(data)

        if std > 1e-5:
            data = (data - mean) / std
        else:
            data = data - mean

        tensor_data = torch.from_numpy(data).float().unsqueeze(0)  # Shape: (1, L)
        tensor_label = torch.tensor(label, dtype=torch.float32)
        return tensor_data, tensor_label


class SyntheticUNetDataset(Dataset):
    """Dataset class for Stage 2 U-Net Synthetic Pre-training.

    Expects root_dir structure:
      - root_dir/
          - clean/  (*.dat)
          - noisy/  (*.dat)
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

        # Convert to Tensors with channel dimension (1, L)
        tensor_noisy = torch.from_numpy(noisy_data).float().unsqueeze(0)
        tensor_clean = torch.from_numpy(clean_data).float().unsqueeze(0)

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

        # Full noisy signal tensor (1, L)
        tensor_noisy = torch.from_numpy(data).float().unsqueeze(0)

        # Extract initial quiet segment as noise profile context (1, noise_len)
        noise_profile = data[: self.noise_sample_len]
        tensor_noise_profile = (
            torch.from_numpy(noise_profile).float().unsqueeze(0)
        )

        return tensor_noisy, tensor_noise_profile
