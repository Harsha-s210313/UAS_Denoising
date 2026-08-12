import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import NoiseAdaptationUNetDataset
from unet import Configurable1DUNet
from utils import set_seed


class UnsupervisedNoiseSuppressionLoss(nn.Module):
    """
    Suppresses ambient noise power in non-signal regions while preserving
    overall structural power of acoustic echo signatures.
    """
    def __init__(self, noise_weight=2.0, energy_weight=0.1):
        super(UnsupervisedNoiseSuppressionLoss, self).__init__()
        self.noise_weight = noise_weight
        self.energy_weight = energy_weight

    def forward(self, outputs, noisy_batch, noise_profile):
        noise_len = noise_profile.size(-1)
        pred_noise = outputs[:, :, :noise_len]

        # Suppress leading edge noise window power
        noise_loss = torch.mean(torch.square(pred_noise))

        # Soft regularizer to prevent zero-collapse
        energy_loss = torch.abs(
            torch.mean(torch.square(outputs)) - (0.5 * torch.mean(torch.square(noisy_batch)))
        )

        return (self.noise_weight * noise_loss) + (self.energy_weight * energy_loss)


def finetune_denoiser():
    set_seed(Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print(f"--- Stage 2: Fine-tuning U-Net Denoiser on Real Trial Data [{Config.DEVICE}] ---")

    dataset = NoiseAdaptationUNetDataset(
        Config.TRIAL_DATA_DIR, Config.SIGNAL_LENGTH, Config.IS_COMPLEX_SIGNAL
    )
    if len(dataset) == 0:
        print(f"Error: No trial dataset files found in {Config.TRIAL_DATA_DIR}")
        return

    train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.UNET_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.UNET_BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = Configurable1DUNet(Config.UNET_CONFIG).to(Config.DEVICE)
    pretrained_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=Config.DEVICE))
        print(f"Successfully loaded pre-trained checkpoint: {pretrained_path}")
    else:
        print(f"Warning: {pretrained_path} not found. Training from initialization.")

    criterion = UnsupervisedNoiseSuppressionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.UNET_LR * 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, "unet_finetuned.pth")

    for epoch in range(1, Config.FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for noisy_batch, noise_profile in train_loader:
            noisy_batch = noisy_batch.to(Config.DEVICE)
            noise_profile = noise_profile.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(noisy_batch)

            loss = criterion(outputs, noisy_batch, noise_profile)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, noise_profile in val_loader:
                noisy_batch = noisy_batch.to(Config.DEVICE)
                noise_profile = noise_profile.to(Config.DEVICE)

                outputs = model(noisy_batch)
                loss = criterion(outputs, noisy_batch, noise_profile)
                val_loss += loss.item() * noisy_batch.size(0)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{Config.FINETUNE_EPOCHS:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Fine-tuned U-Net Checkpoint: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.UNET_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break


if __name__ == "__main__":
    finetune_denoiser()
