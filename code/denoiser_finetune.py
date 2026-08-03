import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import NoiseAdaptationUNetDataset
from unet import Configurable1DUNet
from utils import set_seed


def finetune_denoiser():
    set_seed(Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print(
        f"--- Phase 2: U-Net Fine-tuning on Real Trial Data Starting on {Config.DEVICE} ---"
    )

    # 1. Load Dataset
    dataset = NoiseAdaptationUNetDataset(
        Config.TRIAL_DATA_DIR, Config.SIGNAL_LENGTH
    )
    if len(dataset) == 0:
        print(
            f"Error: No trial dataset found in directory: {Config.TRIAL_DATA_DIR}"
        )
        return

    train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.UNET_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.UNET_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 2. Initialize Model and Load Pre-trained Weights
    model = Configurable1DUNet(Config.UNET_CONFIG).to(Config.DEVICE)
    pretrained_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")

    if os.path.exists(pretrained_path):
        model.load_state_dict(
            torch.load(pretrained_path, map_location=Config.DEVICE)
        )
        print(f"Successfully loaded pre-trained weights from {pretrained_path}")
    else:
        print(
            f"Warning: Pre-trained weights not found at {pretrained_path}. Fine-tuning from scratch."
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.UNET_LR * 0.1
    )  # Lower learning rate for fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, "unet_finetuned.pth")

    # 3. Fine-tuning Loop
    for epoch in range(1, Config.FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for noisy_batch, noise_profile in train_loader:
            noisy_batch = noisy_batch.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(noisy_batch)

            # Match target tensor dimensions to prediction outputs (batch_size, 1, 14113)
            target = torch.zeros_like(outputs)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, noise_profile in val_loader:
                noisy_batch = noisy_batch.to(Config.DEVICE)

                outputs = model(noisy_batch)
                target = torch.zeros_like(outputs)

                loss = criterion(outputs, target)
                val_loss += loss.item() * noisy_batch.size(0)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{Config.FINETUNE_EPOCHS:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Fine-tuned U-Net Checkpoint to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.UNET_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break


if __name__ == "__main__":
    finetune_denoiser()
