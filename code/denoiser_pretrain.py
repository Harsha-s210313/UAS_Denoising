import os

# Intel Xeon OpenMP thread optimization for multi-socket execution
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["MKL_DYNAMIC"] = "FALSE"

import torch
import torch.nn as nn

torch.set_num_threads(16)

from torch.utils.data import DataLoader
from config import Config
from unet import UNetDenoiser
from dataset import SyntheticUNetDataset

# Enable Autograd Anomaly Detection to trace any NaNs or Infs back to source
torch.autograd.set_detect_anomaly(True)


class SafeDenoisingLoss(nn.Module):
    """
    Robust loss wrapper that prevents zero-division and log(0) instabilities.
    Compares reconstructed signal against clean ground truth target.
    """
    def __init__(self, eps=1e-8):
        super(SafeDenoisingLoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Non-finite values detected in model prediction or ground truth target.")

        loss = self.mse(pred, target)
        loss = torch.clamp(loss, min=self.eps)
        return loss


def train_denoiser():
    device = torch.device(Config.DEVICE if hasattr(Config, 'DEVICE') else "cpu")
    print(f"Executing denoiser pretraining on device: {device}")

    # Load configuration parameters safely
    batch_size = getattr(Config, "DENOISER_BATCH_SIZE", 32)
    epochs = getattr(Config, "DENOISER_EPOCHS", 50)
    learning_rate = getattr(Config, "DENOISER_LR", 1e-4)

    # Initialize Dataset and DataLoader
    train_dataset = SyntheticUNetDataset(
        data_dir=getattr(Config, "SYNTHETIC_DATA_DIR", "data/synthetic"),
        signal_length=getattr(Config, "SIGNAL_LENGTH", 2048)
    )

    if len(train_dataset) == 0:
        print("Error: No synthetic clean/noisy data found for pretraining.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    # Initialize UNet Model, Loss, and Optimizer
    model = UNetDenoiser().to(device)
    criterion = SafeDenoisingLoss(eps=1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for batch_idx, (noisy_signals, clean_signals) in enumerate(train_loader):
            noisy_signals = noisy_signals.to(device)
            clean_signals = clean_signals.to(device)

            optimizer.zero_grad()

            predictions = model(noisy_signals.float())
            loss = criterion(predictions, clean_signals.float())

            if not torch.isfinite(loss):
                print(f"CRITICAL ERROR: Infinite loss encountered at Epoch {epoch}, Batch {batch_idx}")
                raise RuntimeError("Training aborted due to non-finite loss.")

            loss.backward()

            # Gradient Clipping: Prevent exploding gradients in UNet skip connections
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * noisy_signals.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Denoising Loss: {epoch_loss:.6f}")

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--> Pretraining complete. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    train_denoiser()
