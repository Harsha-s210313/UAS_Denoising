import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import project modules
from config import TRAIN_CONFIG
from unet import UNetDenoiser
from dataset import SignalDataset

# 1. Enable Autograd Anomaly Detection to trace any NaNs or Infs back to source
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
        # Guarantee inputs are finite before loss computation
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Non-finite values detected in model prediction or ground truth target.")

        # Primary MSE loss calculation
        loss = self.mse(pred, target)

        # Numerical guard: avoid log(0) or division by zero if computing relative loss
        loss = torch.clamp(loss, min=self.eps)
        return loss


def normalize_batch(x, eps=1e-8):
    """
    Normalizes each signal sample in the batch to zero mean and unit variance.
    Prevents large dynamic range mismatches in real trial data.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    # Add epsilon to prevent division by zero on silent/flat signal frames
    return (x - mean) / (std + eps)


def train_denoiser():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing denoiser pretraining on device: {device}")

    # Load configuration
    batch_size = TRAIN_CONFIG.get("batch_size", 32)
    epochs = TRAIN_CONFIG.get("epochs", 50)
    # Conservative learning rate to prevent optimizer instability
    learning_rate = TRAIN_CONFIG.get("learning_rate", 1e-4)

    # Initialize Dataset and DataLoader
    train_dataset = SignalDataset(mode="pretrain")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

            # 2. Normalize inputs to constrain dynamic range across real/trial data
            noisy_signals = normalize_batch(noisy_signals)
            clean_signals = normalize_batch(clean_signals)

            # Sanity check for data bounds on initial batch
            if epoch == 1 and batch_idx == 0:
                print(f"[Sanity Check] Noisy Max: {noisy_signals.max().item():.4f}, "
                      f"Min: {noisy_signals.min().item():.4f}, "
                      f"Has NaN: {torch.isnan(noisy_signals).any().item()}")

            optimizer.zero_grad()

            # 3. Disable Mixed Precision (AMP) force FP32 execution for stability
            with torch.cuda.amp.autocast(enabled=False):
                predictions = model(noisy_signals.float())
                loss = criterion(predictions, clean_signals.float())

            # Catch infinite loss immediately before backward step
            if not torch.isfinite(loss):
                print(f"CRITICAL ERROR: Infinite loss encountered at Epoch {epoch}, Batch {batch_idx}")
                print(f"Prediction range: [{predictions.min()}, {predictions.max()}]")
                raise RuntimeError("Training aborted due to non-finite loss.")

            loss.backward()

            # 4. Gradient Clipping: Prevent exploding gradients in UNet skip connections
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f}")

    # Save stable checkpoint
    os.makedirs("code/models", exist_ok=True)
    save_path = "code/models/unet_pretrained.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pretraining complete. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    train_denoiser()
