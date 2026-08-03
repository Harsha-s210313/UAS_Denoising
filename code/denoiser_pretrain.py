
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import SyntheticUNetDataset
from unet import Configurable1DUNet
from utils import set_seed

def pretrain_denoiser():
    set_seed(Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    print(f"--- Phase 1: U-Net Pre-training on Synthetic Data Starting on {Config.DEVICE} ---")
    
    # 1. Load Synthetic Dataset (Requires root_dir with 'clean/' and 'noisy/' subfolders)
    dataset = SyntheticUNetDataset(Config.SYNTHETIC_DATA_DIR, Config.SIGNAL_LENGTH)
    if len(dataset) == 0:
        print(f"Error: No synthetic dataset found at {Config.SYNTHETIC_DATA_DIR}")
        return

    train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(Config.SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.UNET_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.UNET_BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    model = Configurable1DUNet(Config.UNET_CONFIG).to(Config.DEVICE)
    criterion = nn.MSELoss() # Supervised MSE: Comparing output directly to clean ground truth
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.UNET_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, "unet_pretrained.pth")

    for epoch in range(1, Config.PRETRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(Config.DEVICE)
            clean_batch = clean_batch.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(noisy_batch)
            
            # Loss minimizes difference between network reconstruction and pure synthetic clean signal
            loss = criterion(outputs, clean_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(Config.DEVICE)
                clean_batch = clean_batch.to(Config.DEVICE)
                
                outputs = model(noisy_batch)
                loss = criterion(outputs, clean_batch)
                val_loss += loss.item() * noisy_batch.size(0)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{Config.PRETRAIN_EPOCHS:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Pre-trained U-Net Checkpoint to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.UNET_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break

if __name__ == "__main__":
    pretrain_denoiser()
