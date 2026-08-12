import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import SignalDetectionDataset
from detector import SignalDetectorCNN
from utils import set_seed, compute_classification_metrics


def train_detector():
    set_seed(Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print(f"--- Stage 1: Detector Training Starting on [{Config.DEVICE}] ---")

    dataset = SignalDetectionDataset(
        Config.SYNTHETIC_DATA_DIR,
        signal_length=Config.SIGNAL_LENGTH,
        is_complex=Config.IS_COMPLEX_SIGNAL,
    )

    if len(dataset) == 0:
        print(f"Error: No detector dataset found in {Config.SYNTHETIC_DATA_DIR}")
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
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SignalDetectorCNN(Config.DETECTOR_CONFIG).to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.DETECTOR_LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, "detector_model.pth")

    for epoch in range(1, Config.DETECTOR_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(Config.DEVICE)
            y_batch = y_batch.to(Config.DEVICE)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(Config.DEVICE)
                y_batch = y_batch.to(Config.DEVICE)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        metrics = compute_classification_metrics(
            val_targets, val_preds, threshold=Config.DETECTOR_THRESHOLD
        )

        print(
            f"Epoch {epoch:02d}/{Config.DETECTOR_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Detector Weights to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.DETECTOR_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break


if __name__ == "__main__":
    train_detector()
