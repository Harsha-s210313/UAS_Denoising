import os
import torch
import torch.nn as nn
from config import Config
from dataset import TrialDetectorDataset
from detector import Configurable1DCNN
from torch.utils.data import DataLoader, Subset
from utils import compute_classification_metrics, set_seed


def pri_grouped_split(dataset, train_ratio=0.8):
    signal_pri_groups = {}
    noise_pri_groups = {}

    for idx, filepath in enumerate(dataset.file_list):
        label = dataset.labels[idx]
        pri_folder = os.path.basename(os.path.dirname(filepath))

        if label == 1.0:
            if pri_folder not in signal_pri_groups:
                signal_pri_groups[pri_folder] = []
            signal_pri_groups[pri_folder].append(idx)
        else:
            if pri_folder not in noise_pri_groups:
                noise_pri_groups[pri_folder] = []
            noise_pri_groups[pri_folder].append(idx)

    def split_groups(group_dict):
        sorted_keys = sorted(list(group_dict.keys()))
        num_train = max(1, int(len(sorted_keys) * train_ratio))
        train_keys = set(sorted_keys[:num_train])

        t_idx, v_idx = [], []
        for key, indices in group_dict.items():
            if key in train_keys:
                t_idx.extend(indices)
            else:
                v_idx.extend(indices)
        return t_idx, v_idx

    sig_train, sig_val = split_groups(signal_pri_groups)
    noise_train, noise_val = split_groups(noise_pri_groups)

    train_indices = sig_train + noise_train
    val_indices = sig_val + noise_val

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_detector():
    set_seed(Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print(f"--- Stage 1: Detector Training Starting on {Config.DEVICE} ---")

    full_dataset = TrialDetectorDataset(
        Config.TRIAL_DATA_DIR, Config.SIGNAL_LENGTH
    )
    if len(full_dataset) == 0:
        print(
            f"Error: No trial data found in directory: {Config.TRIAL_DATA_DIR}"
        )
        return

    train_dataset, val_dataset = pri_grouped_split(
        full_dataset, Config.TRAIN_VAL_SPLIT
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.DETECTOR_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.DETECTOR_BATCH_SIZE, shuffle=False
    )

    model = Configurable1DCNN(Config.DETECTOR_CONFIG).to(Config.DEVICE)
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.DETECTOR_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_metrics = None
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, "detector_best.pth")

    for epoch in range(1, Config.DETECTOR_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(Config.DEVICE)
            y_batch = y_batch.to(Config.DEVICE).view(-1)

            optimizer.zero_grad()
            preds = model(x_batch).view(-1)
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)

            loss = criterion(preds, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(Config.DEVICE)
                y_batch = y_batch.to(Config.DEVICE).view(-1)

                preds = model(x_batch).view(-1)
                preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)

                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                all_targets.extend(y_batch.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        metrics = compute_classification_metrics(
            list(all_targets), list(all_preds), Config.DETECTOR_THRESHOLD
        )

        print(
            f"Epoch {epoch:02d}/{Config.DETECTOR_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | Val AUC: {metrics['roc_auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model Checkpoint to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.DETECTOR_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break

    if best_metrics:
        print("\n--- Final Best Model Validation Evaluation ---")
        print(f"Accuracy:        {best_metrics['accuracy']:.4f}")
        print(f"Precision:       {best_metrics['precision']:.4f}")
        print(f"Recall:          {best_metrics['recall']:.4f}")
        print(f"F1-Score:        {best_metrics['f1_score']:.4f}")
        print(f"ROC-AUC:         {best_metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(best_metrics["confusion_matrix"])
        print("---------------------------------------------\n")


if __name__ == "__main__":
    train_detector()
