import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def set_seed(seed):
    """Sets random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dat_file(filepath, expected_length=30452):
    """
    Reads raw binary float32 data from .dat file.
    Adjust dtype if raw file uses float64 or int16.
    """
    data = np.fromfile(filepath, dtype=np.float32)
    if len(data) != expected_length:
        # Fallback/padding/truncation if data size slightly deviates
        if len(data) > expected_length:
            data = data[:expected_length]
        else:
            data = np.pad(data, (0, expected_length - len(data)), 'constant')
    return data

def save_dat_file(filepath, data):
    """Saves numpy array to binary .dat format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.astype(np.float32).tofile(filepath)

def compute_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculates classification metrics for Stage 1 evaluation."""
    y_pred_probs = np.array(y_pred_probs)
    y_true = np.array(y_true)
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }
def plot_signals(original, denoised, save_path, pri_name, beam_name, proba):
    """Generates, displays, and saves comparison plots for detected/denoised sonar signals."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.plot(original, label="Received Signal (Noisy)", alpha=0.6, color="gray")
    plt.plot(denoised, label="Denoised Echo (UNet)", alpha=0.9, color="blue")
    plt.title(f"PRI: {pri_name} | Beam: {beam_name} | Signal Probability: {proba*100:.2f}%")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save the plot file to disk
    plt.savefig(save_path)
    
    # Render and display the plot interactively using matplotlib GUI window
    plt.show()
    
    # Close the figure cleanly to free up memory across batch iterations
    plt.close()

