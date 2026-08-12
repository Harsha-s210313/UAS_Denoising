import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


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


def load_dat_file(filepath, expected_length=14113, is_complex=True):
    """
    Reads raw binary float32 data from .dat file.
    Converts interleaved IQ complex samples [I, Q, I, Q...] to magnitude envelope if needed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Binary file not found: {filepath}")

    data = np.fromfile(filepath, dtype=np.float32)
    
    # Handle complex interleaved pairs (length = 2 * expected_length)
    if is_complex and len(data) == 2 * expected_length:
        i_samples = data[0::2]
        q_samples = data[1::2]
        data = np.sqrt(i_samples**2 + q_samples**2)
    
    # Enforce strict vector length
    if len(data) != expected_length:
        if len(data) > expected_length:
            data = data[:expected_length]
        else:
            data = np.pad(data, (0, expected_length - len(data)), 'constant')

    # Z-score normalization for numerical stability
    std_dev = np.std(data)
    if std_dev > 1e-8:
        data = (data - np.mean(data)) / std_dev
    else:
        data = data - np.mean(data)

    return data


def save_dat_file(filepath, data):
    """Saves numpy array to binary .dat format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.astype(np.float32).tofile(filepath)


def compute_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculates evaluation metrics for Stage 1 detector."""
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


def enable_scroll_zoom(ax):
    """Enables mouse scroll-wheel zooming centered on cursor position."""
    def on_scroll(event):
        if event.inaxes != ax:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        scale_factor = 1.15 if event.button == 'down' else 1 / 1.15
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)
