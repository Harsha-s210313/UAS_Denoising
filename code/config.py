import os
import torch


class Config:
    # --- PATH DIRECTORIES ---
    TRIAL_DATA_DIR = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project"
    SYNTHETIC_DATA_DIR = (
        r"C:\Users\HARSHA\Documents\nstl\signal_generation\Synthetic_Data"
    )
    OUTPUT_DIR = "Results"
    MODEL_DIR = "models"

    # --- SIGNAL & EXECUTION SETTINGS ---
    SIGNAL_LENGTH = 14113
    FILE_DTYPE = "float32"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- STAGE 1: DETECTOR CONFIGURATION ---
    DETECTOR_BATCH_SIZE = 32
    DETECTOR_EPOCHS = 50
    DETECTOR_LR = 3e-4  # Slightly increased learning rate for steady convergence
    DETECTOR_PATIENCE = 10
    DETECTOR_THRESHOLD = 0.50
    TRAIN_VAL_SPLIT = 0.8

    DETECTOR_CONFIG = {
        "input_channels": 1,
        "conv_filters": [16, 32, 64, 128],
        "kernel_sizes": [7, 5, 5, 3],
        "fc_units": 64,
        "dropout": 0.3,
    }

    # --- STAGE 2: UNET DENOISER CONFIGURATION ---
    UNET_BATCH_SIZE = 16
    PRETRAIN_EPOCHS = 30
    FINETUNE_EPOCHS = 20
    UNET_LR = 1e-3
    UNET_PATIENCE = 5

    UNET_CONFIG = {
        "in_channels": 1,
        "out_channels": 1,
        "features": [32, 64, 128, 256],
    }
