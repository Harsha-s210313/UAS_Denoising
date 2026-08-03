# Signal Generation & Denoising Pipeline for UAS Signal Processing

A modular Python framework designed for synthetic signal generation, impairment modeling (noise/propagation), and signal restoration using UNet-based deep learning architectures.

---

## 📌 Features

* **Synthetic Signal Generation:** Generate Linear Frequency Modulated (LFM) and custom waveforms.
* **Realistic Impairment Models:** Simulate real-world noise and propagation attenuation channels.
* **Deep Learning Denoising Pipeline:** PyTorch-based UNet architecture for noise suppression and signal reconstruction.
* **Modular Codebase:** Fully decoupled modules for dataset generation, model pretraining, fine-tuning, and inference execution.

---

## 📁 Repository Structure

```text
├── code/
│   ├── models/                  # Core UNet model architectures
│   ├── config.py                # Pipeline parameters and hyperparameters
│   ├── dataset.py               # PyTorch Dataset and DataLoader classes
│   ├── denoiser_pretrain.py     # Unsupervised/synthetic pretraining script
│   ├── denoiser_finetune.py     # Supervised fine-tuning script
│   ├── detector.py              # Signal detection module
│   ├── detector_train.py        # Detection model training script
│   ├── generate_and_test.py     # End-to-end dataset generation & validation
│   ├── inference.py             # Inference pipeline for trained checkpoints
│   ├── unet.py                  # Core UNet network definitions
│   └── utils.py                 # Signal processing helper utilities
│
├── signal_generation/
│   ├── config.py                # Waveform generation parameters
│   ├── filters.py               # DSP filtering utilities
│   ├── generator.py             # Base signal generation routines
│   ├── lfm.py                   # LFM waveform generation routines
│   ├── main.py                  # Main generation entrypoint
│   ├── noise_models.py          # Noise injection models (AWGN, phase noise, etc.)
│   └── propagation.py          # Channel propagation and fading models
│
├── .gitignore                   # Version control exclusion rules
└── requirements.txt             # Python dependencies
