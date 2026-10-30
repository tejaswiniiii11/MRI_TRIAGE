# Automated Brain MRI Triage System - Module 1 (General Abnormality Detection)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/pytorch-2.3-EE4C2C.svg)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/encoder-DINOv2--Base-green.svg)](https://github.com/facebookresearch/dinov2)
[![Docker Support](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📌 Executive Overview

This repository contains the containerized implementation of **Module 1 (General Abnormality Detection)** for the **Automated Brain MRI Triage System**. 

The system analyzes **Coronal T2-FLAIR** brain MRI studies to perform patient-level binary classification into **Normal** vs **Abnormal**. It uses a frozen **DINOv2-Base** Vision Transformer (`dinov2_vitb14`) for 2D slice feature extraction combined with **Position-Aware Attention Pooling** to aggregate variable slice counts (17 to 33 slices) into a unified 768-D patient representation.

Two main clinical experiments (ablations) are supported:
1. **Ablation 1 (Age-Aware)**: Uses `bridge.xlsx` (400 Normal including 41 `non_significant` mild age-expected cases vs 456 Abnormal). Patient age is normalized and concatenated to the patient MRI embedding before the classification head.
2. **Ablation 2 (Age-Independent)**: Uses `bridge_modified.xlsx` (359 Normal vs 497 Abnormal, where all 41 `non_significant` cases are relabeled Abnormal). Uses image data only (no age feature), stratified patient-level splitting, and class-imbalance weighting.

---

## 🚀 Quick Start Guide

### 1. Run with Docker & Docker Compose (Recommended)

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
- (Optional for GPU acceleration) NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

#### Step 1: Build the Docker Image
```bash
docker-compose build
```

#### Step 2: Run Verification Script inside Container
```bash
docker-compose run --rm brain-mri-module1 python -m src.verify_dataset
```

#### Step 3: Run Full Preprocessing & DINOv2 Feature Caching
```bash
docker-compose run --rm brain-mri-module1 python -m src.preprocess --batch_size 16
```

#### Step 4: Train Ablation 1 (Age-Aware Model)
```bash
docker-compose run --rm brain-mri-module1 python -m src.train --ablation age_aware --epochs 50 --lr 1e-3
```

#### Step 5: Train Ablation 2 (Age-Independent Model)
```bash
docker-compose run --rm brain-mri-module1 python -m src.train --ablation age_independent --epochs 50 --lr 1e-3
```

---

### 2. Run Locally without Docker (Alternative)

If you prefer running directly in your local Python environment:

#### Prerequisites
- Python 3.10 or 3.11+
- CUDA-enabled GPU (optional but recommended)

#### Setup Environment
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

#### Commands
```bash
# Verify Dataset
python -m src.verify_dataset

# Preprocess DICOMs & Cache DINOv2 Features
python -m src.preprocess

# Train Age-Aware Model (Ablation 1)
python -m src.train --ablation age_aware

# Train Age-Independent Model (Ablation 2)
python -m src.train --ablation age_independent

# Evaluate Trained Models
python -m src.evaluate --ablation age_aware
```

---

## 📁 Repository Structure

```
CODE/
├── Dockerfile                   # PyTorch CUDA-enabled container environment
├── docker-compose.yml           # Docker service definition with volume mounts
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── .dockerignore                # Docker build ignore rules
├── README.md                    # Setup and execution guide
├── TECHNICAL_DESIGN_AND_MODULE1_ROADMAP.md # Comprehensive Technical Specification
└── src/                         # Source code directory
    ├── __init__.py              # Package initializer
    ├── config.py                # Hyperparameters, paths, and global settings
    ├── utils.py                 # Logging, random seed, metrics calculation utilities
    ├── verify_dataset.py        # Dataset & metadata QC verification tool
    ├── dataset.py               # DICOM slice dataset & embedding dataloaders
    ├── models.py                # DINOv2, Position-Aware Attention Pooling & MLP models
    ├── preprocess.py            # Preprocessing & feature caching pipeline
    ├── train.py                 # Patient-level training and validation loop
    └── evaluate.py              # Performance evaluation and metrics reporting
```

---

## 📖 Documentation & Technical Specification

For the complete technical blueprint, clinical rationale, mathematical formulas for position-aware attention, slice cropping, age normalization, and phase-by-phase implementation details, refer to:

👉 **[TECHNICAL_DESIGN_AND_MODULE1_ROADMAP.md](./TECHNICAL_DESIGN_AND_MODULE1_ROADMAP.md)**

---

## 📊 Evaluation Metrics

The system reports the following clinical & statistical metrics on unseen validation/test sets:
- **Accuracy & Balanced Accuracy**
- **Sensitivity / Recall** (Abnormal detection sensitivity)
- **Specificity** (Normal detection specificity)
- **Precision & F1-Score**
- **ROC-AUC Score & ROC Curve**
- **Confusion Matrix**
