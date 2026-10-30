# MRI TRIAGE: Automated Brain MRI Triage System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/pytorch-2.3-EE4C2C.svg)](https://pytorch.org/)
[![Encoder DINOv2](https://img.shields.io/badge/encoder-DINOv2--Base-green.svg)](https://github.com/facebookresearch/dinov2)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

An end-to-end, deep-learning-based triage pipeline for **Coronal T2-FLAIR Brain MRI** studies. The system performs patient-level binary classification into **Normal** vs. **Abnormal**, designed for emergency radiological prioritization.

---

## 📌 Architecture & Methodology

The triage system utilizes a modern foundation model approach combining self-supervised visual representations with position-aware attention aggregation:

```
Raw DICOM Series (17-33 slices)
            │
            ▼
[Preprocess: Sorting, Brain Coverage Filter (>5%), 
 Robust Normalization, 224x224 Aspect-Preserved Resize]
            │
            ▼
[Frozen DINOv2-Base Backbone (vit_base_patch14_dinov2)] ──► Slice Embeddings (N x 768)
            │
            ▼
[Relative Slice Positional Encoding (0.0 -> 1.0)]
            │
            ▼
[Position-Aware Multi-Head Attention Pooling] ──► Unified Patient Representation (768-D)
            │
            ▼
[Classification Head (with / without Normalized Patient Age)] ──► Normal vs Abnormal (+ Attention Heatmaps)
```

### Key Highlights
- **Frozen Foundation Model**: Employs `dinov2_vitb14.lvd142m` (768-D) without destructive fine-tuning, preventing overfitting on small medical cohorts.
- **Variable Slice Count Handling**: Dynamically processes variable clinical acquisitions (typically 17 to 33 slices per patient) using attention masking.
- **Position-Aware Aggregation**: Incorporates normalized slice positions along the patient's anatomical axis ($p_i \in [0.0, 1.0]$) to capture spatial slice context.
- **Dual Clinical Ablations**:
  - **Ablation 1 (Age-Aware)**: Integrates standardized patient age into the downstream classification head (`bridge.xlsx`).
  - **Ablation 2 (Age-Independent)**: Pure imaging evaluation with weighted cross-entropy loss (`bridge_modified.xlsx`).

---

## 🏆 Model Checkpoints & Performance

Pre-trained model checkpoint weights are provided in `CODE/checkpoints/`:

| Model Checkpoint | Description | Val ROC-AUC | Val Accuracy | Val Sensitivity | Val Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `best_module1_age_aware.pt` | Position-aware pooling + normalized age feature | **0.8566** | **76.0%** | **77.9%** | **73.8%** |
| `best_module1_age_independent.pt` | Position-aware pooling on imaging features only | **0.8711** | **70.3%** | **52.7%** | **94.4%** |

---

## 📁 Repository Structure

```
MRI_TRIAGE/
├── CODE/
│   ├── checkpoints/                     # Saved PyTorch model checkpoint bundles (.pt)
│   │   ├── best_module1_age_aware.pt
│   │   └── best_module1_age_independent.pt
│   ├── src/                             # Core Python package
│   │   ├── config.py                    # Global configurations & paths
│   │   ├── dataset.py                   # PyTorch dataset & collate functions
│   │   ├── evaluate.py                  # Evaluation metrics (ROC-AUC, Sens, Spec)
│   │   ├── inference.py                 # Single-study end-to-end inference engine
│   │   ├── models.py                    # Attention pooling & classification heads
│   │   ├── preprocess.py                # DICOM parsing & DINOv2 feature caching
│   │   ├── train.py                     # Training loop with early stopping
│   │   ├── utils.py                     # Seeding and logging utilities
│   │   └── verify_dataset.py            # Cohort integrity & DICOM validator
│   ├── run_module1_inference.py         # Quick demo script for inference
│   ├── Dockerfile                       # Container definition
│   ├── docker-compose.yml               # Container orchestrator
│   └── requirements.txt                 # Python dependencies
├── datasets/                            # Local datasets folder (git-ignored)
│   ├── BrainGen_deidentified/           # DICOM patient folders
│   └── test_data_mri1/                  # Test set patient studies
├── doc/                                 # Documentation & specifications
│   ├── module1_technical_details.md     # Detailed pipeline and ablation specifications
│   ├── Brain_MRI_Module1_Dataset_and_Modeling_Guide.pdf
│   ├── POSSIBLE_METHODOLOGY_WE_MAY_FOLLOW.pdf
│   └── proj_exp.pdf
├── bridge.xlsx                          # Clinical ground truth & labels (Ablation 1)
├── bridge_modified.xlsx                 # Re-annotated clinical ground truth (Ablation 2)
├── .gitignore                           # Repository ignore rules
└── README.md                            # Main project overview
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
cd CODE
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Run Inference on a Patient Study

Use the trained checkpoint to predict directly from a raw DICOM folder:

```bash
cd CODE
python run_module1_inference.py
```

Or via Python code:
```python
from src.inference import Module1InferenceEngine

# Load checkpoint
engine = Module1InferenceEngine("checkpoints/best_module1_age_aware.pt")

# Run prediction directly on raw DICOM series
result = engine.predict_from_dicom_folder("../datasets/BrainGen_deidentified/995030-000259", age=40)

print("Prediction:", result["prediction"])
print("Abnormality Probability:", result["abnormality_probability"])
print("Confidence:", result["confidence"])
```

### 3. Verification & Training

To verify cohort data integrity:
```bash
python -m src.verify_dataset
```

To extract and cache DINOv2 embeddings:
```bash
python -m src.preprocess --batch_size 16
```

To train from scratch:
```bash
# Age-Aware Model
python -m src.train --ablation age_aware --epochs 50 --lr 1e-3

# Age-Independent Model
python -m src.train --ablation age_independent --epochs 50 --lr 1e-3
```

---

## 🐳 Docker Support

To run everything in an isolated container environment:

```bash
cd CODE
docker-compose build
docker-compose run --rm brain-mri-module1 python run_module1_inference.py
```

---

## 📄 License & Attribution

Designed and developed for clinical brain MRI abnormality triage and radiological emergency prioritization.
