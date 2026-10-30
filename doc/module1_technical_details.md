Viewed pdf2_dump.txt:1-66
Viewed dataset.py:1-100
Viewed dataset.py:101-167
Ran command: `python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`
Checked task Check CUDA device info

### 1. Preprocessing Pipeline (Comparison with PDF)

**Yes, the implemented pipeline matches the guide PDF ([`Brain_MRI_Module1_Dataset_and_Modeling_Guide.pdf`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/Brain_MRI_Module1_Dataset_and_Modeling_Guide.pdf) and [`POSSIBLE METHODOLOGY WE MAY FOLLOW.pdf`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/POSSIBLE%20METHODOLOGY%20WE%20MAY%20FOLLOW.pdf)) exactly.**

Here is the exact step-by-step pipeline executed in [`CODE/src/preprocess.py`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/src/preprocess.py#L26-L134):

1. **Patient DICOM Ingestion & Anatomical Ordering**:
   - Reads all Coronal T2-FLAIR `.dcm` files for each patient.
   - Slices are sorted along the patient's anatomical axis using `ImagePositionPatient[2]` (or `InstanceNumber`), ensuring slices are correctly ordered from anterior to posterior.
2. **Brain Coverage Filtering (Non-brain slice exclusion)**:
   - Slices with less than 5% brain tissue signal (`MIN_BRAIN_COVERAGE_RATIO = 0.05`) are discarded so empty/background skull-edge slices don't pollute the aggregation.
3. **Robust Intensity Normalization**:
   - For non-zero brain pixels, outliers are clipped to the 1st and 99th percentiles ($p_1, p_{99}$).
   - Intensity z-score standardization: $(I - \mu) / \sigma$.
4. **Aspect-Ratio Preserving Resize & Square Padding**:
   - Resized while maintaining native anatomical aspect ratio and zero-padded symmetrically into a square matrix of $224 \times 224$.
5. **Channel Replication**:
   - As mandated by the PDF, single-channel grayscale MR intensities are replicated across 3 identical channels (`[X, X, X]`) to meet ImageNet/ViT input requirements without inventing artificial false colors.
6. **Relative Slice Position Indexing**:
   - For a patient with $N$ valid slices, each slice $i$ receives a normalized relative position $p_i = \frac{i}{N-1} \in [0.0, 1.0]$.

---

### 2. DINOv2 Feature Extraction (Comparison with PDF)

**Yes, the DINOv2 feature extraction strictly implements the PDF specifications.**

- **Frozen Backbone Extraction**:
  - Model is loaded via `timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)`.
  - Placed in `.eval()` mode with `torch.no_grad()`. The ViT weights remain completely frozen, preventing catastrophic forgetting and overfitting.
- **Slice Representation**:
  - Every 2D slice is encoded into a high-dimensional semantic feature vector of shape **$(1, 768)$**.
  - A patient study with $N$ slices produces an embedding matrix of shape **$(N, 768)$**, accompanied by a position vector of shape **$(N,)$**.
- **Pre-cached Embeddings**:
  - All **856 patients** have been pre-extracted and cached as `.pt` files in [`CODE/processed/embeddings/`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/processed/embeddings).

---

### 3. Training Configuration & Complete Details

The training setup across all 856 patients follows the configuration in [`CODE/src/config.py`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/src/config.py), [`train.py`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/src/train.py), and [`dataset.py`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/src/dataset.py):

| Parameter / Component | Configuration Details | Rationale / PDF Reference |
| :--- | :--- | :--- |
| **Deep Learning Framework** | **PyTorch 2.x** (`torch`, `torchvision`, `timm`) | Native standard |
| **Optimizer** | **AdamW** | `weight_decay = 1e-4` for regularization |
| **Learning Rate** | **`1e-3` (0.001)** with **Cosine Annealing LR Scheduler** | Smooth annealing over epochs |
| **Loss Function** | **BCEWithLogitsLoss** | For Age-Independent ablation: loss is weighted using `pos_weight = N_neg / N_pos` (359/497) to balance class penalization. |
| **Batch Size** | **16 patients** | Studies are padded dynamically with a mask per batch |
| **Number of Epochs** | **50 Epochs** max | Sufficient convergence on frozen embeddings |
| **Early Stopping** | **Patience = 10 epochs** | Monitored on **Validation ROC-AUC**; saves the best checkpoint bundle automatically |
| **Hardware Used** | **GPU**: NVIDIA GeForce MX450 (CUDA enabled)<br>**Host OS**: Windows | Training downstream heads takes < 2 minutes |
| **Augmentation Details** | **Deterministic Feature Domain Training**: In the frozen feature caching regime, raw pixel transformations are omitted during head training to maintain deterministic embeddings. The PDF recommends mild spatial shifts/jitter only if training end-to-end; here, dropout ($p=0.3$ and $p=0.2$) is applied in the classification head for regularization. |
| **Variable Slice Handling** | **Dynamic Batch Padding + Attention Mask**: In each batch of size 16, patients with fewer slices are zero-padded to `max_len`. A boolean mask ($B, \text{max\_N}$) masks out padded slices with $-10^9$ before Softmax so they receive zero attention. |

---

### 4. Patient Split Details (Strict Zero-Leakage)

Splits are strictly partitioned at the **patient level** (all slices for any single patient are kept in only one set):

#### **Ablation 1: Age-Aware Model (`bridge.xlsx`)**
- **Total Cohort**: 856 Patients (400 Normal, 456 Abnormal)
- **Special `non_significant` Cohort Allocation (41 cases)**:
  - Exactly **34** allocated to the **Training set**
  - Exactly **7** allocated to the **Validation set** (as requested by the guide PDF)
  - **0** in the final Test set
- **Split Distribution**:
  - **Train**: 604 patients (70.5%)
  - **Validation**: 124 patients (14.5%)
  - **Test**: 128 patients (15.0%)
- **Age Feature**: Patient age is normalized using **training-set statistics only** ($\mu_{\text{train}} = 46.85, \sigma_{\text{train}} = 16.52$). Validation and test sets use these training stats to eliminate data leakage.

#### **Ablation 2: Age-Independent Model (`bridge_modified.xlsx`)**
- **Total Cohort**: 856 Patients (359 Normal, 497 Abnormal)
- **41 `non_significant` cases**: Relabeled as **ABNORMAL**
- **Stratified Split**:
  - **Train**: 599 patients (70%)
  - **Validation**: 128 patients (15%)
  - **Test**: 129 patients (15%)
- **Age Feature**: Completely omitted from model input.

---

### 5. Checkpoint Status

Both primary models have already completed training and are saved in [`CODE/checkpoints/`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/checkpoints):
1. **Age-Aware Best Checkpoint** ([`best_module1_age_aware.pt`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/checkpoints/best_module1_age_aware.pt)):
   - **Validation ROC-AUC**: **0.8566** | Accuracy: **76.0%** | Sensitivity: **77.9%** | Specificity: **73.8%**
2. **Age-Independent Best Checkpoint** ([`best_module1_age_independent.pt`](file:///c:/Users/Akash/OneDrive/Desktop/Project/MRI%20Stroke%20(Bro)/CODE/checkpoints/best_module1_age_independent.pt)):
   - **Validation ROC-AUC**: **0.8711** | Accuracy: **70.3%** | Sensitivity: **52.7%** | Specificity: **94.4%**