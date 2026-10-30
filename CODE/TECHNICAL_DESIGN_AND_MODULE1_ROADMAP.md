# Automated Brain MRI Triage System — Module 1 Technical Specification & Implementation Roadmap

## 1. Executive Summary & Clinical Context

### 1.1 Clinical Purpose
Hospital emergency departments and radiology departments receive hundreds of brain MRI examinations daily. Radiologist worklists are typically sorted by arrival time rather than clinical severity. 

The **Automated Brain MRI Triage System** is designed to inspect incoming brain MRI examinations prior to human radiologist review and categorize each study into one of three triage priorities:
- 🔴 **Immediate**: Acute critical findings requiring immediate intervention (e.g., Acute Ischemic Stroke).
- 🟡 **Urgent**: Non-stroke abnormal brain findings requiring prioritized review (e.g., mass effect, chronic stroke, significant edema, hemorrhage, tumors).
- 🟢 **Routine**: Normal or age-expected brain MRI scans.

### 1.2 Two-Module System Architecture
The complete system is composed of two independent diagnostic AI modules followed by a clinical decision fusion engine:

```
                          BRAIN MRI STUDY
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
         CORONAL T2-FLAIR                     DWI + ADC
             MODULE 1                          MODULE 2
       General Abnormality               Acute Ischemic Stroke
        Normal / Abnormal                  Stroke / No Stroke
                │                                 │
                └────────────────┬────────────────┘
                                 │
                          DECISION FUSION
                                 │
                   Routine / Urgent / Immediate
```

- **Module 1 (General Abnormality Module)**: Analyzes **Coronal T2-FLAIR** brain MRI studies to perform patient-level binary classification (**Normal** vs **Abnormal**).
- **Module 2 (Acute Stroke Module)**: Analyzes paired **Axial DWI + ADC** sequences to detect **Acute Ischemic Stroke** vs **No Stroke**.
- **Decision Fusion Rules**:
  - `Stroke = Yes` $\rightarrow$ **Immediate** (Acute Ischemic Stroke overrides all FLAIR findings due to extreme time sensitivity).
  - `Stroke = No` AND `FLAIR = Abnormal` $\rightarrow$ **Urgent**.
  - `Stroke = No` AND `FLAIR = Normal` $\rightarrow$ **Routine**.

---

## 2. Module 1 Task & Specification

This specification governs **Module 1 (General Abnormality Detection)**.

### 2.1 Inputs & Outputs
- **Input**: Coronal T2-FLAIR DICOM slice stack for a patient (variable length $N \in [17, 33]$ slices).
- **Output**: Patient-level binary probability $P(\text{Abnormal} \mid \text{Patient MRI})$, mapped to class label **NORMAL** or **ABNORMAL**.

### 2.2 Key Technical Principles
1. **Patient-Level Classification**: Predictions are made for the entire patient study, not for independent slices.
2. **2D Foundation Model Backbone**: Frozen pretrained **DINOv2-Base** (`dinov2_vitb14`) extracts a fixed 768-D representation per 2D slice.
3. **Variable Slice Handling**: Patient slice counts vary ($N \in [17, 33]$). No forced volume resampling or slice deletion is performed.
4. **Position-Aware Attention Pooling**: Attaches normalized relative slice position $p_i = \frac{i}{N-1}$ to each slice feature vector and learns dynamic attention weights $\alpha_i$ to aggregate slices into a single 768-D patient representation.
5. **No Data Leakage**: All data splits are performed strictly at the patient level. All slices from a single patient reside in the same split partition.

---

## 3. Dataset & Metadata Structure

### 3.1 Dataset Overview
The primary research dataset consists of **856 de-identified patients** with Coronal T2-FLAIR DICOM studies stored in [`BrainGen_deidentified`](file:///c:/Users/Akash/Downloads/Bro%20project/BrainGen_deidentified).

### 3.2 Metadata Spreadsheets & Ablation Formulations

Ground-truth labels are defined across two Excel spreadsheets to enable two distinct clinical ablations:

#### 1. `bridge.xlsx` (**Ablation 1: Age-Aware Clinical Classification**)
- **Target Label**: `Classification` (NORMAL: 400, ABNORMAL: 456).
- **Special Subset**: Among the 400 Normal cases, **41 cases are marked as `non_significant`** under the `Significance` column. These represent mild, age-expected findings (e.g., age-related periventricular white matter hyperintensities) that are clinically normal in elderly patients but would appear abnormal visually.
- **Model Inputs**: Patient MRI Embedding concatenated with **Normalized Patient Age**.
- **Model Output**: NORMAL vs ABNORMAL.
- **Leakage Rules**: `ANON_PatientID`, `ANON_StudyInstanceUID`, `Diagnosis`, and `Significance` are metadata/helper fields and **must not** be used as predictive inputs.

#### 2. `bridge_modified.xlsx` (**Ablation 2: Age-Independent Abnormality Classification**)
- **Target Label**: `Classification` (NORMAL: 359, ABNORMAL: 497).
- **Label Mapping Change**: The 41 `non_significant` age-related cases are **relabeled as ABNORMAL**.
- **Model Inputs**: Patient MRI Embedding **ONLY** (Age is **NOT** supplied to the model).
- **Model Output**: NORMAL vs ABNORMAL.
- **Imbalance Handling**: Addressed via stratified patient-level splitting, training-only data augmentation, and class-weighted cross-entropy loss.

| Feature / Property | Ablation 1 (Age-Aware) | Ablation 2 (Age-Independent) |
| :--- | :--- | :--- |
| **Excel Spreadsheet** | `bridge.xlsx` | `bridge_modified.xlsx` |
| **Total Cohort** | 856 patients | 856 patients |
| **Normal Count** | 400 | 359 |
| **Abnormal Count** | 456 | 497 |
| **41 `non_significant` Cases** | Retained as **NORMAL** | Relabeled as **ABNORMAL** |
| **Age Supplied to Model?** | **YES** (Normalized Age) | **NO** (Image data only) |
| **Model Input** | `[MRI Embedding (768-D) + Age (1-D)]` | `[MRI Embedding (768-D)]` |
| **Imbalance Strategy** | Stratified split | Stratified split + Class weights + Augmentation |

---

## 4. Preprocessing & Feature Caching Pipeline

Each patient study is processed through the following deterministic pipeline:

```
[Coronal T2-FLAIR DICOM Stack]
             │
             ▼
1. DICOM Slice Sorting & Metadata Inspection
   - Order slices spatially by ImagePositionPatient[2] (Z-coordinate) or SliceLocation
             │
             ▼
2. Brain Localization & Bounding Box Cropping (SynthStrip)
   - Compute brain mask across slices
   - Compute 3D bounding box over non-background brain region with a 10-pixel safety margin
   - Crop slices to bounding box
             │
             ▼
3. Brain Coverage Slice Filtering
   - Calculate brain pixel coverage ratio per slice: ratio = (brain_pixels / total_pixels)
   - Exclude extreme edge slices where ratio < 0.05 (5% brain coverage)
             │
             ▼
4. Intensity Normalization
   - Clip intensity values at 1st and 99th percentiles within brain mask
   - Apply z-score normalization: x_norm = (x - mean_brain) / std_brain
             │
             ▼
5. Aspect-Ratio Preserving Resize & Padding
   - Resize longest dimension to 224 pixels preserving aspect ratio
   - Pad background symmetrically to yield square 224 × 224 resolution
             │
             ▼
6. 3-Channel Grayscale Conversion
   - Duplicate normalized grayscale slice X across 3 channels: [X, X, X]
             │
             ▼
7. DINOv2 Feature Extraction & Embedding Caching
   - Pass tensor (N, 3, 224, 224) through frozen DINOv2-Base (dinov2_vitb14)
   - Extract CLS token output: (N, 768)
   - Save feature tensor and relative positions p_i to disk (.pt files)
```

---

## 5. DINOv2 Vision Foundation Feature Extraction

We utilize **DINOv2-Base** (`dinov2_vitb14`), a self-supervised Vision Transformer foundation model pretrained on over 142 million images.

- **Architecture**: ViT-Base/14 (patch size $14 \times 14$).
- **Output Embedding Dimension**: $D = 768$.
- **Backbone Status**: **Frozen** during initial pipeline training (weights remain static).
- **Slice Vector Representation**: For a patient with $N$ preprocessed slices, DINOv2 produces slice embeddings:
  $$\mathbf{Z} = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_N] \in \mathbb{R}^{N \times 768}$$

---

## 6. Variable-Length Slice Aggregation: Position-Aware Attention Pooling

### 6.1 Relative Slice Position Computation
For a study containing $N$ valid slices indexed $i \in \{0, 1, \dots, N-1\}$, the normalized relative position $p_i$ is defined as:
$$p_i = \frac{i}{N - 1} \in [0.0, 1.0]$$
- $p_i \approx 0.0$: Anterior / initial slices of stack.
- $p_i \approx 0.5$: Mid-brain slices.
- $p_i \approx 1.0$: Posterior / end slices of stack.

This relative encoding ensures invariance to slice count differences ($N=21$ vs $N=29$).

### 6.2 Position-Aware Attention Pooling Mathematical Formulation
Let $\mathbf{z}_i \in \mathbb{R}^{768}$ be the DINOv2 feature vector of slice $i$, and $p_i \in [0, 1]$ be its scalar position.

1. **Positional Feature Vector**:
   Project scalar $p_i$ via a small linear projection or concatenate directly:
   $$\mathbf{x}_i = [\mathbf{z}_i \,\|\, p_i] \in \mathbb{R}^{769}$$

2. **Attention Scoring**:
   Pass $\mathbf{x}_i$ through a 2-layer MLP scoring network:
   $$e_i = \mathbf{w}_2^\top \tanh(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1)$$
   where $\mathbf{W}_1 \in \mathbb{R}^{d_a \times 769}$, $\mathbf{b}_1 \in \mathbb{R}^{d_a}$, $\mathbf{w}_2 \in \mathbb{R}^{d_a}$, and $d_a = 128$ is the attention hidden dimension.

3. **Softmax Normalization**:
   $$\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^N \exp(e_j)}, \quad \text{such that } \sum_{i=1}^N \alpha_i = 1.0$$

4. **Patient-Level Embedding Aggregation**:
   $$\mathbf{z}_{\text{patient}} = \sum_{i=1}^N \alpha_i \mathbf{z}_i \in \mathbb{R}^{768}$$

---

## 7. Model Classification Heads (MLP)

### 7.1 Ablation 1 Head (Age-Aware Model)
1. **Age Normalization**:
   $$a_{\text{norm}} = \frac{\text{Age} - \mu_{\text{train\_age}}}{\sigma_{\text{train\_age}}}$$
   *(Note: $\mu$ and $\sigma$ are calculated strictly from training split patients).*
2. **Feature Fusion**:
   $$\mathbf{h} = [\mathbf{z}_{\text{patient}} \,\|\, a_{\text{norm}}] \in \mathbb{R}^{769}$$
3. **MLP Classifier**:
   $$\hat{y} = \text{Sigmoid}(\mathbf{W}_3 \text{ReLU}(\text{Dropout}(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2)) + b_3)$$
   - Hidden Layer 1: $769 \rightarrow 256$ + BatchNorm + ReLU + Dropout(0.3)
   - Hidden Layer 2: $256 \rightarrow 64$ + ReLU + Dropout(0.2)
   - Output Layer: $64 \rightarrow 1$ (Logits / Sigmoid binary probability)

### 7.2 Ablation 2 Head (Age-Independent Model)
1. **Input**: $\mathbf{z}_{\text{patient}} \in \mathbb{R}^{768}$ (No age feature).
2. **MLP Classifier**:
   - Hidden Layer 1: $768 \rightarrow 256$ + BatchNorm + ReLU + Dropout(0.3)
   - Hidden Layer 2: $256 \rightarrow 64$ + ReLU + Dropout(0.2)
   - Output Layer: $64 \rightarrow 1$ (Logits / Sigmoid binary probability)

---

## 8. Patient-Level Splitting & Class Imbalance Rules

### 8.1 Partitioning Rules
- **Split Ratio**: 70% Train (~600 patients), 15% Validation (~128 patients), 15% Test (~128 patients).
- **Patient-Level Stratification**: Stratified by `Classification` label.
- **Special Rule for 41 `non_significant` Cases (Ablation 1)**:
  - Exactly **~34 cases** assigned to Training split.
  - Exactly **~7 cases** assigned to Validation split.
  - Ensures model performance on age-expected mild findings is rigorously evaluated.

### 8.2 Data Augmentation & Imbalance Handling (Ablation 2)
- **Order of Operations**: Split patients FIRST $\rightarrow$ Apply augmentation ONLY to Training split.
- **Augmentation Techniques (Training Only)**:
  - Mild 2D rotations ($\pm 10^\circ$).
  - Mild translations ($\pm 5\%$).
  - Small scaling ($\pm 5\%$).
  - Limited Gaussian noise ($\sigma \le 0.01$).
  - *Do NOT apply aggressive distortion or erasing that could obscure small focal lesions.*
- **Class Weighting**: Apply weighted binary cross-entropy loss:
  $$w_{\text{Normal}} = \frac{N_{\text{total}}}{2 \cdot N_{\text{Normal}}}, \quad w_{\text{Abnormal}} = \frac{N_{\text{total}}}{2 \cdot N_{\text{Abnormal}}}$$

---

---

## 9. Model Serialization, Persistence & Deployment Interface

To ensure models can be reused for future inference, clinical worklists, and Module 2 decision fusion, complete model checkpoint bundles are saved upon validation improvement during training.

### 9.1 Model Checkpoint Bundle Specification
Saved checkpoints (`checkpoints/best_module1_age_aware.pt` and `checkpoints/best_module1_age_independent.pt`) contain full self-contained state dictionaries and metadata:

```python
bundle = {
    'epoch': best_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_metrics': val_metrics,
    'ablation': 'age_aware',              # or 'age_independent'
    'pooling': 'position_attention',       # or 'attention', 'mean'
    'is_age_aware': True,
    'embedding_dim': 768,
    'age_mean': 52.41,                     # Training set age mean (for Age-Aware model)
    'age_std': 14.85,                      # Training set age std (for Age-Aware model)
}
```

### 9.2 Standalone Python Inference Interface
The codebase provides `Module1InferenceEngine` ([`src/inference.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/inference.py)) for one-line prediction on raw DICOM patient folders or precomputed embeddings:

```python
from src.inference import Module1InferenceEngine

# 1. Load saved model checkpoint bundle
engine = Module1InferenceEngine("checkpoints/best_module1_age_aware.pt")

# 2. End-to-end prediction from raw DICOM folder
res = engine.predict_from_dicom_folder("path/to/patient_dicom_dir", age=48)

print(res)
# Output:
# {
#    "prediction": "ABNORMAL",
#    "abnormality_probability": 0.942,
#    "confidence": 0.942,
#    "slice_attention_weights": [0.01, 0.02, 0.45, 0.38, 0.14, ...],
#    "ablation": "age_aware"
# }
```

---

## 10. Evaluation Metrics & Clinical Verification

The model is evaluated using the following comprehensive metrics suite:

1. **Accuracy**: Overall classification accuracy.
2. **Balanced Accuracy**: Macro-average of Sensitivity and Specificity (Primary metric for Ablation 2).
3. **Sensitivity / Recall**: $\frac{TP}{TP + FN}$ (Ability to detect Abnormal cases).
4. **Specificity**: $\frac{TN}{TN + FP}$ (Ability to correctly identify Normal cases).
5. **Precision**: $\frac{TP}{TP + FP}$.
6. **F1-Score**: Harmonic mean of Precision and Sensitivity.
7. **ROC-AUC**: Area under Receiver Operating Characteristic curve.
8. **Confusion Matrix**: 2x2 matrix $[[TN, FP], [FN, TP]]$.

---

## 11. Phase-by-Phase Implementation Roadmap

```
                                  ROADMAP PHASES
                                  ==============

┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│        PHASE 1         │      │        PHASE 2         │      │        PHASE 3         │
│ Data Verification & QC │ ───► │ Preprocessing & DINOv2 │ ───► │ Patient-Level Dataset  │
│    Metadata Integrity  │      │    Embedding Cache     │      │ & Stratified Splitter  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                            │
                                                                            ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│        PHASE 7         │      │        PHASE 6         │      │        PHASE 5         │
│ Model Serialization &  │ ◄─── │  Ablation Evaluation   │ ◄─── │ Model Training Loop &  │
│ Inference Interface    │      │    & Comparisons       │      │   Checkpoint Saving    │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

### Phase 1: Data Verification & QC Integrity (Completed ✅)
- Verify matching patient IDs between DICOM folders and Excel sheets (`bridge.xlsx` & `bridge_modified.xlsx`).
- Inspect DICOM tags (`SeriesDescription`, `StudyInstanceUID`, `Modality`, `ImagePositionPatient`).
- Output: [`src/verify_dataset.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/verify_dataset.py).

### Phase 2: Preprocessing & DINOv2 Embedding Cache
- Build DICOM reader, brain bounding box cropper, low-coverage slice filter, z-score normalizer, and 224x224 padder.
- Load frozen DINOv2 model (`dinov2_vitb14`) and precompute 768-D slice embeddings + relative positions $p_i$ saved to `processed/embeddings/`.
- Output: [`src/preprocess.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/preprocess.py).

### Phase 3: Patient-Level Dataset & Stratified Splitter
- Create PyTorch `Dataset` loading cached embeddings and relative positions.
- Implement stratified patient-level splitter supporting the 34/7 split rule for `non_significant` cases.
- Output: [`src/dataset.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/dataset.py).

### Phase 4: Model Architectures & Attention Aggregators
- Implement `MeanPooling`, `AttentionPooling`, and `PositionAwareAttentionPooling`.
- Implement `AgeAwareClassifier` (Ablation 1) and `AgeIndependentClassifier` (Ablation 2).
- Output: [`src/models.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/models.py).

### Phase 5: Model Training Loop & Checkpoint Saving
- Build training engine with AdamW optimizer, cosine annealing scheduler, binary cross-entropy loss with class weights, early stopping, and metric tracking.
- Save full model bundle (`model_state_dict`, hyperparameters, normalization stats) to `checkpoints/best_module1_{ablation}.pt`.
- Output: [`src/train.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/train.py).

### Phase 6: Ablation Comparison & Final Evaluation
- Run side-by-side evaluation of Ablation 1 (Age-Aware) vs Ablation 2 (Age-Independent).
- Report accuracy, sensitivity, specificity, F1, ROC-AUC, and confusion matrix.
- Output: [`src/evaluate.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/evaluate.py).

### Phase 7: Model Serialization & Inference Interface
- Provide standalone `Module1InferenceEngine` API for one-line predictions from raw DICOM folders or embeddings.
- Output: [`src/inference.py`](file:///c:/Users/Akash/Downloads/Bro%20project/CODE/src/inference.py).
### Phase 8: Enhanced v2 Training (>750 Images) & External Cohort Benchmark
- Partitioned dataset into v2 split with **756 training cases** (> 750), **100 validation cases** (held-out BrainGen rest), and **130 external test cases** (datasets/test_data_mri1 with 	est_groundtruth.xlsx).
- Preserved existing v1 models (est_module1_age_aware.pt and est_module1_age_independent.pt) and serialized new models with _v2 suffix:
  - checkpoints/best_module1_age_aware_v2.pt: Test ROC-AUC **0.9045**, Test Accuracy **82.31%**, Sensitivity **78.67%**, Specificity **87.27%**.
  - checkpoints/best_module1_age_independent_v2.pt: Test ROC-AUC **0.9336**, Test Accuracy **84.62%**, Sensitivity **92.00%**, Specificity **74.55%**.
