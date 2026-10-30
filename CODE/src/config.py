import os

class Config:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # Root Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Dataset Paths (Mount points inside container or local paths)
    DATASET_DIR = os.environ.get(
        "DATASET_DIR", 
        os.path.join(BASE_DIR, "data", "BrainGen_deidentified")
        if os.path.exists(os.path.join(BASE_DIR, "data", "BrainGen_deidentified"))
        else (
            os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "BrainGen_deidentified"))
            if os.path.exists(os.path.join(BASE_DIR, "..", "datasets", "BrainGen_deidentified"))
            else os.path.abspath(os.path.join(BASE_DIR, "..", "BrainGen_deidentified"))
        )
    )
    
    BRIDGE_EXCEL = os.environ.get(
        "BRIDGE_EXCEL",
        os.path.join(BASE_DIR, "data", "bridge.xlsx")
        if os.path.exists(os.path.join(BASE_DIR, "data", "bridge.xlsx"))
        else os.path.abspath(os.path.join(BASE_DIR, "..", "bridge.xlsx"))
    )
    
    BRIDGE_MODIFIED_EXCEL = os.environ.get(
        "BRIDGE_MODIFIED_EXCEL",
        os.path.join(BASE_DIR, "data", "bridge_modified.xlsx")
        if os.path.exists(os.path.join(BASE_DIR, "data", "bridge_modified.xlsx"))
        else os.path.abspath(os.path.join(BASE_DIR, "..", "bridge_modified.xlsx"))
    )
    
    # Output Directories
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
    EMBEDDINGS_DIR = os.path.join(PROCESSED_DIR, "embeddings")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Preprocessing Specs
    IMAGE_SIZE = 224
    DINOV2_MODEL_NAME = "dinov2_vitb14"
    EMBEDDING_DIM = 768
    MIN_BRAIN_COVERAGE_RATIO = 0.05  # Filter slices with < 5% brain pixels
    
    # Training & Split Specs
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    NON_SIG_TRAIN_COUNT = 34
    NON_SIG_VAL_COUNT = 7
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUTS_DIR, exist_ok=True)
