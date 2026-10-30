import os
from src.inference import Module1InferenceEngine

# Load trained Age-Aware model bundle
engine = Module1InferenceEngine("checkpoints/best_module1_age_aware.pt")

# Predict directly from raw DICOM folder
sample_dicom_dir = "../datasets/BrainGen_deidentified/995030-000259"
if not os.path.exists(sample_dicom_dir):
    sample_dicom_dir = "../BrainGen_deidentified/995030-000259"

res = engine.predict_from_dicom_folder(sample_dicom_dir, age=40)

print("\n================ INFERENCE RESULT ================")
print(f"Patient ID:              995030-000259")
print(f"Predicted Classification: {res['prediction']}")
print(f"Abnormality Probability:  {res['abnormality_probability']:.4f}")
print(f"Confidence:               {res['confidence']:.4f}")
print(f"Slice Attention Weights:  {[round(w, 4) for w in res['slice_attention_weights'][:5]]}")
print("==================================================")
