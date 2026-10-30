import os
from src.inference import Module1InferenceEngine

script_dir = os.path.dirname(os.path.abspath(__file__))

# Prefer v2 trained model bundle if available, fallback to v1
ckpt_path = os.path.join(script_dir, "checkpoints", "best_module1_age_aware_v2.pt")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(script_dir, "checkpoints", "best_module1_age_aware.pt")

print(f"Loading checkpoint: {ckpt_path}")
engine = Module1InferenceEngine(ckpt_path)

# Predict directly from raw DICOM folder
sample_dicom_dir = os.path.abspath(os.path.join(script_dir, "..", "datasets", "BrainGen_deidentified", "995030-000259"))
if not os.path.exists(sample_dicom_dir):
    sample_dicom_dir = os.path.abspath(os.path.join(script_dir, "..", "datasets", "test_data_mri1", "996299-000001"))

patient_id = os.path.basename(sample_dicom_dir)
res = engine.predict_from_dicom_folder(sample_dicom_dir, age=40)

print("\n================ INFERENCE RESULT ================")
print(f"Patient ID:              {patient_id}")
print(f"Predicted Classification: {res['prediction']}")
print(f"Abnormality Probability:  {res['abnormality_probability']:.4f}")
print(f"Confidence:               {res['confidence']:.4f}")
print(f"Slice Attention Weights:  {[round(w, 4) for w in res['slice_attention_weights'][:5]]}")
print("==================================================")
