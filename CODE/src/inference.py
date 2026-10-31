import os
import torch
import numpy as np
import timm

from src.config import Config
from src.utils import get_logger
from src.models import Module1Classifier
from src.preprocess import load_patient_dicom_slices, extract_patient_embeddings

logger = get_logger("inference")

class Module1InferenceEngine:
    """
    Inference Engine for Module 1 (General Abnormality Classifier).
    Loads a saved model checkpoint bundle and provides easy prediction APIs.
    """
    def __init__(self, checkpoint_path, device=Config.DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint bundle not found at {checkpoint_path}")
            
        logger.info(f"Loading Model Checkpoint Bundle from {checkpoint_path}...")
        bundle = torch.load(checkpoint_path, map_location=device)
        
        self.is_age_aware = bundle.get('is_age_aware', True)
        self.pooling_type = bundle.get('pooling', 'position_attention')
        self.ablation = bundle.get('ablation', 'age_aware')
        self.embedding_dim = bundle.get('embedding_dim', 768)
        self.age_mean = bundle.get('age_mean', None)
        self.age_std = bundle.get('age_std', None)
        
        # Instantiate and load model weights
        self.model = Module1Classifier(
            is_age_aware=self.is_age_aware,
            pooling_type=self.pooling_type,
            emb_dim=self.embedding_dim
        ).to(device)
        
        self.model.load_state_dict(bundle['model_state_dict'])
        self.model.eval()
        
        # Feature extractor for raw DICOM inference
        self._dinov2 = None
        logger.info(f"✓ Module 1 Model Loaded successfully! (Ablation: {self.ablation}, Pooling: {self.pooling_type})")

    def _get_dinov2(self):
        if self._dinov2 is None:
            logger.info("Initializing DINOv2 Feature Extractor for inference...")
            self._dinov2 = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
            self._dinov2 = self._dinov2.to(self.device)
            self._dinov2.eval()
        return self._dinov2

    def predict_from_embeddings(self, embeddings, positions, age=None, threshold=0.5):
        """
        Predicts binary classification label and abnormality probability from precomputed embeddings.
        Args:
            embeddings: Tensor (N, 768) or (1, N, 768)
            positions: Tensor (N,) or (1, N)
            age: raw patient age (float) if age-aware
            threshold: classification threshold (default 0.5)
        Returns:
            dict containing probability, predicted_class, attention_weights
        """
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)  # (1, N, 768)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)    # (1, N)
            
        embeddings = embeddings.to(self.device)
        positions = positions.to(self.device)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=self.device)
        
        age_tensor = None
        if self.is_age_aware:
            if age is None:
                raise ValueError("Patient age is required for Age-Aware model inference!")
            if self.age_mean is not None and self.age_std is not None:
                norm_age = (float(age) - self.age_mean) / self.age_std
            else:
                norm_age = float(age)
            age_tensor = torch.tensor([norm_age], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, alpha = self.model(embeddings, positions, mask=mask, age=age_tensor)
            prob = torch.sigmoid(logits).item()
            
        pred_label = "ABNORMAL" if prob >= threshold else "NORMAL"
        attn_weights = alpha.squeeze(0).cpu().numpy().tolist()
        pos_list = positions.squeeze(0).cpu().numpy().tolist()
        
        # Build slice-by-slice detailed breakdown
        slice_breakdown = [
            {
                "slice_index": i,
                "relative_position": round(pos_list[i], 3),
                "attention_weight": round(attn_weights[i], 6),
                "attention_percentage": round(attn_weights[i] * 100.0, 2)
            }
            for i in range(len(attn_weights))
        ]
        
        return {
            "prediction": pred_label,
            "abnormality_probability": prob,
            "confidence": prob if pred_label == "ABNORMAL" else (1.0 - prob),
            "slice_attention_weights": attn_weights,
            "slice_attention_breakdown": slice_breakdown,
            "ablation": self.ablation
        }

    def get_patient_slice_attention(self, patient_id, age=None):
        """
        Convenience method: Retrieves slice attention percentages and ranks them
        for any patient by their Patient ID using precomputed embeddings.
        """
        emb_path = os.path.join(Config.EMBEDDINGS_DIR, f"{patient_id}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found for patient {patient_id} at {emb_path}")
            
        data = torch.load(emb_path, map_location=self.device)
        res = self.predict_from_embeddings(data['embeddings'], data['positions'], age=age)
        
        breakdown = res['slice_attention_breakdown']
        ranked = sorted(breakdown, key=lambda x: x['attention_percentage'], reverse=True)
        
        return {
            "patient_id": patient_id,
            "prediction": res["prediction"],
            "abnormality_probability": round(res["abnormality_probability"], 4),
            "total_slices": len(breakdown),
            "slices": breakdown,
            "top_contributing_slices": ranked
        }

    def predict_from_dicom_folder(self, dicom_folder, age=None, threshold=0.5):
        """
        End-to-end inference directly from a raw patient DICOM directory.
        """
        slices = load_patient_dicom_slices(dicom_folder)
        if not slices:
            raise ValueError(f"No valid DICOM slices found in {dicom_folder}")
            
        dinov2 = self._get_dinov2()
        embeddings, positions = extract_patient_embeddings(dinov2, slices, device=self.device)
        
        return self.predict_from_embeddings(embeddings, positions, age=age, threshold=threshold)


if __name__ == "__main__":
    import sys
    ckpt = os.path.join(Config.CHECKPOINTS_DIR, "best_module1_age_aware.pt")
    if os.path.exists(ckpt):
        engine = Module1InferenceEngine(ckpt)
        print("Engine initialized successfully.")
    else:
        print(f"Checkpoint not found at {ckpt}. Run training first.")
