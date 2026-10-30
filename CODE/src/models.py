import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAwareAttentionPooling(nn.Module):
    """
    Position-Aware Attention Pooling Aggregator.
    Combines slice embedding (768-D) with relative position (1-D)
    to dynamically learn slice attention weights alpha_i.
    """
    def __init__(self, in_dim=768, hidden_dim=128):
        super().__init__()
        # Input dimension is 768 + 1 (positional feature)
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, embeddings, positions, mask=None):
        """
        Args:
            embeddings: Tensor of shape (B, N, 768)
            positions: Tensor of shape (B, N)
            mask: Tensor of shape (B, N), True for valid slices, False for padding
        Returns:
            patient_embeddings: Tensor of shape (B, 768)
            alpha: Tensor of shape (B, N) attention weights
        """
        B, N, D = embeddings.shape
        
        # Concatenate position value (B, N, 1) to feature vectors (B, N, 768)
        pos_expanded = positions.unsqueeze(-1)  # (B, N, 1)
        x = torch.cat([embeddings, pos_expanded], dim=-1)  # (B, N, 769)
        
        # Calculate raw scores e_i
        attn_logits = self.attn_mlp(x).squeeze(-1)  # (B, N)
        
        # Mask out padded positions prior to Softmax
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
            
        # Compute normalized attention weights alpha_i
        alpha = F.softmax(attn_logits, dim=-1)  # (B, N)
        
        # Aggregate slice embeddings: sum_i (alpha_i * z_i)
        patient_embeddings = torch.bmm(alpha.unsqueeze(1), embeddings).squeeze(1)  # (B, 768)
        
        return patient_embeddings, alpha


class MeanPooling(nn.Module):
    """Baseline Mean Pooling Aggregator."""
    def __init__(self):
        super().__init__()
        
    def forward(self, embeddings, positions=None, mask=None):
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()  # (B, N, 1)
            sum_emb = (embeddings * mask_float).sum(dim=1)
            count = mask_float.sum(dim=1).clamp(min=1.0)
            patient_embeddings = sum_emb / count
        else:
            patient_embeddings = embeddings.mean(dim=1)
        return patient_embeddings, None


class StandardAttentionPooling(nn.Module):
    """Standard Attention Pooling Aggregator (without position info)."""
    def __init__(self, in_dim=768, hidden_dim=128):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, embeddings, positions=None, mask=None):
        attn_logits = self.attn_mlp(embeddings).squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
        alpha = F.softmax(attn_logits, dim=-1)
        patient_embeddings = torch.bmm(alpha.unsqueeze(1), embeddings).squeeze(1)
        return patient_embeddings, alpha


class Module1Classifier(nn.Module):
    """
    Module 1 Patient-Level General Abnormality Classifier.
    Supports both Ablation 1 (Age-Aware) and Ablation 2 (Age-Independent).
    """
    def __init__(self, is_age_aware=True, pooling_type="position_attention", emb_dim=768):
        super().__init__()
        self.is_age_aware = is_age_aware
        
        # Select pooling aggregator
        if pooling_type == "position_attention":
            self.pooler = PositionAwareAttentionPooling(in_dim=emb_dim)
        elif pooling_type == "attention":
            self.pooler = StandardAttentionPooling(in_dim=emb_dim)
        elif pooling_type == "mean":
            self.pooler = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
            
        # MLP Classifier Head
        head_in_dim = emb_dim + (1 if is_age_aware else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, embeddings, positions, mask=None, age=None):
        """
        Args:
            embeddings: (B, N, 768)
            positions: (B, N)
            mask: (B, N)
            age: (B,) normalized age if age-aware
        """
        patient_emb, alpha = self.pooler(embeddings, positions, mask=mask)
        
        if self.is_age_aware:
            if age is None:
                raise ValueError("Age feature is required for age-aware classifier!")
            h = torch.cat([patient_emb, age.unsqueeze(-1)], dim=-1)  # (B, 769)
        else:
            h = patient_emb  # (B, 768)
            
        logits = self.classifier(h).squeeze(-1)  # (B,)
        return logits, alpha
