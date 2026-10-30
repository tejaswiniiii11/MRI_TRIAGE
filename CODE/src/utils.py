import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name="brain_mri_module1"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes standard clinical & statistical evaluation metrics.
    """
    y_pred_probs = np.array(y_pred_probs)
    y_true = np.array(y_true)
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    
    # Specificity calculation from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0
        
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }
