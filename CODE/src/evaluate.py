import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_logger, compute_metrics
from src.dataset import create_patient_splits, PatientEmbeddingDataset, pad_collate_fn
from src.models import Module1Classifier

logger = get_logger("evaluate")

def evaluate_checkpoint(ablation="age_aware"):
    set_seed(Config.SEED)
    is_age_aware = (ablation == "age_aware")
    excel_path = Config.BRIDGE_EXCEL if is_age_aware else Config.BRIDGE_MODIFIED_EXCEL
    ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, f"best_module1_{ablation}.pt")
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Please run training first!")
        return
        
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
    pooling_type = checkpoint.get('pooling', 'position_attention')
    
    _, _, test_df = create_patient_splits(excel_path, is_age_aware=is_age_aware, seed=Config.SEED)
    test_dataset = PatientEmbeddingDataset(test_df, is_age_aware=is_age_aware)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    model = Module1Classifier(is_age_aware=is_age_aware, pooling_type=pooling_type).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_targets, all_probs, all_ids = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(Config.DEVICE)
            positions = batch['positions'].to(Config.DEVICE)
            mask = batch['mask'].to(Config.DEVICE)
            targets = batch['targets'].to(Config.DEVICE)
            ages = batch['ages'].to(Config.DEVICE) if 'ages' in batch else None
            
            logits, _ = model(embeddings, positions, mask=mask, age=ages)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(batch['patient_ids'])
            
    metrics = compute_metrics(all_targets, all_probs)
    
    logger.info("================ EVALUATION SUMMARY ================")
    logger.info(f"Ablation Strategy:  {ablation.upper()}")
    logger.info(f"Test Patient Count: {len(test_df)}")
    logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Sensitivity:        {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity:        {metrics['specificity']:.4f}")
    logger.info(f"Precision:          {metrics['precision']:.4f}")
    logger.info(f"F1 Score:           {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    logger.info(f"Confusion Matrix:   {metrics['confusion_matrix']}")
    logger.info("====================================================")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Module 1 Classifier Checkpoint")
    parser.add_argument("--ablation", type=str, choices=["age_aware", "age_independent"], default="age_aware")
    args = parser.parse_args()
    evaluate_checkpoint(ablation=args.ablation)
