import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_logger, compute_metrics
from src.dataset import create_patient_splits, PatientEmbeddingDataset, pad_collate_fn
from src.models import Module1Classifier

logger = get_logger("evaluate")

def evaluate_checkpoint(ablation="age_aware", version="v2", checkpoint_path=None):
    set_seed(Config.SEED)
    is_age_aware = (ablation == "age_aware")
    excel_path = Config.BRIDGE_EXCEL if is_age_aware else Config.BRIDGE_MODIFIED_EXCEL
    
    if checkpoint_path is not None:
        ckpt_path = checkpoint_path
    else:
        suffix = "_v2" if version == "v2" else ""
        ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, f"best_module1_{ablation}{suffix}.pt")
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}. Please run training first!")
        return
        
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
    pooling_type = checkpoint.get('pooling', 'position_attention')
    
    train_df, val_df, test_df = create_patient_splits(
        excel_path, 
        is_age_aware=is_age_aware, 
        seed=Config.SEED, 
        version=version
    )
    
    model = Module1Classifier(is_age_aware=is_age_aware, pooling_type=pooling_type).to(Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def _eval_df(df_split, name):
        dataset = PatientEmbeddingDataset(df_split, is_age_aware=is_age_aware)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
        all_targets, all_probs, all_ids = [], [], []
        with torch.no_grad():
            for batch in loader:
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
        logger.info(f"================ EVALUATION: {name} ================")
        logger.info(f"Patient Count:     {len(df_split)} (Normal: {(df_split['Classification']=='NORMAL').sum()}, Abnormal: {(df_split['Classification']=='ABNORMAL').sum()})")
        logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"Sensitivity:       {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity:       {metrics['specificity']:.4f}")
        logger.info(f"Precision:         {metrics['precision']:.4f}")
        logger.info(f"F1 Score:          {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"Confusion Matrix:  {metrics['confusion_matrix']}")
        logger.info("====================================================")
        return metrics

    logger.info(f"Evaluating Ablation: {ablation.upper()} | Version: {version.upper()}")
    val_metrics = _eval_df(val_df, "Validation Set (BrainGen Rest)")
    test_metrics = _eval_df(test_df, "Test Set (datasets/test_data_mri1)")
    
    return {'val': val_metrics, 'test': test_metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Module 1 Classifier Checkpoint")
    parser.add_argument("--ablation", type=str, choices=["age_aware", "age_independent"], default="age_aware")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to checkpoint .pt")
    args = parser.parse_args()
    evaluate_checkpoint(ablation=args.ablation, version=args.version, checkpoint_path=args.checkpoint)
