import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_logger, compute_metrics
from src.dataset import create_patient_splits, PatientEmbeddingDataset, pad_collate_fn
from src.models import Module1Classifier

logger = get_logger("train")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_targets, all_probs = [], []
    
    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        ages = batch['ages'].to(device) if 'ages' in batch else None
        
        optimizer.zero_grad()
        logits, _ = model(embeddings, positions, mask=mask, age=ages)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_probs)
    metrics['loss'] = avg_loss
    return metrics


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets, all_probs = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            ages = batch['ages'].to(device) if 'ages' in batch else None
            
            logits, _ = model(embeddings, positions, mask=mask, age=ages)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * targets.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_probs)
    metrics['loss'] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Module 1 Classifier")
    parser.add_argument("--ablation", type=str, choices=["age_aware", "age_independent"], default="age_aware",
                        help="Select Ablation 1 (age_aware) or Ablation 2 (age_independent)")
    parser.add_argument("--pooling", type=str, choices=["position_attention", "attention", "mean"], default="position_attention")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    args = parser.parse_args()

    set_seed(Config.SEED)
    Config.ensure_dirs()
    
    is_age_aware = (args.ablation == "age_aware")
    excel_path = Config.BRIDGE_EXCEL if is_age_aware else Config.BRIDGE_MODIFIED_EXCEL
    
    logger.info(f"Starting Training for Module 1 | Ablation: {args.ablation.upper()} | Pooling: {args.pooling.upper()}")
    
    # Create patient splits
    train_df, val_df, test_df = create_patient_splits(excel_path, is_age_aware=is_age_aware, seed=Config.SEED)
    
    # Create datasets and dataloaders
    train_dataset = PatientEmbeddingDataset(train_df, is_age_aware=is_age_aware)
    val_dataset = PatientEmbeddingDataset(val_df, is_age_aware=is_age_aware)
    test_dataset = PatientEmbeddingDataset(test_df, is_age_aware=is_age_aware)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # Compute pos_weight for loss if imbalanced
    if not is_age_aware:
        num_neg = (train_df['Classification'] == 'NORMAL').sum()
        num_pos = (train_df['Classification'] == 'ABNORMAL').sum()
        pos_weight = torch.tensor([num_neg / float(num_pos)]).to(Config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    model = Module1Classifier(is_age_aware=is_age_aware, pooling_type=args.pooling).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_auc = 0.0
    best_ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, f"best_module1_{args.ablation}.pt")
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_metrics = evaluate_epoch(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()
        
        logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['roc_auc']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")
        
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0
            
            # Save comprehensive model bundle for persistence and deployment
            bundle = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'ablation': args.ablation,
                'pooling': args.pooling,
                'is_age_aware': is_age_aware,
                'embedding_dim': Config.EMBEDDING_DIM,
                'age_mean': train_df['Age'].mean() if ('Age' in train_df.columns and is_age_aware) else None,
                'age_std': train_df['Age'].std() if ('Age' in train_df.columns and is_age_aware) else None,
            }
            torch.save(bundle, best_ckpt_path)
            logger.info(f"  --> Saved new best checkpoint bundle to {best_ckpt_path} (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Final evaluation on test set
    logger.info("Evaluating Best Checkpoint on Unseen Test Partition...")
    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_epoch(model, test_loader, criterion, Config.DEVICE)
    
    logger.info("================ FINAL TEST RESULTS ================")
    logger.info(f"Ablation Strategy:  {args.ablation.upper()}")
    logger.info(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Sensitivity:        {test_metrics['sensitivity']:.4f}")
    logger.info(f"Specificity:        {test_metrics['specificity']:.4f}")
    logger.info(f"F1 Score:           {test_metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    logger.info(f"Confusion Matrix:   {test_metrics['confusion_matrix']}")
    logger.info("====================================================")

if __name__ == "__main__":
    main()
