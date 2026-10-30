import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.config import Config
from src.utils import get_logger

logger = get_logger("dataset")

class PatientEmbeddingDataset(Dataset):
    """
    PyTorch Dataset loading pre-cached DINOv2 slice embeddings and relative positions.
    """
    def __init__(self, metadata_df, embeddings_dir=Config.EMBEDDINGS_DIR, is_age_aware=True, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir
        self.is_age_aware = is_age_aware
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['ANON_PatientID']
        
        # Load cached embedding file: contains 'embeddings' (N, 768) and 'positions' (N,)
        emb_path = os.path.join(self.embeddings_dir, f"{pid}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Cached embedding file not found for patient {pid} at {emb_path}")
            
        data = torch.load(emb_path)
        embeddings = data['embeddings']  # Tensor of shape (N, 768)
        positions = data['positions']    # Tensor of shape (N,)
        
        # Binary target label (NORMAL -> 0, ABNORMAL -> 1)
        label_str = str(row['Classification']).upper().strip()
        target = torch.tensor(1.0 if label_str == 'ABNORMAL' else 0.0, dtype=torch.float32)
        
        sample = {
            'patient_id': pid,
            'embeddings': embeddings,
            'positions': positions,
            'target': target
        }
        
        if self.is_age_aware:
            # Normalized age feature if age-aware experiment
            norm_age = torch.tensor(row['normalized_age'], dtype=torch.float32)
            sample['age'] = norm_age
            
        return sample


def create_patient_splits(excel_path, is_age_aware=True, seed=Config.SEED):
    """
    Creates patient-level train, validation, and test splits.
    Guarantees no slice or patient leakage across split partitions.
    For Age-Aware (Ablation 1): Enforces ~34 non_significant cases in train and ~7 in val.
    """
    df = pd.read_excel(excel_path)
    np.random.seed(seed)
    
    # 1. Age normalization (computed using training statistics later, initializing raw age)
    df['normalized_age'] = 0.0
    
    if is_age_aware and 'Significance' in df.columns:
        # Separate non_significant cases for controlled allocation
        non_sig_df = df[df['Significance'] == 'non_significant'].sample(frac=1, random_state=seed)
        other_df = df[df['Significance'] != 'non_significant'].sample(frac=1, random_state=seed)
        
        # Split 34 train, 7 val from the 41 non_significant cases
        non_sig_train = non_sig_df.iloc[:Config.NON_SIG_TRAIN_COUNT]
        non_sig_val = non_sig_df.iloc[Config.NON_SIG_TRAIN_COUNT:]
        
        # Stratified split for the remaining 815 cases
        from sklearn.model_selection import train_test_split
        train_other, val_test_other = train_test_split(
            other_df, 
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            stratify=other_df['Classification'],
            random_state=seed
        )
        val_other, test_other = train_test_split(
            val_test_other,
            test_size=0.5,
            stratify=val_test_other['Classification'],
            random_state=seed
        )
        
        train_df = pd.concat([train_other, non_sig_train]).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df = pd.concat([val_other, non_sig_val]).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = test_other.sample(frac=1, random_state=seed).reset_index(drop=True)
        
    else:
        # Standard stratified patient-level split for Ablation 2
        from sklearn.model_selection import train_test_split
        train_df, val_test_df = train_test_split(
            df,
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            stratify=df['Classification'],
            random_state=seed
        )
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=0.5,
            stratify=val_test_df['Classification'],
            random_state=seed
        )
        
    # Calculate age normalization stats STRICTLY from training set
    if 'Age' in df.columns:
        age_mean = train_df['Age'].mean()
        age_std = train_df['Age'].std()
        
        train_df['normalized_age'] = (train_df['Age'] - age_mean) / age_std
        val_df['normalized_age'] = (val_df['Age'] - age_mean) / age_std
        test_df['normalized_age'] = (test_df['Age'] - age_mean) / age_std
        
    logger.info(f"Created Patient Splits (Age-Aware={is_age_aware}):")
    logger.info(f"  Train Patients: {len(train_df)} (Normal: {(train_df['Classification']=='NORMAL').sum()}, Abnormal: {(train_df['Classification']=='ABNORMAL').sum()})")
    logger.info(f"  Val Patients:   {len(val_df)} (Normal: {(val_df['Classification']=='NORMAL').sum()}, Abnormal: {(val_df['Classification']=='ABNORMAL').sum()})")
    logger.info(f"  Test Patients:  {len(test_df)} (Normal: {(test_df['Classification']=='NORMAL').sum()}, Abnormal: {(test_df['Classification']=='ABNORMAL').sum()})")
    
    return train_df, val_df, test_df


def pad_collate_fn(batch):
    """
    Custom collate function to handle variable slice counts within a batch via padding.
    """
    patient_ids = [b['patient_id'] for b in batch]
    targets = torch.stack([b['target'] for b in batch])
    
    has_age = 'age' in batch[0]
    ages = torch.stack([b['age'] for b in batch]) if has_age else None
    
    # Get slice count per patient in batch
    lengths = [b['embeddings'].size(0) for b in batch]
    max_len = max(lengths)
    emb_dim = batch[0]['embeddings'].size(1)
    
    padded_embeddings = torch.zeros(len(batch), max_len, emb_dim)
    padded_positions = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        l = lengths[i]
        padded_embeddings[i, :l] = b['embeddings']
        padded_positions[i, :l] = b['positions']
        mask[i, :l] = True  # True indicates valid slice, False indicates padded position
        
    out = {
        'patient_ids': patient_ids,
        'embeddings': padded_embeddings,   # Shape: (B, max_N, 768)
        'positions': padded_positions,     # Shape: (B, max_N)
        'mask': mask,                       # Shape: (B, max_N)
        'targets': targets
    }
    
    if ages is not None:
        out['ages'] = ages
        
    return out
