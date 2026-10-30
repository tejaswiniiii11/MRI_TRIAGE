import os
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import timm
from concurrent.futures import ThreadPoolExecutor

from src.config import Config
from src.utils import get_logger, set_seed

logger = get_logger("preprocess")

def load_single_dicom(fpath):
    try:
        ds = pydicom.dcmread(fpath)
        pos_z = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else float(getattr(ds, 'InstanceNumber', 0))
        return (pos_z, ds)
    except Exception:
        return None

def load_patient_dicom_slices(patient_dir, max_workers=8):
    """Reads and orders DICOM slices spatially for a patient using multi-threading."""
    dcm_files = []
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.endswith(('.dcm', '.DCM', '')) and not f.startswith('.'):
                dcm_files.append(os.path.join(root, f))
                
    if not dcm_files:
        return []
        
    slices = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_single_dicom, dcm_files)
        for res in results:
            if res is not None:
                slices.append(res)
            
    slices.sort(key=lambda x: x[0])
    return [s[1] for s in slices]


def preprocess_slice(pixel_array, target_size=224):
    """
    Preprocesses a single 2D slice:
    - Robust intensity z-score normalization
    - Aspect ratio preserving resize & square padding
    - Duplicate grayscale to 3 identical RGB channels
    """
    img = pixel_array.astype(np.float32)
    
    # Brain tissue intensity normalization
    non_zero = img[img > 0]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, (1, 99))
        img = np.clip(img, p1, p99)
        mean_val, std_val = non_zero.mean(), non_zero.std() + 1e-8
        img = (img - mean_val) / std_val
    else:
        img = img - img.mean()

    # Convert to 8-bit image for PIL operations
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
        
    pil_img = Image.fromarray(img_norm).convert("RGB")
    
    # Aspect-ratio preserving resize & padding
    w, h = pil_img.size
    scale = target_size / float(max(w, h))
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded_img.paste(resized_img, (pad_left, pad_top))
    
    return padded_img


def extract_patient_embeddings(model, dicom_slices, device=Config.DEVICE):
    """
    Filters low-coverage slices, extracts DINOv2 embeddings,
    and computes relative slice positions.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_images = []
    
    for ds in dicom_slices:
        arr = ds.pixel_array
        
        # Simple threshold for brain tissue coverage check (>5% non-background)
        brain_ratio = np.mean(arr > (arr.max() * 0.1))
        if brain_ratio < Config.MIN_BRAIN_COVERAGE_RATIO:
            continue
            
        prep_img = preprocess_slice(arr, target_size=Config.IMAGE_SIZE)
        tensor_img = transform(prep_img)
        valid_images.append(tensor_img)
        
    # If all slices filtered out, keep at least middle slices
    if not valid_images:
        for ds in dicom_slices:
            prep_img = preprocess_slice(ds.pixel_array, target_size=Config.IMAGE_SIZE)
            valid_images.append(transform(prep_img))

    N = len(valid_images)
    batch_tensors = torch.stack(valid_images).to(device)  # (N, 3, 224, 224)
    
    # Calculate relative positions p_i in [0.0, 1.0]
    if N > 1:
        positions = torch.linspace(0.0, 1.0, steps=N, dtype=torch.float32)
    else:
        positions = torch.tensor([0.5], dtype=torch.float32)
        
    with torch.no_grad():
        embeddings = model(batch_tensors)  # (N, 768)
        
    return embeddings.cpu(), positions.cpu()


def run_preprocessing():
    set_seed(Config.SEED)
    Config.ensure_dirs()
    
    logger.info("Initializing Frozen DINOv2 Feature Extractor (dinov2_vitb14)...")
    device = Config.DEVICE
    dinov2 = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
    dinov2 = dinov2.to(device)
    dinov2.eval()
    
    df1 = pd.read_excel(Config.BRIDGE_EXCEL)
    patient_ids = df1['ANON_PatientID'].tolist()
    
    logger.info(f"Processing and caching DINOv2 slice embeddings for {len(patient_ids)} patients...")
    
    cached_count = 0
    for pid in tqdm(patient_ids, desc="Caching DINOv2 Embeddings"):
        save_path = os.path.join(Config.EMBEDDINGS_DIR, f"{pid}.pt")
        if os.path.exists(save_path):
            cached_count += 1
            continue
            
        p_dir = os.path.join(Config.DATASET_DIR, pid)
        slices = load_patient_dicom_slices(p_dir)
        
        if not slices:
            logger.warning(f"No valid DICOM slices found for patient {pid}")
            continue
            
        embeddings, positions = extract_patient_embeddings(dinov2, slices, device=device)
        
        torch.save({
            'patient_id': pid,
            'embeddings': embeddings,  # Shape: (N, 768)
            'positions': positions     # Shape: (N,)
        }, save_path)
        
        cached_count += 1
        
    logger.info(f"✓ Feature pre-caching complete. Total cached patient files: {cached_count}/{len(patient_ids)}")

if __name__ == "__main__":
    run_preprocessing()
