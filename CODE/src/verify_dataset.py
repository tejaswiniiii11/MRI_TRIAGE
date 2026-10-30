import os
import pandas as pd
import pydicom
from src.config import Config
from src.utils import get_logger

logger = get_logger("verify_dataset")

def verify():
    logger.info("Starting Dataset and Metadata Verification...")
    
    if not os.path.exists(Config.BRIDGE_EXCEL):
        logger.error(f"bridge.xlsx not found at {Config.BRIDGE_EXCEL}")
        return False
        
    if not os.path.exists(Config.BRIDGE_MODIFIED_EXCEL):
        logger.error(f"bridge_modified.xlsx not found at {Config.BRIDGE_MODIFIED_EXCEL}")
        return False
        
    if not os.path.exists(Config.DATASET_DIR):
        logger.error(f"Dataset directory not found at {Config.DATASET_DIR}")
        return False

    df1 = pd.read_excel(Config.BRIDGE_EXCEL)
    df2 = pd.read_excel(Config.BRIDGE_MODIFIED_EXCEL)

    logger.info(f"Loaded bridge.xlsx shape: {df1.shape}")
    logger.info(f"Loaded bridge_modified.xlsx shape: {df2.shape}")

    # Check ID alignment
    if not (df1['ANON_PatientID'] == df2['ANON_PatientID']).all():
        logger.error("Patient IDs in bridge.xlsx and bridge_modified.xlsx do NOT match!")
        return False
    logger.info("✓ Patient IDs between bridge.xlsx and bridge_modified.xlsx match 100%.")

    # Check Class distributions
    logger.info(f"Ablation 1 (bridge.xlsx) class distribution: {df1['Classification'].value_counts().to_dict()}")
    logger.info(f"Ablation 2 (bridge_modified.xlsx) class distribution: {df2['Classification'].value_counts().to_dict()}")

    # Check 41 non_significant mapping
    non_sig_df1 = df1[df1['Significance'] == 'non_significant']
    non_sig_pids = set(non_sig_df1['ANON_PatientID'])
    logger.info(f"Total non_significant cases identified in bridge.xlsx: {len(non_sig_pids)}")
    
    df2_non_sig_classes = df2[df2['ANON_PatientID'].isin(non_sig_pids)]['Classification'].value_counts().to_dict()
    logger.info(f"Classes of these 41 cases in bridge_modified.xlsx: {df2_non_sig_classes}")
    if df2_non_sig_classes.get('ABNORMAL', 0) == 41:
        logger.info("✓ All 41 non_significant cases are correctly relabeled to ABNORMAL in bridge_modified.xlsx.")

    # Check folder matching
    excel_patients = list(df1['ANON_PatientID'])
    dir_folders = set(os.listdir(Config.DATASET_DIR))
    
    missing_folders = [pid for pid in excel_patients if pid not in dir_folders]
    if missing_folders:
        logger.error(f"Missing {len(missing_folders)} patient folders in dataset directory!")
        return False
    logger.info(f"✓ All {len(excel_patients)} patients present in {Config.DATASET_DIR}.")

    # Inspect DICOM headers
    slice_counts = []
    series_descs = set()
    uid_mismatches = 0
    
    for idx, row in df1.iterrows():
        pid = row['ANON_PatientID']
        p_uid = row['ANON_StudyInstanceUID']
        p_dir = os.path.join(Config.DATASET_DIR, pid)
        
        dcms = []
        for root, _, files in os.walk(p_dir):
            for f in files:
                if f.endswith(('.dcm', '.DCM', '')) and not f.startswith('.'):
                    dcms.append(os.path.join(root, f))
                    
        if not dcms:
            logger.error(f"No DICOM files found for patient {pid}!")
            return False
            
        slice_counts.append(len(dcms))
        
        try:
            ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
            d_uid = getattr(ds, 'StudyInstanceUID', None)
            s_desc = getattr(ds, 'SeriesDescription', 'N/A')
            series_descs.add(str(s_desc))
            
            if d_uid and str(d_uid).strip() != str(p_uid).strip():
                uid_mismatches += 1
        except Exception as e:
            logger.error(f"Error reading DICOM for patient {pid}: {e}")
            return False

    logger.info(f"✓ All {len(slice_counts)} patient DICOM sets inspected successfully.")
    logger.info(f"✓ Slice counts per patient: Min={min(slice_counts)}, Max={max(slice_counts)}, Mean={sum(slice_counts)/len(slice_counts):.2f}")
    logger.info(f"✓ StudyInstanceUID mismatches: {uid_mismatches}")
    logger.info(f"✓ Series descriptions found: {list(series_descs)}")

    # Verify Test Dataset if available
    if os.path.exists(Config.TEST_GROUNDTRUTH_EXCEL) and os.path.exists(Config.TEST_DATASET_DIR):
        logger.info("\n--- Verifying Test Cohort (datasets/test_data_mri1) ---")
        df_test = pd.read_excel(Config.TEST_GROUNDTRUTH_EXCEL)
        pid_col = 'ANON_PatientID' if 'ANON_PatientID' in df_test.columns else 'ANON-PatientID'
        test_patients = list(df_test[pid_col])
        test_dir_folders = set(os.listdir(Config.TEST_DATASET_DIR))
        missing_test = [pid for pid in test_patients if pid not in test_dir_folders]
        if missing_test:
            logger.error(f"Missing {len(missing_test)} patient folders in test dataset directory!")
            return False
        logger.info(f"✓ All {len(test_patients)} test patients found in {Config.TEST_DATASET_DIR}.")
        logger.info(f"✓ Test class distribution: {df_test['Classification'].value_counts().to_dict()}")

    logger.info("==========================================")
    logger.info("VERIFICATION PASSED! Dataset is READY.")
    logger.info("==========================================")
    return True

if __name__ == "__main__":
    verify()
