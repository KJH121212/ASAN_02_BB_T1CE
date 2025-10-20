# ================================================================
# scripts/move_t1ce_files.py
# Move or copy T1CE files to match BB dataset structure
# ================================================================

import os
import shutil
from glob import glob
from tqdm import tqdm

def move_t1ce_files(bb_dir, t1ce_src_dir, t1ce_dst_dir):
    """
    BB/T1CE 쌍 데이터 정리를 위해 동일 환자 ID 기반으로 T1CE 파일 이동.
    예: 12345_bb_coreg.nii.gz → 12345_t1ce_coreg.nii.gz
    """
    os.makedirs(t1ce_dst_dir, exist_ok=True)
    bb_files = sorted(glob(os.path.join(bb_dir, "*.nii.gz")))

    for bb_file in tqdm(bb_files):
        fname = os.path.basename(bb_file)
        patient_id = fname.split("_")[0]
        t1ce_pattern = f"{patient_id}_*_t1ce_coreg.nii.gz"
        t1ce_matches = glob(os.path.join(t1ce_src_dir, t1ce_pattern))
        if len(t1ce_matches) == 1:
            shutil.copy2(t1ce_matches[0], os.path.join(t1ce_dst_dir, os.path.basename(t1ce_matches[0])))
        elif len(t1ce_matches) > 1:
            print(f"⚠️ Multiple T1CE matches for {patient_id}")
        else:
            print(f"❌ No T1CE found for {patient_id}")

if __name__ == "__main__":
    BB_DIR = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-bb"
    T1CE_SRC = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/source_t1ce"
    T1CE_DST = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-t1ce"
    move_t1ce_files(BB_DIR, T1CE_SRC, T1CE_DST)
