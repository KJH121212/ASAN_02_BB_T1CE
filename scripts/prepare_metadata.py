# ================================================================
# scripts/prepare_metadata.py
# Generate metadata CSV for BB→T1CE dataset
# ================================================================

import os
import csv
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def prepare_metadata(bb_dir, t1ce_dir, save_csv):
    """
    BB / T1CE 폴더를 순회하며 환자 ID, 파일 경로, 이미지 크기 등을 CSV로 저장
    """
    bb_files = sorted(glob(os.path.join(bb_dir, "*.nii.gz")))
    t1ce_files = sorted(glob(os.path.join(t1ce_dir, "*.nii.gz")))

    assert len(bb_files) == len(t1ce_files), "❌ BB / T1CE pair count mismatch"

    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "bb_path", "t1ce_path", "shape_z", "shape_y", "shape_x"])
        for bb_path, t1ce_path in tqdm(zip(bb_files, t1ce_files), total=len(bb_files)):
            pid = os.path.basename(bb_path).split("_")[0]
            bb_img = sitk.ReadImage(bb_path)
            arr = sitk.GetArrayFromImage(bb_img)
            z, y, x = arr.shape
            writer.writerow([pid, bb_path, t1ce_path, z, y, x])

    print(f"✅ Metadata CSV saved: {save_csv}")

if __name__ == "__main__":
    BB_DIR = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-bb"
    T1CE_DIR = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-t1ce"
    SAVE_CSV = "./metadata_bb2t1ce.csv"
    prepare_metadata(BB_DIR, T1CE_DIR, SAVE_CSV)
