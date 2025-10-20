# ================================================================
# scripts/visualize_sample.py
# Quick sample visualization of BB–T1CE dataset
# ================================================================

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob

def visualize_pair(bb_path, t1ce_path, slice_idx=None):
    bb_img = sitk.ReadImage(bb_path)
    t1ce_img = sitk.ReadImage(t1ce_path)
    bb_arr = sitk.GetArrayFromImage(bb_img)
    t1ce_arr = sitk.GetArrayFromImage(t1ce_img)

    if slice_idx is None:
        slice_idx = bb_arr.shape[0] // 2

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(bb_arr[slice_idx], cmap="gray")
    plt.title(f"BB Slice {slice_idx}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(t1ce_arr[slice_idx], cmap="gray")
    plt.title(f"T1CE Slice {slice_idx}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BB_DIR = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-bb"
    T1CE_DIR = "/workspace/nas100/forGPU2/Kimjihoo/ASAN_02_BB_T1CE/data/0_RAW_DATA/meta-t1ce"
    bb_files = sorted(glob(os.path.join(BB_DIR, "*.nii.gz")))
    t1ce_files = sorted(glob(os.path.join(T1CE_DIR, "*.nii.gz")))
    visualize_pair(bb_files[0], t1ce_files[0])
