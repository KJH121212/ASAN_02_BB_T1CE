# ================================================================
# datasets/bb2t1ce_dataset.py
# ASAN MRI 데이터용 최종 버전
# - (H, W, D) → (D, H, W)
# - adaptive percentile normalization (-1 ~ 1)
# - 중심 기준 슬라이스 n개 추출
# ================================================================

import os
import torch
import SimpleITK as sitk
import numpy as np
from glob import glob
from torch.utils.data import Dataset


# ------------------------------------------------------------
# adaptive intensity normalization helper
# ------------------------------------------------------------
def adaptive_rescale(volume, lower=0.5, upper=99.5):
    """
    각 볼륨의 intensity 분포를 기준으로 [-1, 1]로 정규화

    Args:
        volume (ndarray): 3D MRI array (D, H, W)
        lower (float): 하위 백분위수 (기본 0.5%)
        upper (float): 상위 백분위수 (기본 99.5%)
    """
    vmin, vmax = np.percentile(volume, (lower, upper))
    volume = np.clip(volume, vmin, vmax)
    volume = 2.0 * (volume - vmin) / (vmax - vmin) - 1.0
    return volume.astype(np.float32)


# ------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------
class BB2T1CE_Dataset(Dataset):
    """
    BB → T1CE 변환용 Dataset 클래스
    - NIfTI(.nii.gz)을 로드
    - (H, W, D) → (D, H, W) 변환
    - adaptive intensity normalization 적용
    - 중심 기준 n_slices 만큼 2D 슬라이스 반환
    """

    def __init__(self, bb_dir, t1ce_dir, n_slices=8):
        """
        Args:
            bb_dir (str): BB MRI 폴더 경로 (예: 'meta-bb')
            t1ce_dir (str): T1CE MRI 폴더 경로 (예: 'meta-t1ce')
            n_slices (int): 한 번에 추출할 슬라이스 개수
        """
        self.bb_files = sorted(glob(os.path.join(bb_dir, "*_bb_coreg.nii.gz")))
        self.t1ce_files = [
            f.replace(bb_dir, t1ce_dir).replace("_bb_coreg", "_t1ce_coreg")
            for f in self.bb_files
        ]
        self.n_slices = n_slices

    def __len__(self):
        return len(self.bb_files)

    def __getitem__(self, idx):
        # --------------------------------------------------------
        # 1️⃣ 파일 로드
        # --------------------------------------------------------
        bb_path = self.bb_files[idx]
        t1ce_path = self.t1ce_files[idx]

        # (H, W, D) 형태 → (D, H, W)로 변환
        bb_vol = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(bb_path)), -1, 0)
        t1ce_vol = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path)), -1, 0)

        # --------------------------------------------------------
        # 2️⃣ Adaptive normalization [-1, 1]
        # --------------------------------------------------------
        bb_vol = adaptive_rescale(bb_vol)
        t1ce_vol = adaptive_rescale(t1ce_vol)

        # --------------------------------------------------------
        # 3️⃣ 중심 기준 슬라이스 선택 (D축)
        # --------------------------------------------------------
        D = bb_vol.shape[0]
        center = D // 2
        start = center - self.n_slices // 2
        end = center + self.n_slices // 2
        bb_slices = bb_vol[start:end]       # (n_slices, H, W)
        t1ce_slices = t1ce_vol[start:end]

        # --------------------------------------------------------
        # 4️⃣ PyTorch Tensor 변환 (B,C,H,W)
        # --------------------------------------------------------
        bb_slices = torch.from_numpy(bb_slices[:, None, :, :])     # (n_slices, 1, H, W)
        t1ce_slices = torch.from_numpy(t1ce_slices[:, None, :, :]) # (n_slices, 1, H, W)

        return bb_slices, t1ce_slices
