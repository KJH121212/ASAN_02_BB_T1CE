# ================================================================
# datasets/domain_dataset.py  (기존 bb2t1ce_dataset.py 대체)
# ================================================================
import os
import torch
import SimpleITK as sitk
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from datasets.transforms import resample_img, zscore


# ------------------------------------------------------------
# Generic MRI Translation Dataset
# ------------------------------------------------------------
class Train_Dataset(Dataset):
    """
    Source → Target 변환용 Dataset 클래스
    - NIfTI(.nii.gz)을 로드
    - (H, W, D) → (D, H, W)로 변환
    - z-score normalization 적용
    """

    def __init__(self, source_dir, target_dir, img_size=256):
        """
        Args:
            source_dir (str): Source MRI 폴더 경로 (예: 'meta-bb')
            target_dir (str): Target MRI 폴더 경로 (예: 'meta-t1ce')
            img_size (int): 이미지 resampling 크기
        """
        self.source_paths = sorted(glob(os.path.join(source_dir, "*.nii.gz")))
        self.target_paths = sorted(glob(os.path.join(target_dir, "*.nii.gz")))
        assert len(self.source_paths) == len(self.target_paths), \
            f"❌ Source({len(self.source_paths)})와 Target({len(self.target_paths)}) 개수가 다릅니다!"
        self.img_size = img_size

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, idx):
        # --------------------------------------------------------
        # 1️⃣ 파일 로드
        # --------------------------------------------------------
        src_itk = sitk.ReadImage(self.source_paths[idx])
        tar_itk = sitk.ReadImage(self.target_paths[idx])

        # --------------------------------------------------------
        # 2️⃣ Resampling
        # --------------------------------------------------------
        original_size = src_itk.GetSize()  # (x, y, z)
        out_size = (self.img_size, self.img_size, original_size[2])  # Z는 그대로 유지

        src_resampled = resample_img(src_itk, out_size)
        tar_resampled = resample_img(tar_itk, out_size)

        # --------------------------------------------------------
        # 3️⃣ NumPy 변환 및 정규화
        # --------------------------------------------------------
        src_arr = sitk.GetArrayFromImage(src_resampled).astype(np.float32)
        tar_arr = sitk.GetArrayFromImage(tar_resampled).astype(np.float32)

        src_arr = zscore(src_arr)
        tar_arr = zscore(tar_arr)

        # --------------------------------------------------------
        # 4️⃣ Tensor 변환 및 채널 결합
        # --------------------------------------------------------
        src_tensor = torch.from_numpy(src_arr).unsqueeze(0)  # (1, Z, H, W)
        tar_tensor = torch.from_numpy(tar_arr).unsqueeze(0)
        concat_tensor = torch.cat([src_tensor, tar_tensor], dim=0)  # (C=2, Z, H, W)

        return concat_tensor


class Val_Dataset(Dataset):
    """Validation용 Dataset (구조 동일)"""
    def __init__(self, source_dir, target_dir, img_size=256):
        self.source_paths = sorted(glob(os.path.join(source_dir, "*.nii.gz")))
        self.target_paths = sorted(glob(os.path.join(target_dir, "*.nii.gz")))
        assert len(self.source_paths) == len(self.target_paths), \
            f"❌ Source({len(self.source_paths)})와 Target({len(self.target_paths)}) 개수가 다릅니다!"
        self.img_size = img_size

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, idx):
        src_itk = sitk.ReadImage(self.source_paths[idx])
        tar_itk = sitk.ReadImage(self.target_paths[idx])

        original_size = src_itk.GetSize()
        out_size = (self.img_size, self.img_size, original_size[2])

        src_resampled = resample_img(src_itk, out_size)
        tar_resampled = resample_img(tar_itk, out_size)

        src_arr = sitk.GetArrayFromImage(src_resampled).astype(np.float32)
        tar_arr = sitk.GetArrayFromImage(tar_resampled).astype(np.float32)

        src_arr = zscore(src_arr)
        tar_arr = zscore(tar_arr)

        src_tensor = torch.from_numpy(src_arr).unsqueeze(0)
        tar_tensor = torch.from_numpy(tar_arr).unsqueeze(0)
        concat_tensor = torch.cat([src_tensor, tar_tensor], dim=0)

        return concat_tensor
