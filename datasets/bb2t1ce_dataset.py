import os                                                   # 파일 경로 조작을 위한 모듈
import torch                                                # PyTorch 텐서 및 연산용 모듈
import SimpleITK as sitk                                    # 의료영상(NIfTI 등) 읽기 위한 라이브러리
import numpy as np                                          # 수치 계산용 NumPy 모듈
from glob import glob                                       # 파일 패턴 매칭을 위한 glob

from functions.resampling import resample_img
from functions.z_score import zscore

from torch.utils.data import Dataset                        # PyTorch Dataset 클래스 상속용
import torch.nn.functional as F                             # padding resize에 필요


# ------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------
class Train_Dataset(Dataset):                         # PyTorch Dataset 클래스 정의
    """
    BB → T1CE 변환용 Dataset 클래스
    - NIfTI(.nii.gz)을 로드
    - (H, W, D) → (D, H, W)로 변환
    - z-score normalization 적용
    """

    def __init__(self, bb_dir, t1ce_dir, img_size=256):    # 초기화 함수 정의
        """
        Args:
            bb_dir (str): BB MRI 폴더 경로 (예: 'meta-bb')
            t1ce_dir (str): T1CE MRI 폴더 경로 (예: 'meta-t1ce')
            img_size (int): 이미지 resampling 크기
        """
        self.bb_paths = sorted(glob(os.path.join(bb_dir, "*.nii.gz")))
        self.t1ce_paths = sorted(glob(os.path.join(t1ce_dir, "*.nii.gz")))
        assert len(self.bb_paths) == len(self.t1ce_paths), \
            f"❌ BB({len(self.bb_paths)})와 T1CE({len(self.t1ce_paths)}) 개수가 다릅니다!"
        self.img_size = img_size

    def __len__(self):                                   # 전체 데이터 개수 반환
        return len(self.bb_paths)

    def __getitem__(self, idx):                          # 인덱스로 데이터 하나 불러오기
        # 파일 로드
        bb_itk = sitk.ReadImage(self.bb_paths[idx])
        t1ce_itk = sitk.ReadImage(self.t1ce_paths[idx])

        # out_size 계산
        original_size = bb_itk.GetSize()  # (x, y, z)
        out_size = (self.img_size, self.img_size, original_size[2])  # Z는 그대로 유지

        # resampling
        bb_resampled = resample_img(bb_itk, out_size)
        t1ce_resampled = resample_img(t1ce_itk, out_size)

        # 배열 변환
        bb_arr = sitk.GetArrayFromImage(bb_resampled).astype(np.float32)
        t1ce_arr = sitk.GetArrayFromImage(t1ce_resampled).astype(np.float32)
        
        # 정규화
        bb_arr = zscore(bb_arr)
        t1ce_arr = zscore(t1ce_arr)

        # (Z, H, W) → (1, Z, H, W)
        bb_tensor = torch.from_numpy(bb_arr).unsqueeze(0)
        t1ce_tensor = torch.from_numpy(t1ce_arr).unsqueeze(0)

        # (C=2, Z, H, W)
        concat_tensor = torch.cat([bb_tensor, t1ce_tensor], dim=0)
            
        return concat_tensor

class Val_Dataset(Dataset):                         # PyTorch Dataset 클래스 정의
    def __init__(self, bb_dir, t1ce_dir, img_size=256):    # 초기화 함수 정의
        """
        Args:
            bb_dir (str): BB MRI 폴더 경로 (예: 'meta-bb')
            t1ce_dir (str): T1CE MRI 폴더 경로 (예: 'meta-t1ce')
            img_size (int): 이미지 resampling 크기
        """
        self.bb_paths = sorted(glob(os.path.join(bb_dir, "*.nii.gz")))
        self.t1ce_paths = sorted(glob(os.path.join(t1ce_dir, "*.nii.gz")))
        assert len(self.bb_paths) == len(self.t1ce_paths), \
            f"❌ BB({len(self.bb_paths)})와 T1CE({len(self.t1ce_paths)}) 개수가 다릅니다!"
        self.img_size = img_size

    def __len__(self):                                   # 전체 데이터 개수 반환
        return len(self.bb_paths)

    def __getitem__(self, idx):                          # 인덱스로 데이터 하나 불러오기
        # 파일 로드
        bb_itk = sitk.ReadImage(self.bb_paths[idx])
        t1ce_itk = sitk.ReadImage(self.t1ce_paths[idx])

        # out_size 계산
        original_size = bb_itk.GetSize()  # (x, y, z)
        out_size = (self.img_size, self.img_size, original_size[2])  # Z는 그대로 유지

        # resampling
        bb_resampled = resample_img(bb_itk, out_size)
        t1ce_resampled = resample_img(t1ce_itk, out_size)

        # 배열 변환
        bb_arr = sitk.GetArrayFromImage(bb_resampled).astype(np.float32)
        t1ce_arr = sitk.GetArrayFromImage(t1ce_resampled).astype(np.float32)
        
        # 정규화
        bb_arr = zscore(bb_arr)
        t1ce_arr = zscore(t1ce_arr)

        # (Z, H, W) → (1, Z, H, W)
        bb_tensor = torch.from_numpy(bb_arr).unsqueeze(0)
        t1ce_tensor = torch.from_numpy(t1ce_arr).unsqueeze(0)

        # (C=2, Z, H, W)
        concat_tensor = torch.cat([bb_tensor, t1ce_tensor], dim=0)
            
        return concat_tensor
