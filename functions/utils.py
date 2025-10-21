# ================================================================
# functions/utils.py
# ================================================================
"""
Diffusion 실험 전반에서 공통으로 사용하는 유틸리티 함수 모음
- 스케일링, 리샘플링, 통계 계산 등
"""

import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import random


# ---------------------------------------------------------------
# 📏 Scaling / Normalization
# ---------------------------------------------------------------

def scaling(x, old_min, old_max, new_min, new_max):
    """
    입력 텐서의 값을 (old_min, old_max) → (new_min, new_max) 범위로 선형 스케일링.
    """
    return ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def rescale(arr, src_min, src_max, tar_min, tar_max):
    """
    numpy 배열을 새로운 범위로 변환.
    """
    return (arr - src_min) / (src_max - src_min) * (tar_max - tar_min) + tar_min


# ---------------------------------------------------------------
# 🧮 Diffusion math helpers
# ---------------------------------------------------------------

def compute_alpha(beta, t, device=None):
    """
    DDPM의 누적 α_t 계산.
    Args:
        beta: β 스케줄 텐서
        t: timestep (LongTensor)
        device: 선택적 디바이스 지정
    Returns:
        α_t 누적곱 텐서 (B, 1, 1, 1)
    """
    device = device or beta.device
    beta = torch.cat([torch.zeros(1).to(device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


# ---------------------------------------------------------------
# 🧠 MRI / Image related helpers
# ---------------------------------------------------------------

def resample_img(itk_image, out_size, is_label=False):
    """
    SimpleITK 이미지를 지정한 크기로 리샘플링.
    Args:
        itk_image: SimpleITK Image 객체
        out_size: (x, y, z) 형태의 출력 크기
        is_label: True인 경우 nearest neighbor interpolation 사용
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_spacing = [
        original_spacing[0] * (original_size[0] / out_size[0]),
        original_spacing[1] * (original_size[1] / out_size[1]),
        original_spacing[2] * (original_size[2] / out_size[2]),
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resample.Execute(itk_image)


# ---------------------------------------------------------------
# 🔀 Tensor manipulation helpers
# ---------------------------------------------------------------

def random_crop(tensor, crop_size=(64, 512, 512)):
    """
    3D 텐서를 랜덤 위치에서 crop.
    Args:
        tensor: (B, C, D, H, W)
        crop_size: (depth, height, width)
    """
    b, c, d, h, w = tensor.shape
    cd, ch, cw = crop_size
    z = random.randint(0, d - cd)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    return tensor[:, :, z:z+cd, y:y+ch, x:x+cw]


def select_random_axis(tensor, num_axis=8):
    """
    5D MRI 텐서 (B, Z, H, W) 중 임의의 Z 슬라이스 영역을 선택.
    """
    start_index = random.randint(0, tensor.size(1) - num_axis)
    return tensor[:, start_index:start_index + num_axis, :, :]


def cosine_similarity(x1, x2):
    """
    두 텐서 간 코사인 유사도 계산.
    """
    x1_flat, x2_flat = x1.view(-1), x2.view(-1)
    return F.cosine_similarity(x1_flat.unsqueeze(0), x2_flat.unsqueeze(0)).item()


def torch2hwcuint8(x, clip=False):
    """
    텐서를 [0,1] → [0,255] 형태의 uint8로 변환.
    """
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5,
                        beta_end ** 0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64)
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas
