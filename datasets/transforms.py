# ================================================================
# datasets/transforms.py
# BB–T1CE Dataset 전처리 및 Augmentation 함수 정의 (5D 대응)
# ================================================================

import torch
import random

# ---------------------------------------------------------------
# (1) 랜덤 슬라이스 선택
# ---------------------------------------------------------------
def random_slice(bb_tensor, t1ce_tensor, n_slices=8):
    """3D 볼륨에서 연속된 슬라이스 n개를 랜덤하게 선택
    Args:
        bb_tensor (torch.Tensor): (B, C, Z, H, W)
        t1ce_tensor (torch.Tensor): (B, C, Z, H, W)
    Returns:
        (torch.Tensor, torch.Tensor): (B, C, n_slices, H, W)
    """
    z = bb_tensor.shape[2]
    if z <= n_slices:
        return bb_tensor, t1ce_tensor

    start_idx = random.randint(0, z - n_slices)
    end_idx = start_idx + n_slices
    bb_crop = bb_tensor[:, :, start_idx:end_idx, :, :]
    t1ce_crop = t1ce_tensor[:, :, start_idx:end_idx, :, :]
    return bb_crop, t1ce_crop


# ---------------------------------------------------------------
# (2) 중앙 crop
# ---------------------------------------------------------------
def center_crop(tensor, crop_size=(256, 256)):
    h, w = tensor.shape[-2:]
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return tensor[..., top:top + ch, left:left + cw]


# ---------------------------------------------------------------
# (3) 랜덤 crop
# ---------------------------------------------------------------
def random_crop(tensor, crop_size=(256, 256)):
    h, w = tensor.shape[-2:]
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return tensor
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    return tensor[..., top:top + ch, left:left + cw]


# ---------------------------------------------------------------
# (4) 랜덤 flip
# ---------------------------------------------------------------
def random_flip(bb_tensor, t1ce_tensor, p=0.5):
    if random.random() < p:
        bb_tensor = torch.flip(bb_tensor, dims=[-1])  # 좌우 flip
        t1ce_tensor = torch.flip(t1ce_tensor, dims=[-1])
    if random.random() < p:
        bb_tensor = torch.flip(bb_tensor, dims=[-2])  # 상하 flip
        t1ce_tensor = torch.flip(t1ce_tensor, dims=[-2])
    return bb_tensor, t1ce_tensor


# ---------------------------------------------------------------
# (5) 전체 transform pipeline
# ---------------------------------------------------------------
def bb2t1ce_transform(bb_tensor, t1ce_tensor):
    assert bb_tensor.ndim == 4, f"Expected 4D (B,C,H,W), got {bb_tensor.shape}"
    assert t1ce_tensor.ndim == 4, f"Expected 4D (B,C,H,W), got {t1ce_tensor.shape}"
    # 여기에 2D augment (flip, crop 등)만 적용
    return bb_tensor, t1ce_tensor
