# ================================================================
# datasets/utils.py
# ================================================================
import numpy as np
import torch
import torch.nn.functional as F

def rescale_intensity(arr, new_min=-1, new_max=1):
    """numpy 배열 intensity를 [new_min, new_max] 범위로 정규화"""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def normalize_tensor(t):
    """Tensor를 [-1,1]로 정규화"""
    return 2 * (t - t.min()) / (t.max() - t.min()) - 1

def resize_2d(bb, t1ce, target_size=(256, 256)):
    """bilinear resize (두 입력 동기화)"""
    if bb.dim() != 4 or t1ce.dim() != 4:
        raise ValueError(f"Expected 4D tensors, got bb={bb.shape}, t1ce={t1ce.shape}")
    bb = F.interpolate(bb, size=target_size, mode="bilinear", align_corners=False)
    t1ce = F.interpolate(t1ce, size=target_size, mode="bilinear", align_corners=False)
    return bb, t1ce
