# ================================================================
# datasets/transforms.py
# ================================================================
import torch
import random
from .utils import resize_2d

def bb2t1ce_transform(bb_tensor, t1ce_tensor, crop_size=(256, 256), flip_prob=0.5, use_resize=True):
    """
    BB → T1CE 학습용 transform
    Args:
        bb_tensor, t1ce_tensor: (H,W) or (1,H,W) or (1,1,H,W)
    Returns:
        (bb_t, t1ce_t): 동일 크기의 2D Tensor (1,1,H,W)
    """

    # ------------------------------------------------------------
    # 1️⃣ 입력 차원 보정
    # ------------------------------------------------------------
    # numpy에서 불러온 경우 (H, W)
    if bb_tensor.ndim == 2:
        bb_tensor = bb_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif bb_tensor.ndim == 3:
        bb_tensor = bb_tensor.unsqueeze(0)  # (1,1,H,W)
    # 이미 (1,1,H,W)면 그대로 둠

    if t1ce_tensor.ndim == 2:
        t1ce_tensor = t1ce_tensor.unsqueeze(0).unsqueeze(0)
    elif t1ce_tensor.ndim == 3:
        t1ce_tensor = t1ce_tensor.unsqueeze(0)

    # ------------------------------------------------------------
    # 2️⃣ Resize or Random Crop
    # ------------------------------------------------------------
    if use_resize:
        bb_tensor, t1ce_tensor = resize_2d(bb_tensor, t1ce_tensor, target_size=crop_size)
    else:
        h, w = bb_tensor.shape[-2:]
        top = torch.randint(0, h - crop_size[0] + 1, (1,)).item()
        left = torch.randint(0, w - crop_size[1] + 1, (1,)).item()
        bb_tensor = bb_tensor[..., top:top + crop_size[0], left:left + crop_size[1]]
        t1ce_tensor = t1ce_tensor[..., top:top + crop_size[0], left:left + crop_size[1]]

    # ------------------------------------------------------------
    # 3️⃣ Random horizontal flip
    # ------------------------------------------------------------
    if random.random() < flip_prob:
        bb_tensor = torch.flip(bb_tensor, dims=[-1])
        t1ce_tensor = torch.flip(t1ce_tensor, dims=[-1])

    # ------------------------------------------------------------
    # 4️⃣ 출력
    # ------------------------------------------------------------
    return bb_tensor, t1ce_tensor
