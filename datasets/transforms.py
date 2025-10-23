# ================================================================
# datasets/transforms.py
# ================================================================
import torch                                               # 텐서 연산용
import random                                              # 확률적 변환용
import torch.nn.functional as F                            # 보간, 패딩 함수
# from .utils import resize_2d  # ❌ 더 이상 필요 없음


# ------------------------------------------------------------
# 1️⃣ 비율 유지 + 패딩 resize 함수
# ------------------------------------------------------------
def resize_with_padding(tensor, target_size=(256, 256), pad_value=-1.0):
    """
    비율(Aspect Ratio)을 유지하면서 target_size로 resize하고
    남는 영역은 pad_value로 채우는 함수.
    """
    _, _, h, w = tensor.shape                             # 입력 크기 확인
    target_h, target_w = target_size                      # 목표 크기 분리

    scale = min(target_h / h, target_w / w)               # 비율 유지 위한 스케일 비율 계산
    new_h, new_w = int(h * scale), int(w * scale)         # resize 후 크기 계산

    # bilinear 보간으로 리사이즈
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # 중앙 기준 패딩 계산
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2

    # 패딩 적용 (좌,우,상,하)
    tensor = F.pad(
        tensor,
        pad=(pad_w, pad_w, pad_h, pad_h),
        mode="constant",
        value=pad_value,                                  # 빈 공간은 -1.0으로 채움 (normalize 일관성 유지)
    )

    # 혹시 모자라면 잘라서 정확히 target 크기로 맞춤
    tensor = tensor[..., :target_h, :target_w]

    return tensor


# ------------------------------------------------------------
# 2️⃣ BB/T1CE 쌍을 동시에 패딩 resize
# ------------------------------------------------------------
def resize_2d_with_padding(bb_tensor, t1ce_tensor, target_size=(256, 256)):
    """
    BB / T1CE 두 영상을 동일한 방식으로 resize + padding 적용
    """
    bb_tensor = resize_with_padding(bb_tensor, target_size)
    t1ce_tensor = resize_with_padding(t1ce_tensor, target_size)
    return bb_tensor, t1ce_tensor


# ------------------------------------------------------------
# 3️⃣ BB→T1CE transform 함수
# ------------------------------------------------------------
def bb2t1ce_transform(bb_tensor, t1ce_tensor,
                      crop_size=(256, 256),
                      flip_prob=0.5,
                      use_resize=True):
    """
    BB → T1CE 학습용 데이터 변환 함수
    Args:
        bb_tensor, t1ce_tensor: (H,W), (1,H,W) 또는 (1,1,H,W) 형태의 입력 텐서
        crop_size (tuple): 출력 이미지 크기 (기본 256×256)
        flip_prob (float): 좌우 반전 확률 (기본 0.5)
        use_resize (bool): True → padding resize / False → random crop
    Returns:
        (bb_t, t1ce_t): 동일 크기의 2D Tensor 쌍 (1,1,H,W)
    """

    # ------------------------------------------------------------
    # 1️⃣ 입력 차원 보정
    # ------------------------------------------------------------
    if bb_tensor.ndim == 2:
        bb_tensor = bb_tensor.unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
    elif bb_tensor.ndim == 3:
        bb_tensor = bb_tensor.unsqueeze(0)                 # (1,1,H,W)
    # 이미 (1,1,H,W)이면 그대로 유지

    if t1ce_tensor.ndim == 2:
        t1ce_tensor = t1ce_tensor.unsqueeze(0).unsqueeze(0)
    elif t1ce_tensor.ndim == 3:
        t1ce_tensor = t1ce_tensor.unsqueeze(0)

    # ------------------------------------------------------------
    # 2️⃣ Resize (padding 방식) 또는 Random Crop
    # ------------------------------------------------------------
    if use_resize:
        # ✅ padding 기반 resize로 교체
        bb_tensor, t1ce_tensor = resize_2d_with_padding(
            bb_tensor, t1ce_tensor, target_size=crop_size)
    else:
        # ❗ 원본 크기에서 랜덤 crop
        h, w = bb_tensor.shape[-2:]
        top = torch.randint(0, h - crop_size[0] + 1, (1,)).item()
        left = torch.randint(0, w - crop_size[1] + 1, (1,)).item()
        bb_tensor = bb_tensor[..., top:top + crop_size[0], left:left + crop_size[1]]
        t1ce_tensor = t1ce_tensor[..., top:top + crop_size[0], left:left + crop_size[1]]

    # ------------------------------------------------------------
    # 3️⃣ Random horizontal flip (데이터 증강)
    # ------------------------------------------------------------
    if random.random() < flip_prob:
        bb_tensor = torch.flip(bb_tensor, dims=[-1])        # 좌우 반전
        t1ce_tensor = torch.flip(t1ce_tensor, dims=[-1])

    # ------------------------------------------------------------
    # 4️⃣ 최종 반환
    # ------------------------------------------------------------
    return bb_tensor, t1ce_tensor
