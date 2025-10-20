# ================================================================
# functions/losses.py
# Diffusion noise prediction loss functions
# ================================================================

import torch
import torch.nn.functional as F

def noise_prediction_loss(pred_noise, true_noise, loss_type="l2"):
    """
    기본 DDPM 손실: 모델이 예측한 노이즈와 실제 노이즈 간의 차이.
    Args:
        pred_noise (torch.Tensor): 모델이 예측한 노이즈 ε_pred
        true_noise (torch.Tensor): 실제 노이즈 ε
        loss_type (str): "l1", "l2", "hybrid"
    """
    if loss_type == "l2":
        return F.mse_loss(pred_noise, true_noise)
    elif loss_type == "l1":
        return F.l1_loss(pred_noise, true_noise)
    elif loss_type == "hybrid":
        l1 = F.l1_loss(pred_noise, true_noise)
        l2 = F.mse_loss(pred_noise, true_noise)
        return 0.5 * (l1 + l2)
    else:
        raise ValueError(f"❌ Unknown loss type: {loss_type}")

def reconstruction_loss(pred, target, loss_type="l1"):
    """
    복원된 T1CE와 GT T1CE를 직접 비교할 때 사용 (선택사항)
    """
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    elif loss_type == "l2":
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"❌ Unknown loss type: {loss_type}")
