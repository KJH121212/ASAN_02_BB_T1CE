# ================================================================
# functions/metrics.py
# Evaluation metrics for generated images
# ================================================================

import torch
import torch.nn.functional as F

def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio
    """
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def mae(pred, target):
    """
    Mean Absolute Error
    """
    return torch.mean(torch.abs(pred - target))

def mse(pred, target):
    """
    Mean Squared Error
    """
    return F.mse_loss(pred, target)

# (선택) MONAI SSIM metric 사용
try:
    from monai.metrics import SSIMMetric
    def ssim(pred, target):
        metric = SSIMMetric(spatial_dims=2, data_range=2.0)
        return metric(pred, target)
except ImportError:
    def ssim(pred, target):
        raise RuntimeError("❌ MONAI not installed. SSIM unavailable.")
