# ================================================================
# functions/calc_fid.py
# ================================================================
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet Inception Distance(FID) 계산
    두 분포의 평균(mu)과 공분산(sigma)을 이용하여 거리 계산
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors must have same length"
    assert sigma1.shape == sigma2.shape, "Covariances must have same dimension"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations(tensor, model, batch_size=50, dims=2048, device='cpu'):
    """InceptionV3에서 feature 추출"""
    model.eval()
    pred_arr = np.empty((tensor.size(0), dims))
    for i in range(0, tensor.size(0), batch_size):
        batch = tensor[i:i + batch_size].to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, (1, 1))
        pred_arr[i:i + batch_size] = pred.squeeze(3).squeeze(2).cpu().numpy()
    return pred_arr


def calculate_activation_statistics(tensor, model, batch_size=50, dims=2048, device='cpu'):
    """FID 계산용 평균(mu) 및 공분산(sigma) 계산"""
    act = get_activations(tensor, model, batch_size, dims, device)
    return np.mean(act, axis=0), np.cov(act, rowvar=False)


def calculate_fid(tensor1, tensor2, batch_size, device, dims):
    """최종 FID 계산"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(tensor1, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(tensor2, model, batch_size, dims, device)
    return calculate_frechet_distance(m1, s1, m2, s2)
