# ================================================================
# functions/noise_scheduler.py
# Beta schedule and diffusion parameter utilities
# ================================================================

import torch

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    선형 beta 스케줄 (DDPM 기본)
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine 스케줄 (Improved DDPM, Nichol et al.)
    """
    import math
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def get_alpha_params(betas):
    """
    β 스케줄로부터 α, ᾱ, √ᾱ, 1−ᾱ 등 주요 파라미터 계산
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    return alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
