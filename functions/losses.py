# ================================================================
# functions/losses.py
# 학습 손실
# ================================================================
import torch


import torch

def noise_estimation_loss(model, x0, x_c, t, e, b):
    """
    Args:
        model: diffusion model
        x0: target image (T1CE)       → (B, 1, H, W)
        x_c: conditional image (BB)   → (B, 1, H, W)
        t: timestep tensor            → (B,)
        e: random noise tensor        → (B, 1, H, W)
        b: beta schedule              → (T,)
    """
    device = x0.device

    # 1️⃣ Alpha 계산
    a = (1 - b).cumprod(dim=0).to(device)             # (T,)
    a_t = a[t].unsqueeze(1).unsqueeze(2).unsqueeze(3) # (B, 1, 1, 1)

    # 2️⃣ Forward noising step
    xt = x0 * a_t.sqrt() + e * (1.0 - a_t).sqrt()     # (B, 1, H, W)

    # 3️⃣ Conditional 입력 결합
    x_in = torch.cat([xt, x_c], dim=1)                # (B, 2, H, W)

    # 4️⃣ Model prediction
    output, _ = model(x_in, t.float())                # (B, 1, H, W)

    # 5️⃣ Loss 계산
    loss = torch.mean((e - output) ** 2)
    return loss


# 손실 함수 레지스트리
loss_registry = {
    "simple": noise_estimation_loss,
}
