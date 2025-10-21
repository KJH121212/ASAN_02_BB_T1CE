# sampling (generalized_steps_condition)
# ================================================================
# functions/denoising.py
# ================================================================
import torch


def compute_alpha(beta, t):
    """누적 alpha 계산"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


def generalized_steps_condition(x, c, e, seq, model, b):
    """
    조건부 DDPM 샘플링 (conditional diffusion)
    x: noisy target
    c: condition (e.g., BB)
    e: random noise
    seq: timestep sequence
    """
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []

        for idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            if idx == 0:
                xs = [x * at.sqrt() + e * (1.0 - at).sqrt()]

            xt = xs[-1].to(x.device)
            c = c.to(x.device)
            et, _ = model(torch.cat([xt, c], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))

            c1 = 0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to("cpu"))

        return xs, x0_preds
