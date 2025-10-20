# ================================================================
# runners/tester.py
# Inference / Testing for BB→T1CE Diffusion model
# ================================================================

import os
import torch
from tqdm import tqdm
from functions.noise_scheduler import linear_beta_schedule, get_alpha_params
from functions.visualization import save_sample_pair


class Tester:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Scheduler setup
        self.timesteps = config["diffusion"]["timesteps"]
        betas = linear_beta_schedule(
            self.timesteps,
            config["diffusion"]["beta_start"],
            config["diffusion"]["beta_end"],
        ).to(device)
        self.alphas, self.alphas_cumprod, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = get_alpha_params(betas)

        os.makedirs(config["data"]["save_dir"], exist_ok=True)

    # ---------------------------------------------------------------
    # Reverse Diffusion sampling: p(x_{t-1}|x_t)
    # ---------------------------------------------------------------
    def p_sample(self, x, t, bb_cond):
        beta_t = 1 - self.alphas[t][:, None, None, None]
        pred_noise = self.model(x, t, x_cond=bb_cond)
        coef1 = 1 / torch.sqrt(self.alphas[t][:, None, None, None])
        coef2 = beta_t / torch.sqrt(1 - self.alphas_cumprod[t][:, None, None, None])
        mean = coef1 * (x - coef2 * pred_noise)
        noise = torch.randn_like(x) if t[0] > 0 else 0
        return mean + torch.sqrt(beta_t) * noise

    def sample(self, bb):
        b = bb.size(0)
        x = torch.randn_like(bb)
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, bb)
        return x

    def run(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                bb = batch["bb"].to(self.device)
                pred_t1ce = self.sample(bb)
                save_sample_pair(bb, torch.zeros_like(bb), pred_t1ce,
                                 os.path.join(self.config["data"]["save_dir"], f"sample_{i:03d}.png"))
