# ================================================================
# runners/trainer.py
# BB→T1CE Diffusion model training loop
# ================================================================

import os
import torch
from tqdm import tqdm
from functions.losses import noise_prediction_loss
from functions.noise_scheduler import linear_beta_schedule, get_alpha_params
from functions.visualization import save_sample_pair, plot_loss_curve
from models.ema import EMAHelper


class Trainer:
    def __init__(self, model, dataloader, config, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device

        # Scheduler
        self.timesteps = config["diffusion"]["timesteps"]
        betas = linear_beta_schedule(
            self.timesteps,
            config["diffusion"]["beta_start"],
            config["diffusion"]["beta_end"],
        ).to(device)
        self.alphas, self.alphas_cumprod, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = get_alpha_params(betas)

        # Optimizer
        opt_cfg = config["optimizer"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0)
        )

        # EMA helper
        self.ema = EMAHelper(mu=0.9999)
        self.ema.register(model)

        self.ckpt_dir = config["training"]["ckpt_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.loss_history = []

    # ---------------------------------------------------------------
    # (1) Forward diffusion: q(x_t | x_0)
    # ---------------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise, noise

    # ---------------------------------------------------------------
    # (2) Training Loop
    # ---------------------------------------------------------------
    def train(self):
        epochs = self.config["training"]["n_epochs"]
        loss_type = self.config["training"]["loss_type"]

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                bb = batch["bb"].to(self.device)
                t1ce = batch["t1ce"].to(self.device)

                # Random t sampling
                t = torch.randint(0, self.timesteps, (bb.size(0),), device=self.device).long()

                # q(x_t | x_0)
                x_t, noise = self.q_sample(t1ce, t)

                # Noise prediction
                pred_noise = self.model(x_t, t, x_cond=bb)
                loss = noise_prediction_loss(pred_noise, noise, loss_type=loss_type)

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # EMA update
                self.ema.update(self.model)

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            print(f"✅ Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

            # Visualization sample (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
                save_sample_pair(bb, t1ce, x_t, f"./outputs/train/epoch_{epoch+1:03d}.png")

        plot_loss_curve(self.loss_history, save_path="./outputs/train/loss_curve.png")

    # ---------------------------------------------------------------
    # (3) Checkpoint save
    # ---------------------------------------------------------------
    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")
