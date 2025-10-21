import os
import time
import torch
from torch.utils.data import DataLoader
from models import DiffusionModel
from functions import get_optimizer
from functions.metrics import psnr

# ================================================================
# TrainerDDPM : Train / Save / Evaluate
# ================================================================
class TrainerDDPM:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.diffusion = DiffusionModel(config, device)
        self.optimizer = get_optimizer(config, self.diffusion.model.parameters())

        # log/checkpoint directory
        self.save_dir = os.path.join(config.training.ckpt_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1️⃣ Train loop
    # ------------------------------------------------------------
    def train(self, train_loader: DataLoader):
        print("🚀 Starting DDPM Training...")
        for epoch in range(self.config.training.n_epochs):
            epoch_loss = 0.0
            start_time = time.time()

            for step, (bb, t1ce) in enumerate(train_loader):
                bb, t1ce = bb.to(self.device), t1ce.to(self.device)
                loss = self.diffusion.forward_loss(t1ce, bb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.diffusion.update_ema()

                epoch_loss += loss.item()

                if step % 10 == 0:
                    print(f"[Epoch {epoch:03d}] Step {step:04d} | Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            duration = time.time() - start_time
            print(f"✅ Epoch {epoch} done | Avg Loss={avg_loss:.4f} | Time={duration:.1f}s")

            self.save_checkpoint(epoch)

    # ------------------------------------------------------------
    # 2️⃣ Checkpoint save/load
    # ------------------------------------------------------------
    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        states = {
            "model": self.diffusion.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.diffusion.ema_helper.state_dict(),
            "epoch": epoch,
        }
        torch.save(states, ckpt_path)
        print(f"💾 Saved checkpoint → {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        print(f"📦 Loading checkpoint from {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.device)
        self.diffusion.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.diffusion.ema_helper.load_state_dict(states["ema"])
        print(f"✅ Loaded epoch {states['epoch']}")

    # ------------------------------------------------------------
    # 3️⃣ Validation metric (PSNR)
    # ------------------------------------------------------------
    @torch.no_grad()
    def validate(self, val_loader):
        self.diffusion.model.eval()
        psnr_scores = []

        for bb, t1ce in val_loader:
            bb, t1ce = bb.to(self.device), t1ce.to(self.device)
            pred = self.diffusion.sample(bb, shape=t1ce.shape)
            score = psnr(pred, t1ce)
            psnr_scores.append(score.item())

        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        print(f"🔎 Validation PSNR: {avg_psnr:.2f} dB")
        self.diffusion.model.train()
        return avg_psnr
