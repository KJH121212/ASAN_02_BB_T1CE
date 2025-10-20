# ================================================================
# runners/validator.py
# Validation routine for BB→T1CE Diffusion
# ================================================================

import torch
from tqdm import tqdm
from functions.metrics import psnr, mae
from functions.visualization import save_sample_pair

class Validator:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

    def validate(self, epoch=0):
        self.model.eval()
        total_psnr, total_mae = 0, 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"[Validation Epoch {epoch}]"):
                bb = batch["bb"].to(self.device)
                t1ce = batch["t1ce"].to(self.device)

                # 단순 forward inference (conditioning)
                pred = self.model(t1ce, torch.zeros(bb.size(0), device=self.device).long(), x_cond=bb)

                total_psnr += psnr(pred, t1ce).item()
                total_mae += mae(pred, t1ce).item()

            avg_psnr = total_psnr / len(self.dataloader)
            avg_mae = total_mae / len(self.dataloader)
            print(f"📊 Validation — PSNR: {avg_psnr:.3f}, MAE: {avg_mae:.5f}")

        save_sample_pair(bb, t1ce, pred, f"./outputs/validation/epoch_{epoch:03d}.png")
