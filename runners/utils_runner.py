import os
import torch

# ================================================================
# runners/utils_runner.py
# ================================================================

def save_checkpoint(model, optimizer, epoch, path, ema=None):
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    if ema:
        states["ema"] = ema.state_dict()
    torch.save(states, path)
    print(f"💾 Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path, device="cuda", ema=None):
    states = torch.load(path, map_location=device)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])
    if ema and "ema" in states:
        ema.load_state_dict(states["ema"])
    print(f"📦 Checkpoint loaded from {path}")
    return states.get("epoch", 0)
