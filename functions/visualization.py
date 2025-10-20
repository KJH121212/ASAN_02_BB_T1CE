# ================================================================
# functions/visualization.py
# Visualization utilities for monitoring training progress
# ================================================================

import os
import torch
import matplotlib.pyplot as plt

def save_sample_pair(bb, t1ce, pred_t1ce, save_path, idx=0):
    """
    BB / T1CE / Generated T1CE를 나란히 시각화하여 저장.
    Args:
        bb (Tensor): BB 이미지 (B, C, H, W)
        t1ce (Tensor): GT T1CE
        pred_t1ce (Tensor): 예측된 T1CE
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bb_np = bb[idx, 0].detach().cpu().numpy()
    gt_np = t1ce[idx, 0].detach().cpu().numpy()
    pred_np = pred_t1ce[idx, 0].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(bb_np, cmap="gray"); axs[0].set_title("BB Input"); axs[0].axis("off")
    axs[1].imshow(gt_np, cmap="gray"); axs[1].set_title("T1CE Ground Truth"); axs[1].axis("off")
    axs[2].imshow(pred_np, cmap="gray"); axs[2].set_title("Predicted T1CE"); axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_loss_curve(loss_history, save_path="./outputs/loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Train Loss")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
