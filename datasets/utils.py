# ================================================================
# datasets/utils.py
# Dataset utility functions for preprocessing and scaling
# ================================================================

import numpy as np
import torch

def rescale_to_unit(data, data_min, data_max):
    """clip and rescale to [-1, 1]"""
    data = np.clip(data, data_min, data_max)
    return 2.0 * (data - data_min) / (data_max - data_min) - 1.0

def tensor_info(tensor, name="Tensor"):
    """print basic tensor info"""
    print(f"[{name}] shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
          f"range=({tensor.min().item():.3f}, {tensor.max().item():.3f})")

def save_slice_preview(tensor, save_path):
    """save a middle slice (for quick visual debugging)"""
    import matplotlib.pyplot as plt
    mid = tensor.shape[1] // 2
    plt.imshow(tensor[0, mid].cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
