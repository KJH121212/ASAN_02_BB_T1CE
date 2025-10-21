import os
import torch
import numpy as np
import SimpleITK as sitk

# ================================================================
# SamplerDDPM : Generate 2D slices or full 3D volume
# ================================================================
class SamplerDDPM:
    def __init__(self, diffusion_model, device="cuda"):
        self.diffusion = diffusion_model
        self.device = device

    @torch.no_grad()
    def sample_2d(self, bb_tensor, out_dir=None, fname_prefix="sample"):
        """
        bb_tensor: (B,1,H,W) torch.Tensor
        """
        self.diffusion.model.eval()
        x_c = bb_tensor.to(self.device)
        samples = self.diffusion.sample(x_c, shape=x_c.shape)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for i, s in enumerate(samples):
                npy_path = os.path.join(out_dir, f"{fname_prefix}_{i:03d}.npy")
                np.save(npy_path, s.cpu().numpy())
        return samples

    @torch.no_grad()
    def sample_3d(self, bb_volume, out_path):
        """
        bb_volume: (1, D, H, W) torch.Tensor
        """
        slices = []
        for i in range(bb_volume.shape[1]):
            bb_slice = bb_volume[:, i:i+1, :, :]
            pred = self.sample_2d(bb_slice)
            slices.append(pred.cpu().numpy()[0, 0])
        volume = np.stack(slices, axis=0)
        sitk.WriteImage(sitk.GetImageFromArray(volume), out_path)
        print(f"💾 Saved 3D reconstruction → {out_path}")
        return volume
