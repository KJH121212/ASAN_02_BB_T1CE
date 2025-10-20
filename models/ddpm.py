# ================================================================
# models/ddpm.py
# Conditional Diffusion Model Wrapper
# ================================================================

import torch
import torch.nn as nn

class ConditionalUNet(nn.Module):
    """기본적인 Conditional UNet 구조"""
    def __init__(self, in_channels=1, cond_channels=1, base_channels=64, channel_mults=[1,2,4], num_res_blocks=2, dropout=0.1):
        super().__init__()
        # TODO: 실제 UNet 구현은 diffusion_unet.py 참조
        self.encoder = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)
        self.decoder = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t, x_cond):
        x_in = torch.cat([x, x_cond], dim=1)
        h = torch.relu(self.encoder(x_in))
        return self.decoder(h)


class DiffusionModel(nn.Module):
    """Forward diffusion + Conditional noise prediction"""
    def __init__(self, unet, timesteps=1000):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward(self, x_t, t, x_cond):
        return self.unet(x_t, t, x_cond)
