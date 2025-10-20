# ================================================================
# models/diffusion_unet_custom.py
# Custom PyTorch Diffusion U-Net (BB → T1CE)
# ================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# (1) Timestep Embedding
# ---------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def Normalize(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)  # Swish activation


# ---------------------------------------------------------------
# (2) Basic Blocks
# ---------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, h_, w_ = q.shape
        q = q.reshape(b, c, h_ * w_).permute(0, 2, 1)
        k = k.reshape(b, c, h_ * w_)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        v = v.reshape(b, c, h_ * w_)
        attn_out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h_, w_)
        return x + self.proj(attn_out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------
# (3) Diffusion UNet (Custom)
# ---------------------------------------------------------------
class DiffusionUNetCustom(nn.Module):
    """
    Custom PyTorch Diffusion U-Net (BB condition)
    입력: noisy T1CE
    조건: BB
    출력: predicted noise ε_pred
    """
    def __init__(self, in_channels=1, cond_channels=1, out_channels=1,
                 base_channels=128, ch_mult=(1, 2, 4), num_res_blocks=2,
                 dropout=0.1, attn_res=(16,)):
        super().__init__()

        self.ch = base_channels
        self.temb_ch = base_channels * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.temb = nn.Sequential(
            nn.Linear(self.ch, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        self.conv_in = nn.Conv2d(in_channels + cond_channels, self.ch, 3, padding=1)

        # Downsampling
        in_ch_mult = (1,) + ch_mult
        curr_res = 256
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            in_ch = self.ch * in_ch_mult[i_level]
            out_ch = self.ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch, temb_channels=self.temb_ch, dropout=dropout))
                in_ch = out_ch
                if curr_res in attn_res:
                    attn.append(AttnBlock(in_ch))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_ch)
                curr_res //= 2
            self.down.append(down)

        # Middle
        self.mid = nn.ModuleDict({
            "block1": ResnetBlock(in_ch, in_ch, temb_channels=self.temb_ch, dropout=dropout),
            "attn1": AttnBlock(in_ch),
            "block2": ResnetBlock(in_ch, in_ch, temb_channels=self.temb_ch, dropout=dropout),
        })

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch = self.ch * ch_mult[i_level]
            skip_in = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_ch + skip_in, out_ch, temb_channels=self.temb_ch, dropout=dropout)
                )
                in_ch = out_ch
                if curr_res in attn_res:
                    attn.append(AttnBlock(in_ch))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(in_ch)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x_noisy, t, x_cond=None):
        if x_cond is not None:
            x = torch.cat([x_noisy, x_cond], dim=1)
        else:
            x = x_noisy
        temb = self.temb(get_timestep_embedding(t, self.ch))

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid["block1"](h, temb)
        h = self.mid["attn1"](h)
        h = self.mid["block2"](h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))
