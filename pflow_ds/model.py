"""TinyUNet model for PFlow-T dataset version.

Identical architecture to pft-v11 (which worked on the synthetic '8'):
small U-Net, sinusoidal time embedding injected at every encoder stage and
the bottleneck, sigmoid output. Predicts x_0 directly from (x_t, t).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class TinyUNet(nn.Module):
    """Small U-Net for 28x28 MNIST. ~150k params at ch=32."""

    def __init__(self, ch: int = 32, t_dim: int = 128, in_channels: int = 1):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.t_to_enc1 = nn.Linear(t_dim, ch)
        self.t_to_enc2 = nn.Linear(t_dim, ch * 2)
        self.t_to_enc3 = nn.Linear(t_dim, ch * 4)
        self.t_to_bot = nn.Linear(t_dim, ch * 4)

        self.enc1 = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.enc2 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)
        self.bot = nn.Conv2d(ch * 4, ch * 4, 3, padding=1)
        self.dec3 = nn.Conv2d(ch * 4 + ch * 2, ch * 2, 3, padding=1)
        self.dec2 = nn.Conv2d(ch * 2 + ch, ch, 3, padding=1)
        self.head = nn.Conv2d(ch, 1, 1)

    def _inject(self, x, t_emb_proj):
        return x + t_emb_proj[:, :, None, None]

    def forward(self, x, t):
        te = self.t_emb(t)
        e1 = F.silu(self._inject(self.enc1(x), self.t_to_enc1(te)))
        e2 = F.silu(self._inject(self.enc2(e1), self.t_to_enc2(te)))
        e3 = F.silu(self._inject(self.enc3(e2), self.t_to_enc3(te)))
        b = F.silu(self._inject(self.bot(e3), self.t_to_bot(te)))
        d3 = F.silu(self.dec3(torch.cat(
            [F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), e2], 1)))
        d2 = F.silu(self.dec2(torch.cat(
            [F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), e1], 1)))
        return torch.sigmoid(self.head(d2))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
