"""Conditioning baseline: standard DDPM with persistence-diagram conditioning.

This is the TAGG / TopoDiffusionNet / Hu-et-al.-style competitor:
  - Forward process: Gaussian noise (topology-blind)
  - Reverse process: same TinyUNet backbone as PFlow-T (parameter parity)
  - Topology enters as a CONDITIONING signal (5-d PD descriptor) added to
    the time embedding.

The point of comparison: at controllability time, can this model match a
requested β1 as reliably as PFlow-T's substrate-based approach?

Parameterization: epsilon-prediction (standard DDPM). This is the canonical
choice for Gaussian-noise diffusion. Sampling is iterative T-step ancestral.
"""

from __future__ import annotations
import math
import os
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SinusoidalTimeEmb
from .dataset import MNISTPFlowDataset


# =========================================================================
# 1. PD encoder + conditioned U-Net
# =========================================================================

class PDEncoder(nn.Module):
    """Map a 5-d persistence descriptor to a t_dim embedding.

    Designed to live in the same vector space as the time embedding so we
    can simply add them. The 5-d input vector is normalized by typical
    magnitudes before being passed in; see normalize_pd_vec().
    """

    def __init__(self, t_dim: int = 128, hidden: int = 64, in_dim: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

    def forward(self, pd_vec):
        return self.mlp(pd_vec)


def normalize_pd_vec(pd: torch.Tensor) -> torch.Tensor:
    """Center/scale a descriptor for stable training.

    For the 5-d summary, divide by typical magnitudes.
    For higher-dim descriptors (landscape, etc.) we assume the caller
    has chosen a sane range and just pass through with a global scale.
    """
    if pd.shape[-1] == 5:
        scale = pd.new_tensor([3.0, 2.0, 0.5, 1.0, 1.0])
        return pd / scale
    # Landscape: values are already in [0, ~0.5]. Scale by 2 to roughly
    # match the 5-d summary's effective range.
    return pd * 2.0


class CondTinyUNet(nn.Module):
    """TinyUNet conditioned on both time and a persistence descriptor.

    The descriptor can be either the 5-d summary (default) or the 64-d
    persistence landscape, controlled by `cond_dim`. Architecturally
    identical to TinyUNet plus a PDEncoder; parameter count differs
    only in the PDEncoder's input layer.
    """

    def __init__(self, ch: int = 32, t_dim: int = 128, in_channels: int = 1,
                 cond_dim: int = 5):
        super().__init__()
        self.cond_dim = cond_dim
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.pd_enc = PDEncoder(t_dim=t_dim, in_dim=cond_dim)

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

    def forward(self, x, t, pd_vec):
        # Combined embedding: time + persistence descriptor.
        te = self.t_emb(t) + self.pd_enc(normalize_pd_vec(pd_vec))
        e1 = F.silu(self._inject(self.enc1(x), self.t_to_enc1(te)))
        e2 = F.silu(self._inject(self.enc2(e1), self.t_to_enc2(te)))
        e3 = F.silu(self._inject(self.enc3(e2), self.t_to_enc3(te)))
        b = F.silu(self._inject(self.bot(e3), self.t_to_bot(te)))
        d3 = F.silu(self.dec3(torch.cat(
            [F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), e2], 1)))
        d2 = F.silu(self.dec2(torch.cat(
            [F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), e1], 1)))
        return self.head(d2)  # NB: no sigmoid - predicting epsilon, not x_0

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =========================================================================
# 2. DDPM scheduler (linear betas)
# =========================================================================

class DDPMScheduler:
    """Standard linear-beta DDPM scheduler.

    Stores beta, alpha, alpha_bar tensors on a device. Provides forward
    noising (add_noise) and the reverse step (step).
    """

    def __init__(self, T: int = 200, beta_min: float = 1e-4, beta_max: float = 0.02,
                 device: Optional[str] = None):
        self.T = T
        device = device or 'cpu'
        self.betas = torch.linspace(beta_min, beta_max, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: Optional[torch.Tensor] = None):
        """x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return x_t, noise

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_pred: torch.Tensor) -> torch.Tensor:
        """Ancestral DDPM reverse step: produce x_{t-1} from x_t and predicted ε."""
        alpha = self.alphas[t]
        ab = self.alpha_bars[t]
        beta = self.betas[t]
        # μ_{t-1|t} = (1/√α_t) (x_t - (β_t/√(1-ᾱ_t)) ε)
        mean = (x_t - beta / (1 - ab).sqrt() * eps_pred) / alpha.sqrt()
        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + beta.sqrt() * noise
        return mean


# =========================================================================
# 3. Training loop
# =========================================================================

def train_cond_baseline(
    out_dir: str = 'runs/baseline',
    data_root: str = './data',
    digits=(0, 1, 8),
    subset: Optional[int] = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 2e-3,
    base_channels: int = 32,
    T: int = 200,
    weight_decay: float = 1e-5,
    num_workers: int = 0,
    log_every: int = 100,
    cond_type: str = 'pd5',   # 'pd5' or 'landscape'
    dataset_name: str = 'mnist',
    device: Optional[str] = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    print(f'device: {device}  cond_type: {cond_type}  dataset: {dataset_name}')

    ds = MNISTPFlowDataset(
        root=data_root, train=True, digit_filter=list(digits), subset=subset,
        dataset_name=dataset_name,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    print(f'dataset: {len(ds)} samples | batches/epoch: {len(loader)}')

    # Probe the descriptor dimension from the dataset.
    probe = ds[0]
    if cond_type == 'pd5':
        cond_dim = probe['pd_vec'].numel()
        cond_key = 'pd_vec'
    elif cond_type == 'landscape':
        cond_dim = probe['landscape'].numel()
        cond_key = 'landscape'
    else:
        raise ValueError(f'unknown cond_type: {cond_type}')
    print(f'conditioning dim: {cond_dim}  (from key "{cond_key}")')

    model = CondTinyUNet(ch=base_channels, cond_dim=cond_dim).to(device)
    print(f'baseline parameters: {model.num_parameters():,}')

    sched_dm = DDPMScheduler(T=T, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(loader))
    )

    log_path = os.path.join(out_dir, 'train_log.csv')
    log_f = open(log_path, 'w')
    log_f.write('step,epoch,loss\n')

    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'baseline epoch {epoch}')
        for batch in pbar:
            x0 = batch['target'].to(device)
            cond = batch[cond_key].to(device)
            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device)
            x_t, eps = sched_dm.add_noise(x0, t)
            t_norm = t.float() / T
            eps_pred = model(x_t, t_norm, cond)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched_lr.step()

            if step % log_every == 0:
                pbar.set_description(f'baseline epoch {epoch} | loss {loss.item():.5f}')
                log_f.write(f'{step},{epoch},{loss.item():.6f}\n')
                log_f.flush()
            step += 1

        ckpt = os.path.join(out_dir, f'baseline_epoch{epoch}.pt')
        torch.save({
            'model': model.state_dict(),
            'hparams': {
                'base_channels': base_channels,
                'T': T,
                'digits': list(digits),
                'cond_type': cond_type,
                'cond_dim': cond_dim,
                'dataset_name': dataset_name,
            },
        }, ckpt)
        print(f'  saved {ckpt}  ({time.time() - t0:.1f}s)')

    log_f.close()
    return model, sched_dm, ds


def get_baseline_cond(test_ds, idx: int, hp: dict):
    """Return the descriptor (numpy array) matching a baseline checkpoint's
    cond_type, so we feed it the same descriptor type it was trained on."""
    cond_type = hp.get('cond_type', 'pd5')
    if cond_type == 'landscape':
        return test_ds.get_landscape(idx)
    return test_ds.get_pd_vec(idx)


def load_cond_baseline(ckpt_path: str, device: Optional[str] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hparams', {})
    cond_dim = hp.get('cond_dim', 5)
    model = CondTinyUNet(ch=hp.get('base_channels', 32), cond_dim=cond_dim).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    scheduler = DDPMScheduler(T=hp.get('T', 200), device=device)
    return model, scheduler, hp


# =========================================================================
# 4. Sampling
# =========================================================================

@torch.no_grad()
def sample_cond_baseline(
    model: CondTinyUNet,
    scheduler: DDPMScheduler,
    pd_vec: np.ndarray,
    n_samples: int = 1,
    image_size: int = 28,
    device: Optional[str] = None,
) -> np.ndarray:
    """Iterative DDPM ancestral sampling, conditioned on a PD descriptor.

    Returns (n_samples, image_size, image_size) numpy array in [0, 1].
    """
    device = device or next(model.parameters()).device
    model.eval()
    pd = torch.from_numpy(np.asarray(pd_vec, dtype=np.float32))[None].to(device)
    pd = pd.expand(n_samples, -1)

    x = torch.randn(n_samples, 1, image_size, image_size, device=device)
    for t in reversed(range(scheduler.T)):
        t_norm = torch.full((n_samples,), t / scheduler.T,
                            dtype=torch.float32, device=device)
        eps_pred = model(x, t_norm, pd)
        x = scheduler.step(x, t, eps_pred)

    # Clamp to [0, 1] for visualization / metrics. Standard DDPM produces
    # roughly in [-1, 1] but our MNIST is in [0, 1] so the model learned
    # noise around that distribution.
    return x.clamp(0, 1).squeeze(1).cpu().numpy()
