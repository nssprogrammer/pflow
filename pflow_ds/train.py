"""Training and one-shot sampling for PFlow-T dataset version.

Training:
    - Each batch: sample t ~ U(0, 1) per item, build x_t via the cached
      persistence schedule, train model(x_t, t) to predict the original
      target with MSE loss.

Sampling:
    - One-shot: model(x_T, t=1.0) -> predicted x_0.
    - For unconditional generation, we need an x_T that doesn't depend on
      a specific test image. We sample x_T from the empirical distribution
      of training x_T images. Strategy: pick a random training image,
      build its x_T (= fully filled blob), feed to the model. Different
      training images give different blobs; the model recovers the digit.
    - For class-conditional or "fill in the holes of this specific image",
      pass the x_T directly.
"""

from __future__ import annotations
import os
import time
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TinyUNet
from .dataset import MNISTPFlowDataset, make_x_T


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 fg_threshold: float = 0.3, fg_weight: float = 5.0) -> torch.Tensor:
    """MSE with extra weight on foreground (ink) pixels.

    Without this, the model can hit a low-but-not-zero MSE by predicting
    all-zeros — a known pathology on sparse-foreground datasets like MNIST.
    Setting fg_weight > 1 makes that local minimum strictly worse than
    actually predicting the digit.
    """
    fg_mask = (target >= fg_threshold).float()
    weight = 1.0 + (fg_weight - 1.0) * fg_mask
    return (weight * (pred - target).pow(2)).mean()


def train(
    out_dir: str = 'runs/pflow_ds',
    data_root: str = './data',
    digit_filter: Optional[Union[int, List[int]]] = 8,
    subset: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-3,
    base_channels: int = 32,
    weight_decay: float = 1e-5,
    num_workers: int = 0,
    log_every: int = 100,
    dataset_name: str = 'mnist',
    device: Optional[str] = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    print(f'device: {device}  dataset: {dataset_name}')

    ds = MNISTPFlowDataset(
        root=data_root, train=True, digit_filter=digit_filter, subset=subset,
        dataset_name=dataset_name,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    print(f'dataset: {len(ds)} samples | batches/epoch: {len(loader)}')

    model = TinyUNet(ch=base_channels).to(device)
    print(f'model parameters: {model.num_parameters():,}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(loader))
    )

    step = 0
    t0 = time.time()
    log_path = os.path.join(out_dir, 'train_log.csv')
    log_f = open(log_path, 'w')
    log_f.write('step,epoch,loss\n')

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'epoch {epoch}')
        for batch in pbar:
            target = batch['target'].to(device)
            x_t = batch['x_t'].to(device)
            t = batch['t'].to(device)

            pred = model(x_t, t)
            loss = weighted_mse(pred, target, fg_threshold=0.3, fg_weight=5.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % log_every == 0:
                pbar.set_description(f'epoch {epoch} | loss {loss.item():.5f}')
                log_f.write(f'{step},{epoch},{loss.item():.6f}\n')
                log_f.flush()
            step += 1

        ckpt = os.path.join(out_dir, f'pflow_ds_epoch{epoch}.pt')
        torch.save({
            'model': model.state_dict(),
            'hparams': {
                'base_channels': base_channels,
                'digit_filter': digit_filter,
                'dataset_name': dataset_name,
            },
        }, ckpt)
        print(f'saved {ckpt}  ({time.time() - t0:.1f}s)')

    log_f.close()
    return model, ds


def load_model(ckpt_path: str, device: Optional[str] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hparams', {})
    model = TinyUNet(ch=hp.get('base_channels', 32)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, hp


@torch.no_grad()
def sample_one_shot(
    model: TinyUNet,
    x_T: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict x_0 directly from x_T. x_T should be the fully-filled-in
    starting point — built either from a specific test image's schedule
    or from a sampled-from-empirical-distribution x_T.

    Returns: (H, W) numpy array in [0, 1].
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(x_T).float()[None, None].to(device)
    t = torch.tensor([1.0], dtype=torch.float32, device=device)
    pred = model(x, t)
    return pred.squeeze().cpu().numpy()


@torch.no_grad()
def diagnose_per_t(
    model: TinyUNet,
    target_img: np.ndarray,
    schedule,
    n_t: int = 6,
    fill_radius: float = 2.5,
    device: Optional[str] = None,
):
    """Call model(x_t, t) for n_t evenly spaced t values. A well-trained
    model should output an image close to the target at every t."""
    from .forward import persistence_melt
    device = device or next(model.parameters()).device
    model.eval()
    ts = np.linspace(0, 1, n_t)
    inputs, outputs = [], []
    for tv in ts:
        x_t = persistence_melt(target_img, float(tv), schedule=schedule,
                               fill_radius=fill_radius)
        x = torch.from_numpy(x_t).float()[None, None].to(device)
        tt = torch.tensor([tv], dtype=torch.float32, device=device)
        pred = model(x, tt).squeeze().cpu().numpy()
        inputs.append(x_t)
        outputs.append(pred)
    return ts, inputs, outputs
