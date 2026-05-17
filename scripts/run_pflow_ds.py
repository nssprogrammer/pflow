"""End-to-end PFlow-T dataset version: train, diagnose, sample.

Usage:
    python -m scripts.run_pflow_ds --epochs 5 --subset 1000

For a first end-to-end check, --subset 1000 is plenty to verify the
pipeline works on real MNIST 8s. Scale up after.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from pflow_ds.dataset import MNISTPFlowDataset, make_x_T
from pflow_ds.train import train, sample_one_shot, diagnose_per_t
from pflow_ds.viz import plot_forward_sweep, plot_diagnose, plot_grid
from pflow_ds.persistence import count_betti


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='runs/pflow_ds')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--digits', type=int, nargs='+', default=[8],
                   help='MNIST digits to include. For controllability '
                        'experiments use e.g. --digits 0 1 8.')
    p.add_argument('--subset', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--base_channels', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--n_samples', type=int, default=16)
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'fashion'],
                   help='Source dataset (default: mnist).')
    return p


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Train.
    digit_filter = args.digits[0] if len(args.digits) == 1 else list(args.digits)
    model, ds = train(
        out_dir=args.out_dir,
        data_root=args.data_root,
        digit_filter=digit_filter,
        subset=args.subset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        dataset_name=args.dataset,
    )

    # 2. Forward-sweep figure for one image.
    print('\n=== Forward sweep on a sample image ===')
    item0 = ds[0]
    img0 = item0['target'].numpy()[0]
    sch0 = ds.get_schedule(0)
    plot_forward_sweep(img0, sch0,
                       save_path=os.path.join(args.out_dir, 'forward_sweep.png'))
    plt.close('all')

    # 3. Per-t diagnostic.
    print('\n=== Per-t diagnostic ===')
    ts, ins, outs = diagnose_per_t(model, img0, sch0, n_t=6)
    plot_diagnose(img0, ts, ins, outs,
                  save_path=os.path.join(args.out_dir, 'diagnose.png'))
    plt.close('all')

    # 4. Sampling: one-shot reconstructions.
    # For each of `n_samples` images, build its x_T and reconstruct.
    print('\n=== Sampling ===')
    device = next(model.parameters()).device
    n = min(args.n_samples, len(ds))
    targets, generated = [], []
    for i in range(n):
        item = ds[i]
        img_i = item['target'].numpy()[0]
        sch_i = ds.get_schedule(i)
        x_T = make_x_T(img_i, sch_i)
        gen = sample_one_shot(model, x_T, device=device)
        targets.append(img_i)
        generated.append(gen)

    # Stack target/generated side-by-side.
    pairs = []
    titles = []
    for t_img, g_img in zip(targets, generated):
        b0_t, b1_t = count_betti(t_img, threshold=0.5)
        b0_g, b1_g = count_betti(g_img, threshold=0.5)
        pairs.append(t_img)
        titles.append(f'target β1={b1_t}')
        pairs.append(g_img)
        titles.append(f'pflow β1={b1_g}')
    plot_grid(pairs, titles=titles, cols=8,
              suptitle='Target  /  PFlow-T generated (one-shot from x_T)',
              save_path=os.path.join(args.out_dir, 'samples.png'))
    plt.close('all')

    # 5. Compute aggregate β1 statistics.
    print('\n=== β1 statistics on this batch ===')
    b1_target = np.array([count_betti(t, threshold=0.5)[1] for t in targets])
    b1_gen = np.array([count_betti(g, threshold=0.5)[1] for g in generated])
    print(f'targets   β1 mean={b1_target.mean():.2f}  std={b1_target.std():.2f}')
    print(f'generated β1 mean={b1_gen.mean():.2f}  std={b1_gen.std():.2f}')
    print(f'match rate (β1 exactly equal): '
          f'{(b1_target == b1_gen).mean()*100:.1f}%')


if __name__ == '__main__':
    main(build_argparser().parse_args())
