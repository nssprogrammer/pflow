"""Train the conditioning baseline (DDPM with PD conditioning).

Usage:
    python -m scripts.train_baseline --digits 0 1 8 --subset 3000 --epochs 8
"""

from __future__ import annotations
import argparse
import os

from pflow_ds.baseline_cond import train_cond_baseline


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='runs/baseline')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--digits', type=int, nargs='+', default=[0, 1, 8],
                   help='MNIST digits to include (default: 0 1 8 -> β1=1, 0, 2)')
    p.add_argument('--subset', type=int, default=3000,
                   help='Cap training set size after digit filter')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--base_channels', type=int, default=32)
    p.add_argument('--T', type=int, default=200)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--cond_type', type=str, default='pd5',
                   choices=['pd5', 'landscape'],
                   help='Conditioning descriptor: 5-d summary (pd5) or '
                        '64-d persistence landscape.')
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'fashion'],
                   help='Source dataset (default: mnist).')
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_cond_baseline(
        out_dir=args.out_dir,
        data_root=args.data_root,
        digits=tuple(args.digits),
        subset=args.subset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        T=args.T,
        num_workers=args.num_workers,
        cond_type=args.cond_type,
        dataset_name=args.dataset,
    )


if __name__ == '__main__':
    main()
