"""Probe Fashion-MNIST classes for usable H1 topology.

Fashion-MNIST classes:
    0 T-shirt/top    5 Sandal
    1 Trouser        6 Shirt
    2 Pullover       7 Sneaker
    3 Dress          8 Bag
    4 Coat           9 Ankle boot

For PFlow-T's substrate claim to apply, we need classes with distinct
distributions over β1. This script samples some images per class and
reports the empirical β1 distribution — so we can pick which classes
to use for the controllability experiment without guessing.

Usage:
    python -m scripts.inspect_fashion
"""

from __future__ import annotations
import argparse
import os
from collections import Counter
import numpy as np

from pflow_ds.dataset import MNISTPFlowDataset
from pflow_ds.persistence import count_betti


CLASS_NAMES = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',  4: 'Coat',
    5: 'Sandal',     6: 'Shirt',   7: 'Sneaker',  8: 'Bag',    9: 'Ankle boot',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--n_per_class', type=int, default=100,
                   help='Samples per class to probe.')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--threshold_sweep', action='store_true',
                   help='Also report β1 distribution at thresholds {0.3, 0.4, 0.5, 0.6}.')
    args = p.parse_args()

    print('Loading Fashion-MNIST (full test set so we can stratify by class)...')
    ds = MNISTPFlowDataset(
        root=args.data_root, train=False, digit_filter=None,
        dataset_name='fashion',
    )
    print(f'  {len(ds)} samples cached.\n')

    # Stratify by class.
    indices_by_class = {c: [] for c in CLASS_NAMES}
    for i in range(len(ds)):
        item = ds[i]
        c = item['label']
        if len(indices_by_class[c]) < args.n_per_class:
            indices_by_class[c].append(i)

    print(f'β1 distribution per class ({args.n_per_class} samples each):')
    print(f'  threshold = {args.threshold}\n')
    print(f'  {"class":<14}  {"β1=0":>6} {"β1=1":>6} {"β1=2":>6} {"β1=3+":>6}'
          f'  {"modal β1":<8}  {"verdict"}')
    print('  ' + '-' * 80)

    summary = {}
    for c, indices in sorted(indices_by_class.items()):
        b1_counts = Counter()
        for idx in indices:
            img = ds[idx]['target'].numpy()[0]
            _, b1 = count_betti(img, threshold=args.threshold)
            b1_counts[min(b1, 3)] += 1  # bin β1 ≥ 3 together

        total = sum(b1_counts.values())
        modal = max(b1_counts, key=b1_counts.get)
        modal_frac = b1_counts[modal] / total
        verdict = ('CLEAN' if modal_frac > 0.8 else
                   'mixed' if modal_frac > 0.5 else
                   'noisy')
        summary[c] = (modal, modal_frac, verdict)
        name = CLASS_NAMES[c]
        print(f'  {c} {name:<12}  '
              f'{b1_counts[0]:>6} {b1_counts[1]:>6} '
              f'{b1_counts[2]:>6} {b1_counts[3]:>6}'
              f'  {modal:<8}  {verdict} ({modal_frac*100:.0f}%)')
    print()

    # Recommend a 3-class subset with non-overlapping modal β1 values.
    by_b1 = {0: [], 1: [], 2: []}
    for c, (modal, frac, verdict) in summary.items():
        if modal in by_b1 and verdict in ('CLEAN', 'mixed'):
            by_b1[modal].append((c, frac, CLASS_NAMES[c]))
    # Pick the cleanest per β1 bucket
    chosen = []
    for b1_val, candidates in by_b1.items():
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[1])  # highest modal fraction first
        chosen.append((b1_val, candidates[0]))

    print('Suggested 3-class subset for the controllability experiment:')
    for b1_val, (c, frac, name) in chosen:
        print(f'  β1={b1_val}: class {c} ({name}) — modal at {frac*100:.0f}%')
    print()
    if len(chosen) == 3:
        digits = ' '.join(str(b1_val_chosen[1][0]) for b1_val_chosen in chosen)
        print(f'Command: python -m scripts.run_pflow_ds --dataset fashion '
              f'--digits {digits} --subset 6000 --epochs 20 --base_channels 64')
    else:
        print('WARNING: could not find a clean 3-class triple covering '
              'β1 ∈ {0,1,2}. Consider:')
        print('  - lowering --threshold to capture lighter strokes')
        print('  - using only 2 classes (β1=0 vs β1=1)')


    if args.threshold_sweep:
        print('\nThreshold sweep — class 1 (Trouser) and class 8 (Bag):')
        for cls in [1, 8]:
            print(f'  class {cls} ({CLASS_NAMES[cls]}):')
            for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                b1_counts = Counter()
                for idx in indices_by_class[cls]:
                    img = ds[idx]['target'].numpy()[0]
                    _, b1 = count_betti(img, threshold=thr)
                    b1_counts[min(b1, 3)] += 1
                modal = max(b1_counts, key=b1_counts.get) if b1_counts else 0
                total = sum(b1_counts.values()) or 1
                print(f'    thr={thr}: β1=0:{b1_counts[0]:3d}  β1=1:{b1_counts[1]:3d}  '
                      f'β1=2:{b1_counts[2]:3d}  β1=3+:{b1_counts[3]:3d}'
                      f'   modal β1={modal} ({b1_counts[modal]*100//total}%)')


if __name__ == '__main__':
    main()
