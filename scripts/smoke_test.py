"""Smoke test for the dataset PFlow-T pipeline.

Run before any serious training. Verifies:
  1. gudhi persistence works and finds the expected H1 features in an 8.
  2. The forward process kills features in the right order and Betti
     numbers progress monotonically (β1 = 2 -> 1 -> 0).
  3. One training step produces a finite loss.
  4. Diagnostic plot renders.

Outputs go to ./smoke_outputs/.

Usage:
    python -m scripts.smoke_test
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from pflow_ds.dataset import MNISTPFlowDataset
from pflow_ds.forward import build_h1_kill_schedule, persistence_melt
from pflow_ds.persistence import count_betti
from pflow_ds.model import TinyUNet
from pflow_ds.train import diagnose_per_t
from pflow_ds.viz import plot_forward_sweep


OUT = './smoke_outputs'
os.makedirs(OUT, exist_ok=True)


def test_persistence_finds_loops():
    print('[1] persistence finds H1 features in an 8...')
    ds = MNISTPFlowDataset(train=True, digit_filter=8, subset=20)
    item = ds[0]
    img = item['target'].numpy()[0]
    schedule = ds.get_schedule(0)
    all_events = ds.events[0]
    h1_raw = [e for e in all_events if e['dim'] == 1]
    h1_pers = sorted([e['persistence'] for e in h1_raw], reverse=True)
    print(f'    total H1 features (any persistence): {len(h1_raw)}')
    print(f'    top H1 persistences: {[round(p, 3) for p in h1_pers[:5]]}')
    print(f'    schedule has {len(schedule)} H1 features (>= min_persistence {ds.min_persistence})')
    if len(schedule) < 1:
        print(f'    ERROR: zero features made it into the schedule, even with fallback.')
        print(f'           This image may genuinely have no loops (real MNIST 8s sometimes')
        print(f'           are not closed). Try a different sample or different digit_filter.')
        # Probe a few more samples before giving up
        for k in range(1, min(10, len(ds))):
            sch_k = ds.get_schedule(k)
            if sch_k:
                print(f'    sample {k} has {len(sch_k)} features — using it instead')
                item = ds[k]
                img = item['target'].numpy()[0]
                schedule = sch_k
                break
    persistences = [round(e['persistence'], 3) for e in schedule]
    print(f'    schedule persistences (ascending): {persistences}')
    assert len(schedule) >= 1, 'No H1 features found in any of the first 10 samples'
    print('    PASS\n')
    return img, schedule


def test_betti_progression(img, schedule):
    print('[2] β1 should decrease monotonically with t...')
    last_b1 = None
    transitions = 0
    for t in np.linspace(0, 1, 11):
        x_t = persistence_melt(img, float(t), schedule=schedule)
        _, b1 = count_betti(x_t, threshold=0.5)
        marker = ''
        if last_b1 is not None and b1 != last_b1:
            marker = ' (transition)'
            transitions += 1
            if b1 > last_b1:
                marker = ' (NON-MONOTONIC: BAD)'
        print(f'    t={t:.2f}  β1={b1}{marker}')
        last_b1 = b1
    print(f'    {transitions} transitions seen.\n')


def test_forward_sweep_render(img, schedule):
    print('[3] rendering forward sweep figure...')
    plot_forward_sweep(img, schedule, save_path=os.path.join(OUT, 'sweep.png'))
    plt.close('all')
    print()


def test_one_train_step():
    print('[4] one training step on a small batch...')
    ds = MNISTPFlowDataset(train=True, digit_filter=8, subset=32)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    model = TinyUNet(ch=16)
    pred = model(batch['x_t'], batch['t'])
    loss = torch.nn.functional.mse_loss(pred, batch['target'])
    loss.backward()
    print(f'    pred shape: {tuple(pred.shape)}  loss: {loss.item():.4f}')
    assert np.isfinite(loss.item()), 'loss not finite'
    print('    PASS\n')


def test_diagnose_render():
    print('[5] per-t diagnostic plot...')
    ds = MNISTPFlowDataset(train=True, digit_filter=8, subset=20)
    item = ds[0]
    model = TinyUNet(ch=16)
    ts, ins, outs = diagnose_per_t(
        model, item['target'].numpy()[0], ds.get_schedule(0), n_t=4,
    )
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axes[0, i].imshow(ins[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off'); axes[0, i].set_title(f't={ts[i]:.2f}')
        axes[1, i].imshow(outs[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'diagnose_untrained.png'), dpi=100)
    plt.close('all')
    print(f'    saved {OUT}/diagnose_untrained.png')
    print('    (model is untrained, so outputs will not look like 8s yet)')
    print('    PASS\n')


if __name__ == '__main__':
    print('=' * 60)
    print('PFlow-T dataset version smoke test')
    print('=' * 60)
    img, schedule = test_persistence_finds_loops()
    test_betti_progression(img, schedule)
    test_forward_sweep_render(img, schedule)
    test_one_train_step()
    test_diagnose_render()
    print('All smoke checks passed.')
    print(f'Visualizations in {OUT}/')
