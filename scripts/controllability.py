"""Controllability comparison: PFlow-T vs conditioning baseline.

For each target β1 ∈ {0, 1, 2}:
  1. Pick N held-out test images whose actual β1 equals the target.
  2. For each, extract its conditioning signal:
       - PFlow-T: build x_T from the image and its schedule.
       - Baseline: compute the 5-d PD descriptor from the image's events.
  3. Generate one sample from each model using that conditioning.
  4. Measure: what fraction of generated samples have the requested β1?

This is the load-bearing experiment for the paper. Without a clear gap
between the two columns, the substrate-vs-conditioning claim does not hold.

Usage:
    python -m scripts.controllability \\
        --pflow_ckpt runs/pflow_ds/pflow_ds_epoch4.pt \\
        --baseline_ckpt runs/baseline/baseline_epoch7.pt \\
        --digits 0 1 8 \\
        --n_per_betti 30
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from pflow_ds.dataset import MNISTPFlowDataset, make_x_T, pd_to_vec
from pflow_ds.train import load_model, sample_one_shot
from pflow_ds.baseline_cond import load_cond_baseline, sample_cond_baseline
from pflow_ds.persistence import count_betti


# =========================================================================
# 1. Partition the test set by ground-truth β1
# =========================================================================

def partition_by_betti(dataset: MNISTPFlowDataset, threshold: float = 0.5,
                       max_per_class: int = 200):
    """Return {β1: [indices...]} grouping test items by ground-truth β1."""
    buckets = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(dataset)):
        item = dataset[i]
        _, b1 = count_betti(item['target'].numpy()[0], threshold=threshold)
        if b1 in buckets and len(buckets[b1]) < max_per_class:
            buckets[b1].append(i)
    return buckets


# =========================================================================
# 2. Run head-to-head on a single β1 class
# =========================================================================

def evaluate_class(
    target_b1: int,
    indices: list,
    test_ds: MNISTPFlowDataset,
    pflow_model,
    baseline_model,
    baseline_sched,
    device: str,
    threshold: float = 0.5,
    baseline_hp: dict = None,
):
    """For each test index, generate one PFlow-T sample and one baseline
    sample, then return both lists plus their β1s."""
    from pflow_ds.baseline_cond import get_baseline_cond
    baseline_hp = baseline_hp or {}
    pflow_samples, base_samples = [], []
    targets = []

    for idx in indices:
        item = test_ds[idx]
        target_img = item['target'].numpy()[0]
        targets.append(target_img)

        # PFlow-T: x_T as the conditioning signal.
        schedule = test_ds.get_schedule(idx)
        x_T = make_x_T(target_img, schedule, fill_radius=test_ds.fill_radius)
        gen_pflow = sample_one_shot(pflow_model, x_T, device=device)
        pflow_samples.append(gen_pflow)

        # Baseline: descriptor matching its trained cond_type.
        cond_vec = get_baseline_cond(test_ds, idx, baseline_hp)
        gen_base = sample_cond_baseline(
            baseline_model, baseline_sched, cond_vec,
            n_samples=1, image_size=target_img.shape[0], device=device,
        )[0]
        base_samples.append(gen_base)

    pflow_b1 = np.array([count_betti(g, threshold=threshold)[1] for g in pflow_samples])
    base_b1 = np.array([count_betti(g, threshold=threshold)[1] for g in base_samples])

    return {
        'target_b1': target_b1,
        'n': len(indices),
        'pflow_match_rate': float((pflow_b1 == target_b1).mean()),
        'baseline_match_rate': float((base_b1 == target_b1).mean()),
        'pflow_b1_distribution': dict(zip(*np.unique(pflow_b1, return_counts=True))),
        'baseline_b1_distribution': dict(zip(*np.unique(base_b1, return_counts=True))),
        'pflow_samples': pflow_samples,
        'base_samples': base_samples,
        'targets': targets,
    }


# =========================================================================
# 3. Plotting
# =========================================================================

def plot_match_rates(per_class_results: dict, save_path: str):
    b1s = sorted(per_class_results.keys())
    pflow_rates = [per_class_results[b]['pflow_match_rate'] for b in b1s]
    base_rates = [per_class_results[b]['baseline_match_rate'] for b in b1s]

    x = np.arange(len(b1s))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, pflow_rates, w, label='PFlow-T (substrate)',
           color='tab:blue')
    ax.bar(x + w / 2, base_rates, w, label='DDPM + PD cond (baseline)',
           color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'β1 = {b}' for b in b1s])
    ax.set_ylabel('β1 match rate')
    ax.set_ylim(0, 1.0)
    ax.set_title('Controllability: requested vs achieved β1')
    ax.legend()
    for i, (p, b) in enumerate(zip(pflow_rates, base_rates)):
        ax.text(i - w / 2, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
        ax.text(i + w / 2, b + 0.02, f'{b:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f'  saved {save_path}')


def plot_samples_grid(per_class_results: dict, save_path: str, n_show: int = 6):
    b1s = sorted(per_class_results.keys())
    rows = 3 * len(b1s)  # target / pflow / baseline rows per β1
    fig, axes = plt.subplots(rows, n_show, figsize=(1.6 * n_show, 1.7 * rows))
    if rows == 1:
        axes = axes[None, :]

    for c, b in enumerate(b1s):
        r = c * 3
        res = per_class_results[b]
        for k in range(n_show):
            if k < len(res['targets']):
                t_img = res['targets'][k]
                p_img = res['pflow_samples'][k]
                b_img = res['base_samples'][k]
                t_b1 = count_betti(t_img)[1]
                p_b1 = count_betti(p_img)[1]
                b_b1 = count_betti(b_img)[1]
                axes[r, k].imshow(t_img, cmap='gray', vmin=0, vmax=1)
                axes[r, k].set_title(f'target β1={t_b1}', fontsize=8)
                axes[r + 1, k].imshow(p_img, cmap='gray', vmin=0, vmax=1)
                axes[r + 1, k].set_title(f'pflow β1={p_b1}', fontsize=8)
                axes[r + 2, k].imshow(b_img, cmap='gray', vmin=0, vmax=1)
                axes[r + 2, k].set_title(f'base β1={b_b1}', fontsize=8)
            for rr in (r, r + 1, r + 2):
                axes[rr, k].axis('off')

        # Left-side labels
        axes[r, 0].set_ylabel(f'β1={b}\ntarget', fontsize=9, rotation=0, labelpad=24)
        axes[r + 1, 0].set_ylabel('PFlow-T', fontsize=9, rotation=0, labelpad=24)
        axes[r + 2, 0].set_ylabel('baseline', fontsize=9, rotation=0, labelpad=24)

    plt.suptitle('Controllability comparison: PFlow-T vs DDPM+PD baseline', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f'  saved {save_path}')


# =========================================================================
# 4. Entrypoint
# =========================================================================

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--pflow_ckpt', type=str, required=True)
    p.add_argument('--baseline_ckpt', type=str, required=True)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--digits', type=int, nargs='+', default=[0, 1, 8])
    p.add_argument('--n_per_betti', type=int, default=30,
                   help='Number of test images per β1 class')
    p.add_argument('--betti_threshold', type=float, default=0.5)
    p.add_argument('--out_dir', type=str, default='runs/controllability')
    p.add_argument('--dataset', type=str, default=None,
                   choices=['mnist', 'fashion', None],
                   help='Source dataset. If None, inferred from the PFlow-T checkpoint.')
    return p


def degeneracy_check(model_fn, name, n_probes=8):
    """Generate a few samples and verify the model isn't degenerate."""
    print(f'\n--- Sanity check on {name} ---')
    samples = [model_fn() for _ in range(n_probes)]
    means = [float(s.mean()) for s in samples]
    maxes = [float(s.max()) for s in samples]
    avg_mean = sum(means) / len(means)
    avg_max = sum(maxes) / len(maxes)
    var_across = float(np.var([s.flatten() for s in samples]))
    print(f'  avg pixel mean across {n_probes} samples: {avg_mean:.4f}')
    print(f'  avg pixel max:                            {avg_max:.4f}')
    print(f'  variance across samples:                   {var_across:.6f}')
    if avg_max < 0.05:
        print(f'  WARNING: {name} appears to be outputting all-zeros (mode collapse).')
        print(f'           Check that the model was retrained with the current')
        print(f'           forward process and that loss converged below ~0.005.')
        return False
    if var_across < 1e-6:
        print(f'  WARNING: {name} outputs are nearly identical across samples.')
        return False
    print(f'  looks healthy')
    return True


def run_controllability(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    print('\n=== Loading models ===')
    pflow_model, pflow_hp = load_model(args.pflow_ckpt, device=device)
    base_model, base_sched, base_hp = load_cond_baseline(args.baseline_ckpt, device=device)
    print(f'  pflow params:    {sum(p.numel() for p in pflow_model.parameters()):,}')
    print(f'  baseline params: {sum(p.numel() for p in base_model.parameters()):,}')

    # Resolve dataset: CLI override > PFlow-T ckpt hparams > 'mnist'.
    dataset_name = args.dataset or pflow_hp.get('dataset_name') or 'mnist'
    print(f'  dataset: {dataset_name}')

    print('\n=== Loading test set ===')
    test_ds = MNISTPFlowDataset(
        root=args.data_root, train=False, digit_filter=list(args.digits),
        dataset_name=dataset_name,
    )
    print(f'  test set: {len(test_ds)} samples on {args.digits}')

    # Run degeneracy checks on each model BEFORE the expensive controllability run.
    def probe_pflow():
        item = test_ds[np.random.randint(0, len(test_ds))]
        img = item['target'].numpy()[0]
        sch = test_ds.get_schedule(np.random.randint(0, len(test_ds)))
        x_T = make_x_T(img, sch, fill_radius=test_ds.fill_radius)
        return sample_one_shot(pflow_model, x_T, device=device)

    def probe_baseline():
        from pflow_ds.baseline_cond import get_baseline_cond
        idx = np.random.randint(0, len(test_ds))
        cond_vec = get_baseline_cond(test_ds, idx, base_hp)
        return sample_cond_baseline(base_model, base_sched, cond_vec,
                                    n_samples=1, image_size=28, device=device)[0]

    pflow_ok = degeneracy_check(probe_pflow, 'PFlow-T')
    base_ok = degeneracy_check(probe_baseline, 'Baseline (DDPM+PD)')

    if not pflow_ok:
        print('\nPFlow-T failed the sanity check. Stopping early — fix training before')
        print('running controllability. Likely causes:')
        print('  (a) Checkpoint was trained with old forward process; retrain on current code.')
        print('  (b) Training loss did not converge — check train_log.csv.')
        print('  (c) The blob -> digit inverse is too hard at current capacity; try '
              '--base_channels 64 and more epochs.')
        return
    if not base_ok:
        print('\nBaseline failed the sanity check. Stopping early.')
        return

    print('\n=== Partitioning test set by β1 ===')
    buckets = partition_by_betti(test_ds, threshold=args.betti_threshold,
                                 max_per_class=args.n_per_betti)
    for b, idx in buckets.items():
        print(f'  β1={b}: {len(idx)} test images')

    print('\n=== Running head-to-head (this may take a few minutes) ===')
    per_class = {}
    for b1, indices in buckets.items():
        if len(indices) < 5:
            continue  # not enough samples for meaningful comparison
        print(f'\n--- target β1 = {b1} ({len(indices)} images) ---')
        res = evaluate_class(b1, indices, test_ds, pflow_model,
                             base_model, base_sched, device,
                             threshold=args.betti_threshold,
                             baseline_hp=base_hp)
        print(f'  PFlow-T match rate:  {res["pflow_match_rate"]*100:5.1f}%')
        print(f'  Baseline match rate: {res["baseline_match_rate"]*100:5.1f}%')
        print(f'  PFlow-T  β1 dist: {res["pflow_b1_distribution"]}')
        print(f'  Baseline β1 dist: {res["baseline_b1_distribution"]}')
        per_class[b1] = res

    # Aggregate stats (averaged across classes)
    if per_class:
        avg_pflow = np.mean([r['pflow_match_rate'] for r in per_class.values()])
        avg_base = np.mean([r['baseline_match_rate'] for r in per_class.values()])
        print(f'\n=== Average across classes ===')
        print(f'  PFlow-T:  {avg_pflow*100:5.1f}%')
        print(f'  Baseline: {avg_base*100:5.1f}%')
        print(f'  delta:    {(avg_pflow-avg_base)*100:+5.1f} pp')

    # Plots
    print('\n=== Plotting ===')
    plot_match_rates(per_class, os.path.join(args.out_dir, 'match_rates.png'))
    plot_samples_grid(per_class, os.path.join(args.out_dir, 'samples.png'))

    # JSON summary (drop big arrays so it's small)
    summary = {}
    for b, r in per_class.items():
        summary[int(b)] = {
            'n': r['n'],
            'pflow_match_rate': r['pflow_match_rate'],
            'baseline_match_rate': r['baseline_match_rate'],
            'pflow_b1_distribution': {int(k): int(v) for k, v in r['pflow_b1_distribution'].items()},
            'baseline_b1_distribution': {int(k): int(v) for k, v in r['baseline_b1_distribution'].items()},
        }
    out_json = os.path.join(args.out_dir, 'summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  saved {out_json}')


if __name__ == '__main__':
    args = build_argparser().parse_args()
    run_controllability(args)
