"""Multi-seed controllability runner.

Runs the controllability experiment N times with different seeds and
aggregates: mean ± std on match rates, plus a Welch t-statistic per β1
class to indicate whether the PFlow-T vs baseline gap is real.

What varies per seed:
  - Test-image subsampling: each seed picks a different random subset
    of n_per_betti images from each β1 bucket.
  - Baseline Gaussian noise: each seed's torch random state produces a
    different DDPM sampling trajectory.

What does NOT vary per seed:
  - Trained model weights. We reuse the same checkpoints for both PFlow-T
    and baseline across all seeds. For a fully paper-grade experiment
    you would also train multiple model seeds; that's a separate runner.

Usage:
    python -m scripts.multiseed_controllability \\
        --pflow_ckpt runs/pflow_ds/pflow_ds_epoch11.pt \\
        --baseline_ckpt runs/baseline/baseline_epoch11.pt \\
        --digits 0 1 8 \\
        --n_per_betti 30 \\
        --seeds 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pflow_ds.dataset import MNISTPFlowDataset
from pflow_ds.train import load_model
from pflow_ds.baseline_cond import load_cond_baseline
from scripts.controllability import (
    partition_by_betti, evaluate_class, degeneracy_check,
)
from pflow_ds.train import sample_one_shot
from pflow_ds.baseline_cond import sample_cond_baseline
from pflow_ds.dataset import make_x_T


# =========================================================================
# Seeding
# =========================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =========================================================================
# One seed = one full controllability evaluation
# =========================================================================

def run_one_seed(
    seed: int,
    pflow_model,
    base_model,
    base_sched,
    test_ds: MNISTPFlowDataset,
    n_per_betti: int,
    threshold: float,
    device: str,
    baseline_hp: dict = None,
):
    set_seed(seed)
    # Partition all test images by β1 (deterministic — same across seeds).
    buckets = partition_by_betti(test_ds, threshold=threshold, max_per_class=10_000)
    # Randomly sub-sample n_per_betti per class.
    rng = np.random.RandomState(seed)
    per_class = {}
    for b1, all_indices in buckets.items():
        if len(all_indices) < 5:
            continue
        chosen = rng.permutation(all_indices)[:n_per_betti].tolist()
        res = evaluate_class(b1, chosen, test_ds, pflow_model, base_model,
                             base_sched, device, threshold=threshold,
                             baseline_hp=baseline_hp)
        # Strip heavy fields before storing — we only need scalar stats.
        per_class[b1] = {
            'n': res['n'],
            'pflow_match_rate': res['pflow_match_rate'],
            'baseline_match_rate': res['baseline_match_rate'],
            'pflow_b1_distribution': {int(k): int(v)
                                      for k, v in res['pflow_b1_distribution'].items()},
            'baseline_b1_distribution': {int(k): int(v)
                                         for k, v in res['baseline_b1_distribution'].items()},
        }
    return per_class


# =========================================================================
# Aggregation
# =========================================================================

def aggregate_across_seeds(all_results: dict) -> dict:
    """Compute mean/std/raw-rates per β1 class across seeds."""
    all_b1s = sorted(set().union(*(r.keys() for r in all_results.values())))
    agg = {}
    for b1 in all_b1s:
        pflow_rates = []
        base_rates = []
        ns = []
        for seed, res in all_results.items():
            if b1 in res:
                pflow_rates.append(res[b1]['pflow_match_rate'])
                base_rates.append(res[b1]['baseline_match_rate'])
                ns.append(res[b1]['n'])
        agg[b1] = {
            'n_seeds': len(pflow_rates),
            'n_per_seed': int(np.mean(ns)) if ns else 0,
            'pflow_mean': float(np.mean(pflow_rates)),
            'pflow_std': float(np.std(pflow_rates, ddof=1)) if len(pflow_rates) > 1 else 0.0,
            'pflow_rates': pflow_rates,
            'base_mean': float(np.mean(base_rates)),
            'base_std': float(np.std(base_rates, ddof=1)) if len(base_rates) > 1 else 0.0,
            'base_rates': base_rates,
            'delta_mean': float(np.mean(pflow_rates) - np.mean(base_rates)),
        }
    return agg


def welch_t(a, b):
    """Welch t-statistic for two independent samples (returns t, dof)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan'), float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        diff = a.mean() - b.mean()
        return (float('inf') if diff > 0 else float('-inf') if diff < 0 else 0.0), float('nan')
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float('nan'), float('nan')
    t = (a.mean() - b.mean()) / se
    df_num = (va / na + vb / nb) ** 2
    df_den = ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1))
    df = df_num / df_den if df_den > 0 else float('nan')
    return float(t), float(df)


# =========================================================================
# Output
# =========================================================================

def print_table(agg: dict):
    print()
    print('=' * 76)
    print('Multi-seed controllability summary')
    print('=' * 76)
    n_seeds = next(iter(agg.values()))['n_seeds']
    n_per = next(iter(agg.values()))['n_per_seed']
    print(f'{n_seeds} seeds × {n_per} test images per β1 class per seed\n')
    header = f'  {"β1":<5} {"PFlow-T (% ± std)":<22} {"Baseline (% ± std)":<22} {"Δ":<10} {"t":<8}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for b1, d in agg.items():
        pflow_str = f'{d["pflow_mean"]*100:5.1f} ± {d["pflow_std"]*100:4.1f}'
        base_str = f'{d["base_mean"]*100:5.1f} ± {d["base_std"]*100:4.1f}'
        delta = d['delta_mean'] * 100
        t_stat, _ = welch_t(d['pflow_rates'], d['base_rates'])
        if np.isinf(t_stat):
            t_str = '   inf'
        elif np.isnan(t_stat):
            t_str = '    NA'
        else:
            t_str = f'{t_stat:6.2f}'
        print(f'  {b1:<5} {pflow_str:<22} {base_str:<22} {delta:+6.1f}pp  {t_str}')
    print()
    avg_pflow = np.mean([d['pflow_mean'] for d in agg.values()])
    avg_base = np.mean([d['base_mean'] for d in agg.values()])
    print(f'  Overall avg across classes:  PFlow-T {avg_pflow*100:5.1f}%   '
          f'Baseline {avg_base*100:5.1f}%   Δ {(avg_pflow-avg_base)*100:+5.1f}pp')
    print('=' * 76)


def print_latex(agg: dict, path: str):
    lines = []
    lines.append(r'% PFlow-T vs Baseline controllability, multi-seed')
    lines.append(r'\begin{tabular}{cccc}')
    lines.append(r'\toprule')
    lines.append(r'$\beta_1$ & PFlow-T (\%) & Baseline (\%) & $\Delta$ (pp) \\')
    lines.append(r'\midrule')
    for b1, d in agg.items():
        lines.append(
            f'{b1} & ${d["pflow_mean"]*100:.1f} \\pm {d["pflow_std"]*100:.1f}$ '
            f'& ${d["base_mean"]*100:.1f} \\pm {d["base_std"]*100:.1f}$ '
            f'& ${d["delta_mean"]*100:+.1f}$ \\\\'
        )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'saved {path}')


def plot_error_bars(agg: dict, save_path: str):
    b1s = sorted(agg.keys())
    pf_m = [agg[b]['pflow_mean'] for b in b1s]
    pf_s = [agg[b]['pflow_std'] for b in b1s]
    bs_m = [agg[b]['base_mean'] for b in b1s]
    bs_s = [agg[b]['base_std'] for b in b1s]

    x = np.arange(len(b1s)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w/2, pf_m, w, yerr=pf_s, capsize=4,
           label='PFlow-T (substrate)', color='tab:blue')
    ax.bar(x + w/2, bs_m, w, yerr=bs_s, capsize=4,
           label='DDPM + PD cond', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'β1 = {b}' for b in b1s])
    ax.set_ylabel('β1 match rate')
    ax.set_ylim(0, 1.05)
    n_seeds = next(iter(agg.values()))['n_seeds']
    ax.set_title(f'Controllability (mean ± std, {n_seeds} seeds)')
    ax.legend(loc='upper right')
    # Numeric labels above each bar
    for i, (p, b) in enumerate(zip(pf_m, bs_m)):
        ax.text(i - w/2, p + max(pf_s[i], 0.02) + 0.01, f'{p*100:.0f}%',
                ha='center', fontsize=9)
        ax.text(i + w/2, b + max(bs_s[i], 0.02) + 0.01, f'{b*100:.0f}%',
                ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f'saved {save_path}')


# =========================================================================
# Entrypoint
# =========================================================================

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--pflow_ckpt', type=str, required=True)
    p.add_argument('--baseline_ckpt', type=str, required=True)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--digits', type=int, nargs='+', default=[0, 1, 8])
    p.add_argument('--n_per_betti', type=int, default=30,
                   help='Test images per β1 class PER SEED.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                   help='Random seeds (3 is the minimum for std).')
    p.add_argument('--betti_threshold', type=float, default=0.5)
    p.add_argument('--out_dir', type=str, default='runs/multiseed')
    p.add_argument('--skip_degeneracy_check', action='store_true')
    p.add_argument('--dataset', type=str, default=None,
                   choices=['mnist', 'fashion', None],
                   help='Source dataset; inferred from PFlow-T checkpoint if not set.')
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(f'seeds: {args.seeds}   n_per_betti: {args.n_per_betti}')

    # Load models once and reuse across seeds.
    print('\n=== Loading models ===')
    pflow_model, pflow_hp = load_model(args.pflow_ckpt, device=device)
    base_model, base_sched, base_hp = load_cond_baseline(args.baseline_ckpt, device=device)
    print(f'  pflow params:    {sum(p.numel() for p in pflow_model.parameters()):,}')
    print(f'  baseline params: {sum(p.numel() for p in base_model.parameters()):,}')

    dataset_name = args.dataset or pflow_hp.get('dataset_name') or 'mnist'
    print(f'  dataset: {dataset_name}')

    # Load test set once.
    print('\n=== Loading test set ===')
    test_ds = MNISTPFlowDataset(
        root=args.data_root, train=False, digit_filter=list(args.digits),
        dataset_name=dataset_name,
    )
    print(f'  test set: {len(test_ds)} samples on {args.digits}')

    # Degeneracy check (optional skip).
    if not args.skip_degeneracy_check:
        def probe_pflow():
            idx = np.random.randint(0, len(test_ds))
            item = test_ds[idx]
            sch = test_ds.get_schedule(idx)
            x_T = make_x_T(item['target'].numpy()[0], sch,
                           fill_radius=test_ds.fill_radius)
            return sample_one_shot(pflow_model, x_T, device=device)

        def probe_baseline():
            from pflow_ds.baseline_cond import get_baseline_cond
            idx = np.random.randint(0, len(test_ds))
            cond_vec = get_baseline_cond(test_ds, idx, base_hp)
            return sample_cond_baseline(base_model, base_sched, cond_vec,
                                        n_samples=1, image_size=28,
                                        device=device)[0]

        pf_ok = degeneracy_check(probe_pflow, 'PFlow-T', n_probes=4)
        bs_ok = degeneracy_check(probe_baseline, 'Baseline', n_probes=4)
        if not (pf_ok and bs_ok):
            print('\nDegeneracy check failed — refusing to run multi-seed evaluation.')
            return

    # Per-seed evaluation.
    print('\n=== Running per-seed evaluation ===')
    all_results = {}
    for seed in args.seeds:
        print(f'\n--- seed {seed} ---')
        per_class = run_one_seed(
            seed, pflow_model, base_model, base_sched, test_ds,
            n_per_betti=args.n_per_betti, threshold=args.betti_threshold,
            device=device, baseline_hp=base_hp,
        )
        for b1, res in per_class.items():
            print(f'  β1={b1}: PFlow-T {res["pflow_match_rate"]*100:5.1f}%   '
                  f'Baseline {res["baseline_match_rate"]*100:5.1f}%')
        all_results[seed] = per_class

    # Aggregate.
    agg = aggregate_across_seeds(all_results)
    print_table(agg)

    # Outputs.
    plot_error_bars(agg, os.path.join(args.out_dir, 'match_rates_multiseed.png'))
    print_latex(agg, os.path.join(args.out_dir, 'table.tex'))

    save = {
        'seeds': list(args.seeds),
        'n_per_betti': args.n_per_betti,
        'digits': list(args.digits),
        'pflow_ckpt': args.pflow_ckpt,
        'baseline_ckpt': args.baseline_ckpt,
        'per_seed': {str(s): {str(b): r for b, r in res.items()}
                     for s, res in all_results.items()},
        'aggregate': {str(b): d for b, d in agg.items()},
    }
    out_path = os.path.join(args.out_dir, 'multiseed_summary.json')
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
