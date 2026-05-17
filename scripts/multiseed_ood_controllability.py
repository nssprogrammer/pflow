"""Multi-seed wrapper for OOD controllability.

Same design pattern as scripts/multiseed_controllability.py, but for the
OOD source-to-target pairs from scripts/ood_controllability.py.

For each seed:
  - Re-shuffle the source-class test indices and pick the first n_per_pair.
  - Re-set torch + numpy RNG so the baseline's Gaussian noise differs.
  - Run evaluate_pair() for each (source, target) pair.

Aggregates across seeds:
  - Mean ± std on per-pair "generated β1 matches source β1" rate.
  - Welch t-statistic for PFlow-T vs baseline per pair.
  - Aggregate over the discriminating pairs (source β1 ≥ 1) — this is
    the paper's headline OOD number.

What does NOT vary per seed:
  - Trained model weights. We reuse the same PFlow-T and baseline
    checkpoints across all seeds.

Usage:
    python -m scripts.multiseed_ood_controllability \\
        --pflow_ckpt runs/pflow_ds/pflow_ds_epoch19.pt \\
        --baseline_ckpt runs/baseline/baseline_epoch19.pt \\
        --digits 0 1 8 \\
        --n_per_pair 30 \\
        --seeds 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt

from pflow_ds.dataset import MNISTPFlowDataset, make_x_T
from pflow_ds.train import load_model, sample_one_shot
from pflow_ds.baseline_cond import (
    load_cond_baseline, sample_cond_baseline,
)
from scripts.ood_controllability import (
    evaluate_pair, indices_by_label, DEFAULT_PAIRS, parse_pairs,
)
from scripts.controllability import degeneracy_check


# =========================================================================
# Seeding
# =========================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =========================================================================
# One seed
# =========================================================================

def run_one_seed(
    seed: int,
    valid_pairs: List[tuple],
    label_buckets: Dict[int, List[int]],
    test_ds: MNISTPFlowDataset,
    pflow_model,
    base_model,
    base_sched,
    n_per_pair: int,
    threshold: float,
    device: str,
    baseline_hp: dict = None,
) -> Dict[str, dict]:
    """Run all valid OOD pairs once with a given seed."""
    set_seed(seed)
    rng = np.random.RandomState(seed)

    per_pair: Dict[str, dict] = {}
    for src, tgt in valid_pairs:
        pname = f'{src}->{tgt}'
        src_indices = rng.permutation(label_buckets[src]).tolist()
        res = evaluate_pair(
            src, tgt, src_indices, test_ds, pflow_model, base_model,
            base_sched, device, threshold=threshold, n_pairs=n_per_pair,
            baseline_hp=baseline_hp,
        )
        # Strip heavy fields (images) — keep only scalar stats.
        per_pair[pname] = {
            'n': res['n'],
            'pflow_match_rate': res['pflow_match_rate'],
            'baseline_match_rate': res['baseline_match_rate'],
            'src_b1_distribution': {int(k): int(v)
                                    for k, v in res['src_b1_distribution'].items()},
            'pflow_b1_distribution': {int(k): int(v)
                                      for k, v in res['pflow_b1_distribution'].items()},
            'baseline_b1_distribution': {int(k): int(v)
                                         for k, v in res['baseline_b1_distribution'].items()},
        }
    return per_pair


# =========================================================================
# Aggregation
# =========================================================================

def aggregate_across_seeds(all_results: Dict[int, Dict[str, dict]]) -> Dict[str, dict]:
    """Compute mean/std per pair across seeds."""
    all_pairs = sorted(set().union(*(r.keys() for r in all_results.values())))
    agg = {}
    for pname in all_pairs:
        pflow_rates, base_rates, ns = [], [], []
        src_b1_collect: Dict[int, int] = {}
        for seed, res in all_results.items():
            if pname in res:
                pflow_rates.append(res[pname]['pflow_match_rate'])
                base_rates.append(res[pname]['baseline_match_rate'])
                ns.append(res[pname]['n'])
                # Aggregate the source β1 distribution (sanity check that the
                # source bucket is stable across seeds).
                for k, v in res[pname]['src_b1_distribution'].items():
                    src_b1_collect[k] = src_b1_collect.get(k, 0) + v
        agg[pname] = {
            'n_seeds': len(pflow_rates),
            'n_per_seed': int(np.mean(ns)) if ns else 0,
            'pflow_mean': float(np.mean(pflow_rates)),
            'pflow_std': float(np.std(pflow_rates, ddof=1)) if len(pflow_rates) > 1 else 0.0,
            'pflow_rates': pflow_rates,
            'base_mean': float(np.mean(base_rates)),
            'base_std': float(np.std(base_rates, ddof=1)) if len(base_rates) > 1 else 0.0,
            'base_rates': base_rates,
            'delta_mean': float(np.mean(pflow_rates) - np.mean(base_rates)),
            'src_b1_pooled': src_b1_collect,
        }
    return agg


def welch_t(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        diff = a.mean() - b.mean()
        return float('inf') if diff > 0 else (float('-inf') if diff < 0 else 0.0)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / se)


# =========================================================================
# Discriminating-pairs aggregate (the paper headline)
# =========================================================================

def discriminating_aggregate(
    agg: Dict[str, dict], min_src_b1: int = 1,
) -> Dict[str, float]:
    """Average across only those pairs whose pooled source-image β1 is ≥ min_src_b1.

    These are the pairs where the baseline can't trivially win by
    "do not add holes" — the test PFlow-T's substrate claim hinges on.
    """
    pflow_means, base_means, deltas = [], [], []
    pairs_used = []
    for pname, d in agg.items():
        # Use the mode of the source distribution as the "typical" source β1.
        src_dist = d['src_b1_pooled']
        if not src_dist:
            continue
        modal_b1 = max(src_dist, key=src_dist.get)
        if modal_b1 >= min_src_b1:
            pflow_means.append(d['pflow_mean'])
            base_means.append(d['base_mean'])
            deltas.append(d['delta_mean'])
            pairs_used.append(pname)
    if not pflow_means:
        return {'pflow_mean': 0.0, 'base_mean': 0.0, 'delta_mean': 0.0,
                'pairs': []}
    return {
        'pflow_mean': float(np.mean(pflow_means)),
        'base_mean': float(np.mean(base_means)),
        'delta_mean': float(np.mean(deltas)),
        'pairs': pairs_used,
    }


# =========================================================================
# Output
# =========================================================================

def print_table(agg: Dict[str, dict], disc: Dict[str, float]):
    print()
    print('=' * 80)
    print('Multi-seed OOD controllability summary')
    print('=' * 80)
    any_pair = next(iter(agg.values()))
    n_seeds = any_pair['n_seeds']
    n_per = any_pair['n_per_seed']
    print(f'{n_seeds} seeds × {n_per} samples per pair per seed\n')

    header = (f'  {"pair":<10} {"PFlow-T (% ± std)":<22} '
              f'{"Baseline (% ± std)":<22} {"Δ":<10} {"t":<8}')
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for pname, d in agg.items():
        pflow_str = f'{d["pflow_mean"]*100:5.1f} ± {d["pflow_std"]*100:4.1f}'
        base_str = f'{d["base_mean"]*100:5.1f} ± {d["base_std"]*100:4.1f}'
        delta = d['delta_mean'] * 100
        t_stat = welch_t(d['pflow_rates'], d['base_rates'])
        if np.isinf(t_stat):
            t_str = '   inf'
        elif np.isnan(t_stat):
            t_str = '    NA'
        else:
            t_str = f'{t_stat:6.2f}'
        print(f'  {pname:<10} {pflow_str:<22} {base_str:<22} {delta:+6.1f}pp  {t_str}')
    print()
    print(f'  Headline: discriminating pairs (source β1 ≥ 1) — '
          f'{len(disc["pairs"])} pair(s): {disc["pairs"]}')
    print(f'    PFlow-T avg: {disc["pflow_mean"]*100:5.1f}%   '
          f'Baseline avg: {disc["base_mean"]*100:5.1f}%   '
          f'Δ: {disc["delta_mean"]*100:+5.1f}pp')
    print('=' * 80)


def print_latex(agg: Dict[str, dict], disc: Dict[str, float], path: str):
    lines = []
    lines.append(r'% OOD controllability, multi-seed')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'pair & PFlow-T (\%) & Baseline (\%) & $\Delta$ (pp) \\')
    lines.append(r'\midrule')
    for pname, d in agg.items():
        lines.append(
            f'{pname} & ${d["pflow_mean"]*100:.1f} \\pm {d["pflow_std"]*100:.1f}$ '
            f'& ${d["base_mean"]*100:.1f} \\pm {d["base_std"]*100:.1f}$ '
            f'& ${d["delta_mean"]*100:+.1f}$ \\\\'
        )
    lines.append(r'\midrule')
    lines.append(
        f'avg ($\\beta_1\\geq 1$ source) & ${disc["pflow_mean"]*100:.1f}$ '
        f'& ${disc["base_mean"]*100:.1f}$ '
        f'& ${disc["delta_mean"]*100:+.1f}$ \\\\'
    )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'saved {path}')


def plot_error_bars(agg: Dict[str, dict], save_path: str):
    pairs = list(agg.keys())
    pf_m = [agg[p]['pflow_mean'] for p in pairs]
    pf_s = [agg[p]['pflow_std'] for p in pairs]
    bs_m = [agg[p]['base_mean'] for p in pairs]
    bs_s = [agg[p]['base_std'] for p in pairs]

    x = np.arange(len(pairs)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w/2, pf_m, w, yerr=pf_s, capsize=4,
           label='PFlow-T (substrate)', color='tab:blue')
    ax.bar(x + w/2, bs_m, w, yerr=bs_s, capsize=4,
           label='DDPM + PD cond', color='tab:orange')
    ax.set_xticks(x); ax.set_xticklabels(pairs)
    ax.set_ylabel('β1(generated) == β1(source) rate')
    ax.set_ylim(0, 1.05)
    n_seeds = next(iter(agg.values()))['n_seeds']
    ax.set_title(f'OOD controllability (mean ± std, {n_seeds} seeds)')
    ax.legend(loc='upper right')
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
    p.add_argument('--pairs', type=str, default=None,
                   help='Comma-separated source->target pairs, e.g. "8->1,1->8,0->1". '
                        'If omitted, uses DEFAULT_PAIRS from ood_controllability.')
    p.add_argument('--n_per_pair', type=int, default=30,
                   help='Samples per (source, target) pair PER SEED.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                   help='Random seeds (3 is the minimum for std).')
    p.add_argument('--betti_threshold', type=float, default=0.5)
    p.add_argument('--min_disc_src_b1', type=int, default=1,
                   help='Minimum (modal) source β1 to include in the '
                        'discriminating-pair aggregate (default 1).')
    p.add_argument('--out_dir', type=str, default='runs/multiseed_ood')
    p.add_argument('--skip_degeneracy_check', action='store_true')
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(f'seeds: {args.seeds}   n_per_pair: {args.n_per_pair}')

    # Load models once.
    print('\n=== Loading models ===')
    pflow_model, _ = load_model(args.pflow_ckpt, device=device)
    base_model, base_sched, base_hp = load_cond_baseline(args.baseline_ckpt, device=device)
    print(f'  pflow params:    {sum(p.numel() for p in pflow_model.parameters()):,}')
    print(f'  baseline params: {sum(p.numel() for p in base_model.parameters()):,}')

    # Load test set once.
    print('\n=== Loading test set ===')
    test_ds = MNISTPFlowDataset(
        root=args.data_root, train=False, digit_filter=list(args.digits),
    )
    label_buckets = indices_by_label(test_ds)
    print(f'  test set: {len(test_ds)} samples')
    for lbl, idx in label_buckets.items():
        print(f'    label {lbl}: {len(idx)} samples')

    # Parse pairs and filter against available labels.
    pair_list = parse_pairs(args.pairs) if args.pairs else DEFAULT_PAIRS
    valid_pairs = []
    for src, tgt in pair_list:
        if src not in label_buckets or len(label_buckets[src]) < 5:
            print(f'  WARNING: skipping {src}->{tgt}, not enough source digits.')
            continue
        valid_pairs.append((src, tgt))
    if not valid_pairs:
        print('No valid pairs to evaluate. Exiting.')
        return

    # Optional degeneracy check.
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
            print('\nDegeneracy check failed — refusing to run multi-seed OOD.')
            return

    # Per-seed evaluation.
    print('\n=== Running per-seed OOD evaluation ===')
    all_results: Dict[int, Dict[str, dict]] = {}
    for seed in args.seeds:
        print(f'\n--- seed {seed} ---')
        per_pair = run_one_seed(
            seed, valid_pairs, label_buckets, test_ds,
            pflow_model, base_model, base_sched,
            n_per_pair=args.n_per_pair, threshold=args.betti_threshold,
            device=device, baseline_hp=base_hp,
        )
        for pname, r in per_pair.items():
            print(f'  {pname}: PFlow-T {r["pflow_match_rate"]*100:5.1f}%   '
                  f'Baseline {r["baseline_match_rate"]*100:5.1f}%')
        all_results[seed] = per_pair

    # Aggregate.
    agg = aggregate_across_seeds(all_results)
    disc = discriminating_aggregate(agg, min_src_b1=args.min_disc_src_b1)
    print_table(agg, disc)

    # Outputs.
    plot_error_bars(agg, os.path.join(args.out_dir, 'ood_match_rates_multiseed.png'))
    print_latex(agg, disc, os.path.join(args.out_dir, 'ood_table.tex'))

    save = {
        'seeds': list(args.seeds),
        'n_per_pair': args.n_per_pair,
        'digits': list(args.digits),
        'pairs': [f'{a}->{b}' for a, b in valid_pairs],
        'pflow_ckpt': args.pflow_ckpt,
        'baseline_ckpt': args.baseline_ckpt,
        'min_disc_src_b1': args.min_disc_src_b1,
        'per_seed': {str(s): r for s, r in all_results.items()},
        'aggregate': agg,
        'discriminating_aggregate': disc,
    }
    out_path = os.path.join(args.out_dir, 'ood_multiseed_summary.json')
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
