# PFlow-T: A Persistence-Driven Forward Process for Topology-Controlled Generation

Reference implementation of **PFlow-T**, a generative model whose
forward (noising) process is defined by the persistent homology of the
data rather than by topology-agnostic Gaussian noise.

The time parameter $t \in [0, 1]$ encodes what fraction of the data's
$H_1$ persistence-mass has been killed; the forward operator destroys
$H_1$ features (digit holes) in strict ascending order of persistence.
The reverse network predicts $x_0$ from $(x_t, t)$ directly and is
invoked one-shot at inference time.

This repository reproduces the headline numbers from the paper:

- **In-distribution controllability on MNIST $\{0, 1, 8\}$**:
  PFlow-T matches the requested $\beta_1$ in $99.6\%/96.0\%/84.8\%$ of
  generations vs. $96.4\%/3.2\%/0.0\%$ for a parameter-matched DDPM
  conditioned on a 64-d persistence landscape (the descriptor used by
  TAGG and TopoDiffusionNet).
- **Out-of-distribution controllability**: PFlow-T preserves the source
  topology in $92.0\%$ of cases on the discriminating pairs vs. $9.7\%$
  for the baseline.

All numbers are mean across 5 evaluation seeds.

Paper: [https://arxiv.org/abs/2605.17555]

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pflow_ds/
│   ├── persistence.py            # gudhi cubical PH with critical-cell locations
│   ├── forward.py                # persistence_melt forward operator (Definition 1)
│   ├── landscape.py              # Bubenik (2015) persistence landscapes
│   ├── dataset.py                # MNIST/Fashion-MNIST with cached events
│   ├── model.py                  # TinyUNet (x_0 prediction)
│   ├── train.py                  # PFlow-T training + one-shot sampling
│   ├── baseline_cond.py          # CondTinyUNet + DDPM scheduler (the baseline)
│   └── viz.py                    # plotting helpers
└── scripts/
    ├── smoke_test.py             # verify the pipeline before training
    ├── run_pflow_ds.py           # train PFlow-T + diagnostics
    ├── train_baseline.py         # train the conditioning baseline
    ├── controllability.py        # single-seed in-distribution comparison
    ├── multiseed_controllability.py    # 5-seed in-distribution evaluation
    ├── ood_controllability.py    # single-seed OOD experiment
    ├── multiseed_ood_controllability.py  # 5-seed OOD evaluation
    └── inspect_fashion.py        # probe Fashion-MNIST class topology
```

---

## Installation

```bash
git clone https://github.com/nssprogrammer/pflow.git
cd pflow-t
pip install -r requirements.txt
```

Dependencies are PyTorch, torchvision, NumPy, gudhi, matplotlib, and
tqdm. The trickiest one is gudhi; on Linux `pip install gudhi` usually
works. If not, use `conda install -c conda-forge gudhi`.

Tested on Python 3.9+, PyTorch 2.0+, gudhi 3.8+. Training reproduces
on a single GPU with 8 GB VRAM; CPU works but is roughly 10x slower.

---

## Reproducing the paper

The recipe below produces every numerical result and figure from the
paper. Run the steps in order; later steps consume the checkpoints
produced by earlier ones.

### Step 1 — Smoke test (~1 minute)

```bash
python -m scripts.smoke_test
```

Verifies that gudhi can compute persistence on MNIST, that the forward
process kills $H_1$ features in the right order, and that one training
step produces a finite loss. If the $\beta_1$ progression is
non-monotonic, bump `fill_radius` from 2.5 to 3.5 before continuing.

### Step 2 — Train PFlow-T (~10 minutes on a single GPU)

```bash
python -m scripts.run_pflow_ds \
    --digits 0 1 8 \
    --subset 6000 \
    --epochs 20 \
    --base_channels 64 \
    --out_dir runs/pflow_ds
```

Trains on MNIST digits 0, 1, 8 ($\beta_1$ values 1, 0, 2 respectively).
The final checkpoint is `runs/pflow_ds/pflow_ds_epoch19.pt`. The script
also produces:

- `runs/pflow_ds/forward_sweep.png` — visualization of the forward
  process at several $t$ values, showing $\beta_1$ progression
- `runs/pflow_ds/diagnose.png` — per-$t$ diagnostic; the model should
  output recognizable digits at every $t$
- `runs/pflow_ds/samples.png` — reconstructions for 16 test images

### Step 3 — Train the conditioning baseline (~10 minutes)

```bash
python -m scripts.train_baseline \
    --digits 0 1 8 \
    --subset 6000 \
    --epochs 20 \
    --base_channels 64 \
    --cond_type landscape \
    --out_dir runs/baseline_landscape
```

Same backbone as PFlow-T, same training data, same parameter count.
The `--cond_type landscape` flag specifies the 64-d persistence
landscape descriptor (Bubenik 2015). The final checkpoint is
`runs/baseline_landscape/baseline_epoch19.pt`.

To reproduce the weaker 5-d baseline in Appendix A, swap
`--cond_type landscape` with `--cond_type pd5` and choose a different
output directory.

### Step 4 — Single-seed in-distribution controllability (~5 minutes)

```bash
python -m scripts.controllability \
    --pflow_ckpt runs/pflow_ds/pflow_ds_epoch19.pt \
    --baseline_ckpt runs/baseline_landscape/baseline_epoch19.pt \
    --digits 0 1 8 \
    --n_per_betti 50 \
    --out_dir runs/ctrl
```

For each target $\beta_1 \in \{0, 1, 2\}$, partitions held-out test
images by ground-truth $\beta_1$, generates samples from both models,
and reports the match rate. Outputs:

- `runs/ctrl/match_rates.png` — bar chart per $\beta_1$ class
- `runs/ctrl/samples.png` — target / PFlow-T / baseline grid
- `runs/ctrl/summary.json` — numerical results

### Step 5 — Multi-seed in-distribution controllability (~25 minutes)

Step 4 produces point estimates; this produces mean ± std across 5 seeds.

```bash
python -m scripts.multiseed_controllability \
    --pflow_ckpt runs/pflow_ds/pflow_ds_epoch19.pt \
    --baseline_ckpt runs/baseline_landscape/baseline_epoch19.pt \
    --digits 0 1 8 \
    --n_per_betti 50 \
    --seeds 0 1 2 3 4 \
    --out_dir runs/multiseed
```

Each seed varies test-image subsampling and the baseline's Gaussian
noise trajectory. Outputs:

- `runs/multiseed/match_rates_multiseed.png` — bar chart with error bars
- `runs/multiseed/table.tex` — LaTeX-ready table for the paper
- `runs/multiseed/multiseed_summary.json` — per-seed + aggregate results

This is what produces Table 1 in the paper.

### Step 6 — Single-seed OOD controllability (~5 minutes)

```bash
python -m scripts.ood_controllability \
    --pflow_ckpt runs/pflow_ds/pflow_ds_epoch19.pt \
    --baseline_ckpt runs/baseline_landscape/baseline_epoch19.pt \
    --digits 0 1 8 \
    --n_per_pair 30 \
    --out_dir runs/ood
```

For each source $\rightarrow$ target pair, conditions on a test image
of the source class and measures whether the generated sample's
$\beta_1$ matches the source's. Tests whether the model honors
topology requests structurally rather than via class prior.

### Step 7 — Multi-seed OOD controllability (~25 minutes)

```bash
python -m scripts.multiseed_ood_controllability \
    --pflow_ckpt runs/pflow_ds/pflow_ds_epoch19.pt \
    --baseline_ckpt runs/baseline_landscape/baseline_epoch19.pt \
    --digits 0 1 8 \
    --n_per_pair 30 \
    --seeds 0 1 2 3 4 \
    --out_dir runs/multiseed_ood
```

The discriminating-pair aggregate (pairs where source $\beta_1 \geq 1$)
is the OOD headline number — 92.0% vs. 9.7% in the paper. Outputs
include an error-bar plot, a LaTeX table, and a JSON dump.

This is what produces Table 2 in the paper.

---

## Total runtime to reproduce

Roughly 80 minutes on a single GPU:

| Step | Description | Time |
|------|-------------|------|
| 1 | Smoke test | 1 min |
| 2 | Train PFlow-T | 10 min |
| 3 | Train baseline | 10 min |
| 4 | Single-seed in-distribution | 5 min |
| 5 | Multi-seed in-distribution | 25 min |
| 6 | Single-seed OOD | 5 min |
| 7 | Multi-seed OOD | 25 min |

CPU runtime is roughly 10x longer. The smoke test and single-seed
experiments are tractable on CPU; the multi-seed runs benefit
substantially from GPU.

---

## Fashion-MNIST (negative result)

We investigated Fashion-MNIST as a candidate second dataset:

```bash
python -m scripts.inspect_fashion --threshold_sweep
```

This loads the Fashion-MNIST test set, computes $\beta_1$ per class,
and reports the distribution. We found that all 10 Fashion-MNIST classes
have modal $\beta_1 = 0$, which makes the dataset unsuitable for the
controllability test as defined here. The script is included for
reproducibility of the negative result; we do not recommend training
PFlow-T on Fashion-MNIST without a synthetic-shape supplement that
covers $\beta_1 \in \{1, 2\}$.

---

## Citing this work

```
@misc{khilar2026pflowt,
  author = {Khilar, Snigdha Chandan},
  title  = {PFlow-T: A Persistence-Driven Forward Process for Topology-Controlled Generation},
  year   = {2026},
  note   = {arXiv preprint [2605.17555]},
}
```

---

## License

MIT. See `LICENSE`.

---

## Contact

Snigdha Chandan Khilar — `snkhilar@gmail.com`

Independent Researcher
