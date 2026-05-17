"""Visualization for PFlow-T dataset version."""

from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from .persistence import count_betti


def plot_forward_sweep(
    img: np.ndarray, schedule, ts=(0.0, 0.25, 0.5, 0.75, 1.0),
    fill_radius: float = 2.5, save_path: str | None = None,
):
    """Visualize the forward process at several t values + Betti numbers."""
    from .forward import persistence_melt
    fig, axes = plt.subplots(1, len(ts), figsize=(2.6 * len(ts), 3.2))
    for ax, t in zip(axes, ts):
        x_t = persistence_melt(img, float(t), schedule=schedule,
                               fill_radius=fill_radius)
        b0, b1 = count_betti(x_t, threshold=0.5)
        ax.imshow(x_t, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't = {t:.2f}\nβ0={b0}  β1={b1}')
        ax.axis('off')
    plt.suptitle('Forward process: H1 features killed in ascending persistence order',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'  saved {save_path}')
    return fig


def plot_diagnose(
    target: np.ndarray, ts, inputs: List[np.ndarray], outputs: List[np.ndarray],
    save_path: str | None = None,
):
    """Per-t diagnostic: model output should look like the target at every t."""
    n = len(ts)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.2))
    for i, t in enumerate(ts):
        b0_in, b1_in = count_betti(inputs[i], threshold=0.5)
        b0_out, b1_out = count_betti(outputs[i], threshold=0.5)
        axes[0, i].imshow(inputs[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'x_t  t={t:.2f}\nβ1={b1_in}')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'pred(x_t, t)\nβ1={b1_out}')
        axes[1, i].axis('off')
    plt.suptitle('Per-t diagnostic: every prediction should resemble the target',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'  saved {save_path}')
    return fig


def plot_grid(images: List[np.ndarray], titles: List[str] | None = None,
              cols: int = 8, save_path: str | None = None,
              suptitle: str | None = None):
    """Generic grid of images."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=8)
        ax.axis('off')
    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'  saved {save_path}')
    return fig
