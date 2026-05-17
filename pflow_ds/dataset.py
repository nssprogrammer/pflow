"""MNIST dataset for PFlow-T with cached persistence events.

Cache stores the FULL events list per image (H0 + H1 with locations).
From the cached events we derive on the fly:
  - schedule: H1 events sorted by ascending persistence (for PFlow-T forward)
  - pd_vec: a 5-d persistence descriptor (for baseline conditioning)

Multi-digit support: digit_filter can be int, list of ints, or None (all).
"""

from __future__ import annotations
import os
import pickle
from typing import Optional, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from .forward import persistence_melt
from .persistence import compute_persistence_with_locations
from .landscape import landscape_descriptor


def pd_to_vec(events: List[dict]) -> np.ndarray:
    """Compress a persistence diagram into a 5-d descriptor.

    Components:
        [n_h0, n_h1, max_pers_h1, sum_pers_h1, max_pers_h0]

    These five summaries are (a) stable under small perturbations and
    (b) n_h1 directly encodes β1, the controllability knob we test.
    """
    h0 = [e for e in events if e['dim'] == 0]
    h1 = [e for e in events if e['dim'] == 1]
    return np.array([
        float(len(h0)),
        float(len(h1)),
        max([e['persistence'] for e in h1], default=0.0),
        sum([e['persistence'] for e in h1]),
        max([e['persistence'] for e in h0], default=0.0),
    ], dtype=np.float32)


def schedule_from_events(
    events: List[dict],
    min_persistence: float = 0.02,
    fallback_top_k: int = 4,
) -> List[dict]:
    """H1 events sorted by ascending persistence (PFlow-T kill schedule).

    If `min_persistence` yields zero features, fall back to the top
    `fallback_top_k` H1 features by persistence. This handles real MNIST
    digits where loop persistences can be small (e.g., 0.02-0.04) — much
    smaller than the persistences of clean synthetic digits.
    """
    h1_all = [e for e in events if e['dim'] == 1]
    h1 = [e for e in h1_all if e['persistence'] >= min_persistence]
    if not h1 and h1_all:
        # Fallback: keep the most persistent H1 features regardless of threshold.
        h1 = sorted(h1_all, key=lambda e: -e['persistence'])[:fallback_top_k]
    h1.sort(key=lambda e: e['persistence'])
    return h1


class MNISTPFlowDataset(Dataset):
    """MNIST + cached persistence events for both PFlow-T and baseline.

    Returns per __getitem__:
        target:   (1, H, W) tensor
        x_t:      (1, H, W) tensor   (random t per call)
        x_T:      (1, H, W) tensor   (fully melted, for PFlow-T inference)
        t:        scalar tensor in [0, 1]
        label:    int
        pd_vec:   (5,) tensor        (baseline conditioning input)
    """

    CACHE_VERSION = 3

    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        digit_filter: Optional[Union[int, List[int]]] = 8,
        subset: Optional[int] = None,
        cache_path: Optional[str] = None,
        min_h1_persistence: float = 0.02,
        fill_radius: float = 2.5,
        dataset_name: str = 'mnist',  # 'mnist' or 'fashion'
    ):
        self.fill_radius = fill_radius
        self.min_persistence = min_h1_persistence
        self.dataset_name = dataset_name

        tfm = transforms.Compose([transforms.ToTensor()])
        if dataset_name == 'mnist':
            mnist = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
        elif dataset_name == 'fashion':
            mnist = datasets.FashionMNIST(root=root, train=train, download=True, transform=tfm)
        else:
            raise ValueError(f'unknown dataset_name: {dataset_name}')

        if isinstance(digit_filter, int):
            digits = [digit_filter]
        elif isinstance(digit_filter, (list, tuple)):
            digits = list(digit_filter)
        else:
            digits = None

        if digits is not None:
            indices = [i for i in range(len(mnist)) if mnist.targets[i].item() in digits]
        else:
            indices = list(range(len(mnist)))

        if subset is not None:
            indices = indices[:subset]

        self.indices = indices
        self.mnist = mnist
        self.digits = digits

        split = 'train' if train else 'test'
        digit_tag = ('_d' + '-'.join(str(d) for d in digits)) if digits else '_all'
        sub_tag = f'_sub{subset}' if subset is not None else ''
        if cache_path is None:
            cache_path = os.path.join(
                root, f'pflow_ds_eventsv{self.CACHE_VERSION}_{dataset_name}_{split}{digit_tag}{sub_tag}.pkl'
            )
        self.cache_path = cache_path
        self._build_cache()

    def _build_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.events = pickle.load(f)
            return
        print(f'Building persistence events for {len(self.indices)} images -> {self.cache_path}')
        self.events = []
        for mnist_idx in tqdm(self.indices):
            img, _ = self.mnist[mnist_idx]
            arr = img.numpy()[0]
            inverted = 1.0 - arr
            ev = compute_persistence_with_locations(
                inverted, dimensions=(0, 1), min_persistence=0.0
            )
            self.events.append(ev)
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.events, f)

    def __len__(self):
        return len(self.indices)

    def get_schedule(self, idx):
        return schedule_from_events(self.events[idx], min_persistence=self.min_persistence)

    def get_pd_vec(self, idx):
        return pd_to_vec(self.events[idx])

    def get_landscape(self, idx, L: int = 4, S: int = 8):
        """64-dim persistence-landscape descriptor for the stronger baseline."""
        return landscape_descriptor(self.events[idx], L=L, S=S)

    def __getitem__(self, idx):
        mnist_idx = self.indices[idx]
        img, label = self.mnist[mnist_idx]
        arr = img.numpy()[0]
        schedule = self.get_schedule(idx)
        t = float(np.random.uniform(0.0, 1.0))
        x_t = persistence_melt(arr, t, schedule=schedule, fill_radius=self.fill_radius)
        x_T = persistence_melt(arr, 1.0, schedule=schedule, fill_radius=self.fill_radius)
        pd_vec = self.get_pd_vec(idx)
        landscape = self.get_landscape(idx)
        return {
            'target': torch.from_numpy(arr).float().unsqueeze(0),
            'x_t': torch.from_numpy(x_t).float().unsqueeze(0),
            'x_T': torch.from_numpy(x_T).float().unsqueeze(0),
            't': torch.tensor(t, dtype=torch.float32),
            'label': int(label),
            'pd_vec': torch.from_numpy(pd_vec).float(),
            'landscape': torch.from_numpy(landscape).float(),
        }


def make_x_T(img: np.ndarray, schedule, fill_radius: float = 2.5) -> np.ndarray:
    return persistence_melt(img, 1.0, schedule=schedule, fill_radius=fill_radius)
