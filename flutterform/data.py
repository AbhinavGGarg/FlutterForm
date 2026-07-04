"""Dataset wrapper over generate_pk.py shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TierADataset(Dataset):
    """Typical-section V-g/V-f trajectories with flutter labels.

    Subsamples the stored velocity grid to n_v points for training speed.
    By default keeps only configs with a flutter point inside the grid
    (the generator reports — never hides — how many were flagged out).
    """

    # param column indices for extrapolation holdouts
    PCOL = {"mu": 0, "sigma": 1, "x_theta": 2, "a": 3, "r2": 4, "mach": 5}

    def __init__(self, path: str | Path, n_v: int = 64,
                 flutter_only: bool = True, split: str = "all",
                 val_frac: float = 0.1, split_seed: int = 1234,
                 holdout_col: str | None = None, holdout_thresh: float = None):
        d = np.load(path)
        keep = d["has_flutter"] if flutter_only else np.ones(len(d["params"]), bool)

        if holdout_col is not None:
            # EXTRAPOLATION split: train below threshold, test at/above it.
            # No random leakage — a whole region of parameter space is unseen.
            col = d["params"][:, self.PCOL[holdout_col]]
            below = col < holdout_thresh
            region = below if split == "train" else ~below
            keep = keep & region
        elif split != "all":
            # deterministic random train/val split over the KEPT configs
            kept_idx = np.where(keep)[0]
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(len(kept_idx))
            n_val = int(round(val_frac * len(kept_idx)))
            sel = perm[n_val:] if split == "train" else perm[:n_val]
            mask = np.zeros(len(d["params"]), bool)
            mask[kept_idx[sel]] = True
            keep = mask

        idx = np.linspace(0, d["V_grid"].size - 1, n_v).round().astype(int)
        self.V = torch.tensor(d["V_grid"][idx], dtype=torch.float32)
        self.params = torch.tensor(d["params"][keep], dtype=torch.float32)
        self.gamma = torch.tensor(d["gamma"][keep][:, :, idx], dtype=torch.float32)
        self.omega = torch.tensor(d["omega"][keep][:, :, idx], dtype=torch.float32)
        self.flutter_V = torch.tensor(d["flutter_V"][keep], dtype=torch.float32)
        self.flutter_omega = torch.tensor(d["flutter_omega"][keep], dtype=torch.float32)
        self.flutter_branch = torch.tensor(d["flutter_branch"][keep], dtype=torch.long)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, i):
        # trajectories transposed to (nV, 2) to match model output layout
        return dict(
            params=self.params[i],
            gamma=self.gamma[i].T.contiguous(),
            omega=self.omega[i].T.contiguous(),
            flutter_V=self.flutter_V[i],
            flutter_omega=self.flutter_omega[i],
            flutter_branch=self.flutter_branch[i],
        )
