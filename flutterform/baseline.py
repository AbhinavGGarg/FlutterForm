"""Capacity-matched black-box baseline.

An MLP that regresses the section parameters directly to (V_F, omega_F) —
the standard neural-surrogate approach, and the thing FlutterForm's
physics-structured glass-box must beat on data-efficiency, extrapolation,
and mechanism recovery. Matched to FlutterForm's ~2.4-6k parameter budget so
the comparison is about *inductive bias*, not capacity.

The baseline predicts only the scalar flutter point; it has no notion of
which modes coalesce, so it structurally cannot do mode identification —
that gap is itself a headline result.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, hidden: int = 48, depth: int = 3, in_dim: int = 6):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 2)]  # (log V_F, log omega_F)
        self.net = nn.Sequential(*layers)
        # standardization buffers (filled from training data)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))

    def set_norm(self, x_mean, x_std):
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std.clamp_min(1e-6))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        x = (params - self.x_mean) / self.x_std
        return self.net(x)  # (B, 2) = [log V_F, log omega_F]

    def predict(self, params: torch.Tensor):
        out = self.forward(params)
        return out[:, 0].exp(), out[:, 1].exp()  # V_F, omega_F

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
