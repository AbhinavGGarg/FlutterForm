"""Modal tokenizer: one token per structural mode of the typical section.

Each mode is embedded from its own structural quantities (natural frequency,
generalized mass/inertia, coupling, geometry, flow). The aerodynamic coupling
between tokens is *not* an input — recovering it is the model's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_MODE_FEATURES = 8


def mode_features(params: torch.Tensor) -> torch.Tensor:
    """Build per-mode feature vectors from section parameters.

    params: (B, 6) columns [mu, sigma, x_theta, a, r2, mach]
    returns (B, 2, N_MODE_FEATURES); mode 0 = plunge, mode 1 = pitch.
    """
    mu, sigma, x_t, a, r2, mach = params.unbind(-1)
    one = torch.ones_like(mu)
    zero = torch.zeros_like(mu)
    inv_mu = 1.0 / mu

    plunge = torch.stack([one, zero, sigma, one, x_t, a, inv_mu, mach], dim=-1)
    pitch = torch.stack([zero, one, one, r2, x_t, a, inv_mu, mach], dim=-1)
    return torch.stack([plunge, pitch], dim=1)


class ModalTokenizer(nn.Module):
    def __init__(self, d_model: int = 12):
        super().__init__()
        self.proj = nn.Linear(N_MODE_FEATURES, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """(B, 6) section params -> (B, 2, d) mode tokens."""
        return self.norm(self.proj(mode_features(params)))
