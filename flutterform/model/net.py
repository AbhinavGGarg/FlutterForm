"""FlutterForm: tokens -> learned coupling operator -> spectral readout."""

from __future__ import annotations

import torch
import torch.nn as nn

from .coupling import EigenCouplingAttention
from .eigenhead import DifferentiablePK, structural_matrices
from .tokenizer import ModalTokenizer


class FlutterForm(nn.Module):
    def __init__(self, d_model: int = 12, n_iters: int = 6):
        super().__init__()
        self.tokenizer = ModalTokenizer(d_model)
        self.coupling = EigenCouplingAttention(d_model)
        self.head = DifferentiablePK(n_iters)

    def forward(self, params: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """params (B, 6), V (nV,) -> complex p (B, nV, 2), freq-sorted branches."""
        tokens = self.tokenizer(params)
        bmaps = self.coupling.bilinear_maps(tokens)
        Ms, Ks = structural_matrices(params)
        return self.head(bmaps, Ms, Ks, V)

    def vg_vf(self, params: torch.Tensor, V: torch.Tensor):
        """Damping gamma = Re p/|p| and frequency omega = Im p, (B, nV, 2)."""
        p = self.forward(params, V)
        gamma = p.real / (p.abs() + 1e-30)
        return gamma, p.imag

    def operator(self, params: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Learned frequency-factored aero operator Ghat(k) — for the
        operator-consistency eval against the analytic Theodorsen matrix."""
        bmaps = self.coupling.bilinear_maps(self.tokenizer(params))
        return EigenCouplingAttention.evaluate(bmaps, k)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def trajectory_loss(gamma_hat, omega_hat, gamma, omega,
                    w_gamma: float = 1.0, w_omega: float = 1.0):
    """Huber loss on frequency-sorted V-g / V-f branch trajectories."""
    order = omega.abs().argsort(dim=-1)
    gamma = torch.gather(gamma, -1, order)
    omega = torch.gather(omega, -1, order)
    huber = nn.functional.huber_loss
    return (
        w_gamma * huber(gamma_hat, gamma, delta=0.05)
        + w_omega * huber(omega_hat, omega, delta=0.1)
    )
