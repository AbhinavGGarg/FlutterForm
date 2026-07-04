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
                    w_gamma: float = 1.0, w_omega: float = 1.0,
                    crossing_upweight: float = 4.0):
    """Weighted Huber loss on frequency-sorted V-g / V-f branch trajectories.

    Grid points whose ground-truth damping is near zero (where the flutter
    crossing lives, and where the V-g curve is shallow so small gamma errors
    move V_F a lot) are upweighted, so the model spends capacity where the
    answer is decided.
    """
    order = omega.abs().argsort(dim=-1)
    gamma = torch.gather(gamma, -1, order)
    omega = torch.gather(omega, -1, order)

    # per-point weight: 1 + upweight * exp(-(gamma/0.05)^2)
    near = torch.exp(-(gamma / 0.05) ** 2)
    w = 1.0 + crossing_upweight * near
    hg = nn.functional.huber_loss(gamma_hat, gamma, delta=0.05, reduction="none")
    ho = nn.functional.huber_loss(omega_hat, omega, delta=0.1, reduction="none")
    denom = w.sum() + 1e-9
    return (w_gamma * (w * hg).sum() + w_omega * (w * ho).sum()) / denom


def flutter_point_loss(gamma_hat, omega_hat, V, flutter_V, flutter_omega):
    """Residual-at-the-known-crossing penalty.

    At the true flutter speed V_F the coalescing branch must have zero damping
    and the true flutter frequency. We interpolate the model's V-g / V-f
    trajectories at V_F, pick the branch whose frequency is closest to the
    true omega_F (that IS the coalescing branch), and penalize:
      - its damping (should be exactly 0 at V_F), and
      - its frequency error (should equal omega_F).
    This gives a strong local gradient exactly where the flutter answer is
    decided, without a fragile soft-argmin over the whole grid.
    """
    from ..flutter_point import interp_at

    valid = torch.isfinite(flutter_V) & (flutter_V > 0)
    if valid.sum() == 0:
        return gamma_hat.new_zeros(())

    gam_vf = interp_at(gamma_hat, V, flutter_V)          # (B, 2)
    om_vf = interp_at(omega_hat.abs(), V, flutter_V)     # (B, 2)

    # coalescing branch = the one whose frequency matches omega_F (index only,
    # no grad through the selection)
    with torch.no_grad():
        b = (om_vf - flutter_omega.unsqueeze(-1)).abs().argmin(dim=-1)
    ar = torch.arange(gam_vf.shape[0], device=gam_vf.device)
    gam_sel = gam_vf[ar, b]                              # (B,)
    om_sel = om_vf[ar, b]

    gv, os_, ov = gam_sel[valid], om_sel[valid], flutter_omega[valid]
    l_damp = (gv ** 2).mean()
    l_freq = (((os_ - ov) / ov.clamp_min(0.05)) ** 2).mean()
    return l_damp + 0.5 * l_freq
