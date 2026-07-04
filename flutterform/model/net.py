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


def flutter_point_loss(gamma_hat, omega_hat, V, flutter_V, flutter_omega,
                       margin: float = 0.03, delta_frac: float = 0.12):
    """Sharp-up-crossing penalty at the known flutter speed.

    At the true V_F the coalescing branch must (a) have zero damping, and
    (b) cross zero with real positive slope — clearly negative just below V_F,
    clearly positive just above. Penalizing only (a) has a degenerate solution:
    flatten the whole damping curve to ~0, which minimizes the residual but
    destroys the crossing (observed: val V_F blew up while trajectory loss
    fell). The margin terms forbid that flat solution.

    Branch selection (which mode coalesces) is by frequency match to omega_F;
    index only, no gradient through the argmin.
    """
    from ..flutter_point import interp_at

    valid = torch.isfinite(flutter_V) & (flutter_V > 0)
    if valid.sum() == 0:
        return gamma_hat.new_zeros(())

    grid_step = (V[1] - V[0])
    dV = (delta_frac * flutter_V).clamp_min(grid_step)
    g_lo = interp_at(gamma_hat, V, (flutter_V - dV).clamp_min(V[0]))   # (B,2)
    g_0 = interp_at(gamma_hat, V, flutter_V)
    g_hi = interp_at(gamma_hat, V, (flutter_V + dV).clamp_max(V[-1]))
    om_0 = interp_at(omega_hat.abs(), V, flutter_V)

    with torch.no_grad():
        b = (om_0 - flutter_omega.unsqueeze(-1)).abs().argmin(dim=-1)
    ar = torch.arange(gamma_hat.shape[0], device=gamma_hat.device)
    gl, g0, gh = g_lo[ar, b], g_0[ar, b], g_hi[ar, b]
    om_sel = om_0[ar, b]

    v = valid
    l_zero = (g0[v] ** 2).mean()
    l_below = torch.relu(margin + gl[v]).mean()   # want damping < -margin below
    l_above = torch.relu(margin - gh[v]).mean()   # want damping > +margin above
    ov = flutter_omega[v]
    l_freq = (((om_sel[v] - ov) / ov.clamp_min(0.05)) ** 2).mean()
    return l_zero + l_below + l_above + 0.5 * l_freq
