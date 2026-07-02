"""Differentiable p-k eigen-solver head.

Solves the flutter equation with the *learned* aero operator on a velocity
grid, fully in-graph:

    [ p^2 Ms + Ks - (kV)^2 Ghat(k) ] x = 0,     k = |Im p| / V   (b = 1)

The 2x2 eigenvalues are closed-form (trace/determinant), so gradients flow
through the entire spectral readout without torch.linalg.eig — no CUDA/ROCm
kernel dependency at all. The per-branch reduced frequency is found by a
short unrolled fixed-point iteration (backprop through the unroll).

Branch convention: branches are sorted by oscillation frequency |Im p| at
every airspeed (branch 0 = lower). Targets are sorted the same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .coupling import EigenCouplingAttention

_K_MIN, _K_MAX = 1e-4, 20.0


def structural_matrices(params: torch.Tensor):
    """Nondimensional Ms, Ks (B, 2, 2) from [mu, sigma, x_theta, a, r2, mach]."""
    mu, sigma, x_t, _a, r2, _m = params.unbind(-1)
    m = mu * torch.pi
    z = torch.zeros_like(m)
    Ms = torch.stack(
        [torch.stack([m, m * x_t], -1), torch.stack([m * x_t, m * r2], -1)], -2
    )
    Ks = torch.stack(
        [torch.stack([m * sigma**2, z], -1), torch.stack([z, m * r2], -1)], -2
    )
    return Ms, Ks


def eig2x2(A: torch.Tensor) -> torch.Tensor:
    """Closed-form eigenvalues of a batched complex 2x2 matrix -> (..., 2)."""
    tr = A[..., 0, 0] + A[..., 1, 1]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    disc = torch.sqrt(tr * tr / 4.0 - det)
    return torch.stack([tr / 2.0 - disc, tr / 2.0 + disc], dim=-1)


def _mode_p(lam: torch.Tensor) -> torch.Tensor:
    """p = sqrt(lambda) mapped to the Im(p) >= 0 physical branch."""
    p = torch.sqrt(lam)
    return torch.where(p.imag < 0, -p, p)


class DifferentiablePK(nn.Module):
    def __init__(self, n_iters: int = 6):
        super().__init__()
        self.n_iters = n_iters

    def forward(self, bmaps: torch.Tensor, Ms: torch.Tensor, Ks: torch.Tensor,
                V: torch.Tensor) -> torch.Tensor:
        """Run the learned-operator p-k sweep.

        bmaps : (B, F, 2, 2) complex   precomputed coupling maps
        Ms/Ks : (B, 2, 2) real         structural matrices
        V     : (nV,)                  velocity grid
        returns p: (B, nV, 2) complex, branches sorted by |Im p|.
        """
        B, nV = bmaps.shape[0], V.shape[0]
        cdtype = bmaps.dtype
        Msc = Ms.to(cdtype)[:, None, None]                    # (B,1,1,2,2)
        Ksc = Ks.to(cdtype)[:, None, None]
        Minv = torch.linalg.inv(Msc)
        Vb = V.view(1, nV, 1).expand(B, nV, 2)                # (B,nV,2)

        # wind-off natural frequencies for the k initialization
        w2 = eig2x2(torch.linalg.solve(Ms.to(cdtype), Ks.to(cdtype))).real
        w0 = torch.sqrt(w2.clamp_min(1e-12)).sort(dim=-1).values  # (B,2)
        k = (w0[:, None, :] / Vb).clamp(_K_MIN, _K_MAX)       # (B,nV,2)

        for _ in range(self.n_iters):
            Ghat = EigenCouplingAttention.evaluate(bmaps, k)  # (B,nV,2,2,2)
            w = (k * Vb).to(cdtype)                           # w = kV
            G = (w * w)[..., None, None] * Ghat
            A = Minv @ (G - Ksc)                              # p^2 = eig(A)
            lam = eig2x2(A)                                   # (B,nV,2branch_k,2eig)
            p_all = _mode_p(lam)
            # branch br keeps the eigenvalue matching its frequency ordering
            p_sorted = p_all.imag.abs().argsort(dim=-1)
            p_all = torch.gather(p_all, -1, p_sorted)
            idx = torch.arange(2, device=k.device).view(1, 1, 2)
            p = torch.gather(p_all, -1, idx.expand(B, nV, 2)[..., None]).squeeze(-1)
            k = (p.imag.abs() / Vb).clamp(_K_MIN, _K_MAX)

        # final consistent ordering by frequency
        order = p.imag.abs().argsort(dim=-1)
        return torch.gather(p, -1, order)
