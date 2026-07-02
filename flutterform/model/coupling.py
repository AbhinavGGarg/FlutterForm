"""Eigen-structured coupling attention.

The signature primitive: a pairwise outer-product between mode tokens is the
generalized aerodynamic coupling operator. For tokens t_i, t_j and reduced
frequency k the model predicts the frequency-factored aero matrix

    Ghat_ij(k) = sum_f  phi_f(k) * ( t_i^T W_f t_j )        (complex)

so the full aero matrix follows the exact physical scaling

    G(k, V) = w^2 * Ghat(k),   w = k V / b            (see kmethod.py)

i.e. the network learns only the k-dependent coupling structure; the airspeed
scaling is hard-wired. Note t_i^T W t_j != t_j^T W t_i: the learned operator
is non-symmetric, as aerodynamic coupling physically is.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_BASIS = 8
_LOGK_LO, _LOGK_HI = math.log10(1e-4), math.log10(20.0)


_K_REF = 0.3  # reference reduced frequency for the circulatory envelope


def k_basis(k: torch.Tensor) -> torch.Tensor:
    """Physics-informed basis in k. k: (...,) -> (..., N_BASIS).

    Theodorsen's circulatory terms scale as C(k)/k and C(k)/k^2 once the
    aero matrix is frequency-factored (see Section.aero_matrix_khat), so the
    basis spans that envelope explicitly — s1 ~ 1/k, s2 ~ 1/k^2 — modulated
    by smooth log-k features. The network then only needs O(1) coefficients:
    the geometric prior does the dynamic-range work.
    """
    k = k.clamp(1e-4, 20.0)
    x = (torch.log10(k) - _LOGK_LO) / (_LOGK_HI - _LOGK_LO)  # [0, 1]
    x = 2.0 * x - 1.0
    s1 = _K_REF / k
    s2 = s1 * s1
    return torch.stack(
        [
            torch.ones_like(x), x,
            torch.sin(math.pi * x), torch.cos(math.pi * x),
            s1, s1 * x,
            s2, s2 * x,
        ],
        dim=-1,
    )


class EigenCouplingAttention(nn.Module):
    """Outer-product coupling: tokens -> complex Ghat(k) operator."""

    def __init__(self, d_model: int = 12, n_basis: int = N_BASIS):
        super().__init__()
        # W[f, :, :, 0] real part, W[f, :, :, 1] imaginary part
        self.W = nn.Parameter(0.02 * torch.randn(n_basis, d_model, d_model, 2))

    def bilinear_maps(self, tokens: torch.Tensor) -> torch.Tensor:
        """Precompute B[b, f, i, j] = t_i^T W_f t_j (complex), k-independent.

        tokens: (B, 2, d) -> (B, n_basis, 2, 2) complex
        """
        maps = torch.einsum("bid,fdec,bje->bfijc", tokens, self.W, tokens)
        return torch.complex(maps[..., 0], maps[..., 1])

    @staticmethod
    def evaluate(bmaps: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Contract precomputed bilinear maps with the k-basis.

        bmaps: (B, n_basis, 2, 2) complex;  k: (B, *S) arbitrary trailing shape
        returns Ghat: (B, *S, 2, 2) complex
        """
        phi = k_basis(k).to(bmaps.dtype)                      # (B, *S, F)
        return torch.einsum("b...f,bfij->b...ij", phi, bmaps)
