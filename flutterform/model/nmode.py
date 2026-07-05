"""N-mode differentiable p-k flutter head (Tier-B).

Extends FlutterForm's eigen-structured readout from the 2-DOF typical section
to an arbitrary N-mode wing. The coupling attention already yields an N x N
aerodynamic operator by construction (outer products over N mode tokens); the
only change is the spectral solve, which here uses batched `torch.linalg.eig`
for general N x N (the closed-form 2x2 path is kept for Tier-A / ROCm).

Given generalized mass/stiffness M, K (real, N x N, known analytically) and an
aerodynamic operator Q(k) (learned or analytic), the flutter equation

    [ p^2 M + K - (kV)^2 Qhat(k) ] q = 0,   k = |Im p| b / V

is marched over airspeed by an unrolled p-k fixed point, fully differentiable.

`differentiable_pk_flutter` below is validated against the (numpy) Tier-B p-k
solver on the Goland wing in tests/test_tierb.py::TestNMode.
"""

from __future__ import annotations

import torch


def _mode_p(lam: torch.Tensor) -> torch.Tensor:
    p = torch.sqrt(lam)
    return torch.where(p.imag < 0, -p, p)


def differentiable_pk_flutter(M, K, Q_of_k, V, b, n_iter: int = 30):
    """Differentiable N-mode p-k sweep for ONE wing.

    M, K : (N, N) real tensors (known structure)
    Q_of_k : callable k(scalar tensor) -> (N, N) complex  frequency-factored AIC
             such that the aero force is (kV)^2 Q_of_k(k)
    V : (nV,) airspeed grid ; b : semichord
    Returns damping gamma (nV, N) and frequency omega (nV, N), branch-sorted by
    frequency. Flutter = first sustained zero-up-crossing of gamma (extract with
    flutter_point.hard_flutter_speed).
    """
    N = M.shape[0]
    cdtype = torch.complex64 if M.dtype == torch.float32 else torch.complex128
    Mc = M.to(cdtype)
    Kc = K.to(cdtype)
    Minv = torch.linalg.inv(Mc)
    nV = V.shape[0]

    w2 = torch.linalg.eigvals(torch.linalg.solve(Mc, Kc)).real
    w0 = torch.sqrt(w2.clamp_min(1e-9)).sort().values          # (N,)

    gam = torch.zeros(nV, N)
    omg = torch.zeros(nV, N)
    for iv in range(nV):
        Vi = V[iv]
        k = (w0 * b / Vi).clamp(1e-4, 50.0)                    # (N,)
        for _ in range(n_iter):
            Qs = torch.stack([Q_of_k(k[br]) for br in range(N)], 0)  # (N,N,N)
            w = (k * Vi / b)                                   # (N,)
            A = torch.stack([
                Minv @ ((w[br] ** 2) * Qs[br] - Kc) for br in range(N)
            ], 0)                                              # (N,N,N)
            lam = torch.linalg.eigvals(A)                      # (N, N)
            ps = _mode_p(lam)
            # branch br keeps the eigenvalue nearest its current freq
            idx = (ps.imag.abs() - (k * Vi / b).unsqueeze(1)).abs().argmin(1)
            p = ps[torch.arange(N), idx]
            k = (p.imag.abs() * b / Vi).clamp(1e-4, 50.0)
        order = p.imag.abs().argsort()
        p = p[order]
        gam[iv] = (p.real / (p.abs() + 1e-30))
        omg[iv] = p.imag.abs()
    return gam, omg
