"""Flutter-point extraction from V-g / V-f trajectories.

Shared by training and eval so both define "the flutter speed" identically:
the lowest airspeed where an oscillatory branch's damping gamma(V) crosses
zero from below.

- `hard_flutter_speed`  : exact piecewise-linear crossing, for eval/reporting.
- `interp_at`           : differentiable linear interpolation of the branch
                          trajectories at a per-config airspeed, for the loss
                          (supervise gamma -> 0 at the known true V_F).
"""

from __future__ import annotations

import numpy as np
import torch

_OSC_MIN = 5e-3


def hard_flutter_speed(gamma, omega, V):
    """Exact first zero-up-crossing per config. (B,nV,2) -> V_F, omega_F (B,)."""
    g = np.asarray(gamma)
    o = np.abs(np.asarray(omega))
    Vn = np.asarray(V)
    B = g.shape[0]
    vf = np.full(B, np.nan)
    wf = np.full(B, np.nan)
    for b in range(B):
        best = None
        for s in range(g.shape[2]):
            gs, ws = g[b, :, s], o[b, :, s]
            up = ((gs[:-1] <= 0) & (gs[1:] > 0)
                  & (ws[:-1] > _OSC_MIN) & (ws[1:] > _OSC_MIN))
            if up.any():
                j = int(np.argmax(up))
                t = -gs[j] / (gs[j + 1] - gs[j] + 1e-30)
                v = Vn[j] + t * (Vn[j + 1] - Vn[j])
                w = ws[j] + t * (ws[j + 1] - ws[j])
                if best is None or v < best[0]:
                    best = (v, w)
        if best is not None:
            vf[b], wf[b] = best
    return vf, wf


def interp_at(traj: torch.Tensor, V: torch.Tensor, v_query: torch.Tensor):
    """Linear-interp branch trajectories at per-config airspeeds.

    traj: (B, nV, 2) ; V: (nV,) ascending ; v_query: (B,) -> (B, 2).
    Differentiable w.r.t. traj (V and v_query treated as constants).
    """
    vq = v_query.clamp(V[0], V[-1])
    j = torch.searchsorted(V, vq, right=True).clamp(1, V.numel() - 1) - 1  # (B,)
    t = ((vq - V[j]) / (V[j + 1] - V[j])).unsqueeze(-1)                    # (B,1)
    ar = torch.arange(traj.shape[0], device=traj.device)
    lo = traj[ar, j]        # (B, 2)
    hi = traj[ar, j + 1]    # (B, 2)
    return lo + t * (hi - lo)
