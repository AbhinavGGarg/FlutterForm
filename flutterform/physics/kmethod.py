"""Classical k-method (V-g) flutter solution — independent cross-check for p-k.

Assume pure harmonic motion with artificial structural damping g:

    [ -w^2 Ms + (1 + i g) Ks - G(k, V) ] x = 0,        V = w b / k.

Because every term of G(k, V) scales as w^2 once V = w b / k is substituted
(G = w^2 Ghat(k), see Section.aero_matrix_khat), this becomes a generalized
eigenproblem at each k:

    eig( Ks^{-1} (Ms + Ghat(k)) ) = Lambda = (1 + i g) / w^2

so  w = 1/sqrt(Re Lambda),  g = Im Lambda / Re Lambda,  V = w b / k.

Flutter is where a branch's g(V) crosses zero. At that point the harmonic
assumption is exact, so the k-method and p-k must agree — which is exactly
the property the test suite uses to validate both implementations.
"""

from __future__ import annotations

import numpy as np

from .section import Section


def kmethod_sweep(sec: Section, k_grid=None):
    """Sweep reduced frequency; return per-branch (V, w, g) curves.

    Branches are tracked by eigenvector MAC continuity along the k sweep.
    Returns dict with arrays of shape (2, nk): V, omega, g.
    """
    if k_grid is None:
        # high k = low airspeed; sweep down toward low k (high V)
        k_grid = np.geomspace(4.0, 5e-3, 400)
    k_grid = np.asarray(k_grid, dtype=float)

    Ms, Ks = sec.mass_matrix(), sec.stiffness_matrix()
    Kinv = np.linalg.inv(Ks)

    nk = k_grid.size
    V = np.zeros((2, nk))
    w = np.zeros((2, nk))
    g = np.zeros((2, nk))
    prev_vecs = None

    for i, k in enumerate(k_grid):
        lam, vecs = np.linalg.eig(Kinv @ (Ms + sec.aero_matrix_khat(k)))
        if prev_vecs is None:
            order = np.argsort(-lam.real)  # big Re(Lambda) = low frequency first
        else:
            # MAC pairing with previous step
            mac = np.abs(prev_vecs.conj().T @ vecs)
            order = np.array([np.argmax(mac[0]), np.argmax(mac[1])])
            if order[0] == order[1]:  # degenerate pairing, fall back
                order = np.argsort(-lam.real)
        lam, vecs = lam[order], vecs[:, order]
        prev_vecs = vecs

        for br in range(2):
            re = lam[br].real
            if re <= 0:
                V[br, i] = w[br, i] = g[br, i] = np.nan
                continue
            wi = 1.0 / np.sqrt(re)
            w[br, i] = wi
            g[br, i] = lam[br].imag / re
            V[br, i] = wi / k  # b = 1

    return {"k": k_grid, "V": V, "omega": w, "g": g}


def kmethod_flutter(sec: Section, k_grid=None):
    """Locate the lowest-V zero crossing of g (from below) across branches.

    Returns (V_F, omega_F, branch) or (None, None, None) if no crossing.
    """
    sw = kmethod_sweep(sec, k_grid)
    best = None
    for br in range(2):
        Vb, wb, gb = sw["V"][br], sw["omega"][br], sw["g"][br]
        for i in range(1, Vb.size):
            if np.any(np.isnan([gb[i - 1], gb[i], Vb[i - 1], Vb[i]])):
                continue
            # k sweeps high->low so V runs low->high along i
            if gb[i - 1] <= 0.0 < gb[i] and Vb[i] > Vb[i - 1]:
                t = -gb[i - 1] / (gb[i] - gb[i - 1])
                Vf = Vb[i - 1] + t * (Vb[i] - Vb[i - 1])
                wf = wb[i - 1] + t * (wb[i] - wb[i - 1])
                if best is None or Vf < best[0]:
                    best = (float(Vf), float(wf), br)
                break
    return best if best is not None else (None, None, None)
