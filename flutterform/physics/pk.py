"""p-k flutter solution for the typical section.

At a fixed airspeed V the p-k method solves

    [ p^2 Ms + Ks - G(k, V) ] x = 0,      k = Im(p) b / V,

where G is the harmonic (Theodorsen) aero matrix evaluated at the reduced
frequency of the mode's own oscillatory part. Each structural mode is tracked
across a V sweep by eigenvector continuity (MAC), giving V-g / V-f branch
trajectories; flutter is the first oscillatory branch whose damping crosses
zero, refined by bisection.

All quantities nondimensional (b = 1, omega_theta = 1, rho = 1): V is the
reduced velocity U* = V/(b omega_theta), frequencies are omega/omega_theta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .section import Section

_K_FLOOR = 1e-6        # keep the aero evaluation away from the k=0 singularity
_OSC_FREQ_MIN = 5e-3   # below this |Im p| a root is treated as non-oscillatory


def _principal_mode_p(lam: complex) -> complex:
    """Map an eigenvalue lam = p^2 to the physical root p with Im(p) >= 0."""
    p = np.sqrt(lam)  # principal: Re >= 0
    if p.imag < 0.0:
        p = -p
    return p


def _solve_at_v(sec: Section, V: float, k_guesses, ref_vecs=None,
                max_iter: int = 60, tol: float = 1e-10):
    """Solve both p-k branches at one airspeed.

    k_guesses : per-branch starting reduced frequencies (continuation).
    ref_vecs  : optional (2,2) previous eigenvectors, columns = branches, used
                to keep branch identity via MAC pairing.

    Returns p (2,), vecs (2,2 columns), k (2,), converged (2,) bool.
    """
    Ms = sec.mass_matrix()
    Ks = sec.stiffness_matrix()
    Minv = np.linalg.inv(Ms)

    p_out = np.zeros(2, dtype=complex)
    v_out = np.zeros((2, 2), dtype=complex)
    k_out = np.zeros(2)
    ok = np.zeros(2, dtype=bool)

    for br in range(2):
        k = max(float(k_guesses[br]), _K_FLOOR)
        p_prev = None
        for it in range(max_iter):
            A = Minv @ (sec.aero_matrix(k, V) - Ks)
            lam, vecs = np.linalg.eig(A)
            ps = np.array([_principal_mode_p(l) for l in lam])

            # pick the eigenpair belonging to THIS branch
            if ref_vecs is not None:
                ref = ref_vecs[:, br]
                mac = np.abs(ref.conj() @ vecs) / (
                    np.linalg.norm(ref) * np.linalg.norm(vecs, axis=0) + 1e-300
                )
                j = int(np.argmax(mac))
            elif p_prev is not None:
                j = int(np.argmin(np.abs(ps - p_prev)))
            else:
                # first ever solve: order branches by frequency (low = plunge-ish)
                j = int(np.argsort(np.abs(ps.imag))[br])

            p = ps[j]
            vec = vecs[:, j]
            k_new = max(abs(p.imag) / V, _K_FLOOR)

            if p_prev is not None and abs(k_new - k) <= tol * max(k, 1.0):
                p_prev, k = p, k_new
                ok[br] = True
                break
            # relaxed fixed-point update on k
            k = 0.5 * k + 0.5 * k_new
            p_prev = p
        p_out[br], v_out[:, br], k_out[br] = p_prev, vec, k
        if not ok[br]:
            # converged "enough" is normal near k floor; flag only if wild
            ok[br] = abs(k_out[br] - max(abs(p_prev.imag) / V, _K_FLOOR)) < 1e-3
    return p_out, v_out, k_out, ok


@dataclass
class PKResult:
    """Branch-tracked p-k sweep for one section."""

    V: np.ndarray                    # (nV,) reduced velocities
    p: np.ndarray                    # (2, nV) complex eigenvalues per branch
    flutter_V: float | None = None   # first oscillatory zero-damping crossing
    flutter_omega: float | None = None
    flutter_branch: int | None = None
    flutter_k: float | None = None
    divergence_V: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def damping(self) -> np.ndarray:  # gamma = Re p / |p|
        return self.p.real / (np.abs(self.p) + 1e-300)

    @property
    def frequency(self) -> np.ndarray:
        return self.p.imag


def pk_sweep(sec: Section, V_grid=None) -> PKResult:
    """Track both branches over a velocity sweep and locate flutter."""
    if V_grid is None:
        V_grid = np.linspace(0.05, 6.0, 240)
    V_grid = np.asarray(V_grid, dtype=float)

    n = V_grid.size
    p_hist = np.zeros((2, n), dtype=complex)

    # start from the wind-off natural frequencies
    Ms, Ks = sec.mass_matrix(), sec.stiffness_matrix()
    w2 = np.linalg.eigvals(np.linalg.solve(Ms, Ks)).real
    w0 = np.sqrt(np.sort(w2))
    k_g = w0 / V_grid[0]
    vecs = None

    for i, V in enumerate(V_grid):
        p, vecs, k_g, _ = _solve_at_v(sec, V, k_g, ref_vecs=vecs)
        p_hist[:, i] = p

    res = PKResult(V=V_grid, p=p_hist)

    # ---- flutter: first upward zero-crossing of damping on an oscillatory branch
    gam = p_hist.real / (np.abs(p_hist) + 1e-300)
    osc = np.abs(p_hist.imag) > _OSC_FREQ_MIN
    best = None
    for br in range(2):
        for i in range(1, n):
            if not (osc[br, i - 1] and osc[br, i]):
                continue
            if gam[br, i - 1] <= 0.0 < gam[br, i]:
                lo, hi = V_grid[i - 1], V_grid[i]
                Vf, pf = _bisect_crossing(sec, lo, hi, br, p_hist[:, i - 1],
                                          V_lo_vecs=None)
                if best is None or Vf < best[0]:
                    best = (Vf, pf.imag, br, abs(pf.imag) / Vf)
                break
    if best is not None:
        res.flutter_V, res.flutter_omega, res.flutter_branch, res.flutter_k = best

    # ---- static divergence: non-oscillatory branch going unstable
    div = (~osc) & (p_hist.real > 1e-8)
    if div.any():
        res.divergence_V = float(V_grid[np.argmax(div.any(axis=0))])

    return res


def _bisect_crossing(sec: Section, V_lo: float, V_hi: float, branch: int,
                     p_bracket, V_lo_vecs=None, iters: int = 48):
    """Refine the damping zero-crossing of one branch by bisection.

    Re-solves the p-k problem at each midpoint, tracking the branch by
    continuation from the bracketing solution.
    """
    # continuation state at V_lo
    k_g = np.maximum(np.abs(p_bracket.imag) / max(V_lo, 1e-9), _K_FLOOR)
    p_lo, vecs, k_lo, _ = _solve_at_v(sec, V_lo, k_g)
    g_lo = p_lo[branch].real / (abs(p_lo[branch]) + 1e-300)

    lo, hi = V_lo, V_hi
    p_mid = p_lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_mid, vecs_m, k_m, _ = _solve_at_v(sec, mid, k_lo, ref_vecs=vecs)
        g_mid = p_mid[branch].real / (abs(p_mid[branch]) + 1e-300)
        if g_mid > 0.0:
            hi = mid
        else:
            lo, vecs, k_lo, g_lo = mid, vecs_m, k_m, g_mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi), p_mid[branch]
