"""Tier-B: 3-D cantilever wing flutter via assumed modes + strip-theory
Theodorsen aerodynamics + an N-mode p-k solve.

The wing is a uniform cantilever of semi-span L and chord c (semichord
b = c/2). We use the exact clamped-free beam mode shapes:
  - bending  phi_i(eta), eta = y/L in [0,1]  (roots lambda_i of cos.cosh = -1)
  - torsion  psi_j(eta) = sqrt(2) sin((2j-1) pi eta / 2)
mass-orthonormalized so int_0^1 phi_i phi_k = delta, int_0^1 psi_j psi_l = delta.

Generalized structure (real, N x N with N = n_bend + n_tors):
  M_bb = m L I,  M_tt = I_alpha L I,  M_bt = S L * <phi_i, psi_j>
  K_bb = diag(EI lambda_i^4 / L^3),  K_tt = diag(GJ ((2j-1)pi/2)^2 / L)
where S = m x_alpha b is the static unbalance per span (CG aft of the EA).

Generalized unsteady aero (strip theory): each spanwise strip carries the
2-D Theodorsen lift/moment of section.py (dimensional here). For a uniform
wing the sectional operator factors out of the spanwise integral, so the
generalized AIC is Q_gen(k) = kron-assembled from the 2-D G_sec(k) and the
mode-shape overlap integrals. Flutter is [p^2 M + K - Q_gen(k,V)] q = 0,
solved by an N-mode p-k iteration (the same method validated in Tier-A).

Validated against the classic Goland wing (see scripts/validate_tierb.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .physics.theodorsen import theodorsen

# clamped-free beam frequency-equation roots cos(l)cosh(l)+1 = 0
_BEND_LAMBda = None


def bending_lambdas(n):
    """First n roots of cos(l) cosh(l) = -1."""
    roots = []
    guess = 1.8
    k = 0
    x = 0.5
    while len(roots) < n:
        a, b = x, x + 0.5
        fa = np.cos(a) * np.cosh(a) + 1
        fb = np.cos(b) * np.cosh(b) + 1
        if fa * fb < 0:
            roots.append(brentq(lambda t: np.cos(t) * np.cosh(t) + 1, a, b))
        x += 0.5
    return np.array(roots[:n])


def _bend_shapes(lam, eta):
    """Mass-orthonormal clamped-free bending mode shapes phi_i(eta)."""
    phis = []
    for l in lam:
        s = (np.cosh(l) + np.cos(l)) / (np.sinh(l) + np.sin(l))
        phi = (np.cosh(l * eta) - np.cos(l * eta)
               - s * (np.sinh(l * eta) - np.sin(l * eta)))
        phi = phi / np.sqrt(np.trapezoid(phi * phi, eta))  # int phi^2 deta = 1
        phis.append(phi)
    return np.array(phis)


def _tors_shapes(n, eta):
    """Mass-orthonormal fixed-free torsion mode shapes psi_j(eta)."""
    psis = []
    for j in range(1, n + 1):
        psi = np.sqrt(2.0) * np.sin((2 * j - 1) * np.pi * eta / 2.0)
        psis.append(psi)
    return np.array(psis)


@dataclass
class Wing:
    """Uniform cantilever wing (SI units)."""

    L: float          # semi-span [m]
    c: float          # chord [m]
    m: float          # mass per span [kg/m]
    I_alpha: float    # polar mass inertia per span about EA [kg m]
    x_alpha: float    # static unbalance: CG aft of EA, in semichords
    a: float          # elastic axis aft of midchord, in semichords
    EI: float         # bending stiffness [N m^2]
    GJ: float         # torsion stiffness [N m^2]
    rho: float = 1.225
    n_bend: int = 3
    n_tors: int = 2
    n_eta: int = 200

    b: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "b", self.c / 2.0)

    # -- modal structure ---------------------------------------------------
    def _shapes(self):
        eta = np.linspace(0, 1, self.n_eta)
        lam = bending_lambdas(self.n_bend)
        phi = _bend_shapes(lam, eta)
        psi = _tors_shapes(self.n_tors, eta)
        return eta, lam, phi, psi

    def structural(self):
        eta, lam, phi, psi = self._shapes()
        nb, nt = self.n_bend, self.n_tors
        N = nb + nt
        S = self.m * self.x_alpha * self.b
        M = np.zeros((N, N))
        K = np.zeros((N, N))
        # bending block
        for i in range(nb):
            M[i, i] = self.m * self.L
            K[i, i] = self.EI * lam[i] ** 4 / self.L ** 3
        # torsion block
        for j in range(nt):
            M[nb + j, nb + j] = self.I_alpha * self.L
            K[nb + j, nb + j] = self.GJ * ((2 * j + 1) * np.pi / 2.0) ** 2 / self.L
        # bending-torsion mass coupling (static unbalance)
        for i in range(nb):
            for j in range(nt):
                ov = np.trapezoid(phi[i] * psi[j], eta) * self.L
                M[i, nb + j] = M[nb + j, i] = S * ov
        return M, K

    # -- strip-theory generalized aero ------------------------------------
    def _section_G(self, k, V):
        """Dimensional 2-D Theodorsen operator: [-L; M_ea] = G [h; alpha]."""
        b, rho, a = self.b, self.rho, self.a
        w = k * V / b
        C = theodorsen(k)
        iw = 1j * w
        pib2 = np.pi * rho * b ** 2
        circ = 2.0 * np.pi * rho * V * b * C
        f11 = pib2 * (-(w ** 2)) + circ * iw
        f12 = pib2 * (iw * V + b * a * w ** 2) + circ * (V + b * (0.5 - a) * iw)
        f21 = pib2 * (-b * a * w ** 2) + circ * b * (a + 0.5) * iw
        f22 = (pib2 * (-iw * V * b * (0.5 - a) + b ** 2 * (0.125 + a ** 2) * w ** 2)
               + circ * b * (a + 0.5) * (V + b * (0.5 - a) * iw))
        return np.array([[-f11, -f12], [f21, f22]], dtype=complex)

    def aero(self, k, V):
        """Generalized AIC Q_gen(k, V): (N, N) complex.

        Generalized force on mode m from mode n = int shape_m . G_sec . shape_n.
        h(eta)=sum phi_i q_bi, alpha(eta)=sum psi_j q_tj; lift acts through
        plunge (bending), moment through pitch (torsion).
        """
        eta, lam, phi, psi = self._shapes()
        nb, nt = self.n_bend, self.n_tors
        N = nb + nt
        G = self._section_G(k, V)
        # overlap integrals (times L)
        Ibb = np.array([[np.trapezoid(phi[i] * phi[k2], eta) for k2 in range(nb)]
                        for i in range(nb)]) * self.L
        Itt = np.array([[np.trapezoid(psi[j] * psi[l], eta) for l in range(nt)]
                        for j in range(nt)]) * self.L
        Ibt = np.array([[np.trapezoid(phi[i] * psi[j], eta) for j in range(nt)]
                        for i in range(nb)]) * self.L
        Q = np.zeros((N, N), dtype=complex)
        Q[:nb, :nb] = G[0, 0] * Ibb
        Q[:nb, nb:] = G[0, 1] * Ibt
        Q[nb:, :nb] = G[1, 0] * Ibt.T
        Q[nb:, nb:] = G[1, 1] * Itt
        return Q

    def aero_khat(self, k):
        """Frequency-factored generalized AIC Qhat(k): Q(k,V) = (kV/b)^2 Qhat(k).

        Every aero term scales as w^2 = (kV/b)^2 once V = w b / k, so Qhat is
        V-independent. Used by the differentiable N-mode head (nmode.py)."""
        V = 1.0
        w2 = (k * V / self.b) ** 2
        return self.aero(k, V) / w2


def _mode_p(lam):
    p = np.sqrt(lam.astype(complex))
    return np.where(p.imag < 0, -p, p)


@dataclass
class TierBResult:
    V: np.ndarray
    p: np.ndarray                 # (N, nV) complex, branch-tracked
    flutter_V: float | None = None
    flutter_omega: float | None = None
    flutter_mode: int | None = None


def pk_flutter(wing: Wing, V_grid, n_iter: int = 40):
    """N-mode p-k flutter sweep. Returns branch-tracked eigenvalues + flutter."""
    M, K = wing.structural()
    Minv = np.linalg.inv(M)
    N = M.shape[0]
    V_grid = np.asarray(V_grid, float)
    p_hist = np.zeros((N, V_grid.size), dtype=complex)

    # wind-off natural frequencies for k init
    w0 = np.sqrt(np.sort(np.abs(np.linalg.eigvals(Minv @ K).real)))
    prev_vecs = None
    for iv, V in enumerate(V_grid):
        k = np.clip(w0 / max(V / wing.b, 1e-6), 1e-4, 50)
        p = np.zeros(N, dtype=complex)
        for br in range(N):
            kk = k[br]
            for _ in range(n_iter):
                A = Minv @ (wing.aero(kk, V) - K)
                lam, vecs = np.linalg.eig(A)
                ps = _mode_p(lam)
                if prev_vecs is not None:
                    ref = prev_vecs[:, br]
                    mac = np.abs(ref.conj() @ vecs) / (
                        np.linalg.norm(ref) * np.linalg.norm(vecs, axis=0) + 1e-30)
                    jb = int(np.argmax(mac))
                else:
                    jb = int(np.argsort(np.abs(ps.imag))[br])
                pnew = ps[jb]
                knew = max(abs(pnew.imag) / V * wing.b, 1e-4)
                if abs(knew - kk) < 1e-8:
                    kk = knew
                    break
                kk = 0.5 * kk + 0.5 * knew
            p[br] = pnew
            if br == 0:
                cur_vecs = np.zeros((N, N), dtype=complex)
            cur_vecs[:, br] = vecs[:, jb]
        prev_vecs = cur_vecs
        p_hist[:, iv] = p

    res = TierBResult(V=V_grid, p=p_hist)
    gam = p_hist.real
    osc = np.abs(p_hist.imag) > 1e-3
    best = None
    for br in range(N):
        for i in range(1, V_grid.size):
            if osc[br, i - 1] and osc[br, i] and gam[br, i - 1] <= 0 < gam[br, i]:
                t = -gam[br, i - 1] / (gam[br, i] - gam[br, i - 1] + 1e-30)
                Vf = V_grid[i - 1] + t * (V_grid[i] - V_grid[i - 1])
                wf = abs(p_hist[br, i - 1].imag)
                if best is None or Vf < best[0]:
                    best = (Vf, wf, br)
                break
    if best:
        res.flutter_V, res.flutter_omega, res.flutter_mode = best
    return res


# The classic Goland wing (metal, heavy) — standard flutter benchmark.
GOLAND = dict(
    L=6.096, c=1.8288, m=35.71, I_alpha=8.64,
    x_alpha=0.2, a=-0.34, EI=9.77e6, GJ=0.988e6, rho=1.225,
)
GOLAND_FLUTTER_V = 137.2      # m/s  (published)
GOLAND_FLUTTER_W = 70.7       # rad/s
