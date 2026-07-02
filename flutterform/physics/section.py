"""Typical-section (pitch-plunge) aeroelastic matrices.

Conventions (Hodges & Pierce / Bisplinghoff):
  h      plunge of the elastic axis (EA), positive DOWN
  theta  pitch about the EA, positive nose-up
  b      semichord
  a      EA position aft of midchord, in semichords (a in [-1, 1])
  x_t    static unbalance: CG aft of EA, in semichords  (S = m*b*x_t)
  r2     r_theta^2: radius of gyration^2 about EA, in semichords (I = m*b^2*r2)
  mu     mass ratio m / (pi * rho * b^2)
  sigma  omega_h / omega_theta

Equations of motion (lift L positive UP, moment M about EA positive nose-up):
  m h'' + S th'' + k_h h  = -L
  S h'' + I th'' + k_t th =  M

Theodorsen unsteady aero for harmonic motion x = x_bar * e^{i w t}:
  w34 = h' + V th + b(1/2 - a) th'                (3/4-chord downwash)
  L   = pi rho b^2 (h'' + V th' - b a th'') + 2 pi rho V b C(k) w34
  M   = pi rho b^2 (b a h'' - V b(1/2-a) th' - b^2(1/8 + a^2) th'')
        + 2 pi rho V b^2 (a + 1/2) C(k) w34

Compressibility (subsonic, crude): Prandtl-Glauert on the circulatory part,
C(k) -> C(k)/beta with beta = sqrt(1 - M_inf^2). Noncirculatory (apparent
mass) terms are left incompressible.

Nondimensional units used throughout the package: b = 1, omega_theta = 1,
rho = 1. Then airspeed V is the reduced velocity U* = V/(b*omega_theta) and
all frequencies come out as omega/omega_theta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .theodorsen import theodorsen


@dataclass(frozen=True)
class Section:
    """Nondimensional typical section (b = 1, omega_theta = 1, rho = 1)."""

    mu: float          # mass ratio m/(pi rho b^2)
    sigma: float       # omega_h / omega_theta
    x_theta: float     # static unbalance (semichords)
    a: float           # elastic axis aft of midchord (semichords)
    r2: float          # radius of gyration^2 about EA (semichords^2)
    mach: float = 0.0  # freestream Mach number (Prandtl-Glauert on C(k))

    def __post_init__(self):
        if self.r2 <= self.x_theta**2:
            raise ValueError(
                f"non-physical section: r2={self.r2} <= x_theta^2={self.x_theta**2}"
                " (mass matrix not positive definite)"
            )
        if not (0.0 <= self.mach < 1.0):
            raise ValueError("mach must be in [0, 1)")

    # -- structure ---------------------------------------------------------
    @property
    def m(self) -> float:
        return self.mu * np.pi  # rho = b = 1

    def mass_matrix(self) -> np.ndarray:
        m = self.m
        return np.array(
            [[m, m * self.x_theta],
             [m * self.x_theta, m * self.r2]]
        )

    def stiffness_matrix(self) -> np.ndarray:
        m = self.m
        # k_h = m sigma^2 (omega_h = sigma), k_theta = I * 1^2 = m r2
        return np.array([[m * self.sigma**2, 0.0], [0.0, m * self.r2]])

    # -- aerodynamics ------------------------------------------------------
    def _ck(self, k) -> complex:
        beta = np.sqrt(1.0 - self.mach**2)
        return theodorsen(k) / beta

    def aero_matrix(self, k, V) -> np.ndarray:
        """Generalized aerodynamic force matrix G(k, V), complex 2x2.

        For harmonic motion [h, theta] e^{i w t} with w = k V / b:
            [-L; M] = G(k, V) @ [h; theta]
        so the flutter equation reads (p^2 Ms + Ks - G) x = 0.
        """
        rho, b = 1.0, 1.0
        a = self.a
        w = k * V / b
        C = self._ck(k)
        iw = 1j * w

        pib2 = np.pi * rho * b**2
        circ = 2.0 * np.pi * rho * V * b * C

        # L = F11 h + F12 theta ; M = F21 h + F22 theta
        f11 = pib2 * (-(w**2)) + circ * iw
        f12 = pib2 * (iw * V + b * a * w**2) + circ * (V + b * (0.5 - a) * iw)
        f21 = pib2 * (-b * a * w**2) + circ * b * (a + 0.5) * iw
        f22 = (
            pib2 * (-iw * V * b * (0.5 - a) + b**2 * (0.125 + a**2) * w**2)
            + circ * b * (a + 0.5) * (V + b * (0.5 - a) * iw)
        )

        return np.array([[-f11, -f12], [f21, f22]], dtype=complex)

    def aero_matrix_khat(self, k) -> np.ndarray:
        """Frequency-factored aero matrix: G(k, V) = w^2 * Ghat(k).

        Every term of G scales as w^2 once V is expressed as w b / k, which is
        what the classical k-method exploits. Used by kmethod.py.
        """
        rho, b = 1.0, 1.0
        a = self.a
        C = self._ck(k)

        pib2 = np.pi * rho * b**2
        # circ * (i w) = 2 pi rho V b C i w = w^2 * (2 pi rho b^2 C i / k)
        circ_iw = 2.0 * np.pi * rho * b**2 * C * 1j / k
        # circ * V = 2 pi rho V^2 b C = w^2 * (2 pi rho b^3 C / k^2)
        circ_v = 2.0 * np.pi * rho * b**3 * C / k**2

        f11 = -pib2 + circ_iw
        f12 = (
            pib2 * (1j * b / k + b * a)
            + circ_v
            + b * (0.5 - a) * circ_iw
        )
        f21 = -pib2 * b * a + b * (a + 0.5) * circ_iw
        f22 = (
            pib2 * (-1j * b**2 * (0.5 - a) / k + b**2 * (0.125 + a**2))
            + b * (a + 0.5) * (circ_v + b * (0.5 - a) * circ_iw)
        )

        return np.array([[-f11, -f12], [f21, f22]], dtype=complex)

    # -- classical anchors ---------------------------------------------------
    def divergence_speed(self) -> float:
        """Static divergence speed (quasi-steady, C = 1/beta).

        k_theta * th = M_static = 2 pi rho V^2 b^2 (a + 1/2) th / beta
        -> V_D = sqrt(k_theta * beta / (2 pi rho b^2 (a + 1/2))),  inf if a <= -1/2.
        """
        if self.a <= -0.5:
            return np.inf
        beta = np.sqrt(1.0 - self.mach**2)
        k_t = self.m * self.r2
        return float(np.sqrt(k_t * beta / (2.0 * np.pi * (self.a + 0.5))))
