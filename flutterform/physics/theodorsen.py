"""Theodorsen's function C(k) for unsteady thin-airfoil aerodynamics.

C(k) = H1(k) / (H1(k) + i*H0(k)),  Hn = Jn - i*Yn  (Hankel fn of the 2nd kind),
k = omega*b/V  (reduced frequency).

Limits: C(0) = 1 (quasi-steady), C(inf) -> 0.5.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv, yv

# Below this k the Bessel-Y terms are numerically nasty and C(k) ~ 1 anyway.
K_MIN = 1e-8


def theodorsen(k):
    """Theodorsen's function C(k) = F(k) + i*G(k).

    Accepts a scalar or ndarray of nonnegative reduced frequencies.
    Returns complex with the same shape (scalar in -> scalar out).
    """
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    if np.any(k_arr < 0):
        raise ValueError("reduced frequency k must be >= 0")

    out = np.ones_like(k_arr, dtype=complex)  # C(0) = 1
    m = k_arr > K_MIN
    kk = k_arr[m]
    h1 = jv(1, kk) - 1j * yv(1, kk)
    h0 = jv(0, kk) - 1j * yv(0, kk)
    out[m] = h1 / (h1 + 1j * h0)

    return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out.reshape(np.shape(k))
