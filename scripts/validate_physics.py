"""Validation report for the flutter physics core.

Runs the internal gates plus the literature anchor:

Hodges & Pierce, *Introduction to Structural Dynamics and Aeroelasticity*,
typical-section example (a = -1/5, e = -1/10 -> x_theta = 1/10, mu = 20,
r^2 = 6/25, sigma = 2/5), Theodorsen aerodynamics, p-k solution:

    published:  V_F / (b w_theta) = 2.165,   w_F / w_theta = 0.6545

Usage:  python scripts/validate_physics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flutterform.physics import Section, kmethod_flutter, pk_sweep  # noqa: E402

HP_PUBLISHED_VF = 2.165
HP_PUBLISHED_WF = 0.6545


def main():
    sec = Section(mu=20.0, sigma=0.4, x_theta=0.1, a=-0.2, r2=6.0 / 25.0)

    res = pk_sweep(sec, np.linspace(0.05, 4.0, 400))
    vk, wk, brk = kmethod_flutter(sec)

    print("Hodges & Pierce typical section (a=-1/5, x_t=1/10, mu=20, r2=6/25, sigma=2/5)")
    print("-" * 74)
    print(f"  p-k      : V_F = {res.flutter_V:.4f}   w_F = {res.flutter_omega:.4f}   "
          f"branch = {res.flutter_branch}   k_F = {res.flutter_k:.4f}")
    print(f"  k-method : V_F = {vk:.4f}   w_F = {wk:.4f}   branch = {brk}")
    print(f"  published: V_F = {HP_PUBLISHED_VF:.4f}   w_F = {HP_PUBLISHED_WF:.4f}")

    dv = abs(res.flutter_V - HP_PUBLISHED_VF) / HP_PUBLISHED_VF
    dw = abs(res.flutter_omega - HP_PUBLISHED_WF) / HP_PUBLISHED_WF
    print(f"  deviation from published: dV = {100*dv:.2f}%   dw = {100*dw:.2f}%")

    ok = dv < 0.02 and dw < 0.02
    print(f"\n  literature anchor: {'PASS (<2%)' if ok else 'CHECK — investigate before trusting'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
