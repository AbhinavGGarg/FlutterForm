"""Validate the Tier-B 3-D wing solver against the classic Goland wing.

Published (metal Goland wing, sea level): flutter speed ~137 m/s,
flutter frequency ~70.7 rad/s. A reduced assumed-modes + strip-theory model
should land in the right ballpark (strip theory omits 3-D/tip effects, so a
few-percent-to-~15% deviation is expected and honest).

    python scripts/validate_tierb.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutterform.tierb import (GOLAND, GOLAND_FLUTTER_V, GOLAND_FLUTTER_W,  # noqa
                               Wing, pk_flutter)


def main():
    print("Goland wing — Tier-B assumed-modes + strip-theory p-k flutter")
    print("-" * 62)
    for (nb, nt) in [(2, 1), (3, 2), (4, 3)]:
        wing = Wing(**GOLAND, n_bend=nb, n_tors=nt)
        # uncoupled 1st bending / 1st torsion natural frequencies (direct)
        f_b1 = (1.8751 ** 2) * np.sqrt(wing.EI / (wing.m * wing.L ** 4)) / (2 * np.pi)
        f_t1 = (np.pi / 2) * np.sqrt(wing.GJ / (wing.I_alpha * wing.L ** 2)) / (2 * np.pi)
        res = pk_flutter(wing, np.linspace(40, 260, 300))
        vf, wf = res.flutter_V, res.flutter_omega
        if vf:
            dv = 100 * (vf - GOLAND_FLUTTER_V) / GOLAND_FLUTTER_V
            print(f"  modes {nb}B+{nt}T (N={nb+nt}): "
                  f"1st bend {f_b1:5.2f} Hz, 1st tors {f_t1:5.2f} Hz | "
                  f"V_F = {vf:6.1f} m/s ({dv:+.1f}%), omega_F = {wf:5.1f} rad/s")
        else:
            print(f"  modes {nb}B+{nt}T: no flutter found in sweep")
    print(f"\n  published: V_F = {GOLAND_FLUTTER_V} m/s, omega_F = {GOLAND_FLUTTER_W} rad/s")


if __name__ == "__main__":
    main()
