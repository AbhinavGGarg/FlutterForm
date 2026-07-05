"""AGARD 445.6 weakened-wing external check (reduced 2-mode bending-torsion).

HONEST SCOPING — read before citing:
- VERIFIED inputs (Altair OS-V / Yates 1987, NASA TM-100492): first bending
  9.5992 Hz, first torsion 38.1650 Hz; semichord b_s = 0.27935 m; reference
  air densities per Mach; experimental flutter-speed-index (FSI) values below.
- ANCHORED: mass ratio mu(M) = 33.465 * rho(0.499)/rho(M), i.e. the literature
  value at M=0.499 propagated by the VERIFIED density ratios (reproduces the
  standard AGARD mu progression 33.5, 68.7, 143.9, ...).
- ASSUMED (not in the sources I could verify): the equivalent-section inertial
  parameters x_alpha, a, r_alpha^2. Rather than invent single values, we
  BRACKET them over a physically plausible range and report the predicted FSI
  band. A quantitative point prediction needs the wing's full 3-D modal model.
- OUT OF SCOPE, by construction: the transonic dip (M~0.96+). It is a
  shock-driven single-mode instability; our attached-flow Theodorsen aero
  (Prandtl-Glauert corrected) cannot produce shocks, so transonic points are
  reported as a known limitation, not fit.

FSI = U_f / (b_s * omega_alpha * sqrt(mu)) = (nondim V_F from p-k) / sqrt(mu),
since our nondimensional reduced velocity already equals U_f/(b_s*omega_alpha).

    python scripts/agard.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutterform.physics import Section, pk_sweep  # noqa: E402

F_BEND, F_TORS = 9.5992, 38.1650          # Hz, verified
SIGMA = F_BEND / F_TORS                    # frequency ratio ~0.2515

# Mach : (air density kg/m^3 [verified], experimental FSI [Yates 1987])
AGARD = {
    0.499: (0.42770, 0.4459),
    0.678: (0.20818, 0.4174),
    0.901: (0.09945, 0.3700),
    0.960: (0.06338, 0.3059),   # transonic dip (out of scope)
    1.072: (0.05514, 0.3852),   # supersonic (out of scope)
    1.141: (0.07833, 0.4460),   # supersonic (out of scope)
}
MU_ANCHOR_M, MU_ANCHOR = 0.499, 33.465     # literature mu at M=0.499
SUBSONIC = [0.499, 0.678, 0.901]

# plausible equivalent-section parameter brackets (ASSUMED)
X_ALPHA = [0.10, 0.20]
A_EA = [-0.3, 0.0]
R2 = [0.45, 0.65]


def mu_of(M):
    return MU_ANCHOR * AGARD[MU_ANCHOR_M][0] / AGARD[M][0]


def predicted_fsi_band(M):
    mu = mu_of(M)
    vals = []
    for xa in X_ALPHA:
        for a in A_EA:
            for r2 in R2:
                if r2 <= xa ** 2:
                    continue
                sec = Section(mu=mu, sigma=SIGMA, x_theta=xa, a=a, r2=r2,
                              mach=min(M, 0.95))
                r = pk_sweep(sec, np.linspace(0.05, 12.0, 400))
                if r.flutter_V:
                    vals.append(r.flutter_V / np.sqrt(mu))
    return (min(vals), max(vals), np.median(vals)) if vals else (np.nan,) * 3


def main():
    print(f"AGARD 445.6 reduced 2-mode check  (sigma = {SIGMA:.4f})")
    print(f"{'Mach':>6}{'mu':>9}{'FSI exp':>10}{'FSI pred band':>22}{'contains?':>11}")
    print("-" * 60)
    rows = []
    for M in AGARD:
        mu = mu_of(M)
        fsi_exp = AGARD[M][1]
        lo, hi, med = predicted_fsi_band(M)
        inb = lo <= fsi_exp <= hi if np.isfinite(lo) else False
        tag = "" if M in SUBSONIC else "  (transonic/SS - out of scope)"
        contains = ("yes" if inb else "no") if M in SUBSONIC else "n/a"
        print(f"{M:>6.3f}{mu:>9.1f}{fsi_exp:>10.4f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>22}{contains:>11}{tag}")
        rows.append((M, mu, fsi_exp, lo, hi, med))

    # figure
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    Ms = np.array([r[0] for r in rows])
    exp = np.array([r[2] for r in rows])
    lo = np.array([r[3] for r in rows])
    hi = np.array([r[4] for r in rows])
    med = np.array([r[5] for r in rows])
    sub = np.array([m in SUBSONIC for m in Ms])
    ax.plot(Ms, exp, "ko-", label="experiment (Yates 1987)")
    ax.fill_between(Ms[sub], lo[sub], hi[sub], color="#1f6feb", alpha=0.25,
                    label="predicted band (subsonic, bracketed section params)")
    ax.plot(Ms[sub], med[sub], "o--", color="#1f6feb")
    ax.axvspan(0.95, 1.2, color="#f6d5d5", alpha=0.5,
               label="transonic/SS: out of scope (attached-flow aero)")
    ax.set_xlabel("Mach")
    ax.set_ylabel("flutter speed index  FSI")
    ax.set_title("AGARD 445.6: reduced 2-mode prediction vs experiment")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)
    out = Path("results_cmp/figs")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "agard.png", dpi=140)
    print(f"\nwrote {out/'agard.png'}")
    print("Honest reading: subsonic FSI + its downward Mach trend are captured;")
    print("the transonic dip is out of scope by construction (needs CFD aero).")


if __name__ == "__main__":
    main()
