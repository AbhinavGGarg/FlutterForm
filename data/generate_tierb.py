"""Tier-B dataset: flutter of a parametric family of 3-D cantilever wings.

Each wing is sampled over physically realistic nondimensional ratios, its
flutter point is computed by the validated Tier-B p-k solver, and we store
the nondimensional descriptors + the flutter-speed index. Fixed reference
geometry (semi-span, chord, density); the flutter physics is governed by the
sampled ratios (mass ratio, bending/torsion frequency ratio, static
unbalance, elastic axis, radius of gyration).

    python data/generate_tierb.py --n 3000 --seed 42 --out data/tierb.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutterform.tierb import Wing, pk_flutter  # noqa: E402

# fixed reference geometry / air
L, C, RHO = 6.0, 1.8, 1.225
B = C / 2.0

RANGES = dict(
    mu=(5.0, 60.0),            # mass ratio m/(pi rho b^2)  (log-uniform)
    freq_ratio=(0.2, 0.8),     # omega_bend1 / omega_tors1
    x_alpha=(0.0, 0.3),        # static unbalance (semichords)
    a=(-0.5, 0.0),             # elastic axis (semichords)
    r2=(0.2, 0.5),             # radius of gyration^2 about EA (semichords^2)
)
FEATURES = ["mu", "freq_ratio", "x_alpha", "a", "r2"]


def sample_wing(rng):
    mu = float(np.exp(rng.uniform(*np.log(RANGES["mu"]))))
    fr = float(rng.uniform(*RANGES["freq_ratio"]))
    xa = float(rng.uniform(*RANGES["x_alpha"]))
    a = float(rng.uniform(*RANGES["a"]))
    r2 = float(rng.uniform(max(0.15, 1.3 * xa ** 2), RANGES["r2"][1]))

    m = mu * np.pi * RHO * B ** 2               # mass per span
    I_alpha = m * r2 * B ** 2                    # polar inertia per span
    # choose GJ so 1st torsion freq ~ a reference, then EI from freq ratio
    w_t1 = 90.0                                  # rad/s reference 1st torsion
    GJ = (w_t1 / (np.pi / 2)) ** 2 * I_alpha * L ** 2
    w_b1 = fr * w_t1
    EI = (w_b1 / 1.8751 ** 2) ** 2 * m * L ** 4
    return dict(L=L, c=C, m=m, I_alpha=I_alpha, x_alpha=xa, a=a,
                EI=EI, GJ=GJ, rho=RHO, n_bend=3, n_tors=2), \
        [mu, fr, xa, a, r2], w_t1


def _solve(task):
    kw, feats, w_t1 = task
    wing = Wing(**kw)
    res = pk_flutter(wing, np.linspace(20, 400, 260))
    if res.flutter_V is None:
        return feats, np.nan, np.nan
    # flutter speed index V_F / (b * omega_t1 * sqrt(mu))
    fsi = res.flutter_V / (wing.b * w_t1 * np.sqrt(feats[0]))
    return feats, fsi, res.flutter_omega / w_t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/tierb.npz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tasks = [sample_wing(rng) for _ in range(args.n)]
    feats = np.zeros((args.n, len(FEATURES)))
    fsi = np.full(args.n, np.nan)
    freq = np.full(args.n, np.nan)

    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (f, s, w) in enumerate(pool.imap(_solve, tasks, chunksize=4)):
            feats[i] = f
            fsi[i], freq[i] = s, w
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{args.n} ({(i+1)/(time.time()-t0):.1f}/s)", flush=True)

    ok = np.isfinite(fsi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feature_names=np.array(FEATURES), features=feats,
                        fsi=fsi, freq_ratio_flutter=freq, has_flutter=ok,
                        seed=np.array(args.seed))
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  flutter found: {ok.sum()}/{args.n} ({100*ok.mean():.1f}%)")
    if ok.any():
        print(f"  FSI range: [{fsi[ok].min():.3f}, {fsi[ok].max():.3f}] "
              f"median {np.median(fsi[ok]):.3f}")


if __name__ == "__main__":
    main()
