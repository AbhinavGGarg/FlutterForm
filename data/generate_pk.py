"""Tier-A dataset generator: typical-section p-k flutter ground truth.

Samples sections from the proposal's parameter ranges, runs the validated
p-k sweep, and stores nondimensional V-g / V-f branch trajectories plus the
flutter point. Configs whose flutter (if any) lies outside the V grid are
kept but flagged; the drop rate is reported, never silent.

Usage:
    python data/generate_pk.py --n 400 --seed 42 --out data/tierA_dev.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flutterform.physics import Section, pk_sweep  # noqa: E402

# Proposal sweep ranges (Tier A)
RANGES = dict(
    mu=(5.0, 100.0),        # log-uniform
    sigma=(0.2, 1.0),
    x_theta=(0.0, 0.4),
    a=(-0.5, 0.2),
    mach=(0.0, 0.7),
)
R2_MAX = 0.6


def sample_params(rng: np.random.Generator) -> dict:
    mu = float(np.exp(rng.uniform(*np.log(RANGES["mu"]))))
    sigma = float(rng.uniform(*RANGES["sigma"]))
    x_t = float(rng.uniform(*RANGES["x_theta"]))
    a = float(rng.uniform(*RANGES["a"]))
    r2_lo = max(0.1, 1.3 * x_t**2)
    r2 = float(rng.uniform(r2_lo, R2_MAX))
    mach = float(rng.uniform(*RANGES["mach"]))
    return dict(mu=mu, sigma=sigma, x_theta=x_t, a=a, r2=r2, mach=mach)


def _solve_one(task):
    """Worker: p-k sweep for one sampled section (top-level for pickling)."""
    kw, v_grid = task
    res = pk_sweep(Section(**kw), v_grid)
    row = [kw["mu"], kw["sigma"], kw["x_theta"], kw["a"], kw["r2"], kw["mach"]]
    if res.flutter_V is not None:
        fl = (res.flutter_V, res.flutter_omega, res.flutter_branch, res.flutter_k)
    else:
        fl = None
    return row, res.damping, res.frequency, fl


def generate(n: int, seed: int, v_max: float, n_v: int, workers: int = 0):
    rng = np.random.default_rng(seed)
    V_grid = np.linspace(0.05, v_max, n_v)

    # sample every config up front in the main process: the dataset is a pure
    # function of the seed, independent of worker count
    tasks = [(sample_params(rng), V_grid) for _ in range(n)]

    params = np.zeros((n, 6))
    gamma = np.zeros((n, 2, n_v))
    omega = np.zeros((n, 2, n_v))
    flutter_V = np.full(n, np.nan)
    flutter_omega = np.full(n, np.nan)
    flutter_branch = np.full(n, -1, dtype=int)
    flutter_k = np.full(n, np.nan)
    has_flutter = np.zeros(n, dtype=bool)

    t0 = time.time()

    def _store(i, out):
        row, gam, om, fl = out
        params[i] = row
        gamma[i] = gam
        omega[i] = om
        if fl is not None:
            has_flutter[i] = True
            flutter_V[i], flutter_omega[i], flutter_branch[i], flutter_k[i] = fl
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  {i+1}/{n}  ({rate:.1f} cfg/s, eta {eta/60:.1f} min)", flush=True)

    if workers and workers > 1:
        with Pool(workers) as pool:
            for i, out in enumerate(pool.imap(_solve_one, tasks, chunksize=8)):
                _store(i, out)
    else:
        for i, task in enumerate(tasks):
            _store(i, _solve_one(task))

    return dict(
        param_names=np.array(["mu", "sigma", "x_theta", "a", "r2", "mach"]),
        params=params, V_grid=V_grid, gamma=gamma, omega=omega,
        flutter_V=flutter_V, flutter_omega=flutter_omega,
        flutter_branch=flutter_branch, flutter_k=flutter_k,
        has_flutter=has_flutter, seed=np.array(seed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/tierA_dev.npz")
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--nv", type=int, default=320)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel p-k workers (dataset is seed-deterministic "
                         "regardless of worker count); 0/1 = serial")
    args = ap.parse_args()

    print(f"generating {args.n} sections (seed {args.seed}, "
          f"{args.workers} workers) ...")
    d = generate(args.n, args.seed, args.vmax, args.nv, workers=args.workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)

    nf = int(d["has_flutter"].sum())
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  flutter found : {nf}/{args.n}  ({100*nf/args.n:.1f}%)")
    print(f"  no flutter in V<= {args.vmax}: {args.n-nf} (kept, flagged — not silently dropped)")
    vf = d["flutter_V"][d["has_flutter"]]
    if nf:
        print(f"  V_F range     : [{vf.min():.2f}, {vf.max():.2f}]   "
              f"median {np.median(vf):.2f}")


if __name__ == "__main__":
    main()
