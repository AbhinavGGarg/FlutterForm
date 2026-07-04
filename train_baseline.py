"""Train the black-box MLP baseline on (params -> V_F, omega_F).

    python train_baseline.py data=data/tierA_50k.npz train.max_steps=8000 \
        train.frac=1.0 device=auto out=results_baseline

train.frac subsamples the TRAIN split (for data-efficiency curves).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from flutterform.baseline import MLPBaseline
from flutterform.data import TierADataset

DEFAULTS = {
    "data": "data/tierA_50k.npz",
    "train.max_steps": 8000,
    "train.batch": 512,
    "train.lr": 3e-3,
    "train.frac": 1.0,
    "train.hidden": 48,
    "train.depth": 3,
    "holdout.col": "",
    "holdout.thresh": 0.0,
    "seed": 42,
    "device": "auto",
    "out": "results_baseline",
}


def parse(argv):
    cfg = dict(DEFAULTS)
    for tok in argv:
        k, v = tok.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown key {k!r}")
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    return cfg


def load_xy(ds):
    x = ds.params
    y = torch.stack([ds.flutter_V.clamp_min(1e-3).log(),
                     ds.flutter_omega.clamp_min(1e-3).log()], dim=1)
    ok = torch.isfinite(y).all(1)
    return x[ok], y[ok]


def main(argv=None):
    cfg = parse(sys.argv[1:] if argv is None else argv)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(cfg["device"])

    ho = ({} if not cfg["holdout.col"]
          else dict(holdout_col=cfg["holdout.col"], holdout_thresh=cfg["holdout.thresh"]))
    tr = TierADataset(cfg["data"], split="train", n_v=8, **ho)
    va = TierADataset(cfg["data"], split="val", n_v=8, **ho)
    xtr, ytr = load_xy(tr)
    xva, yva = load_xy(va)

    # data-efficiency subsample
    if cfg["train.frac"] < 1.0:
        n = max(64, int(cfg["train.frac"] * len(xtr)))
        idx = torch.randperm(len(xtr))[:n]
        xtr, ytr = xtr[idx], ytr[idx]

    model = MLPBaseline(hidden=cfg["train.hidden"], depth=cfg["train.depth"]).to(dev)
    model.set_norm(xtr.mean(0), xtr.std(0))
    xtr, ytr, xva, yva = (t.to(dev) for t in (xtr, ytr, xva, yva))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train.lr"])
    print(f"MLPBaseline: {model.n_parameters()} params | train {len(xtr)} "
          f"(frac {cfg['train.frac']}) | device {dev}", flush=True)

    n = len(xtr)
    for step in range(cfg["train.max_steps"]):
        for pg in opt.param_groups:
            pg["lr"] = 0.5 * cfg["train.lr"] * (
                1 + np.cos(np.pi * step / cfg["train.max_steps"]))
        idx = torch.randint(0, n, (min(cfg["train.batch"], n),), device=dev)
        pred = model(xtr[idx])
        loss = torch.nn.functional.mse_loss(pred, ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == cfg["train.max_steps"] - 1:
            with torch.no_grad():
                vf_p, _ = model.predict(xva)
                rel = ((vf_p - yva[:, 0].exp()).abs() / yva[:, 0].exp())
            print(f"step {step:6d}  loss {loss:.4f}  val V_F med "
                  f"{rel.median()*100:.2f}%  <10% {(rel<0.1).float().mean()*100:.1f}%",
                  flush=True)

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out / "baseline.pt")
    (out / "baseline_cfg.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved baseline to {out}/", flush=True)


if __name__ == "__main__":
    main()
