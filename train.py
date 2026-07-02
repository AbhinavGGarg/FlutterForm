"""FlutterForm training entry point.

Smoke run (proposal contract):
    python train.py mode=tierA train.max_steps=1

Overrides use key=value (no hydra dependency):
    python train.py mode=tierA data=data/tierA_dev.npz train.max_steps=300 \
                    train.batch=32 train.lr=3e-3 seed=42 device=cpu
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flutterform.data import TierADataset
from flutterform.model import FlutterForm, trajectory_loss

DEFAULTS = {
    "mode": "tierA",
    "data": "data/tierA_dev.npz",
    "train.max_steps": 300,
    "train.batch": 32,
    "train.lr": 3e-3,
    "seed": 42,
    "device": "cpu",
    "out": "results",
    "model.d": 12,
    "model.iters": 6,
}


def parse_overrides(argv):
    cfg = dict(DEFAULTS)
    for tok in argv:
        if "=" not in tok:
            raise SystemExit(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown config key {k!r} (known: {sorted(cfg)})")
        old = cfg[k]
        cfg[k] = type(old)(v) if not isinstance(old, str) else v
    return cfg


def main(argv=None):
    cfg = parse_overrides(sys.argv[1:] if argv is None else argv)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device(cfg["device"])

    if cfg["mode"] != "tierA":
        raise SystemExit(f"mode {cfg['mode']!r} not implemented yet (tierA only)")

    ds = TierADataset(cfg["data"])
    dl = DataLoader(ds, batch_size=cfg["train.batch"], shuffle=True, drop_last=True)
    V = ds.V.to(dev)

    model = FlutterForm(d_model=cfg["model.d"], n_iters=cfg["model.iters"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train.lr"])

    print(f"FlutterForm: {model.n_parameters()} parameters "
          f"(glass-box budget: < 10k) | dataset: {len(ds)} sections | device: {dev}")

    step, t0 = 0, time.time()
    losses = []
    while step < cfg["train.max_steps"]:
        for batch in dl:
            if step >= cfg["train.max_steps"]:
                break
            params = batch["params"].to(dev)
            gam_t = batch["gamma"].to(dev)
            om_t = batch["omega"].to(dev)

            gam, om = model.vg_vf(params, V)
            loss = trajectory_loss(gam, om, gam_t, om_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach()))
            if step % 10 == 0 or step == cfg["train.max_steps"] - 1:
                print(f"step {step:5d}  loss {loss:.5f}  "
                      f"({(time.time()-t0):.1f}s)", flush=True)
            step += 1

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out / "flutterform_tierA.pt")
    (out / "train_metrics.json").write_text(
        json.dumps({"cfg": cfg, "final_loss": losses[-1] if losses else None,
                    "loss_curve": losses}, indent=2)
    )
    print(f"saved checkpoint + metrics to {out}/")


if __name__ == "__main__":
    main()
