"""FlutterForm training entry point.

Smoke run (proposal contract):
    python train.py mode=tierA train.max_steps=1

Real run:
    python train.py mode=tierA data=data/tierA_50k.npz train.max_steps=40000 \
        train.batch=256 train.lr=3e-3 train.w_flutter=1.0 device=auto

Overrides are key=value (no hydra dependency). Trains on a deterministic
train split; reports val loss + val flutter-speed error periodically.
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
from flutterform.flutter_point import hard_flutter_speed
from flutterform.model import FlutterForm, flutter_point_loss, trajectory_loss

DEFAULTS = {
    "mode": "tierA",
    "data": "data/tierA_dev.npz",
    "train.max_steps": 300,
    "train.batch": 32,
    "train.lr": 3e-3,
    "train.w_flutter": 1.0,      # weight on the direct flutter-point loss
    "train.w_lowv": 0.5,         # weight on low-airspeed stability regularizer
    "train.warmup": 200,
    "train.eval_every": 500,
    "train.n_v": 64,
    "train.frac": 1.0,       # subsample train split (data-efficiency curves)
    "holdout.col": "",       # e.g. "mu" for an extrapolation split
    "holdout.thresh": 0.0,   # train below, val at/above
    "seed": 42,
    "device": "auto",
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


def lr_at(step, cfg):
    warm, total = cfg["train.warmup"], cfg["train.max_steps"]
    base = cfg["train.lr"]
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return 0.5 * base * (1 + np.cos(np.pi * min(t, 1.0)))


@torch.no_grad()
def val_report(model, val_ds, V, dev, max_configs=2000):
    model.eval()
    n = min(len(val_ds), max_configs)
    P = val_ds.params[:n].to(dev)
    gam_t = val_ds.gamma[:n].transpose(1, 2)  # (n,nV,2)
    om_t = val_ds.omega[:n].transpose(1, 2)
    vf_t = val_ds.flutter_V[:n].numpy()

    gam, om = model.vg_vf(P, V)
    tloss = trajectory_loss(gam, om, gam_t.to(dev), om_t.to(dev)).item()
    vf_hat, _ = hard_flutter_speed(gam.cpu().numpy(), om.cpu().numpy(),
                                   V.cpu().numpy())
    valid = np.isfinite(vf_hat) & np.isfinite(vf_t) & (vf_t > 0)
    rel = np.abs(vf_hat[valid] - vf_t[valid]) / vf_t[valid]
    model.train()
    return dict(
        val_traj=tloss,
        vf_med=float(np.median(rel)) if rel.size else float("nan"),
        vf_p90=float(np.percentile(rel, 90)) if rel.size else float("nan"),
        within10=float((rel < 0.10).mean()) if rel.size else 0.0,
        coverage=float(valid.mean()),
    )


def main(argv=None):
    cfg = parse_overrides(sys.argv[1:] if argv is None else argv)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(cfg["device"])
    if cfg["mode"] != "tierA":
        raise SystemExit(f"mode {cfg['mode']!r} not implemented yet (tierA only)")

    common = dict(n_v=cfg["train.n_v"])
    if cfg["holdout.col"]:
        common.update(holdout_col=cfg["holdout.col"],
                      holdout_thresh=cfg["holdout.thresh"])
    tr = TierADataset(cfg["data"], split="train", **common)
    va = TierADataset(cfg["data"], split="val", **common)
    if cfg["train.frac"] < 1.0:
        n = max(64, int(cfg["train.frac"] * len(tr)))
        keep = torch.randperm(len(tr))[:n]
        for attr in ("params", "gamma", "omega", "flutter_V", "flutter_omega",
                     "flutter_branch"):
            setattr(tr, attr, getattr(tr, attr)[keep])
    dl = DataLoader(tr, batch_size=cfg["train.batch"], shuffle=True, drop_last=True)
    V = tr.V.to(dev)

    model = FlutterForm(d_model=cfg["model.d"], n_iters=cfg["model.iters"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train.lr"])
    print(f"FlutterForm: {model.n_parameters()} params (<10k) | "
          f"train {len(tr)} / val {len(va)} | device {dev}", flush=True)

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    best_vf, best_state = float("inf"), None

    step, t0, losses, history = 0, time.time(), [], []
    while step < cfg["train.max_steps"]:
        for batch in dl:
            if step >= cfg["train.max_steps"]:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, cfg)
            params = batch["params"].to(dev)
            gam_t = batch["gamma"].to(dev)
            om_t = batch["omega"].to(dev)
            fv = batch["flutter_V"].to(dev)

            fw = batch["flutter_omega"].to(dev)
            gam, om = model.vg_vf(params, V)
            l_traj = trajectory_loss(gam, om, gam_t, om_t)
            l_fp = flutter_point_loss(gam, om, V, fv, fw)
            # low-V stability: at very low airspeed the aeroelastic system is
            # stable (V->0 reduces to the undamped structure), so predicted
            # damping there must not be positive. Directly fights the spurious
            # early crossings that wreck V_F and its parameter gradients.
            low = (V.unsqueeze(0) < 0.5 * fv.clamp_min(0.4).unsqueeze(1))  # (B,nV)
            lowmask = low.unsqueeze(-1).to(gam.dtype)            # (B,nV,1)
            l_lowv = (torch.relu(gam) * lowmask).sum() / (lowmask.sum() * 2 + 1e-6)
            loss = l_traj + cfg["train.w_flutter"] * l_fp + cfg["train.w_lowv"] * l_lowv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))

            if step % 100 == 0 or step == cfg["train.max_steps"] - 1:
                print(f"step {step:6d}  loss {loss:.5f}  (traj {l_traj:.5f} "
                      f"fp {float(l_fp):.5f})  lr {lr_at(step,cfg):.1e}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if cfg["train.max_steps"] > 1 and (
                step % cfg["train.eval_every"] == 0
                and step > 0 or step == cfg["train.max_steps"] - 1
            ):
                r = val_report(model, va, V, dev)
                r["step"] = step
                history.append(r)
                flag = ""
                if np.isfinite(r["vf_med"]) and r["vf_med"] < best_vf:
                    best_vf = r["vf_med"]
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    flag = "  <- best"
                print(f"  [val] traj {r['val_traj']:.5f}  V_F med "
                      f"{r['vf_med']*100:.2f}%  p90 {r['vf_p90']*100:.1f}%  "
                      f"<10% {r['within10']*100:.1f}%  cov {r['coverage']*100:.1f}%"
                      f"{flag}", flush=True)
            step += 1

    # save the BEST-val checkpoint (robust to late-training divergence)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({"model": final_state, "cfg": cfg, "best_val_vf": best_vf},
               out / "flutterform_tierA.pt")
    (out / "train_metrics.json").write_text(json.dumps(
        {"cfg": cfg, "final_loss": losses[-1] if losses else None,
         "best_val_vf_median": best_vf, "val_history": history}, indent=2))
    print(f"saved BEST checkpoint (val V_F median {best_vf*100:.2f}%) + "
          f"metrics to {out}/", flush=True)


if __name__ == "__main__":
    main()
