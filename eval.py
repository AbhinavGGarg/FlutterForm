"""Evaluate a FlutterForm checkpoint (and optionally the MLP baseline) on the
held-out val split. Reports flutter-speed / frequency error, coverage,
mode-ID accuracy (FlutterForm only), and a per-region error breakdown.

    python eval.py ckpt=results_d16/flutterform_tierA.pt data=data/tierA_50k.npz \
        baseline=results_baseline/baseline.pt out=results_d16/eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from flutterform.baseline import MLPBaseline
from flutterform.data import TierADataset
from flutterform.flutter_point import hard_flutter_speed
from flutterform.model import FlutterForm

PARAM_NAMES = ["mu", "sigma", "x_theta", "a", "r2", "mach"]


def parse(argv):
    cfg = {"ckpt": "results_d16/flutterform_tierA.pt", "data": "data/tierA_50k.npz",
           "baseline": "", "out": "", "n_v": 96, "device": "auto",
           "holdout.col": "", "holdout.thresh": 0.0}
    for tok in argv:
        k, v = tok.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], str) else v
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def rel_err(pred, true):
    m = np.isfinite(pred) & np.isfinite(true) & (true > 0)
    return np.abs(pred[m] - true[m]) / true[m], m


def summarize(pred_vf, true_vf, tag):
    rel, m = rel_err(pred_vf, true_vf)
    return {
        "tag": tag,
        "coverage": float(m.mean()),
        "vf_median_%": float(np.median(rel) * 100) if rel.size else None,
        "vf_mean_%": float(rel.mean() * 100) if rel.size else None,
        "vf_p90_%": float(np.percentile(rel, 90) * 100) if rel.size else None,
        "within_5%": float((rel < 0.05).mean() * 100) if rel.size else None,
        "within_10%": float((rel < 0.10).mean() * 100) if rel.size else None,
    }


def flutterform_predict(ckpt_path, params, dev, n_v):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ck["cfg"].get("model.d", 12)
    it = ck["cfg"].get("model.iters", 6)
    m = FlutterForm(d_model=d, n_iters=it).to(dev).eval()
    m.load_state_dict(ck["model"])
    V = torch.linspace(0.05, 8.0, n_v, device=dev)
    Vnp = V.cpu().numpy()
    vf = np.full(len(params), np.nan)
    wf = np.full(len(params), np.nan)
    om_at_vf = np.full((len(params), 2), np.nan)
    with torch.no_grad():
        for i in range(0, len(params), 1024):
            P = params[i:i + 1024].to(dev)
            g, o = m.vg_vf(P, V)
            gv, ov = g.cpu().numpy(), o.cpu().numpy()
            v, w = hard_flutter_speed(gv, ov, Vnp)
            vf[i:i + 1024] = v
            wf[i:i + 1024] = w
    return vf, wf


def mode_id_accuracy(ckpt_path, ds, dev, n_v):
    """Does the frequency-sorted branch that flutters match the ground-truth
    coalescing branch? Ground-truth branch is stored in flutter_branch; the
    p-k tracker orders branches by wind-off frequency, same as our sort, so
    the indices are comparable."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = FlutterForm(d_model=ck["cfg"].get("model.d", 12),
                    n_iters=ck["cfg"].get("model.iters", 6)).to(dev).eval()
    m.load_state_dict(ck["model"])
    V = torch.linspace(0.05, 8.0, n_v, device=dev)
    Vnp = V.cpu().numpy()
    gt = ds.flutter_branch.numpy()
    correct = tot = 0
    with torch.no_grad():
        for i in range(0, len(ds.params), 1024):
            P = ds.params[i:i + 1024].to(dev)
            g, o = m.vg_vf(P, V)
            gv, ov = g.cpu().numpy(), np.abs(o.cpu().numpy())
            for b in range(gv.shape[0]):
                gtb = gt[i + b]
                if gtb < 0:
                    continue
                pred_branch, best_v = -1, np.inf
                for s in range(2):
                    gs, ws = gv[b, :, s], ov[b, :, s]
                    up = (gs[:-1] <= 0) & (gs[1:] > 0) & (ws[:-1] > 5e-3)
                    if up.any():
                        j = int(np.argmax(up))
                        t = -gs[j] / (gs[j + 1] - gs[j] + 1e-30)
                        v = Vnp[j] + t * (Vnp[j + 1] - Vnp[j])
                        if v < best_v:
                            best_v, pred_branch = v, s
                if pred_branch >= 0:
                    tot += 1
                    correct += int(pred_branch == gtb)
    return correct / tot if tot else float("nan"), tot


def region_breakdown(pred_vf, true_vf, params):
    rows = []
    for col, name, edges in [
        (0, "mu", [5, 15, 40, 100]),
        (5, "mach", [0.0, 0.25, 0.5, 0.7]),
        (1, "sigma", [0.2, 0.45, 0.7, 1.0]),
    ]:
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (params[:, col] >= lo) & (params[:, col] < hi + 1e-9)
            rel, m = rel_err(pred_vf[sel], true_vf[sel])
            rows.append({
                "region": f"{name} in [{lo},{hi})",
                "n": int(sel.sum()),
                "vf_median_%": float(np.median(rel) * 100) if rel.size else None,
                "within_10%": float((rel < 0.10).mean() * 100) if rel.size else None,
            })
    return rows


def main():
    cfg = parse(sys.argv[1:])
    dev = torch.device(cfg["device"])
    ho = ({} if not cfg["holdout.col"]
          else dict(holdout_col=cfg["holdout.col"], holdout_thresh=cfg["holdout.thresh"]))
    va = TierADataset(cfg["data"], split="val", n_v=cfg["n_v"], **ho)
    params_np = va.params.numpy()
    true_vf = va.flutter_V.numpy()

    report = {"cfg": cfg, "n_val": len(va)}

    ff_vf, _ = flutterform_predict(cfg["ckpt"], va.params, dev, cfg["n_v"])
    report["flutterform"] = summarize(ff_vf, true_vf, "FlutterForm")
    acc, n_mode = mode_id_accuracy(cfg["ckpt"], va, dev, cfg["n_v"])
    report["flutterform"]["mode_id_acc_%"] = float(acc * 100)
    report["flutterform"]["mode_id_n"] = n_mode
    report["flutterform_regions"] = region_breakdown(ff_vf, true_vf, params_np)

    if cfg["baseline"]:
        ck = torch.load(cfg["baseline"], map_location="cpu", weights_only=False)
        b = MLPBaseline(hidden=ck["cfg"]["train.hidden"],
                        depth=ck["cfg"]["train.depth"]).to(dev).eval()
        b.load_state_dict(ck["model"])
        with torch.no_grad():
            bvf, _ = b.predict(va.params.to(dev))
        report["baseline"] = summarize(bvf.cpu().numpy(), true_vf, "MLP baseline")
        report["baseline"]["mode_id_acc_%"] = None  # structurally cannot

    print(json.dumps(report, indent=2))
    if cfg["out"]:
        Path(cfg["out"]).write_text(json.dumps(report, indent=2))
    # human-readable summary
    ff = report["flutterform"]
    print("\n=== FlutterForm (held-out val) ===")
    print(f"  flutter-speed median err : {ff['vf_median_%']:.2f}%  "
          f"(<10%: {ff['within_10%']:.1f}%, p90 {ff['vf_p90_%']:.1f}%)")
    print(f"  coverage                 : {ff['coverage']*100:.1f}%")
    print(f"  mode-ID accuracy         : {ff['mode_id_acc_%']:.1f}%  (n={ff['mode_id_n']})")
    if cfg["baseline"]:
        bl = report["baseline"]
        print(f"  [baseline] V_F median    : {bl['vf_median_%']:.2f}%  "
              f"(<10%: {bl['within_10%']:.1f}%)  |  mode-ID: n/a (black box)")


if __name__ == "__main__":
    main()
