"""Extrapolation crossover figure: median flutter-speed error vs mass ratio
on the held-out mu>=40 set, FlutterForm vs black-box MLP. Both trained only
on mu<40; the shaded band is the training range.

    python scripts/fig_extrapolation.py ff=/tmp/ff_extrap_local/flutterform_tierA.pt \
        bl=/tmp/bl_extrap_local/baseline.pt data=data/tierA_50k.npz
"""
import sys
from pathlib import Path

import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flutterform.baseline import MLPBaseline  # noqa: E402
from flutterform.data import TierADataset  # noqa: E402
from flutterform.flutter_point import hard_flutter_speed  # noqa: E402
from flutterform.model import FlutterForm  # noqa: E402

FG, TR = "#1f6feb", "#d1495b"


def parse(argv):
    c = {"ff": "/tmp/ff_extrap_local/flutterform_tierA.pt",
         "bl": "/tmp/bl_extrap_local/baseline.pt",
         "data": "data/tierA_50k.npz", "out": "results_cmp/figs"}
    for t in argv:
        k, v = t.split("=", 1)
        c[k] = v
    return c


def main():
    c = parse(sys.argv[1:])
    va = TierADataset(c["data"], split="val", n_v=96, holdout_col="mu", holdout_thresh=40)
    mu = va.params[:, 0].numpy()
    vf_true = va.flutter_V.numpy()

    ck = torch.load(c["ff"], map_location="cpu", weights_only=False)
    m = FlutterForm(d_model=ck["cfg"].get("model.d", 16), n_iters=6).eval()
    m.load_state_dict(ck["model"])
    V = torch.linspace(0.05, 8.0, 96)
    ff = np.full(len(va), np.nan)
    with torch.no_grad():
        for i in range(0, len(va), 1024):
            g, o = m.vg_vf(va.params[i:i + 1024], V)
            ff[i:i + 1024], _ = hard_flutter_speed(g.numpy(), o.numpy(), V.numpy())

    bk = torch.load(c["bl"], map_location="cpu", weights_only=False)
    b = MLPBaseline(hidden=bk["cfg"]["train.hidden"], depth=bk["cfg"]["train.depth"]).eval()
    b.load_state_dict(bk["model"])
    with torch.no_grad():
        bl = b.predict(va.params)[0].numpy()

    edges = np.array([40, 48, 56, 64, 72, 80, 90, 101])
    cen, ffm, blm = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mu >= lo) & (mu < hi)
        for arr, out in ((ff, ffm), (bl, blm)):
            mk = sel & np.isfinite(arr) & (vf_true > 0)
            r = np.abs(arr[mk] - vf_true[mk]) / vf_true[mk]
            out.append(np.median(r) * 100 if r.size else np.nan)
        cen.append(0.5 * (lo + hi))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axvspan(5, 40, color="#eee", label="training range (mu<40)")
    ax.plot(cen, ffm, "o-", color=FG, lw=2, label="FlutterForm (physics-structured)")
    ax.plot(cen, blm, "s--", color=TR, lw=2, label="MLP baseline (black box)")
    ax.set_xlabel("mass ratio  mu   (>=40 = extrapolation)")
    ax.set_ylabel("median flutter-speed error (%)")
    ax.set_title("Extrapolation: both trained on mu<40 only")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    outdir = Path(c["out"])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "extrapolation_mu.png", dpi=140)
    print(f"wrote {outdir/'extrapolation_mu.png'}")
    print("FF :", [f"{x:.1f}" for x in ffm])
    print("MLP:", [f"{x:.1f}" for x in blm])


if __name__ == "__main__":
    main()
