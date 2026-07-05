"""Generate the FlutterForm result figures.

    python scripts/make_figures.py ckpt=results_cmp/ff_indist/flutterform_tierA.pt

Figures (-> results_cmp/figs/):
  vg_vf_examples.png : model V-g / V-f trajectories vs p-k ground truth on
                       held-out sections — the whole flutter diagram, which a
                       scalar regressor cannot produce.
  data_efficiency.png: median V_F error vs #train configs, FlutterForm vs MLP
                       (if results_cmp/de_*.json exist).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutterform.data import TierADataset  # noqa: E402
from flutterform.model import FlutterForm  # noqa: E402
from flutterform.physics import Section, pk_sweep  # noqa: E402

FG = "#1f6feb"
TR = "#d1495b"


def parse(argv):
    cfg = {"ckpt": "results_cmp/ff_indist/flutterform_tierA.pt",
           "data": "data/tierA_50k.npz", "out": "results_cmp/figs", "device": "cpu"}
    for tok in argv:
        k, v = tok.split("=", 1)
        cfg[k] = v
    return cfg


def vg_vf_figure(cfg, outdir):
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    m = FlutterForm(d_model=ck["cfg"].get("model.d", 12),
                    n_iters=ck["cfg"].get("model.iters", 6)).eval()
    m.load_state_dict(ck["model"])

    va = TierADataset(cfg["data"], split="val", n_v=120)
    rng = np.random.default_rng(3)
    idx = rng.choice(len(va), 3, replace=False)
    V = torch.linspace(0.05, 8.0, 240)
    Vn = V.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for c, i in enumerate(idx):
        p = va.params[i]
        sec = Section(*p.tolist())
        truth = pk_sweep(sec, Vn)
        with torch.no_grad():
            gam, om = m.vg_vf(p[None], V)
        gam, om = gam[0].numpy(), om[0].numpy().__abs__()

        axg, axf = axes[0, c], axes[1, c]
        for s in range(2):
            axg.plot(Vn, truth.damping[s], color=TR, lw=2,
                     alpha=0.55, label="p-k truth" if s == 0 else None)
            axg.plot(Vn, gam[:, s], color=FG, lw=1.4, ls="--",
                     label="FlutterForm" if s == 0 else None)
            axf.plot(truth.V, truth.frequency[s], color=TR, lw=2, alpha=0.55)
            axf.plot(Vn, om[:, s], color=FG, lw=1.4, ls="--")
        axg.axhline(0, color="#888", lw=0.7)
        # focus on the physically-meaningful pre/at-flutter region; past
        # flutter the linear model is moot and eigen-branches are near-defective
        xhi = min(8.0, (truth.flutter_V or 3.0) * 1.8 + 0.5)
        axg.set_xlim(0, xhi)
        axf.set_xlim(0, xhi)
        if truth.flutter_V:
            axg.axvline(truth.flutter_V, color="#444", lw=0.8, ls=":")
        axg.set_title(f"mu={sec.mu:.0f}  sigma={sec.sigma:.2f}  M={sec.mach:.2f}",
                      fontsize=9)
        if c == 0:
            axg.set_ylabel("damping  g = Re p/|p|")
            axf.set_ylabel("frequency  Im p")
            axg.legend(fontsize=8, loc="upper left")
        axf.set_xlabel("reduced velocity V")
    fig.suptitle("FlutterForm predicts the V-g / V-f flutter diagram through "
                 "the crossing (dashed) vs p-k ground truth (solid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "vg_vf_examples.png", dpi=130)
    plt.close(fig)
    print(f"wrote {outdir/'vg_vf_examples.png'}")


def data_efficiency_figure(cfg, outdir):
    des = sorted(Path("results_cmp").glob("de_*.json"))
    if not des:
        print("(no de_*.json — skipping data-efficiency figure)")
        return
    fr, ff, bl = [], [], []
    for p in des:
        d = json.loads(p.read_text())
        fr.append(float(p.stem.split("_")[1]))
        ff.append(d.get("flutterform", {}).get("vf_median_%"))
        bl.append(d.get("baseline", {}).get("vf_median_%"))
    order = np.argsort(fr)
    fr = np.array(fr)[order]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(fr, np.array(ff)[order], "o-", color=FG, label="FlutterForm")
    ax.plot(fr, np.array(bl)[order], "s--", color=TR, label="MLP baseline")
    ax.set_xscale("log")
    ax.set_xlabel("train fraction")
    ax.set_ylabel("median flutter-speed error (%)")
    ax.set_title("Data efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "data_efficiency.png", dpi=130)
    plt.close(fig)
    print(f"wrote {outdir/'data_efficiency.png'}")


def main():
    cfg = parse(sys.argv[1:])
    outdir = Path(cfg["out"])
    outdir.mkdir(parents=True, exist_ok=True)
    vg_vf_figure(cfg, outdir)
    data_efficiency_figure(cfg, outdir)


if __name__ == "__main__":
    main()
