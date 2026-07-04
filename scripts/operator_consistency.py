"""Operator-consistency: does FlutterForm's LEARNED aero coupling match the
analytic Theodorsen aerodynamic influence-coefficient (AIC) matrix?

Both the model and the physics express the aero force as G(k,V) = w^2 * Ghat(k)
(the frequency-factored form; see Section.aero_matrix_khat). We compare the
model's Ghat(k) against the analytic Ghat(k) across reduced frequency.

Caveat (identifiability): the model is trained only to produce correct
eigenvalues, so it may learn a coupling that is equivalent up to a
structure-preserving transform. We therefore report BOTH:
  (a) raw relative Frobenius error, and
  (b) error after a single global complex scale align (least-squares) —
      invariant to the one gauge the eigenproblem genuinely cannot pin down.

    python scripts/operator_consistency.py ckpt=results_cmp/ff_indist/flutterform_tierA.pt
"""

import sys

import numpy as np
import torch

from flutterform.model import FlutterForm
from flutterform.physics import Section


def parse(argv):
    cfg = {"ckpt": "results_cmp/ff_indist/flutterform_tierA.pt",
           "data": "data/tierA_50k.npz", "n": 200, "device": "cpu"}
    for tok in argv:
        k, v = tok.split("=", 1)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    return cfg


def main():
    cfg = parse(sys.argv[1:])
    dev = cfg["device"]
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    m = FlutterForm(d_model=ck["cfg"].get("model.d", 12),
                    n_iters=ck["cfg"].get("model.iters", 6)).to(dev).eval()
    m.load_state_dict(ck["model"])

    d = np.load(cfg["data"])
    P = d["params"][d["has_flutter"]][: cfg["n"]]
    ks = np.geomspace(0.05, 2.0, 12)

    raw, aligned = [], []
    with torch.no_grad():
        for row in P:
            sec = Section(*row.tolist())
            pt = torch.tensor(row, dtype=torch.float32, device=dev)[None]
            for k in ks:
                kk = torch.tensor([k], dtype=torch.float32, device=dev)
                learned = m.operator(pt, kk)[0].cpu().numpy()          # (2,2) complex
                analytic = sec.aero_matrix_khat(float(k))              # (2,2) complex
                num = np.linalg.norm(learned - analytic)
                den = np.linalg.norm(analytic) + 1e-12
                raw.append(num / den)
                # single global complex scale that best maps learned->analytic
                s = (np.vdot(learned.ravel(), analytic.ravel())
                     / (np.vdot(learned.ravel(), learned.ravel()) + 1e-12))
                aligned.append(np.linalg.norm(s * learned - analytic) / den)

    raw = np.array(raw)
    aligned = np.array(aligned)
    print(f"checkpoint: {cfg['ckpt']}")
    print(f"configs x k-points: {len(P)} x {len(ks)} = {raw.size}")
    print(f"raw    rel-Frobenius error   : median {np.median(raw)*100:.1f}%  "
          f"mean {raw.mean()*100:.1f}%")
    print(f"scale-aligned rel-Frob error : median {np.median(aligned)*100:.1f}%  "
          f"mean {aligned.mean()*100:.1f}%")
    print("\n(Lower = the learned coupling recovers the analytic Theodorsen AIC.\n"
          " Scale-aligned removes the one global gauge the eigenproblem can't fix.)")


if __name__ == "__main__":
    main()
