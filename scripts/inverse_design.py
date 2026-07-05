"""Inverse design: raise a section's flutter speed by backprop through
FlutterForm, and verify the gain with the true p-k solver.

The differentiable-design payoff. p-k gives one flutter speed per section but
no gradient to *redesign* against it; FlutterForm, being differentiable, turns
flutter speed into an optimizable objective. We optimize three physical design
knobs (elastic-axis a, static unbalance x_theta, frequency ratio sigma) to
push the flutter boundary up, then confirm with the exact p-k solver that the
redesigned section really does flutter later. We compare against the classical
route (finite-difference gradients through p-k), counting cost as p-k solves.

    python scripts/inverse_design.py ckpt=/tmp/ff_probe2/flutterform_tierA.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutterform.model import FlutterForm  # noqa: E402
from flutterform.physics import Section, pk_sweep  # noqa: E402

# design variables (index in the 6-vector, physical bounds)
DVARS = [("sigma", 1, 0.2, 1.0), ("x_theta", 2, 0.0, 0.4), ("a", 3, -0.5, 0.2)]
FIXED = {"mu": 25.0, "r2": 0.29, "mach": 0.0}


def parse(argv):
    c = {"ckpt": "/tmp/ff_probe2/flutterform_tierA.pt", "steps": 200, "device": "cpu"}
    for t in argv:
        k, v = t.split("=", 1)
        c[k] = type(c[k])(v) if k in c and not isinstance(c[k], str) else v
    return c


def true_vf(p6):
    """Exact p-k flutter speed for a 6-param section (np array)."""
    r = pk_sweep(Section(*[float(x) for x in p6]), np.linspace(0.05, 8.0, 320))
    return r.flutter_V


def assemble(dv, base):
    p = base.copy()
    for (_, idx, _, _), val in zip(DVARS, dv):
        p[idx] = val
    return p


def ff_optimize(model, base, steps, device):
    """Gradient ascent on a differentiable stable-range proxy through the model."""
    V = torch.linspace(0.05, 8.0, 96, device=device)
    # unconstrained latents -> box via sigmoid
    z = torch.zeros(len(DVARS), requires_grad=True, device=device)
    fixed = torch.tensor([FIXED["mu"], 0, 0, 0, FIXED["r2"], FIXED["mach"]],
                         dtype=torch.float32, device=device)
    opt = torch.optim.Adam([z], lr=0.05)
    tau = 0.03
    for _ in range(steps):
        dv = []
        for i, (_, _, lo, hi) in enumerate(DVARS):
            dv.append(lo + (hi - lo) * torch.sigmoid(z[i]))
        p = fixed.clone()
        for (_, idx, _, _), val in zip(DVARS, dv):
            p = p.clone()
            p[idx] = val
        gam, _ = model.vg_vf(p[None], V)              # (1, nV, 2)
        # stable-range proxy: integral over V of P(all branches stable)
        p_stable = torch.sigmoid(-gam[0] / tau).prod(dim=-1)   # (nV,)
        proxy = p_stable.mean()
        loss = -proxy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = [lo + (hi - lo) * torch.sigmoid(z[i]).item()
                 for i, (_, _, lo, hi) in enumerate(DVARS)]
    return final  # optimized [sigma, x_theta, a]


def fd_optimize(base, steps, lr=0.03, eps=0.02):
    """Classical baseline: finite-difference gradient of true p-k V_F.
    Counts p-k solves — the cost FlutterForm avoids."""
    z = np.zeros(len(DVARS))

    def to_dv(z):
        return [lo + (hi - lo) / (1 + np.exp(-z[i]))
                for i, (_, _, lo, hi) in enumerate(DVARS)]

    solves = 0
    for _ in range(steps):
        base_dv = to_dv(z)
        f0 = true_vf(assemble(base_dv, base)); solves += 1
        if f0 is None:
            break
        grad = np.zeros(len(DVARS))
        for i in range(len(DVARS)):
            zp = z.copy(); zp[i] += eps
            fp = true_vf(assemble(to_dv(zp), base)); solves += 1
            grad[i] = ((fp or f0) - f0) / eps
        z = z + lr * grad
    return to_dv(z), solves


def main():
    c = parse(sys.argv[1:])
    dev = c["device"]
    ck = torch.load(c["ckpt"], map_location="cpu", weights_only=False)
    model = FlutterForm(d_model=ck["cfg"].get("model.d", 16), n_iters=6).to(dev).eval()
    model.load_state_dict(ck["model"])

    base = np.array([FIXED["mu"], 0.5, 0.25, 0.0, FIXED["r2"], FIXED["mach"]])
    base_dv = [0.5, 0.25, 0.0]
    base_full = assemble(base_dv, base)
    vf0 = true_vf(base_full)
    print(f"baseline design  sigma=0.50 x_theta=0.25 a=0.00 -> true V_F = {vf0:.3f}")

    ff_dv = ff_optimize(model, base, c["steps"], dev)
    vf_ff = true_vf(assemble(ff_dv, base))
    print(f"FlutterForm-opt  sigma={ff_dv[0]:.3f} x_theta={ff_dv[1]:.3f} "
          f"a={ff_dv[2]:.3f} -> true V_F = {vf_ff:.3f}   "
          f"({100*(vf_ff-vf0)/vf0:+.1f}%)   cost: {c['steps']} fwd+bwd passes")

    fd_dv, solves = fd_optimize(base, steps=40)
    vf_fd = true_vf(assemble(fd_dv, base))
    print(f"p-k FD-opt       sigma={fd_dv[0]:.3f} x_theta={fd_dv[1]:.3f} "
          f"a={fd_dv[2]:.3f} -> true V_F = {vf_fd:.3f}   "
          f"({100*(vf_fd-vf0)/vf0:+.1f}%)   cost: {solves} p-k solves")

    print("\nVerification is by the EXACT p-k solver on the redesigned sections.")
    print("FlutterForm supplies the gradient in one backward pass; the classical")
    print("route pays (#vars+1) p-k solves per step. Both should raise V_F by")
    print("mass-balancing (x_theta down) — the textbook flutter fix.")


if __name__ == "__main__":
    main()
