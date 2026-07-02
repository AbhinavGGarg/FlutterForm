"""Gates for the FlutterForm model: glass-box budget, differentiability,
and the ability to actually fit flutter physics (overfit micro-check)."""

import numpy as np
import pytest
import torch

from flutterform.model import FlutterForm, structural_matrices, trajectory_loss
from flutterform.physics import Section, pk_sweep


def _params(n=4, seed=0):
    rng = np.random.default_rng(seed)
    cols = np.stack(
        [
            np.exp(rng.uniform(np.log(5), np.log(100), n)),  # mu
            rng.uniform(0.2, 1.0, n),                        # sigma
            rng.uniform(0.0, 0.4, n),                        # x_theta
            rng.uniform(-0.5, 0.2, n),                       # a
            rng.uniform(0.25, 0.6, n),                       # r2
            rng.uniform(0.0, 0.7, n),                        # mach
        ],
        axis=1,
    )
    return torch.tensor(cols, dtype=torch.float32)


class TestBudgetAndShapes:
    def test_glass_box_parameter_budget(self):
        model = FlutterForm()
        assert model.n_parameters() < 10_000

    def test_forward_shapes_and_finiteness(self):
        model = FlutterForm()
        V = torch.linspace(0.1, 6.0, 32)
        p = model(_params(3), V)
        assert p.shape == (3, 32, 2)
        assert torch.isfinite(p.real).all() and torch.isfinite(p.imag).all()

    def test_structural_matrices_match_physics(self):
        params = _params(2)
        Ms, Ks = structural_matrices(params)
        for i in range(2):
            mu, sigma, x_t, a, r2, mach = params[i].tolist()
            sec = Section(mu=mu, sigma=sigma, x_theta=x_t, a=a, r2=r2, mach=mach)
            np.testing.assert_allclose(Ms[i], sec.mass_matrix(), rtol=1e-5)
            np.testing.assert_allclose(Ks[i], sec.stiffness_matrix(), rtol=1e-5)


class TestDifferentiability:
    def test_gradients_flow_and_are_finite(self):
        model = FlutterForm()
        V = torch.linspace(0.1, 6.0, 24)
        gam, om = model.vg_vf(_params(3), V)
        loss = gam.square().mean() + om.square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients reached the parameters"
        assert all(torch.isfinite(g).all() for g in grads)


class TestLearnability:
    def test_overfits_a_microbatch(self):
        """The eigen head must be trainable end-to-end: loss on 4 sections
        should drop substantially within 60 steps of Adam on CPU."""
        torch.manual_seed(0)
        params = _params(4, seed=1)
        V_np = np.linspace(0.1, 5.0, 24)
        V = torch.tensor(V_np, dtype=torch.float32)

        gams, oms = [], []
        for row in params:
            mu, sigma, x_t, a, r2, mach = row.tolist()
            sec = Section(mu=mu, sigma=sigma, x_theta=x_t, a=a, r2=r2, mach=mach)
            res = pk_sweep(sec, V_np)
            gams.append(res.damping.T)   # (nV, 2)
            oms.append(res.frequency.T)
        gam_t = torch.tensor(np.stack(gams), dtype=torch.float32)
        om_t = torch.tensor(np.stack(oms), dtype=torch.float32)

        model = FlutterForm()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = None
        for _ in range(150):
            gam, om = model.vg_vf(params, V)
            loss = trajectory_loss(gam, om, gam_t, om_t)
            if first is None:
                first = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        final = float(loss.detach())
        assert final < 0.5 * first, (
            f"loss did not halve: {first:.4f} -> {final:.4f}"
        )
