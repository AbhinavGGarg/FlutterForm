"""Tier-B 3-D wing solver gates, anchored on the Goland wing."""

import numpy as np
import pytest
import torch

from flutterform.flutter_point import hard_flutter_speed
from flutterform.model.nmode import differentiable_pk_flutter
from flutterform.tierb import GOLAND, GOLAND_FLUTTER_V, Wing, pk_flutter


class TestGoland:
    def test_flutter_speed_matches_published(self):
        wing = Wing(**GOLAND, n_bend=3, n_tors=2)
        res = pk_flutter(wing, np.linspace(40, 260, 300))
        assert res.flutter_V is not None
        # published 137.2 m/s; strip theory + assumed modes, allow 8%
        assert res.flutter_V == pytest.approx(GOLAND_FLUTTER_V, rel=0.08)

    def test_flutter_speed_converges_in_modes(self):
        vs = []
        for nb, nt in [(2, 1), (3, 2), (4, 3)]:
            res = pk_flutter(Wing(**GOLAND, n_bend=nb, n_tors=nt),
                             np.linspace(40, 260, 240))
            vs.append(res.flutter_V)
        vs = np.array(vs)
        assert np.all(np.isfinite(vs))
        assert (vs.max() - vs.min()) / vs.mean() < 0.03  # converged <3%

    def test_first_bending_frequency(self):
        wing = Wing(**GOLAND)
        f_b1 = (1.8751 ** 2) * np.sqrt(wing.EI / (wing.m * wing.L ** 4)) / (2 * np.pi)
        assert f_b1 == pytest.approx(7.66, rel=0.05)


class TestNMode:
    """The differentiable N-mode eigen head reproduces the numpy Tier-B p-k
    solver, and gradients flow through it."""

    def _setup(self, nb=2, nt=1):
        wing = Wing(**GOLAND, n_bend=nb, n_tors=nt)
        M, K = wing.structural()
        Mt = torch.tensor(M, dtype=torch.float64)
        Kt = torch.tensor(K, dtype=torch.float64)

        def Q_of_k(kk):
            qh = wing.aero_khat(float(kk))
            return torch.tensor(qh, dtype=torch.complex128)
        return wing, Mt, Kt, Q_of_k

    def test_matches_numpy_pk_flutter(self):
        wing, M, K, Q_of_k = self._setup()
        Vg = torch.linspace(60, 200, 90, dtype=torch.float64)
        gam, omg = differentiable_pk_flutter(M, K, Q_of_k, Vg, wing.b, n_iter=30)
        vf, _ = hard_flutter_speed(gam.numpy()[None], omg.numpy()[None],
                                   Vg.numpy(), persist=1)
        ref = pk_flutter(wing, np.linspace(60, 200, 300)).flutter_V
        assert vf[0] is not None and np.isfinite(vf[0])
        assert vf[0] == pytest.approx(ref, rel=0.05)

    def test_gradients_flow_through_N_mode_head(self):
        wing, M, K, _ = self._setup()
        scale = torch.ones(1, dtype=torch.complex128, requires_grad=True)

        def Q_of_k(kk):
            return scale * torch.tensor(wing.aero_khat(float(kk)),
                                        dtype=torch.complex128)
        Vg = torch.linspace(60, 200, 40, dtype=torch.float64)
        gam, _ = differentiable_pk_flutter(M, K, Q_of_k, Vg, wing.b, n_iter=12)
        gam.abs().sum().backward()
        assert scale.grad is not None and torch.isfinite(scale.grad).all()


class TestStructure:
    def test_mass_stiffness_symmetric_pd(self):
        M, K = Wing(**GOLAND).structural()
        assert np.allclose(M, M.T) and np.allclose(K, K.T)
        assert np.all(np.linalg.eigvals(M) > 0)

    def test_freq_factorization(self):
        # Q(k,V) must equal (kV/b)^2 Qhat(k) for any V (V-independent Qhat)
        wing = Wing(**GOLAND)
        k = 0.4
        qhat = wing.aero_khat(k)
        for V in (80.0, 160.0):
            w2 = (k * V / wing.b) ** 2
            np.testing.assert_allclose(wing.aero(k, V), w2 * qhat, rtol=1e-9)

    def test_stiffer_wing_flutters_faster(self):
        base = dict(GOLAND)
        soft = pk_flutter(Wing(**{**base, "GJ": base["GJ"] * 0.6}),
                          np.linspace(40, 260, 240)).flutter_V
        stiff = pk_flutter(Wing(**{**base, "GJ": base["GJ"] * 1.6}),
                           np.linspace(40, 320, 300)).flutter_V
        assert soft is not None and stiff is not None
        assert stiff > soft  # more torsional stiffness -> higher flutter speed
