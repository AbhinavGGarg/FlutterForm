"""Tier-B 3-D wing solver gates, anchored on the Goland wing."""

import numpy as np
import pytest

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


class TestStructure:
    def test_mass_stiffness_symmetric_pd(self):
        M, K = Wing(**GOLAND).structural()
        assert np.allclose(M, M.T) and np.allclose(K, K.T)
        assert np.all(np.linalg.eigvals(M) > 0)

    def test_stiffer_wing_flutters_faster(self):
        base = dict(GOLAND)
        soft = pk_flutter(Wing(**{**base, "GJ": base["GJ"] * 0.6}),
                          np.linspace(40, 260, 240)).flutter_V
        stiff = pk_flutter(Wing(**{**base, "GJ": base["GJ"] * 1.6}),
                           np.linspace(40, 320, 300)).flutter_V
        assert soft is not None and stiff is not None
        assert stiff > soft  # more torsional stiffness -> higher flutter speed
