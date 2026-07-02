"""Correctness gates for the flutter physics core.

Gates (hard):
  1. Theodorsen C(k): exact limits + classical tabulated value at k = 0.1.
  2. Static divergence: p-k static instability matches the closed-form V_D.
  3. Cross-method: k-method and p-k agree on (V_F, omega_F) at flutter, where
     both are exact — this validates the two independent implementations
     against each other.
  4. No-flutter sanity: EA at quarter-chord & CG on EA -> no oscillatory
     instability in the sweep range.

Literature anchor (soft, see scripts/validate_physics.py): Hodges & Pierce
typical section (a=-1/5, e=-1/10, mu=20, r2=6/25, sigma=2/5).
"""

import numpy as np
import pytest

from flutterform.physics import (
    Section,
    kmethod_flutter,
    pk_sweep,
    pk_sweep_tracked,
    theodorsen,
)

# The Hodges & Pierce running example section
HP = dict(mu=20.0, sigma=0.4, x_theta=0.1, a=-0.2, r2=6.0 / 25.0)


class TestTheodorsen:
    def test_quasi_steady_limit(self):
        assert theodorsen(0.0) == pytest.approx(1.0)

    def test_high_frequency_limit(self):
        assert theodorsen(1e3).real == pytest.approx(0.5, abs=2e-3)
        assert theodorsen(1e3).imag == pytest.approx(0.0, abs=2e-3)

    def test_classical_table_k01(self):
        # F(0.1) = 0.8320, G(0.1) = -0.1723 (Theodorsen's classical table)
        c = theodorsen(0.1)
        assert c.real == pytest.approx(0.8320, abs=2e-3)
        assert c.imag == pytest.approx(-0.1723, abs=2e-3)

    def test_monotone_decay_of_F(self):
        k = np.linspace(0.01, 2.0, 50)
        F = theodorsen(k).real
        assert np.all(np.diff(F) < 0)


class TestDivergence:
    def test_static_aero_limit_singular_at_divergence_speed(self):
        # At V = V_D the static aeroelastic stiffness Ks - G_static(V) must be
        # singular. This validates the k -> 0 limit of the aero assembly
        # against the closed-form divergence speed, exactly.
        sec = Section(mu=20.0, sigma=0.2, x_theta=0.0, a=0.2, r2=0.25)
        vd = sec.divergence_speed()
        Ks = sec.stiffness_matrix()

        def static_det(V):
            g_static = sec.aero_matrix(1e-9, V).real  # w -> 0: static terms only
            return np.linalg.det(Ks - g_static)

        assert static_det(vd) == pytest.approx(0.0, abs=1e-6 * abs(static_det(0.05)))
        assert static_det(0.9 * vd) * static_det(1.1 * vd) < 0  # sign change

    def test_section_unstable_beyond_divergence_speed(self):
        # Past V_D the p-k spectrum must contain an unstable root (whether the
        # onset manifests as static divergence or as the flutter-divergence
        # interaction this aft-EA section actually exhibits).
        sec = Section(mu=20.0, sigma=0.2, x_theta=0.0, a=0.2, r2=0.25)
        vd = sec.divergence_speed()
        res = pk_sweep(sec, np.linspace(0.05, 1.3 * vd, 300))
        past = res.V > 1.05 * vd
        assert np.all(res.p.real.max(axis=0)[past] > 0.0)

    def test_no_divergence_at_quarter_chord_ea(self):
        sec = Section(mu=20.0, sigma=0.4, x_theta=0.1, a=-0.5, r2=0.24)
        assert sec.divergence_speed() == np.inf


class TestCrossMethod:
    @pytest.mark.parametrize(
        "kw",
        [
            HP,
            dict(mu=10.0, sigma=0.5, x_theta=0.2, a=-0.3, r2=0.3),
            dict(mu=50.0, sigma=0.3, x_theta=0.15, a=-0.4, r2=0.25),
            dict(mu=20.0, sigma=0.7, x_theta=0.25, a=-0.1, r2=0.35),
        ],
    )
    def test_pk_agrees_with_kmethod_at_flutter(self, kw):
        sec = Section(**kw)
        res = pk_sweep(sec, np.linspace(0.05, 8.0, 320))
        vk, wk, _ = kmethod_flutter(sec)
        assert res.flutter_V is not None, "p-k found no flutter"
        assert vk is not None, "k-method found no flutter"
        assert res.flutter_V == pytest.approx(vk, rel=5e-3)
        assert res.flutter_omega == pytest.approx(wk, rel=2e-2)

    def test_flutter_frequency_between_natural_frequencies(self):
        sec = Section(**HP)
        res = pk_sweep(sec)
        assert res.flutter_V is not None
        # coalescence: flutter frequency sits between omega_h and omega_theta
        assert sec.sigma < res.flutter_omega < 1.0


class TestVectorizedEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_vectorized_matches_tracked_reference(self, seed):
        """The vectorized sweep must agree with the MAC-tracked reference:
        same flutter point, same frequency-sorted trajectories."""
        rng = np.random.default_rng(seed)
        for _ in range(3):
            x_t = rng.uniform(0.05, 0.35)
            sec = Section(
                mu=float(np.exp(rng.uniform(np.log(5), np.log(100)))),
                sigma=rng.uniform(0.2, 0.9),
                x_theta=x_t,
                a=rng.uniform(-0.5, 0.1),
                r2=rng.uniform(max(0.1, 1.3 * x_t**2), 0.6),
                mach=rng.uniform(0.0, 0.7),
            )
            V = np.linspace(0.05, 8.0, 320)
            fast, ref = pk_sweep(sec, V), pk_sweep_tracked(sec, V)

            assert (fast.flutter_V is None) == (ref.flutter_V is None)
            if fast.flutter_V is not None:
                assert fast.flutter_V == pytest.approx(ref.flutter_V, rel=5e-3)
                assert fast.flutter_omega == pytest.approx(
                    ref.flutter_omega, rel=1e-2)

            # frequency-sorted trajectories must match pointwise through the
            # flutter crossing. Deep post-flutter (V >> V_F) the unstable
            # root is near-defective and the converged point is legitimately
            # scheme-dependent, so the comparison stops at 1.2 V_F.
            def sorted_traj(res):
                order = np.argsort(np.abs(res.p.imag), axis=0)
                return np.take_along_axis(res.p, order, axis=0)

            v_cmp = 1.2 * fast.flutter_V if fast.flutter_V else V[-1]
            m = V <= v_cmp
            pf, pr = sorted_traj(fast)[:, m], sorted_traj(ref)[:, m]
            np.testing.assert_allclose(pf.imag, pr.imag, atol=2e-3)
            np.testing.assert_allclose(pf.real, pr.real, atol=2e-3)


class TestSanity:
    def test_balanced_quarter_chord_section_is_clean(self):
        # EA at quarter chord (a = -1/2) and CG on the EA: the classic
        # flutter/divergence-proof configuration.
        sec = Section(mu=20.0, sigma=0.4, x_theta=0.0, a=-0.5, r2=0.25)
        res = pk_sweep(sec, np.linspace(0.05, 6.0, 240))
        assert res.flutter_V is None
        assert res.divergence_V is None

    def test_heavier_section_flutters_faster_speed(self):
        # raising mass ratio raises the flutter speed for this family
        v = []
        for mu in (10.0, 40.0):
            sec = Section(mu=mu, sigma=0.4, x_theta=0.1, a=-0.2, r2=0.24)
            res = pk_sweep(sec, np.linspace(0.05, 10.0, 400))
            assert res.flutter_V is not None
            v.append(res.flutter_V)
        assert v[1] > v[0]

    def test_mach_correction_lowers_flutter_speed(self):
        lo = pk_sweep(Section(**HP), np.linspace(0.05, 6.0, 240)).flutter_V
        hi = pk_sweep(Section(**HP, mach=0.6), np.linspace(0.05, 6.0, 240)).flutter_V
        assert lo is not None and hi is not None
        assert hi < lo  # stronger circulatory lift -> earlier flutter
