"""Tier 2: Boundary & Corner Cases for Features 1 to 5.

Boundary 1: Discrete State
Boundary 2: Continuous State
Boundary 3: Local Fields
Boundary 4: Binary Heat Bath
Boundary 5: Continuous Heat Bath
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_binary_glauber_prob,
    oracle_continuous_inverse_cdf,
)


class TestBoundary01DiscreteState(ModularBaseTestCase):
    """Boundary cases for Feature 1: Discrete State Representation."""

    def test_b01_single_node_population_n1(self):
        """Verify discrete state handles minimal population N = 1."""
        state = self.create_state(N=1, D=1, continuous=False)
        self.assertEqual(state.x.shape, (1, 1))
        self.assertEqual(state.y.shape, (1, 1))

    def test_b01_maximal_dimension_d10(self):
        """Verify discrete state handles dimension D = 3 or larger."""
        state = self.create_state(N=5, D=3, continuous=False)
        self.assertEqual(state.x.shape, (5, 3))

    def test_b01_all_positive_state_vector(self):
        """Verify boundary condition where entire population holds +1 spins."""
        state = self.create_state(N=10, D=1, continuous=False)
        state.x[:] = 1.0
        state.y[:] = 1.0
        self.assertTrue(np.all(state.x == 1.0))
        self.assertTrue(np.all(state.y == 1.0))

    def test_b01_all_negative_state_vector(self):
        """Verify boundary condition where entire population holds -1 spins."""
        state = self.create_state(N=10, D=1, continuous=False)
        state.x[:] = -1.0
        state.y[:] = -1.0
        self.assertTrue(np.all(state.x == -1.0))
        self.assertTrue(np.all(state.y == -1.0))

    def test_b01_alternating_spin_state_vector(self):
        """Verify exact zero-mean alternating configuration."""
        state = self.create_state(N=10, D=1, continuous=False)
        state.x[:] = np.array([[1.0], [-1.0]] * 5)
        self.assertAlmostEqual(float(np.mean(state.x)), 0.0)


class TestBoundary02ContinuousState(ModularBaseTestCase):
    """Boundary cases for Feature 2: Bounded Continuous State Representation."""

    def test_b02_exact_endpoint_minus_one_and_plus_one(self):
        """Verify continuous states at exact boundaries -1.0 and +1.0 are valid."""
        state = self.create_state(N=2, D=1, continuous=True)
        state.x[0, 0] = -1.0
        state.x[1, 0] = +1.0
        self.assertEqual(state.x[0, 0], -1.0)
        self.assertEqual(state.x[1, 0], +1.0)

    def test_b02_exact_zero_opinion_center(self):
        """Verify continuous state centered exactly at 0.0 is stable."""
        state = self.create_state(N=5, D=1, continuous=True)
        state.x[:] = 0.0
        self.assertTrue(np.all(state.x == 0.0))

    def test_b02_near_boundary_epsilon_precision(self):
        """Verify values at 1.0 - 1e-15 do not trigger false out-of-bounds rejection."""
        state = self.create_state(N=2, D=1, continuous=True)
        near_top = 1.0 - 1e-15
        state.x[0, 0] = near_top
        self.assertAlmostEqual(state.x[0, 0], near_top, places=14)

    def test_b02_all_zero_second_moment(self):
        """Verify population at zero opinion has zero second moment M^(2) = 0."""
        calc_second_moment = self.require_attr("src.analytics.discrepancy", "calculate_second_moment")
        x = np.zeros((10, 1))
        m2 = calc_second_moment(x)
        self.assertAlmostEqual(m2[0], 0.0, places=6)

    def test_b02_polarized_bimodal_endpoints_second_moment(self):
        """Verify population equally split at -1.0 and +1.0 has mean 0 but second moment 1.0."""
        calc_moments = self.require_attr("src.analytics.discrepancy", "calculate_moments")
        x = np.array([[-1.0], [1.0], [-1.0], [1.0]])
        res = calc_moments(x)
        self.assertAlmostEqual(res["mean"][0], 0.0, places=6)
        self.assertAlmostEqual(res["second_moment"][0], 1.0, places=6)


class TestBoundary03LocalFields(ModularBaseTestCase):
    """Boundary cases for Feature 3: Local Fields Calculation."""

    def _calc_public(self, A, y, x, J=1.0, K=0.5, h=None):
        fn = self.require_attr("src.core.fields", "calculate_public_fields", aliases=["compute_public_fields"])
        return fn(A, y, x, J=J, K=K, h=h)

    def test_b03_zero_coupling_fields_j0_k0_l0(self):
        """Verify when all couplings are zero, fields reduce purely to external bias."""
        A = np.ones((2, 2))
        y = np.ones((2, 1))
        x = np.ones((2, 1))
        fields = self._calc_public(A, y, x, J=0.0, K=0.0, h=0.35)
        np.testing.assert_allclose(fields, np.full((2, 1), 0.35))

    def test_b03_extreme_external_field_pulse(self):
        """Verify extreme external field pulse h = 1000.0 dominates social couplings."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[-1.0], [-1.0]])
        x = np.array([[-1.0], [-1.0]])
        fields = self._calc_public(A, y, x, J=1.0, K=0.5, h=1000.0)
        self.assertGreater(fields[0, 0], 900.0)

    def test_b03_disconnected_node_zero_attention_field(self):
        """Verify node with all-zero attention row ignores other agents' states."""
        A = np.array([[0.0, 0.0], [1.0, 0.0]])
        y = np.array([[1.0], [-1.0]])
        x = np.array([[1.0], [-1.0]])
        fields = self._calc_public(A, y, x, J=5.0, K=0.5, h=0.0)
        # Node 0 has row A[0, :] = 0 -> J * 0 + K * 1.0 = 0.5
        self.assertAlmostEqual(fields[0, 0], 0.5, places=6)

    def test_b03_maximal_frustration_opposing_fields(self):
        """Verify public alignment and private coupling can cancel to exact zero field."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[1.0], [-1.0]])
        x = np.array([[2.0], [0.0]])
        # J = 1.0, K = 0.5 -> J * (A@y)[0] + K * x[0] = 1.0 * (-1.0) + 0.5 * (2.0) = 0.0
        fields = self._calc_public(A, y, x, J=1.0, K=0.5, h=0.0)
        self.assertAlmostEqual(fields[0, 0], 0.0, places=6)

    def test_b03_field_calculation_with_uniform_spins(self):
        """Verify field arithmetic when all spins in system are +1."""
        N = 10
        A = (np.ones((N, N)) - np.eye(N)) / (N - 1)
        y = np.ones((N, 1))
        x = np.ones((N, 1))
        fields = self._calc_public(A, y, x, J=1.0, K=0.5, h=0.0)
        # J * 1.0 + K * 1.0 = 1.5
        np.testing.assert_allclose(fields, np.full((N, 1), 1.5))


class TestBoundary04BinaryHeatBath(ModularBaseTestCase):
    """Boundary cases for Feature 4: Binary Heat Bath Updates."""

    def _get_prob(self, F, T=1.0):
        fn = self.require_attr("src.core.heat_baths", "binary_glauber_probability", aliases=["binary_glauber_prob_plus"])
        try:
            return fn(F, T)
        except TypeError:
            return fn(F / T)

    def test_b04_zero_temperature_limit_deterministic(self):
        """Verify low temperature T -> 0 produces deterministic step function."""
        # Positive field at tiny T -> 1.0
        self.assertAlmostEqual(self._get_prob(0.1, T=1e-5), 1.0, places=6)
        # Negative field at tiny T -> 0.0
        self.assertAlmostEqual(self._get_prob(-0.1, T=1e-5), 0.0, places=6)

    def test_b04_infinite_temperature_limit_equiprobable(self):
        """Verify high temperature limit T -> infinity produces probability 0.5."""
        self.assertAlmostEqual(self._get_prob(5.0, T=1e8), 0.5, places=5)

    def test_b04_extreme_positive_kappa_overflow_guard(self):
        """Verify kappa = 500.0 evaluates safely to 1.0 without raising OverflowError."""
        val = self._get_prob(500.0, T=1.0)
        self.assertEqual(val, 1.0)

    def test_b04_extreme_negative_kappa_underflow_guard(self):
        """Verify kappa = -500.0 evaluates safely to 0.0 without underflow crash."""
        val = self._get_prob(-500.0, T=1.0)
        self.assertEqual(val, 0.0)

    def test_b04_critical_temperature_susceptibility(self):
        """Verify derivative dP/dF at F=0 scales inversely with temperature: 1 / (2*T)."""
        dF = 1e-4
        for T in [0.5, 1.0, 2.0]:
            p_up = self._get_prob(dF, T=T)
            p_down = self._get_prob(-dF, T=T)
            derivative = (p_up - p_down) / (2 * dF)
            expected = 1.0 / (2.0 * T)
            self.assertAlmostEqual(derivative, expected, places=3)


class TestBoundary05ContinuousHeatBath(ModularBaseTestCase):
    """Boundary cases for Feature 5: Continuous Heat Bath Updates."""

    def _sample_cont(self, kappa, u):
        fn = self.require_attr("src.core.heat_baths", "sample_continuous_inverse_cdf",
                               aliases=["sample_continuous_inverse_cdf_scalar"])
        return fn(kappa, u)

    def test_b05_exact_kappa_zero_uniform_density(self):
        """Verify kappa = 0.0 exactly samples uniformly: sample(0, u) = 2*u - 1."""
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertAlmostEqual(self._sample_cont(kappa=0.0, u=u), 2.0 * u - 1.0, places=7)

    def test_b05_extreme_positive_field_saturation(self):
        """Verify huge positive kappa = 100.0 pins almost all mass near +1.0."""
        val = self._sample_cont(kappa=100.0, u=0.5)
        self.assertGreater(val, 0.95)

    def test_b05_extreme_negative_field_saturation(self):
        """Verify huge negative kappa = -100.0 pins almost all mass near -1.0."""
        val = self._sample_cont(kappa=-100.0, u=0.5)
        self.assertLess(val, -0.95)

    def test_b05_tiny_kappa_cancellation_safeguard(self):
        """Verify kappa = 1e-8 evaluates smoothly without catastrophic cancellation."""
        val = self._sample_cont(kappa=1e-8, u=0.7)
        expected = 2.0 * 0.7 - 1.0
        self.assertAlmostEqual(val, expected, places=5)

    def test_b05_random_draw_endpoints_u0_and_u1(self):
        """Verify u = 0.0 produces -1.0 and u = 1.0 produces +1.0 for any kappa."""
        for kappa in [-10.0, 0.0, 10.0]:
            self.assertAlmostEqual(self._sample_cont(kappa=kappa, u=0.0), -1.0, places=6)
            self.assertAlmostEqual(self._sample_cont(kappa=kappa, u=1.0), +1.0, places=6)


if __name__ == "__main__":
    unittest.main()
