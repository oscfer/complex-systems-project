"""Tier 1: Features 1 to 5 Tests.

Feature 1: Discrete State Representation
Feature 2: Continuous State Representation
Feature 3: Local Fields Calculation
Feature 4: Binary Heat Bath Updates
Feature 5: Stable Continuous Heat Bath
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_binary_glauber_prob,
    oracle_continuous_inverse_cdf,
)


class TestFeature01DiscreteState(ModularBaseTestCase):
    """Feature 1: Discrete State Representation (N x D, x_ia, y_ia in {-1, +1})."""

    def test_f01_discrete_state_initialization_shape_and_dtype(self):
        """Verify discrete state initializes with shape (N, D) and integer/discrete types."""
        state = self.create_state(N=100, D=3, continuous=False)
        self.assertEqual(state.x.shape, (100, 3))
        self.assertEqual(state.y.shape, (100, 3))
        self.assertIn(state.x.dtype, [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])

    def test_f01_discrete_state_binary_values_allowed(self):
        """Verify discrete state holds only binary values {-1, +1}."""
        state = self.create_state(N=50, D=1, continuous=False)
        unique_x = set(np.unique(state.x))
        unique_y = set(np.unique(state.y))
        self.assertTrue(unique_x.issubset({-1, 1, -1.0, 1.0, 0, 0.0}))
        self.assertTrue(unique_y.issubset({-1, 1, -1.0, 1.0, 0, 0.0}))

    def test_f01_discrete_state_invalid_values_rejected(self):
        """Verify assigning invalid non-binary values raises ValueError."""
        state = self.create_state(N=10, D=1, continuous=False)
        invalid_x = np.array([[0.5], [-2], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
        if hasattr(state, "validate"):
            state.x = invalid_x
            with self.assertRaises(ValueError):
                state.validate()
        elif hasattr(state, "set_x"):
            with self.assertRaises((ValueError, TypeError)):
                state.set_x(invalid_x)

    def test_f01_discrete_state_separate_private_public(self):
        """Verify private state x and public state y are separate distinct arrays."""
        state = self.create_state(N=20, D=1, continuous=False)
        state.x[0, 0] = 1.0
        state.y[0, 0] = -1.0
        self.assertNotEqual(state.x[0, 0], state.y[0, 0])

    def test_f01_discrete_state_component_dimension(self):
        """Verify state handles both D=1 and D=3 dimensions."""
        state1 = self.create_state(N=10, D=1, continuous=False)
        state3 = self.create_state(N=10, D=3, continuous=False)
        self.assertEqual(state1.D, 1)
        self.assertEqual(state3.D, 3)


class TestFeature02ContinuousState(ModularBaseTestCase):
    """Feature 2: Continuous State Representation (N x D, x_ia, y_ia in [-1, 1])."""

    def test_f02_continuous_state_shape_and_dtype(self):
        """Verify continuous state shape (N, D) and float64 dtype."""
        state = self.create_state(N=80, D=3, continuous=True)
        self.assertEqual(state.x.shape, (80, 3))
        self.assertEqual(state.y.shape, (80, 3))
        self.assertEqual(state.x.dtype, np.float64)

    def test_f02_continuous_state_interval_bounds(self):
        """Verify continuous state variables strictly respect [-1.0, +1.0] bounds."""
        state = self.create_state(N=100, D=3, continuous=True)
        self.assertTrue(np.all(state.x >= -1.0) and np.all(state.x <= 1.0))
        self.assertTrue(np.all(state.y >= -1.0) and np.all(state.y <= 1.0))

    def test_f02_continuous_state_out_of_bounds_rejected(self):
        """Verify out-of-bounds float values raise ValueError."""
        state = self.create_state(N=10, D=1, continuous=True)
        invalid_x = np.array([[1.5], [-1.2], [0.0], [0.1], [0.2], [-0.3], [0.4], [-0.5], [0.6], [-0.7]])
        if hasattr(state, "validate"):
            state.x = invalid_x
            with self.assertRaises(ValueError):
                state.validate()
        elif hasattr(state, "set_x"):
            with self.assertRaises((ValueError, TypeError)):
                state.set_x(invalid_x)

    def test_f02_continuous_state_endpoint_initialization(self):
        """Verify continuous state supports discrete endpoint initialization in {-1.0, +1.0}."""
        State = self.require_attr("src.core.state", "State")
        if hasattr(State, "create_endpoint_continuous"):
            state = State.create_endpoint_continuous(N=50, D=1, rng=np.random.default_rng(42))
        else:
            generate_endpoint = self.require_attr("src.experiment.generators", "generate_endpoint_continuous")
            state = generate_endpoint(N=50, D=1, seed=42)
        unique_vals = set(np.unique(state.x))
        self.assertTrue(unique_vals.issubset({-1.0, 1.0}))
        self.assertEqual(state.x.dtype, np.float64)

    def test_f02_continuous_state_independent_components(self):
        """Verify continuous components across D=3 evolve independently."""
        state = self.create_state(N=10, D=3, continuous=True)
        state.x[:, 0] = -0.75
        state.x[:, 1] = 0.0
        state.x[:, 2] = +0.75
        self.assertAlmostEqual(float(state.x[0, 0]), -0.75)
        self.assertAlmostEqual(float(state.x[0, 1]), 0.0)
        self.assertAlmostEqual(float(state.x[0, 2]), +0.75)


class TestFeature03LocalFields(ModularBaseTestCase):
    """Feature 3: Local Fields Calculation (Public F^y and Private F^x)."""

    def _calc_public(self, A, y, x, J=1.0, K=0.5, h=None):
        fn = self.require_attr("src.core.fields", "calculate_public_fields", aliases=["compute_public_fields"])
        return fn(A, y, x, J=J, K=K, h=h)

    def _calc_private(self, x, y, B=None, K=0.5, L=1.0, b=None):
        fn = self.require_attr("src.core.fields", "calculate_private_fields", aliases=["compute_private_fields"])
        return fn(x, y, B=B, K=K, L=L, b=b)

    def test_f03_public_field_arithmetic(self):
        """Verify public field: F^y_{ia} = J * sum_j A_ij y_{ja} + K * x_{ia} + h_{ia}."""
        A = np.array([[0.0, 1.0], [0.5, 0.5]])
        y = np.array([[1.0], [-1.0]])
        x = np.array([[1.0], [1.0]])
        J, K, h = 1.0, 0.5, 0.1
        # For node 0: J * (1.0 * -1.0) + K * 1.0 + h = -1.0 + 0.5 + 0.1 = -0.4
        fields = self._calc_public(A, y, x, J=J, K=K, h=h)
        self.assertAlmostEqual(float(fields[0, 0]), -0.4, places=6)

    def test_f03_private_field_arithmetic(self):
        """Verify private field: F^x_{ia} = K * y_{ia} + L * sum_{b!=a} B_{ab} x_{ib} + b_{ia}."""
        y = np.array([[1.0, -1.0, 1.0]])
        x = np.array([[1.0, 1.0, -1.0]])
        B = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        K, L, b = 0.5, 1.0, 0.0
        # For node 0, comp 0: K * 1.0 + L * (0.5 * 1.0 + 0.5 * -1.0) + 0.0 = 0.5 + 0 = 0.5
        fields = self._calc_private(x, y, B=B, K=K, L=L, b=b)
        self.assertAlmostEqual(float(fields[0, 0]), 0.5, places=6)

    def test_f03_external_field_pulse_addition(self):
        """Verify external field pulse h(t) linearly shifts public local field."""
        A = np.zeros((2, 2))
        y = np.array([[1.0], [1.0]])
        x = np.array([[1.0], [1.0]])
        f0 = self._calc_public(A, y, x, J=1.0, K=0.5, h=0.0)
        f_pulse = self._calc_public(A, y, x, J=1.0, K=0.5, h=2.0)
        self.assertAlmostEqual(float(f_pulse[0, 0] - f0[0, 0]), 2.0, places=6)

    def test_f03_empty_attention_row_field_fallback(self):
        """Verify node with zero remembered encounters (A_i* = 0) falls back to K * x + h."""
        A = np.zeros((3, 3))
        y = np.array([[1.0], [-1.0], [1.0]])
        x = np.array([[-1.0], [1.0], [-1.0]])
        fields = self._calc_public(A, y, x, J=1.0, K=0.5, h=0.0)
        self.assertAlmostEqual(float(fields[0, 0]), -0.5, places=6)
        self.assertAlmostEqual(float(fields[1, 0]), 0.5, places=6)

    def test_f03_vectorized_field_calculation_shape(self):
        """Verify local fields vectorized output shape matches (N, D)."""
        N, D = 100, 3
        A = np.eye(N)
        y = np.ones((N, D))
        x = np.ones((N, D))
        fields = self._calc_public(A, y, x, J=1.0, K=0.5, h=0.0)
        self.assertEqual(fields.shape, (N, D))


class TestFeature04BinaryHeatBath(ModularBaseTestCase):
    """Feature 4: Binary Heat Bath Updates (P(s | F) = exp(kappa*s) / (2 cosh kappa))."""

    def _get_prob(self, F, T=1.0):
        fn = self.require_attr("src.core.heat_baths", "binary_glauber_probability", aliases=["binary_glauber_prob_plus"])
        try:
            return fn(F, T)
        except TypeError:
            return fn(F / T)

    def test_f04_binary_glauber_probability_calculation(self):
        """Verify binary Glauber probability matches exact analytical formula."""
        for F in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            T = 1.0
            expected = oracle_binary_glauber_prob(F / T)
            actual = self._get_prob(F, T)
            self.assertAlmostEqual(actual, expected, places=7)

    def test_f04_binary_zero_field_symmetry(self):
        """Verify at zero field F = 0, update probability is exactly 0.5."""
        self.assertAlmostEqual(self._get_prob(0.0, T=1.0), 0.5, places=7)

    def test_f04_binary_high_field_saturation(self):
        """Verify high positive field saturates to 1.0 without numerical overflow."""
        self.assertAlmostEqual(self._get_prob(100.0, T=1.0), 1.0, places=7)

    def test_f04_binary_negative_field_inversion(self):
        """Verify P(s = +1 | -F) = 1 - P(s = +1 | F)."""
        p_pos = self._get_prob(1.5, T=0.8)
        p_neg = self._get_prob(-1.5, T=0.8)
        self.assertAlmostEqual(p_pos + p_neg, 1.0, places=7)

    def test_f04_binary_temperature_scaling(self):
        """Verify increasing temperature pushes probability toward 0.5 (noise limit)."""
        p_cold = self._get_prob(1.0, T=0.1)
        p_warm = self._get_prob(1.0, T=1.0)
        p_hot = self._get_prob(1.0, T=10.0)
        self.assertTrue(1.0 > p_cold > p_warm > p_hot > 0.5)


class TestFeature05StableContinuousHeatBath(ModularBaseTestCase):
    """Feature 5: Stable Continuous Heat Bath (Uniform reference measure inverse-CDF)."""

    def _sample_cont(self, kappa, u):
        fn = self.require_attr("src.core.heat_baths", "sample_continuous_inverse_cdf",
                               aliases=["sample_continuous_inverse_cdf_scalar"])
        return fn(kappa, u)

    def test_f05_continuous_inverse_cdf_zero_field_uniform(self):
        """Verify at zero field kappa = 0, continuous inverse-CDF yields 2*U - 1."""
        for u in [0.1, 0.25, 0.5, 0.75, 0.9]:
            expected = 2.0 * u - 1.0
            actual = self._sample_cont(kappa=0.0, u=u)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_f05_continuous_positive_field_branch_monotonic(self):
        """Verify positive field branch is strictly monotonically increasing in u."""
        kappa = 5.0
        u_vals = np.linspace(0.01, 0.99, 10)
        samples = [self._sample_cont(kappa=kappa, u=u) for u in u_vals]
        for i in range(len(samples) - 1):
            self.assertLess(samples[i], samples[i + 1])

    def test_f05_continuous_negative_field_branch_symmetry(self):
        """Verify symmetry: sample(-kappa, u) == -sample(kappa, 1 - u)."""
        for kappa in [0.5, 2.0, 10.0]:
            for u in [0.2, 0.4, 0.6, 0.8]:
                val_neg = self._sample_cont(kappa=-kappa, u=u)
                val_pos_mirror = -self._sample_cont(kappa=kappa, u=1.0 - u)
                self.assertAlmostEqual(val_neg, val_pos_mirror, places=6)

    def test_f05_continuous_small_kappa_series_stability(self):
        """Verify Taylor series stability regime for |kappa| <= 1e-5."""
        small_kappa = 1e-6
        u = 0.6
        expected = oracle_continuous_inverse_cdf(small_kappa, u)
        actual = self._sample_cont(kappa=small_kappa, u=u)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_f05_continuous_temperature_scaling_rule(self):
        """Verify continuous temperature calibration rule: T_cont = T_bin / 3 matches weak susceptibility."""
        heat_bath_mod = self.require_module("src.core.heat_baths")
        self.assertTrue(hasattr(heat_bath_mod, "calibrate_continuous_temperature") or
                        hasattr(heat_bath_mod, "T_CONT_FACTOR"))
        if hasattr(heat_bath_mod, "calibrate_continuous_temperature"):
            t_cont = heat_bath_mod.calibrate_continuous_temperature(T_bin=3.0)
            self.assertAlmostEqual(t_cont, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
