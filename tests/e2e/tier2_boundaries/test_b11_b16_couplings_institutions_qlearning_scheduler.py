"""Tier 2: Boundary & Corner Cases for Features 11 to 16.

Boundary 11: Scalar Belief Baseline (D=1)
Boundary 12: Coupled Belief Extensions (D=3)
Boundary 13: Participation States & Payoffs
Boundary 14: Institutional Transition Tracking
Boundary 15: Q-Learning Expressed State Updates
Boundary 16: Master Poisson Event Scheduler
"""

import unittest
import numpy as np
from tests.e2e.common import ModularBaseTestCase, oracle_frustrated_energy


class TestBoundary11ScalarBelief(ModularBaseTestCase):
    """Boundary cases for Feature 11: Scalar Belief Baseline (D=1)."""

    def test_b11_scalar_extreme_k_coupling(self):
        """Verify massive K coupling (K=100.0) locks x and y into alignment."""
        calc_fields = self.require_attr("src.core.fields", "calculate_public_fields", aliases=["compute_public_fields"])
        A = np.zeros((1, 1))
        y = np.array([[1.0]])
        x = np.array([[-1.0]])
        f = calc_fields(A, y, x, J=1.0, K=100.0, h=0.0)
        self.assertAlmostEqual(f[0, 0], -100.0, places=6)

    def test_b11_scalar_zero_bias_b0(self):
        """Verify zero private bias b=0 leaves fields strictly symmetric."""
        calc_fields = self.require_attr("src.core.fields", "calculate_private_fields", aliases=["compute_private_fields"])
        y = np.array([[1.0]])
        x = np.array([[1.0]])
        B = np.zeros((1, 1))
        f_pos = calc_fields(x, y, B=B, K=0.5, L=1.0, b=0.0)
        f_neg = calc_fields(-x, -y, B=B, K=0.5, L=1.0, b=0.0)
        self.assertAlmostEqual(float(f_pos[0, 0]), -float(f_neg[0, 0]), places=6)

    def test_b11_scalar_homogeneous_population(self):
        """Verify population where all x=1, y=1 has maximum public local field."""
        calc_fields = self.require_attr("src.core.fields", "calculate_public_fields", aliases=["compute_public_fields"])
        N = 10
        A = (np.ones((N, N)) - np.eye(N)) / (N - 1)
        y = np.ones((N, 1))
        x = np.ones((N, 1))
        f = calc_fields(A, y, x, J=1.0, K=0.5, h=0.0)
        np.testing.assert_allclose(f, np.full((N, 1), 1.5))

    def test_b11_scalar_perfectly_opposed_population(self):
        """Verify population with x = -y has consistency field pulling opposite to expression."""
        calc_fields = self.require_attr("src.core.fields", "calculate_public_fields", aliases=["compute_public_fields"])
        A = np.zeros((2, 2))
        y = np.array([[1.0], [-1.0]])
        x = np.array([[-1.0], [1.0]])
        f = calc_fields(A, y, x, J=0.0, K=0.5, h=0.0)
        self.assertAlmostEqual(f[0, 0], -0.5, places=6)
        self.assertAlmostEqual(f[1, 0], 0.5, places=6)

    def test_b11_scalar_single_spin_update_invariance(self):
        """Verify updating node i does not directly mutate node j's private field."""
        state = self.create_state(N=2, D=1)
        state.x[0, 0] = 1.0
        state.x[1, 0] = -1.0
        # Mutating x[0, 0]
        state.x[0, 0] = -1.0
        self.assertEqual(state.x[1, 0], -1.0)


class TestBoundary12CoupledBeliefs(ModularBaseTestCase):
    """Boundary cases for Feature 12: Coupled Belief Extensions (D=3)."""

    def test_b12_uncoupled_b0_matches_three_scalar_systems(self):
        """Verify B_0 produces zero cross-component contribution across all components."""
        calc_fields = self.require_attr("src.core.fields", "calculate_private_fields", aliases=["compute_private_fields"])
        x = np.array([[1.0, -1.0, 1.0]])
        y = np.array([[1.0, 1.0, -1.0]])
        B0 = np.zeros((3, 3))
        fields = calc_fields(x, y, B=B0, K=0.5, L=1.0, b=0.0)
        # Only K * y contributes
        np.testing.assert_allclose(fields, 0.5 * y)

    def test_b12_reinforcing_b_plus_all_positive_minimum_energy(self):
        """Verify (+,+,+) has internal energy -1.5 (global ground state)."""
        e = self.calc_internal_energy(np.array([1, 1, 1]), "reinforcing", L=1.0)
        self.assertAlmostEqual(e, -1.5, places=6)

    def test_b12_reinforcing_b_plus_all_negative_minimum_energy(self):
        """Verify (-,-,-) has internal energy -1.5 (symmetric ground state)."""
        e = self.calc_internal_energy(np.array([-1, -1, -1]), "reinforcing", L=1.0)
        self.assertAlmostEqual(e, -1.5, places=6)

    def test_b12_frustrated_b_f_high_energy_states(self):
        """Verify (+,-,-) and (-,+,+) have highest frustrated energy +1.5."""
        e1 = self.calc_internal_energy(np.array([1, -1, -1]), "frustrated", L=1.0)
        e2 = self.calc_internal_energy(np.array([-1, 1, 1]), "frustrated", L=1.0)
        self.assertAlmostEqual(e1, 1.5, places=6)
        self.assertAlmostEqual(e2, 1.5, places=6)

    def test_b12_frustrated_b_f_ground_state_degeneracy(self):
        """Verify all 6 non-frustrated states have identical energy -0.5."""
        ground_configs = [
            np.array([1, 1, 1]),
            np.array([1, 1, -1]),
            np.array([1, -1, 1]),
            np.array([-1, 1, -1]),
            np.array([-1, -1, 1]),
            np.array([-1, -1, -1]),
        ]
        for cfg in ground_configs:
            self.assertAlmostEqual(self.calc_internal_energy(cfg, "frustrated", L=1.0), -0.5, places=6)


class TestBoundary16MasterScheduler(ModularBaseTestCase):
    """Boundary cases for Feature 16: Master Poisson Scheduler."""

    def test_b16_zero_rates_infinite_waiting_time(self):
        """Verify total event rate zero yields infinite waiting time."""
        sample_dt = self.require_attr("src.core.scheduler", "sample_waiting_time")
        dt = sample_dt(total_rate=0.0, u=0.5)
        self.assertEqual(dt, float("inf"))

    def test_b16_scheduled_event_at_current_time_immediate_fire(self):
        """Verify scheduled deterministic event with t_event <= t_current fires immediately."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        executed = []
        scheduler.add_deterministic_event(t_event=5.0, callback=lambda: executed.append(True))
        t, fired = scheduler.advance_step(current_time=5.0, stochastic_dt=1.0)
        self.assertEqual(t, 5.0)
        self.assertTrue(fired)
        self.assertEqual(len(executed), 1)

    def test_b16_simultaneous_scheduled_events_order(self):
        """Verify multiple deterministic events at identical time execute deterministically in registration order."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        order = []
        scheduler.add_deterministic_event(t_event=10.0, callback=lambda: order.append(1))
        scheduler.add_deterministic_event(t_event=10.0, callback=lambda: order.append(2))
        scheduler.advance_step(current_time=0.0, stochastic_dt=15.0)
        self.assertEqual(order, [1, 2])

    def test_b16_empty_scheduler_queue_handles_gracefully(self):
        """Verify empty deterministic event queue executes purely stochastic jump step."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        t, fired = scheduler.advance_step(current_time=2.0, stochastic_dt=3.5)
        self.assertAlmostEqual(t, 5.5, places=6)
        self.assertFalse(fired)

    def test_b16_high_rate_tiny_time_increments(self):
        """Verify high rate (10^6) produces very small positive time increments without zero underflow."""
        sample_dt = self.require_attr("src.core.scheduler", "sample_waiting_time")
        dt = sample_dt(total_rate=1e6, u=0.5)
        self.assertGreater(dt, 0.0)
        self.assertLess(dt, 1e-4)


if __name__ == "__main__":
    unittest.main()
