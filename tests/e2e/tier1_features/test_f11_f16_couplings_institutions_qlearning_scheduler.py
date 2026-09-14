"""Tier 1: Features 11 to 16 Tests.

Feature 11: Scalar Belief Baseline (D=1)
Feature 12: Coupled Belief Extensions (D=3)
Feature 13: Participation States & Payoffs
Feature 14: Institutional Transition Tracking
Feature 15: Q-Learning Expressed State Updates
Feature 16: Master Poisson Event Scheduler
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_frustrated_energy,
    oracle_reinforcing_energy,
    oracle_participation_payoff,
)


class TestFeature11ScalarBeliefBaseline(ModularBaseTestCase):
    """Feature 11: Scalar Belief Baseline (D=1)."""

    def test_f11_scalar_belief_zero_cross_coupling(self):
        """Verify D=1 has no cross-component coupling matrix (B = 0 or absent)."""
        get_coupling = self.require_attr("src.core.couplings", "get_coupling_matrix")
        B = get_coupling(D=1)
        self.assertEqual(B.shape, (1, 1))
        self.assertEqual(B[0, 0], 0.0)


    def test_f11_scalar_continuous_field_consistency(self):
        """Verify continuous scalar local field matches J*A*y + K*x + h."""
        calc_fields = self.require_attr("src.core.fields", "calculate_public_fields")
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[0.5], [-0.5]])
        x = np.array([[0.2], [-0.2]])
        fields = calc_fields(A, y, x, J=1.0, K=0.5, h=0.0)
        # Node 0: 1.0 * (-0.5) + 0.5 * 0.2 = -0.4
        self.assertAlmostEqual(float(fields[0, 0]), -0.4, places=6)


class TestFeature12CoupledBeliefs(ModularBaseTestCase):
    """Feature 12: Coupled Belief Extensions (D=3, B_0, B_+, B_f)."""

    def test_f12_uncoupled_matrix_b0(self):
        """Verify uncoupled matrix B_0 is completely zero."""
        B0 = self.get_coupling("uncoupled", D=3)
        self.assertEqual(B0.shape, (3, 3))
        self.assertTrue(np.all(B0 == 0.0))

    def test_f12_reinforcing_matrix_b_plus_symmetry_and_trace(self):
        """Verify reinforcing matrix B_+ has zero diagonal, off-diagonals 0.5, and is symmetric."""
        B_plus = self.get_coupling("reinforcing", D=3)
        self.assertEqual(B_plus.shape, (3, 3))
        self.assertTrue(np.all(np.diag(B_plus) == 0.0))
        np.testing.assert_allclose(B_plus, B_plus.T)
        self.assertAlmostEqual(B_plus[0, 1], 0.5)

    def test_f12_reinforcing_matrix_ground_state_energy(self):
        """Verify ground states (+,+,+) and (-,-,-) under B_+ yield energy -1.5 for L=1."""
        x_ground = np.array([1, 1, 1])
        energy = self.calc_internal_energy(x_ground, "reinforcing", L=1.0)
        self.assertAlmostEqual(energy, -1.5, places=6)

    def test_f12_frustrated_matrix_b_f_structure(self):
        """Verify frustrated matrix B_f structure: 1 couples + to 2 and 3, 2 couples - to 3."""
        B_f = self.get_coupling("frustrated", D=3)
        self.assertAlmostEqual(B_f[0, 1], 0.5)
        self.assertAlmostEqual(B_f[0, 2], 0.5)
        self.assertAlmostEqual(B_f[1, 2], -0.5)
        self.assertAlmostEqual(B_f[2, 1], -0.5)

    def test_f12_frustrated_matrix_energy_levels(self):
        """Verify B_f has 6 ground states at -0.5 and 2 frustrated states at +1.5."""
        # (+, -, -) is a frustrated state: E = +1.5
        e_frust = self.calc_internal_energy(np.array([1, -1, -1]), "frustrated", L=1.0)
        self.assertAlmostEqual(e_frust, 1.5, places=6)
        # (+, +, +) is one of 6 degenerate ground states: E = -0.5
        e_ground = self.calc_internal_energy(np.array([1, 1, 1]), "frustrated", L=1.0)
        self.assertAlmostEqual(e_ground, -0.5, places=6)


class TestFeature13ParticipationStates(ModularBaseTestCase):
    """Feature 13: Participation States & Payoffs (n_i in {0, 1}, Delta u_i)."""

    def test_f13_participation_state_binary_action(self):
        """Verify participation array n holds binary actions in {0, 1}."""
        state = self.create_state(N=50, D=1)
        unique_n = set(np.unique(state.n))
        self.assertTrue(unique_n.issubset({0, 1}))


class TestFeature15QLearningUpdates(ModularBaseTestCase):
    """Feature 15: Q-Learning Expressed State Updates (Chosen-action update, softmax selection)."""

    def test_f15_q_table_zero_initialization(self):
        """Verify Q-table initializes to 0.0 with shape (N, D, 2)."""
        state = self.create_state(N=20, D=3)
        self.assertEqual(state.Q.shape, (20, 3, 2))
        self.assertTrue(np.all(state.Q == 0.0))


class TestFeature16MasterScheduler(ModularBaseTestCase):
    """Feature 16: Master Poisson Event Scheduler with Preemption."""

    def test_f16_total_event_rate_summation(self):
        """Verify master Poisson rate R_total = R_x + R_y + R_enc + R_forget + R_n."""
        calc_total_rate = self.require_attr("src.core.scheduler", "calculate_total_event_rate")
        rates = calc_total_rate(N=100, D=1, tau_x=5.0, tau_y=1.0, nu=2.0, delta=0.1, total_memory=500, tau_n=1.0)
        # R_x = 100 * 1 / 5 = 20
        # R_y = 100 * 1 / 1 = 100
        # R_enc = 100 * 2.0 = 200
        # R_forget = 0.1 * 500 = 50
        # R_n = 100 * 1 = 100
        # Total = 470
        self.assertAlmostEqual(rates["total"], 470.0, places=6)

    def test_f16_exponential_time_increment_sampling(self):
        """Verify stochastic waiting time delta_t ~ Exp(R_total)."""
        sample_dt = self.require_attr("src.core.scheduler", "sample_waiting_time")
        dt = sample_dt(total_rate=100.0, u=0.5)
        expected_dt = -np.log(0.5) / 100.0
        self.assertAlmostEqual(dt, expected_dt, places=7)

    def test_f16_event_channel_selection_proportions(self):
        """Verify event channel probabilities match propensities."""
        calc_channel_probs = self.require_attr("src.core.scheduler", "calculate_channel_probabilities")
        propensities = {"x": 20.0, "y": 100.0, "enc": 200.0, "forget": 50.0}
        probs = calc_channel_probs(propensities)
        self.assertAlmostEqual(probs["y"], 100.0 / 370.0, places=6)

    def test_f16_deterministic_scheduled_event_preemption(self):
        """Verify stochastic increment is preempted when exceeding next scheduled deterministic event."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        executed = []
        scheduler.add_deterministic_event(t_event=5.0, callback=lambda: executed.append(True))
        # Current time 0.0, stochastic dt = 10.0 -> preempted at 5.0
        new_time, fired = scheduler.advance_step(current_time=0.0, stochastic_dt=10.0)
        self.assertAlmostEqual(new_time, 5.0, places=6)
        self.assertTrue(fired)
        self.assertEqual(len(executed), 1)

    def test_f16_clock_advancement_monotonicity(self):
        """Verify simulation time advances strictly monotonically."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        t = 0.0
        for _ in range(5):
            t, _ = scheduler.advance_step(current_time=t, stochastic_dt=1.2)
        self.assertAlmostEqual(t, 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
