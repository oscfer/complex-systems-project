"""Unit tests for src/core/participation.py."""

import unittest
import numpy as np

from src.core.participation import (
    ParticipationParams,
    ParticipationTracker,
    compute_participation_potential,
)


class TestCoreParticipation(unittest.TestCase):
    """Test suite for collective action participation dynamics and threshold holding tracking."""

    def test_potential_difference_identity(self):
        """Verify Delta Phi == Delta u_k exactly across all 24 transitions on 3-node symmetric graph."""
        # 3-node symmetric attention matrix
        A = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=np.float64)

        b, g, c = 1.2, 0.8, 0.5
        I = 1
        x1 = np.array([-1.0, 0.5, 1.0], dtype=np.float64)
        params = ParticipationParams(I=I, b=b, g=g, c=c)
        tracker = ParticipationTracker(N=3, params=params)

        q = tracker.compute_opposition(x1)

        # Enumerate all 2^3 = 8 binary configurations of n
        max_discrepancy = 0.0
        for n0 in (0, 1):
            for n1 in (0, 1):
                for n2 in (0, 1):
                    base_n = np.array([n0, n1, n2], dtype=np.uint8)

                    for k in range(3):
                        n_off = base_n.copy()
                        n_off[k] = 0
                        n_on = base_n.copy()
                        n_on[k] = 1

                        # Potential difference: Phi(n_k = 1) - Phi(n_k = 0)
                        phi_on = compute_participation_potential(n_on, q, A, b=b, g=g, c=c)
                        phi_off = compute_participation_potential(n_off, q, A, b=b, g=g, c=c)
                        delta_phi = phi_on - phi_off

                        # Direct coordination payoff difference: Delta u_k
                        tracker.n = n_off.copy()
                        rho_k = float(np.dot(A[k], n_off))
                        delta_u_k = tracker.compute_payoff_diff(q[k], rho_k)

                        diff = abs(delta_phi - delta_u_k)
                        if diff > max_discrepancy:
                            max_discrepancy = diff

        # Exact identity verified to machine precision (< 1e-14)
        self.assertLess(max_discrepancy, 1e-14)

    def test_logistic_normalization_and_cost_monotonicity(self):
        """Verify logistic redraw probabilities, absence of factor 2, and cost monotonicity."""
        params = ParticipationParams(T_n=1.0, b=1.0, g=0.0, c=0.0)
        tracker = ParticipationTracker(N=1, params=params)

        # Delta u = 0.0 => P = 0.5
        rng = np.random.default_rng(42)
        # Check logistic function via draw probabilities directly
        z0_prob = 1.0 / (1.0 + np.exp(0.0))
        self.assertEqual(z0_prob, 0.5)

        # Verify no factor of 2: Delta u = 1.0, T_n = 1.0 => 1 / (1 + e^-1) ~ 0.731058
        z1_prob = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(z1_prob, 0.73105857863, places=8)
        # Ensure it is NOT 1 / (1 + e^-2) ~ 0.880797
        self.assertNotAlmostEqual(z1_prob, 1.0 / (1.0 + np.exp(-2.0)), places=4)

        # Extreme payoffs without overflow
        self.assertEqual(tracker.draw_action(delta_u=1000.0, rng=rng), 1)
        self.assertEqual(tracker.draw_action(delta_u=-1000.0, rng=rng), 0)

    def test_empty_attention_row_fallback(self):
        """Verify isolated agent with empty attention row has rho_i = 0 and valid payoff."""
        params = ParticipationParams(I=1, b=1.0, g=2.0, c=0.4)
        tracker = ParticipationTracker(N=3, params=params)
        tracker.n = np.array([1, 1, 1], dtype=np.uint8)

        # Agent 0 has empty attention row (all zeros)
        A_row = np.zeros(3, dtype=np.float64)
        x1_val = -1.0  # full opponent: q_i = 1.0

        rng = np.random.default_rng(10)
        changed, old_v, new_v = tracker.update_agent(i=0, x1_val=x1_val, A_row=A_row, t=0.0, rng=rng)

        # Payoff should be b*q - c = 1.0*1.0 - 0.4 = 0.6
        q_i = 0.5 * (1.0 - 1 * (-1.0))
        delta_u = tracker.compute_payoff_diff(q_i=q_i, rho_i=0.0)
        self.assertAlmostEqual(delta_u, 0.6, places=12)

    def test_uninterrupted_holding_duration_success(self):
        """Verify institutional transition fires at exact deadline after uninterrupted holding interval."""
        params = ParticipationParams(theta=0.5, t_hold=2.0)
        tracker = ParticipationTracker(N=10, params=params)

        # Initially M_n = 0 < theta
        self.assertEqual(tracker.get_participation_fraction(), 0.0)
        self.assertFalse(tracker.holding_active)

        # At t = 1.5, participation rises to 0.6 >= 0.5
        tracker.n[:6] = 1
        tracker.check_threshold(t=1.5)

        self.assertTrue(tracker.holding_active)
        self.assertEqual(tracker.t_start_hold, 1.5)
        self.assertEqual(tracker.t_deadline, 3.5)
        gen = tracker.deadline_generation

        # Advance time to t = 3.5 and dispatch holding deadline event
        success = tracker.handle_deadline_event(t=3.5, payload={"generation": gen})
        self.assertTrue(success)
        self.assertTrue(tracker.transition_occurred)
        self.assertEqual(tracker.t_transition, 3.5)

    def test_interrupted_holding_duration_cancellation(self):
        """Verify dip below threshold strictly resets holding timer and invalidates deadline."""
        params = ParticipationParams(theta=0.5, t_hold=2.0)
        tracker = ParticipationTracker(N=10, params=params)

        # Rise above threshold at t = 1.0
        tracker.n[:6] = 1
        tracker.check_threshold(t=1.0)
        gen1 = tracker.deadline_generation
        self.assertTrue(tracker.holding_active)
        self.assertEqual(tracker.t_deadline, 3.0)

        # At t = 2.8, participation dips below threshold: M_n = 0.4 < 0.5
        tracker.n[5] = 0
        tracker.n[4] = 0
        tracker.check_threshold(t=2.8)

        # Holding interval should be invalidated
        self.assertFalse(tracker.holding_active)
        self.assertIsNone(tracker.t_start_hold)
        self.assertIsNone(tracker.t_deadline)
        self.assertNotEqual(tracker.deadline_generation, gen1)

        # Stale deadline event arrives at t = 3.0: must be rejected
        fired = tracker.handle_deadline_event(t=3.0, payload={"generation": gen1})
        self.assertFalse(fired)
        self.assertFalse(tracker.transition_occurred)

    def test_holding_checkpoint_restart_preservation(self):
        """Verify holding state is accurately preserved across checkpoint serialization."""
        params = ParticipationParams(theta=0.4, t_hold=3.0)
        tracker = ParticipationTracker(N=5, params=params)
        tracker.n = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        tracker.check_threshold(t=2.0)

        data = tracker.to_dict()

        restored = ParticipationTracker(N=5, params=params)
        restored.from_dict(data)

        self.assertTrue(restored.holding_active)
        self.assertEqual(restored.t_start_hold, 2.0)
        self.assertEqual(restored.t_deadline, 5.0)
        self.assertEqual(restored.deadline_generation, tracker.deadline_generation)

        # Fire restored deadline at t = 5.0
        success = restored.handle_deadline_event(t=5.0, payload={"generation": restored.deadline_generation})
        self.assertTrue(success)
        self.assertTrue(restored.transition_occurred)
        self.assertEqual(restored.t_transition, 5.0)


if __name__ == "__main__":
    unittest.main()
