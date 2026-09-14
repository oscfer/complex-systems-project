"""Unit tests for src/core/q_learning.py."""

import math
import unittest
import numpy as np

from src.core.q_learning import (
    QLearningParams,
    action_to_index,
    compute_immediate_reward,
    index_to_action,
    init_q_table,
    select_action,
    softmax_probabilities,
    step_q_learning,
    update_chosen_q_value,
)


class TestCoreQLearning(unittest.TestCase):
    """Test suite for Q-learning expressed-state updates, softmax selection, and Glauber equivalence."""

    def test_q_table_initialization_and_indexing(self):
        """Verify Q-table dimensions, initial zeros, and bidirectional action indexing."""
        N, D = 15, 3
        q_table = init_q_table(N, D)
        self.assertEqual(q_table.shape, (15, 3, 2))
        self.assertEqual(q_table.dtype, np.float64)
        self.assertTrue(np.all(q_table == 0.0))

        # Index mapping verification: {-1 -> 0, +1 -> 1}
        self.assertEqual(action_to_index(-1), 0)
        self.assertEqual(action_to_index(1), 1)
        self.assertEqual(index_to_action(0), -1)
        self.assertEqual(index_to_action(1), 1)

    def test_softmax_probabilities_and_stability(self):
        """Verify softmax probabilities, logistic reduction, and overflow resistance."""
        T_Q = 1.0

        # Equal Q-values: P(+1) == P(-1) == 0.5
        q_pair_equal = np.array([0.0, 0.0], dtype=np.float64)
        p_minus, p_plus = softmax_probabilities(q_pair_equal, T_Q)
        self.assertEqual(p_minus, 0.5)
        self.assertEqual(p_plus, 0.5)

        # Known difference: Delta Q = 1.0 => P(+1) = 1 / (1 + e^-1) ~ 0.7310585786
        q_pair_diff = np.array([1.0, 2.0], dtype=np.float64)
        p_minus, p_plus = softmax_probabilities(q_pair_diff, T_Q)
        expected_plus = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(p_plus, expected_plus, places=10)
        self.assertAlmostEqual(p_minus + p_plus, 1.0, places=12)

        # Extreme Q-values without overflow
        q_pair_extreme = np.array([0.0, 1000.0], dtype=np.float64)
        p_minus, p_plus = softmax_probabilities(q_pair_extreme, T_Q)
        self.assertEqual(p_plus, 1.0)
        self.assertEqual(p_minus, 0.0)

        # Large negative difference
        q_pair_neg = np.array([1000.0, 0.0], dtype=np.float64)
        p_minus, p_plus = softmax_probabilities(q_pair_neg, T_Q)
        self.assertEqual(p_plus, 0.0)
        self.assertEqual(p_minus, 1.0)

    def test_chosen_action_update_and_counterfactual_isolation(self):
        """Verify TD update modifies chosen action while unchosen action is strictly untouched."""
        q_pair = np.array([0.0, 0.0], dtype=np.float64)
        chosen_action = 1
        reward = 1.5
        eta = 0.1

        # Pre-update check
        unchosen_val_before = q_pair[0]

        update_chosen_q_value(q_pair, chosen_action, reward, eta)

        # Chosen action Q(+1) updated: 0.0 + 0.1 * (1.5 - 0.0) = 0.15
        self.assertAlmostEqual(q_pair[1], 0.15, places=12)

        # Unchosen action Q(-1) MUST remain strictly byte-for-byte identical
        self.assertEqual(q_pair[0], unchosen_val_before)
        self.assertEqual(q_pair[0], 0.0)

    def test_q_value_convergence_under_constant_field(self):
        """Verify Q-value converges exponentially to s * F^y under repeated exposure."""
        q_pair = np.array([0.0, 0.0], dtype=np.float64)
        field = 2.0
        eta = 0.05

        # Repeatedly select action +1 with constant reward R = +1 * 2.0 = 2.0
        for _ in range(350):
            reward = compute_immediate_reward(action=1, field=field)
            update_chosen_q_value(q_pair, chosen_action=1, reward=reward, eta=eta)

        # Should converge to 2.0 within 1e-6
        self.assertAlmostEqual(q_pair[1], 2.0, places=6)
        # Unchosen action still 0.0
        self.assertEqual(q_pair[0], 0.0)

    def test_analytical_glauber_equivalence(self):
        """Verify softmax selection with Q(+-1) = +-F^y strictly recovers the Glauber heat bath."""
        fields = [-2.5, -1.0, 0.0, 0.5, 1.8, 3.0]
        temperatures = [0.2, 0.5, 1.0, 2.0]

        for fy in fields:
            for T in temperatures:
                # Equilibrium learned values: Q(+1) = +F^y, Q(-1) = -F^y
                q_pair = np.array([-fy, fy], dtype=np.float64)
                _, p_softmax_plus = softmax_probabilities(q_pair, T_Q=T)

                # Glauber probability: 1 / (1 + exp(-2 * F^y / T))
                p_glauber_plus = 1.0 / (1.0 + math.exp(-2.0 * fy / T))

                self.assertAlmostEqual(p_softmax_plus, p_glauber_plus, places=12)

    def test_step_q_learning_assigns_expressed_state(self):
        """Verify step_q_learning samples action, computes reward, updates Q, and assigns y."""
        N, D = 5, 2
        q_table = init_q_table(N, D)
        y = np.zeros((N, D), dtype=np.float64)
        params = QLearningParams(eta=0.2, T_Q=1.0)
        rng = np.random.default_rng(42)

        action, reward = step_q_learning(
            i=2,
            alpha=1,
            Q_table=q_table,
            y=y,
            field=1.2,
            params=params,
            rng=rng,
        )

        self.assertIn(action, (-1, 1))
        self.assertEqual(y[2, 1], action)
        self.assertAlmostEqual(reward, action * 1.2, places=10)

        # Check Q-table value
        idx = action_to_index(action)
        self.assertAlmostEqual(q_table[2, 1, idx], 0.2 * reward, places=10)
        # Unchosen index remains 0.0
        self.assertEqual(q_table[2, 1, 1 - idx], 0.0)


if __name__ == "__main__":
    unittest.main()
