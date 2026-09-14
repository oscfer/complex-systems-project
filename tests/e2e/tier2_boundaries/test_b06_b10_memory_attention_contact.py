"""Tier 2: Boundary & Corner Cases for Features 6 to 10.

Boundary 6: Adaptive Interaction Memory W
Boundary 7: Attention Normalization A_ij
Boundary 8: Memory Reinforcement & Biases
Boundary 9: Memory Decay & Deletion
Boundary 10: Contact Mechanisms & Closure
"""

import unittest
import numpy as np
from tests.e2e.common import ModularBaseTestCase


class TestBoundary06AdaptiveMemory(ModularBaseTestCase):
    """Boundary cases for Feature 6: Adaptive Interaction Memory W."""

    def test_b06_completely_empty_memory_matrix(self):
        """Verify memory matrix at t=0 has all zeros."""
        state = self.create_state(N=10, D=1)
        self.assertTrue(np.all(state.W == 0))

    def test_b06_saturated_memory_matrix_large_counts(self):
        """Verify memory matrix handles large count values (e.g. 10^6) without integer overflow."""
        state = self.create_state(N=5, D=1)
        state.W[0, 1] = 1_000_000
        self.assertEqual(state.W[0, 1], 1_000_000)

    def test_b06_self_encounter_attempt_forbidden(self):
        """Verify attempting to increment self-encounter raises error or is ignored."""
        record_encounter = self.require_attr("src.core.memory", "record_encounter")
        W = np.zeros((4, 4), dtype=np.int32)
        with self.assertRaises((ValueError, AssertionError)):
            record_encounter(W, i=2, j=2)

    def test_b06_single_active_edge_isolated(self):
        """Verify graph with exactly one directed edge operates correctly."""
        W = np.zeros((5, 5), dtype=np.int32)
        W[0, 1] = 3
        calc_sums = self.require_attr("src.core.memory", "calculate_out_degrees")
        s = calc_sums(W)
        self.assertEqual(s[0], 3)
        self.assertEqual(np.sum(s[1:]), 0)

    def test_b06_dense_complete_memory_matrix(self):
        """Verify memory matrix where all pairs (i != j) have positive weights."""
        N = 10
        W = np.ones((N, N), dtype=np.int32) - np.eye(N, dtype=np.int32)
        calc_sums = self.require_attr("src.core.memory", "calculate_out_degrees")
        s = calc_sums(W)
        np.testing.assert_array_equal(s, np.full(N, N - 1))


class TestBoundary07AttentionNormalization(ModularBaseTestCase):
    """Boundary cases for Feature 7: Attention Normalization."""

    def test_b07_all_isolated_nodes_f_active_zero(self):
        """Verify when all nodes are isolated, f_active is exactly 0.0."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.zeros((8, 8), dtype=np.int32)
        A, f_active = calc_attention(W)
        self.assertEqual(f_active, 0.0)
        self.assertTrue(np.all(A == 0.0))

    def test_b07_single_edge_row_normalized_to_one(self):
        """Verify node attending to only one neighbor has attention weight exactly 1.0."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.zeros((4, 4), dtype=np.int32)
        W[0, 2] = 42  # Only one target
        A, _ = calc_attention(W)
        self.assertEqual(A[0, 2], 1.0)
        self.assertEqual(A[0, :].sum(), 1.0)

    def test_b07_equal_weights_uniform_attention(self):
        """Verify node with equal memory counts to k neighbors splits attention as 1/k."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.zeros((5, 5), dtype=np.int32)
        W[0, 1] = W[0, 2] = W[0, 3] = W[0, 4] = 7  # 4 neighbors
        A, _ = calc_attention(W)
        np.testing.assert_allclose(A[0, 1:], np.full(4, 0.25))

    def test_b07_extreme_weight_asymmetry_row(self):
        """Verify extreme memory ratio (e.g. 1 vs 10^5) evaluates accurately in attention."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.zeros((3, 3), dtype=np.int64)
        W[0, 1] = 1
        W[0, 2] = 100_000
        A, _ = calc_attention(W)
        self.assertAlmostEqual(A[0, 2], 100000.0 / 100001.0, places=7)

    def test_b07_received_attention_sum_zero_when_w_zero(self):
        """Verify received attention vector r is all zeros when W = 0."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        calc_r = self.require_attr("src.core.memory", "calculate_received_attention")
        W = np.zeros((6, 6), dtype=np.int32)
        A, _ = calc_attention(W)
        r = calc_r(A)
        self.assertEqual(float(r.sum()), 0.0)


class TestBoundary08MemoryReinforcement(ModularBaseTestCase):
    """Boundary cases for Feature 8: Memory Reinforcement Biases."""

    def test_b08_pure_exploration_epsilon_one(self):
        """Verify epsilon = 1.0 ignores prestige and matching biases completely."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 100, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        r = np.array([0.0, 10.0, 0.1])
        y = np.array([[1.0], [1.0], [-1.0]])
        probs = calc_probs(i=0, W=W, r=r, y=y, a=10.0, c=10.0, epsilon=1.0)
        # Uniform over remaining 2 nodes
        self.assertAlmostEqual(probs[1], 0.5, places=5)
        self.assertAlmostEqual(probs[2], 0.5, places=5)

    def test_b08_zero_prestige_bias_a0(self):
        """Verify a = 0.0 disables received attention advantage."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        r = np.array([0.0, 100.0, 0.01])  # Massive difference in r
        y = np.array([[1.0], [1.0], [1.0]])
        probs = calc_probs(i=0, W=W, r=r, y=y, a=0.0, c=0.0, epsilon=0.0)
        # With a=0 and identical W and y, targets 1 and 2 must have identical probability
        self.assertAlmostEqual(probs[1], probs[2], places=5)

    def test_b08_zero_matching_bias_c0(self):
        """Verify c = 0.0 disables homophily / state similarity influence."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        r = np.array([1.0, 1.0, 1.0])
        y = np.array([[1.0], [1.0], [-1.0]])  # Node 1 aligns, Node 2 opposes
        probs = calc_probs(i=0, W=W, r=r, y=y, a=0.0, c=0.0, epsilon=0.0)
        self.assertAlmostEqual(probs[1], probs[2], places=5)


    def test_b08_fallback_to_uniform_when_no_eligible_ties(self):
        """Verify fallback V_i = U_i when agent has no remembered interactions."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.zeros((4, 4), dtype=np.int32)
        r = np.zeros(4)
        y = np.ones((4, 1))
        probs = calc_probs(i=0, W=W, r=r, y=y, a=1.0, c=1.0, epsilon=0.1)
        # Must fall back to uniform 1/3 over 3 other nodes
        for j in [1, 2, 3]:
            self.assertAlmostEqual(probs[j], 1.0 / 3.0, places=5)


class TestBoundary09MemoryDecay(ModularBaseTestCase):
    """Boundary cases for Feature 9: Memory Decay & Forgetting."""

    def test_b09_zero_decay_rate_delta_zero(self):
        """Verify delta = 0 yields zero forgetting propensity (infinite memory)."""
        calc_decay_propensity = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W = np.ones((5, 5), dtype=np.int32)
        prop = calc_decay_propensity(W, delta=0.0)
        self.assertEqual(prop, 0.0)

    def test_b09_large_decay_rate_instant_forgetting(self):
        """Verify large delta = 100.0 yields high decay propensity."""
        calc_decay_propensity = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W = np.array([[0, 2], [1, 0]], dtype=np.int32)
        prop = calc_decay_propensity(W, delta=100.0)
        self.assertAlmostEqual(prop, 300.0, places=6)

    def test_b09_decay_on_single_token_edge_extinguishes_edge(self):
        """Verify decay event on edge with weight 1 removes edge entirely (weight 0)."""
        execute_decay = self.require_attr("src.core.memory", "execute_memory_decay")
        W = np.array([[0, 1], [0, 0]], dtype=np.int32)
        execute_decay(W, i=0, j=1)
        self.assertEqual(W[0, 1], 0)
        self.assertEqual(np.sum(W), 0)

    def test_b09_empty_memory_zero_decay_propensity(self):
        """Verify empty network W = 0 has decay propensity exactly 0.0."""
        calc_decay_propensity = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W = np.zeros((10, 10), dtype=np.int32)
        self.assertEqual(calc_decay_propensity(W, delta=0.5), 0.0)

    def test_b09_decay_preserves_zero_counts_non_negative(self):
        """Verify decay is never attempted on zero-weight edge (no negative counts)."""
        execute_decay = self.require_attr("src.core.memory", "execute_memory_decay")
        W = np.zeros((3, 3), dtype=np.int32)
        with self.assertRaises((ValueError, AssertionError)):
            execute_decay(W, i=0, j=1)


class TestBoundary10ContactMechanisms(ModularBaseTestCase):
    """Boundary cases for Feature 10: Contact Mechanisms & Closure."""

    def test_b10_completely_isolated_population_eligibility_zero(self):
        """Verify an isolated population with zero cross-eligibility generates null events."""
        build_eligibility = self.require_attr("src.core.memory", "build_eligibility_matrix")
        groups = np.array([0, 0, 1, 1])
        E = np.zeros((2, 2))  # Zero contact anywhere
        C = build_eligibility(groups, E)
        self.assertTrue(np.all(C == 0.0))


    def test_b10_multi_population_all_to_all_eligibility(self):
        """Verify all-to-all eligibility matrix E = 1 treats all agents as eligible partners."""
        build_eligibility = self.require_attr("src.core.memory", "build_eligibility_matrix")
        groups = np.array([0, 1, 2])
        E = np.ones((3, 3))
        C = build_eligibility(groups, E)
        np.testing.assert_array_equal(C, np.ones((3, 3)) - np.eye(3))


if __name__ == "__main__":
    unittest.main()
