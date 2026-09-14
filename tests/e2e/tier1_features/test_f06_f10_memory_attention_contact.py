"""Tier 1: Features 6 to 10 Tests.

Feature 6: Adaptive Interaction Memory W
Feature 7: Attention Normalization A_ij
Feature 8: Memory Reinforcement & Biases
Feature 9: Memory Decay & Deletion
Feature 10: Contact Mechanisms & Closure
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_attention_matrix,
    oracle_influence_metrics,
)


class TestFeature06AdaptiveMemory(ModularBaseTestCase):
    """Feature 6: Adaptive Interaction Memory Matrix W."""

    def test_f06_memory_matrix_integer_counts(self):
        """Verify memory matrix holds non-negative integer counts."""
        state = self.create_state(N=50, D=1)
        self.assertTrue(np.issubdtype(state.W.dtype, np.integer))
        self.assertTrue(np.all(state.W >= 0))

    def test_f06_memory_matrix_zero_diagonal_enforced(self):
        """Verify memory matrix diagonal W_ii is strictly zero (no self-encounters)."""
        state = self.create_state(N=30, D=1)
        self.assertTrue(np.all(np.diag(state.W) == 0))

    def test_f06_memory_matrix_shape_and_row_sums(self):
        """Verify W has shape (N, N) and out-degree row sums s_i match sum_k W_ik."""
        state = self.create_state(N=40, D=1)
        self.assertEqual(state.W.shape, (40, 40))
        calc_sums = self.require_attr("src.core.memory", "calculate_out_degrees", aliases=["compute_out_degrees"])
        s = calc_sums(state.W)
        np.testing.assert_array_equal(s, state.W.sum(axis=1))

    def test_f06_memory_single_encounter_increment(self):
        """Verify an encounter from i to j increments W_ij by exactly 1."""
        increment_mem = self.require_attr("src.core.memory", "record_encounter")
        W = np.zeros((5, 5), dtype=np.int32)
        increment_mem(W, i=1, j=3)
        self.assertEqual(W[1, 3], 1)
        self.assertEqual(W[3, 1], 0)  # Directed

    def test_f06_memory_total_mass_tracking(self):
        """Verify total memory mass S(t) = sum_{i!=j} W_ij is accurately tracked."""
        get_total_memory = self.require_attr("src.core.memory", "get_total_memory_mass")
        W = np.array([[0, 2, 1], [0, 0, 3], [1, 0, 0]], dtype=np.int32)
        self.assertEqual(get_total_memory(W), 7)


class TestFeature07AttentionNormalization(ModularBaseTestCase):
    """Feature 7: Attention Normalization A_ij = W_ij / s_i."""

    def test_f07_row_stochastic_normalization(self):
        """Verify active rows in A sum to exactly 1.0."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.array([[0, 3, 1], [2, 0, 2], [0, 0, 0]], dtype=np.int32)
        A, f_active = calc_attention(W)
        self.assertAlmostEqual(float(A[0, :].sum()), 1.0, places=7)
        self.assertAlmostEqual(float(A[1, :].sum()), 1.0, places=7)

    def test_f07_empty_memory_row_yields_zero_vector(self):
        """Verify empty memory row (s_i = 0) yields A_i* = 0 vector without NaN."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.zeros((4, 4), dtype=np.int32)
        A, f_active = calc_attention(W)
        self.assertTrue(np.all(A == 0.0))
        self.assertFalse(np.any(np.isnan(A)))

    def test_f07_received_attention_column_sum(self):
        """Verify received attention r_j equals column sums of A."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.array([[0, 2, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        A, _ = calc_attention(W)
        calc_r = self.require_attr("src.core.memory", "calculate_received_attention")
        r = calc_r(A)
        np.testing.assert_allclose(r, A.sum(axis=0), rtol=1e-7)

    def test_f07_active_row_fraction_tracking(self):
        """Verify f_active equals fraction of nodes with s_i > 0."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
        _, f_active = calc_attention(W)
        self.assertAlmostEqual(f_active, 2.0 / 3.0, places=7)

    def test_f07_total_received_attention_equals_n_active(self):
        """Verify sum_j r_j == N * f_active."""
        calc_attention = self.require_attr("src.core.memory", "calculate_attention_matrix")
        W = np.array([[0, 5, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32)
        A, f_active = calc_attention(W)
        r = A.sum(axis=0)
        self.assertAlmostEqual(float(r.sum()), 4 * f_active, places=7)


class TestFeature08MemoryReinforcement(ModularBaseTestCase):
    """Feature 8: Memory Reinforcement & Biases (Prestige a, Matching c, Exploration epsilon)."""

    def test_f08_prestige_bias_weighting(self):
        """Verify candidate probability increases monotonically with received attention r_j when a > 0."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        r = np.array([0.5, 2.0, 0.5])
        y = np.array([[1.0], [1.0], [1.0]])
        probs = calc_probs(i=0, W=W, r=r, y=y, a=1.0, c=0.0, epsilon=0.0)
        # Node 1 has higher received attention, so probs[1] > probs[2]
        self.assertGreater(probs[1], probs[2])

    def test_f08_matching_bias_state_similarity(self):
        """Verify candidate probability increases with state similarity s_ij when c > 0."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        r = np.array([1.0, 1.0, 1.0])
        y = np.array([[1.0], [1.0], [-1.0]])  # Node 0 aligns with Node 1, opposes Node 2
        probs = calc_probs(i=0, W=W, r=r, y=y, a=0.0, c=1.0, epsilon=0.0)
        self.assertGreater(probs[1], probs[2])

    def test_f08_exploration_parameter_weighting(self):
        """Verify epsilon = 1.0 produces uniform exploration across available targets."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 10, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        r = np.array([1.0, 10.0, 0.1])
        y = np.array([[1.0], [1.0], [-1.0]])
        probs = calc_probs(i=0, W=W, r=r, y=y, a=2.0, c=2.0, epsilon=1.0)
        # Target 1 and 2 should be equally likely: 0.5 each
        self.assertAlmostEqual(probs[1], 0.5, places=5)
        self.assertAlmostEqual(probs[2], 0.5, places=5)

    def test_f08_unbiased_uniform_candidate_selection(self):
        """Verify a = 0, c = 0 weights candidates proportional to existing memory W_ij."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W = np.array([[0, 3, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        r = np.array([1.0, 1.0, 1.0])
        y = np.array([[1.0], [1.0], [1.0]])
        probs = calc_probs(i=0, W=W, r=r, y=y, a=0.0, c=0.0, epsilon=0.0)
        # W_01 = 3, W_02 = 1 -> ratio 3:1
        self.assertAlmostEqual(probs[1] / probs[2], 3.0, places=5)

    def test_f08_candidate_selection_probability_normalization(self):
        """Verify candidate distribution P_i sums to exactly 1.0 with self-prob P_ii = 0."""
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        N = 10
        W = np.ones((N, N), dtype=np.int32) - np.eye(N, dtype=np.int32)
        r = np.ones(N)
        y = np.ones((N, 1))
        probs = calc_probs(i=3, W=W, r=r, y=y, a=0.5, c=1.0, epsilon=0.1)
        self.assertEqual(probs[3], 0.0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=7)


class TestFeature09MemoryDecay(ModularBaseTestCase):
    """Feature 9: Memory Decay & Deletion at Poisson rate delta."""

    def test_f09_memory_token_decay_rate(self):
        """Verify per-token forgetting propensity equals delta * W_ij."""
        calc_decay_propensity = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W = np.array([[0, 4, 1], [2, 0, 3], [0, 0, 0]], dtype=np.int32)
        delta = 0.1
        # Total tokens = 10 -> total propensity = 1.0
        propensity = calc_decay_propensity(W, delta=delta)
        self.assertAlmostEqual(propensity, 1.0, places=7)

    def test_f09_edge_extinction_when_memory_reaches_zero(self):
        """Verify decaying edge with W_ij = 1 sets W_ij to 0 and removes tie."""
        execute_decay = self.require_attr("src.core.memory", "execute_memory_decay")
        W = np.array([[0, 1], [0, 0]], dtype=np.int32)
        execute_decay(W, i=0, j=1)
        self.assertEqual(W[0, 1], 0)

    def test_f09_decay_propensity_proportional_to_total_memory(self):
        """Verify decay propensity scales strictly linearly with total memory tokens."""
        calc_decay_propensity = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W1 = np.ones((4, 4), dtype=np.int32) - np.eye(4, dtype=np.int32)  # 12 tokens
        W2 = 2 * W1  # 24 tokens
        p1 = calc_decay_propensity(W1, delta=0.5)
        p2 = calc_decay_propensity(W2, delta=0.5)
        self.assertAlmostEqual(p2 / p1, 2.0, places=7)

    def test_f09_uniform_token_selection_probability(self):
        """Verify each memory token has equal probability 1 / S(t) of being selected for decay."""
        sample_decay_edge = self.require_attr("src.core.memory", "sample_decay_edge")
        W = np.array([[0, 10, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
        edge = sample_decay_edge(W, seed=42)
        self.assertEqual(edge, (0, 1))

    def test_f09_decay_preserves_diagonal_zeros(self):
        """Verify memory decay never modifies diagonal W_ii = 0."""
        execute_decay = self.require_attr("src.core.memory", "execute_memory_decay")
        W = np.array([[0, 2], [1, 0]], dtype=np.int32)
        execute_decay(W, i=0, j=1)
        self.assertEqual(W[0, 0], 0)
        self.assertEqual(W[1, 1], 0)


class TestFeature10ContactMechanisms(ModularBaseTestCase):
    """Feature 10: Contact Mechanisms & Closure (Triadic closure, Reciprocal memory, Multi-pop)."""

    def test_f10_population_eligibility_matrix_filtering(self):
        """Verify multi-population contact matrix C_ij = E_{g_i, g_j} filters eligible targets."""
        build_eligibility = self.require_attr("src.core.memory", "build_eligibility_matrix")
        groups = np.array([0, 0, 1, 1])
        E = np.array([[1.0, 0.0], [0.0, 1.0]])  # Fully closed populations
        C = build_eligibility(groups, E)
        # Cross population entries must be 0.0
        self.assertEqual(C[0, 2], 0.0)
        self.assertEqual(C[0, 1], 1.0)
        self.assertEqual(C[0, 0], 0.0)  # Self-tie excluded

    def test_f10_null_encounter_attempt_when_eligibility_zero(self):
        """Verify node with zero eligible targets (c_i = 0) results in a null event."""
        sample_target = self.require_attr("src.core.memory", "sample_encounter_target")
        C = np.zeros((4, 4))
        W = np.zeros((4, 4), dtype=np.int32)
        r = np.ones(4)
        y = np.ones((4, 1))
        target = sample_target(i=0, C=C, W=W, r=r, y=y)
        self.assertIsNone(target)

    def test_f10_triadic_closure_two_step_path_matrix(self):
        """Verify two-step closure path matrix M_ik * M_kj correctly identifies friends-of-friends."""
        calc_closure = self.require_attr("src.core.memory", "calculate_closure_weights")
        # 0 -> 1 -> 2
        A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        H_star = calc_closure(A)
        # Node 2 is a two-step target for node 0
        self.assertGreater(H_star[0, 2], 0.0)

    def test_f10_closure_reinforces_only_direct_link(self):
        """Verify triadic closure encounter increments only direct link W_ij, not intermediate paths."""
        apply_closure_encounter = self.require_attr("src.core.memory", "apply_closure_encounter")
        W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        apply_closure_encounter(W, i=0, j=2)
        self.assertEqual(W[0, 2], 1)
        self.assertEqual(W[0, 1], 1)  # Intermediate path unchanged
        self.assertEqual(W[1, 2], 1)

    def test_f10_reciprocal_memory_pair_token_creation(self):
        """Verify reciprocal variant increments both W_ij and W_ji symmetrically."""
        record_reciprocal = self.require_attr("src.core.memory", "record_reciprocal_encounter")
        W = np.zeros((3, 3), dtype=np.int32)
        record_reciprocal(W, i=0, j=1)
        self.assertEqual(W[0, 1], 1)
        self.assertEqual(W[1, 0], 1)


if __name__ == "__main__":
    unittest.main()
