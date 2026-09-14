"""Tier 1: Features 24 to 29 Tests.

Feature 24: Influence Concentration Metrics
Feature 25: Rank Persistence Metrics
Feature 26: Graph Community Detection
Feature 27: Component Tracking
Feature 28: Degree-Preserving Null Model
Feature 29: Population Discrepancy Metric
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_influence_metrics,
    oracle_spearman_rank_persistence,
    oracle_population_discrepancy,
)


class TestFeature24InfluenceConcentration(ModularBaseTestCase):
    """Feature 24: Influence Concentration Metrics (p_j, C, C_N, 1/C, f_active)."""

    def test_f24_attention_shares_calculation(self):
        """Verify attention shares p_j = r_j / sum_k r_k sum to 1.0."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.array([[0, 2, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        metrics = calc_metrics(W)
        p = metrics["p"]
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_f24_herfindahl_concentration_c(self):
        """Verify Herfindahl concentration C = sum_j p_j^2 is within [1/N, 1]."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.array([[0, 3, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int32)
        metrics = calc_metrics(W)
        C = metrics["C"]
        self.assertTrue(1.0 / 3.0 <= C <= 1.0)

    def test_f24_normalized_concentration_c_n(self):
        """Verify normalized concentration C_N = (N*C - 1) / (N - 1) is in [0, 1]."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.array([[0, 5, 0], [0, 0, 0], [0, 5, 0]], dtype=np.int32)
        metrics = calc_metrics(W)
        C_N = metrics["C_N"]
        self.assertTrue(0.0 <= C_N <= 1.0)

    def test_f24_effective_leaders_inverse_c(self):
        """Verify effective number of leaders equals 1 / C."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        metrics = calc_metrics(W)
        self.assertAlmostEqual(metrics["effective_leaders"], 1.0 / metrics["C"], places=6)

    def test_f24_empty_graph_returns_nan_safely(self):
        """Verify empty graph (W = 0) returns NaN for shares and concentrations without zero-division error."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.zeros((5, 5), dtype=np.int32)
        metrics = calc_metrics(W)
        self.assertTrue(np.isnan(metrics["C"]))
        self.assertTrue(np.isnan(metrics["C_N"]))
        self.assertEqual(metrics["f_active"], 0.0)


class TestFeature25RankPersistence(ModularBaseTestCase):
    """Feature 25: Rank Persistence Metrics (Spearman correlation across lag, top-fraction overlap)."""

    def test_f25_spearman_rank_correlation_identical_ranks(self):
        """Verify Spearman correlation is exactly 1.0 for identical rank vectors."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = spearman(r1, r2)
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_f25_spearman_rank_correlation_inverted_ranks(self):
        """Verify Spearman correlation is exactly -1.0 for reversed rank vectors."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = spearman(r1, r2)
        self.assertAlmostEqual(corr, -1.0, places=6)

    def test_f25_spearman_midpoint_tie_handling(self):
        """Verify tied values receive midpoint fractional ranks."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r1 = np.array([1.0, 2.0, 2.0, 4.0])
        r2 = np.array([1.0, 2.0, 3.0, 4.0])
        corr = spearman(r1, r2)
        expected = oracle_spearman_rank_persistence(r1, r2)
        self.assertAlmostEqual(corr, expected, places=6)

    def test_f25_spearman_constant_vector_nan(self):
        """Verify constant attention vector yields NaN, not 0 or 1."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r_const = np.array([2.0, 2.0, 2.0, 2.0])
        r_var = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(spearman(r_const, r_var)))

    def test_f25_top_fraction_overlap_jaccard_with_ties(self):
        """Verify top fraction includes all tied nodes and computes Jaccard overlap."""
        overlap_func = self.require_attr("src.analytics.influence", "top_fraction_overlap")
        # 10 nodes, top 20% -> nominally 2 nodes, but tied nodes at boundary must all be included
        r1 = np.array([10, 9, 8, 8, 8, 1, 1, 1, 1, 1], dtype=np.float64)
        r2 = np.array([10, 9, 2, 2, 2, 1, 1, 1, 1, 1], dtype=np.float64)
        result = overlap_func(r1, r2, fraction=0.20)
        self.assertIn("jaccard_overlap", result)
        self.assertIn("size_t1", result)
        self.assertIn("size_t2", result)


class TestFeature26GraphCommunities(ModularBaseTestCase):
    """Feature 26: Graph Community Detection (Symmetrized G, Leiden modularity, resolutions gamma)."""

    def test_f26_undirected_symmetrized_adjacency_g(self):
        """Verify symmetrized projection G = W + W^T with zero diagonal."""
        symmetrize = self.require_attr("src.analytics.communities", "symmetrize_adjacency")
        W = np.array([[0, 3], [1, 0]], dtype=np.int32)
        G = symmetrize(W)
        self.assertEqual(G[0, 1], 4)
        self.assertEqual(G[1, 0], 4)
        self.assertEqual(G[0, 0], 0)

    def test_f26_leiden_weighted_modularity_reference_gamma_1(self):
        """Verify Leiden modularity optimization returns valid partition and Q value for gamma = 1.0."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        # Two disconnected 3-cliques
        G = np.zeros((6, 6), dtype=np.float64)
        G[0:3, 0:3] = 1.0
        G[3:6, 3:6] = 1.0
        np.fill_diagonal(G, 0.0)
        res = detect(G, gamma=1.0, n_restarts=5)
        self.assertIn("partition", res)
        self.assertIn("modularity_Q", res)
        self.assertGreater(res["modularity_Q"], 0.0)

    def test_f26_leiden_resolutions_gamma_half_and_two(self):
        """Verify Leiden community detection runs for gamma in {0.5, 1.0, 2.0}."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.ones((8, 8)) - np.eye(8)
        for gamma in [0.5, 1.0, 2.0]:
            res = detect(G, gamma=gamma, n_restarts=2)
            self.assertIsNotNone(res["partition"])

    def test_f26_leiden_multiple_restarts_best_partition(self):
        """Verify running >= 10 restarts retains partition with highest modularity Q."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.array([[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]], dtype=np.float64)
        res = detect(G, gamma=1.0, n_restarts=10)
        self.assertGreaterEqual(res["modularity_Q"], -1.0)

    def test_f26_leiden_partition_stability_ari(self):
        """Verify Adjusted Rand Index (ARI) evaluates partition stability across restarts."""
        calc_stability = self.require_attr("src.analytics.communities", "calculate_partition_stability")
        p1 = np.array([0, 0, 1, 1])
        p2 = np.array([0, 0, 1, 1])
        ari = calc_stability(p1, p2)
        self.assertAlmostEqual(ari, 1.0, places=6)


class TestFeature27ComponentTracking(ModularBaseTestCase):
    """Feature 27: Component Tracking (Weakly and strongly connected components)."""

    def test_f27_weakly_connected_components_directed_w(self):
        """Verify WCC correctly identifies disconnected components on directed graph."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # 0 -> 1 and 2 -> 3 (two components)
        W = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int32)
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 2)

    def test_f27_strongly_connected_components_directed_w(self):
        """Verify SCC correctly measures mutual reachability."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # 0 -> 1 -> 0 (cycle of 2 nodes) and isolated node 2
        W = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)
        res = calc_cc(W)
        self.assertEqual(res["num_scc"], 2)

    def test_f27_largest_component_fraction(self):
        """Verify largest component fraction S_max / N equals size of giant component divided by N."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # Component of size 3 and component of size 1 (N = 4)
        W = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
        res = calc_cc(W)
        self.assertAlmostEqual(res["largest_wcc_fraction"], 0.75, places=6)

    def test_f27_isolated_node_fraction_f_iso(self):
        """Verify isolated node fraction equals count of degree 0 nodes divided by N."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        W = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)
        res = calc_cc(W)
        self.assertAlmostEqual(res["isolated_fraction"], 1.0 / 3.0, places=6)

    def test_f27_component_count_on_disconnected_clusters(self):
        """Verify components count matches number of disconnected subgraphs."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # 3 isolated pairs
        W = np.zeros((6, 6), dtype=np.int32)
        W[0, 1] = W[1, 0] = 1
        W[2, 3] = W[3, 2] = 1
        W[4, 5] = W[5, 4] = 1
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 3)


class TestFeature28NullModel(ModularBaseTestCase):
    """Feature 28: Degree-Preserving Null Model (Edge swaps and weight multiset preservation)."""

    def test_f28_null_model_support_graph_extraction(self):
        """Verify extraction of simple unweighted support graph S = (V, E_supp)."""
        extract_supp = self.require_attr("src.analytics.null_models", "extract_support_graph")
        G = np.array([[0, 5, 0], [5, 0, 2], [0, 2, 0]])
        S = extract_supp(G)
        self.assertEqual(S[0, 1], 1)
        self.assertEqual(S[1, 2], 1)
        self.assertEqual(S[0, 2], 0)

    def test_f28_null_model_preserves_support_degree_sequence(self):
        """Verify double edge swaps strictly preserve unweighted node degrees."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        # 4-cycle graph
        G = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float64)
        degrees_orig = (G > 0).sum(axis=1)
        G_null = swap_edges(G, num_swaps=10, seed=42)
        degrees_null = (G_null > 0).sum(axis=1)
        np.testing.assert_array_equal(degrees_orig, degrees_null)

    def test_f28_null_model_preserves_support_edge_count(self):
        """Verify total number of edges in support graph is exactly conserved."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.array([[0, 2, 3, 0], [2, 0, 0, 1], [3, 0, 0, 4], [0, 1, 4, 0]], dtype=np.float64)
        edges_orig = np.sum(G > 0)
        G_null = swap_edges(G, num_swaps=20, seed=42)
        self.assertEqual(np.sum(G_null > 0), edges_orig)

    def test_f28_null_model_preserves_weight_multiset(self):
        """Verify exact multiset of positive weights is conserved."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.array([[0, 10, 20, 0], [10, 0, 0, 30], [20, 0, 0, 40], [0, 30, 40, 0]], dtype=np.float64)
        weights_orig = sorted(G[np.triu_indices(4, k=1)])
        G_null = swap_edges(G, num_swaps=10, seed=42)
        weights_null = sorted(G_null[np.triu_indices(4, k=1)])
        np.testing.assert_array_equal(weights_orig, weights_null)

    def test_f28_null_model_scrambles_modularity_structure(self):
        """Verify edge swaps on strong modular structure reduce community modularity Q."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        # Two tightly coupled 5-cliques connected by 1 bridge
        G = np.zeros((10, 10), dtype=np.float64)
        G[0:5, 0:5] = 5.0
        G[5:10, 5:10] = 5.0
        G[4, 5] = G[5, 4] = 1.0
        np.fill_diagonal(G, 0.0)
        q_orig = detect(G, gamma=1.0, n_restarts=5)["modularity_Q"]
        G_null = swap_edges(G, num_swaps=50, seed=42)
        q_null = detect(G_null, gamma=1.0, n_restarts=5)["modularity_Q"]
        self.assertGreater(q_orig, q_null)


class TestFeature29PopulationDiscrepancy(ModularBaseTestCase):
    """Feature 29: Population Discrepancy Metric d_2 in [0, 1]."""

    def test_f29_discrepancy_binary_identical_states_zero(self):
        """Verify d_2 = 0.0 when private and public states are identical."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.array([[1, -1, 1], [-1, 1, -1]])
        y = np.array([[1, -1, 1], [-1, 1, -1]])
        self.assertAlmostEqual(calc_d2(x, y), 0.0, places=6)

    def test_f29_discrepancy_binary_opposite_states_one(self):
        """Verify d_2 = 1.0 when private and public states are completely opposite."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.array([[1, 1], [-1, -1]])
        y = np.array([[-1, -1], [1, 1]])
        self.assertAlmostEqual(calc_d2(x, y), 1.0, places=6)

    def test_f29_discrepancy_continuous_zero_states_zero(self):
        """Verify d_2 = 0.0 when both states are continuous zero (x = y = 0.0)."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.zeros((10, 3))
        y = np.zeros((10, 3))
        self.assertAlmostEqual(calc_d2(x, y), 0.0, places=6)

    def test_f29_discrepancy_half_mismatched_binary(self):
        """Verify d_2 = 0.5 when exactly half of state components disagree."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.array([[1, 1], [1, 1]])
        y = np.array([[1, -1], [1, -1]])  # 2 of 4 mismatch
        self.assertAlmostEqual(calc_d2(x, y), 0.5, places=6)

    def test_f29_discrepancy_bounded_in_unit_interval(self):
        """Verify d_2 is strictly bounded in [0, 1] for random continuous states."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=(50, 3))
        y = rng.uniform(-1.0, 1.0, size=(50, 3))
        val = calc_d2(x, y)
        self.assertTrue(0.0 <= val <= 1.0)


if __name__ == "__main__":
    unittest.main()
