"""Tier 2: Boundary & Corner Cases for Features 24 to 29.

Boundary 24: Influence Concentration Metrics
Boundary 25: Rank Persistence Metrics
Boundary 26: Graph Community Detection
Boundary 27: Component Tracking
Boundary 28: Degree-Preserving Null Model
Boundary 29: Population Discrepancy Metric
"""

import unittest
import numpy as np
from tests.e2e.common import ModularBaseTestCase


class TestBoundary24InfluenceMetrics(ModularBaseTestCase):
    """Boundary cases for Feature 24: Influence Concentration Metrics."""

    def test_b24_empty_graph_concentration_returns_nan(self):
        """Verify concentration C and C_N return NaN (never 0.0) when W = 0."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.zeros((10, 10), dtype=np.int32)
        metrics = calc_metrics(W)
        self.assertTrue(np.isnan(metrics["C"]))
        self.assertTrue(np.isnan(metrics["C_N"]))

    def test_b24_monopolistic_hub_c_n_equals_one(self):
        """Verify single monopolistic leader receiving all attention yields C_N = 1.0."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        N = 5
        W = np.zeros((N, N), dtype=np.int32)
        # Everyone attends exclusively to node 0
        for i in range(1, N):
            W[i, 0] = 5
        metrics = calc_metrics(W)
        self.assertAlmostEqual(metrics["C_N"], 1.0, places=6)

    def test_b24_perfectly_uniform_shares_c_n_equals_zero(self):
        """Verify perfectly uniform attention distribution yields C_N = 0.0."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        N = 6
        # Complete balanced directed cycle
        W = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            W[i, (i + 1) % N] = 1
        metrics = calc_metrics(W)
        self.assertAlmostEqual(metrics["C_N"], 0.0, places=6)

    def test_b24_single_agent_population_returns_nan(self):
        """Verify single agent population (N = 1) returns NaN for C_N (denominator N - 1 is zero)."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.zeros((1, 1), dtype=np.int32)
        metrics = calc_metrics(W)
        self.assertTrue(np.isnan(metrics["C_N"]))

    def test_b24_all_isolated_nodes_f_active_zero(self):
        """Verify active-row fraction f_active is 0.0 for completely isolated network."""
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        W = np.zeros((4, 4), dtype=np.int32)
        metrics = calc_metrics(W)
        self.assertEqual(metrics["f_active"], 0.0)


class TestBoundary25RankPersistence(ModularBaseTestCase):
    """Boundary cases for Feature 25: Rank Persistence Metrics."""

    def test_b25_constant_rank_vectors_spearman_returns_nan(self):
        """Verify Spearman correlation on constant rank vector returns NaN."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r1 = np.full(10, 5.0)
        r2 = np.arange(10, dtype=np.float64)
        corr = spearman(r1, r2)
        self.assertTrue(np.isnan(corr))

    def test_b25_all_tied_ranks_spearman_returns_nan(self):
        """Verify two all-tied vectors return NaN (zero variance)."""
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        r1 = np.ones(8)
        r2 = np.ones(8)
        self.assertTrue(np.isnan(spearman(r1, r2)))

    def test_b25_all_tied_top_fraction_includes_all_nodes(self):
        """Verify top-fraction overlap includes all nodes when all received attentions are tied."""
        overlap = self.require_attr("src.analytics.influence", "top_fraction_overlap")
        r1 = np.ones(10)
        r2 = np.ones(10)
        res = overlap(r1, r2, fraction=0.10)
        # Even with fraction 10%, all 10 nodes are tied for first place -> size must be 10
        self.assertEqual(res["size_t1"], 10)
        self.assertEqual(res["size_t2"], 10)

    def test_b25_top_fraction_zero_percent_empty(self):
        """Verify top fraction 0.0 returns empty set or raises ValueError."""
        overlap = self.require_attr("src.analytics.influence", "top_fraction_overlap")
        r = np.arange(10, dtype=np.float64)
        try:
            res = overlap(r, r, fraction=0.0)
            self.assertEqual(res["size_t1"], 0)
        except ValueError:
            pass

    def test_b25_top_fraction_one_hundred_percent_full(self):
        """Verify top fraction 1.0 includes all N nodes."""
        overlap = self.require_attr("src.analytics.influence", "top_fraction_overlap")
        N = 12
        r = np.arange(N, dtype=np.float64)
        res = overlap(r, r, fraction=1.0)
        self.assertEqual(res["size_t1"], N)


class TestBoundary26CommunityDetection(ModularBaseTestCase):
    """Boundary cases for Feature 26: Graph Community Detection."""

    def test_b26_empty_graph_modularity_returns_nan(self):
        """Verify modularity Q is NaN or undefined when graph has zero edges (2m = 0)."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.zeros((6, 6), dtype=np.float64)
        res = detect(G, gamma=1.0)
        self.assertTrue(np.isnan(res["modularity_Q"]) or res["modularity_Q"] == 0.0)

    def test_b26_complete_clique_modularity(self):
        """Verify complete clique graph yields single community at gamma = 1.0."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        N = 8
        G = np.ones((N, N)) - np.eye(N)
        res = detect(G, gamma=1.0)
        # All in one community
        unique_comms = len(np.unique(res["partition"]))
        self.assertEqual(unique_comms, 1)

    def test_b26_disconnected_islands_partition_exact(self):
        """Verify two completely disconnected components form exactly 2 communities."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.zeros((8, 8))
        G[0:4, 0:4] = 1.0
        G[4:8, 4:8] = 1.0
        np.fill_diagonal(G, 0.0)
        res = detect(G, gamma=1.0)
        p = res["partition"]
        self.assertEqual(p[0], p[1])
        self.assertEqual(p[4], p[5])
        self.assertNotEqual(p[0], p[4])

    def test_b26_all_isolated_nodes_singleton_communities(self):
        """Verify all isolated nodes are partitioned into singleton communities."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.zeros((5, 5))
        res = detect(G, gamma=1.0)
        # Each node in own community
        self.assertEqual(len(np.unique(res["partition"])), 5)

    def test_b26_extreme_resolution_gamma_limit(self):
        """Verify very high gamma (e.g. 50.0) splits nodes into singletons."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.ones((6, 6)) - np.eye(6)
        res = detect(G, gamma=50.0)
        self.assertEqual(len(np.unique(res["partition"])), 6)


class TestBoundary27ComponentTracking(ModularBaseTestCase):
    """Boundary cases for Feature 27: Component Tracking."""

    def test_b27_empty_graph_all_singletons_wcc_n(self):
        """Verify graph with 0 edges has N weakly connected components."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        W = np.zeros((10, 10), dtype=np.int32)
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 10)
        self.assertEqual(res["num_scc"], 10)
        self.assertAlmostEqual(res["isolated_fraction"], 1.0)

    def test_b27_complete_graph_single_component_wcc_one(self):
        """Verify complete graph has exactly 1 WCC and 1 SCC."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        N = 7
        W = np.ones((N, N), dtype=np.int32) - np.eye(N, dtype=np.int32)
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 1)
        self.assertEqual(res["num_scc"], 1)
        self.assertEqual(res["largest_wcc_fraction"], 1.0)

    def test_b27_directed_cycle_wcc_one_scc_one(self):
        """Verify directed cycle of N nodes has 1 WCC and 1 SCC."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        N = 5
        W = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            W[i, (i + 1) % N] = 1
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 1)
        self.assertEqual(res["num_scc"], 1)

    def test_b27_directed_star_hub_wcc_one_scc_n(self):
        """Verify directed out-star (hub attends to everyone, no back ties) has 1 WCC but N SCCs."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        N = 5
        W = np.zeros((N, N), dtype=np.int32)
        for j in range(1, N):
            W[0, j] = 1
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 1)
        self.assertEqual(res["num_scc"], N)

    def test_b27_two_disconnected_equal_clusters(self):
        """Verify two equal clusters have largest component fraction 0.5."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        W = np.zeros((8, 8), dtype=np.int32)
        W[0:4, 0:4] = 1
        W[4:8, 4:8] = 1
        np.fill_diagonal(W, 0)
        res = calc_cc(W)
        self.assertEqual(res["num_wcc"], 2)
        self.assertAlmostEqual(res["largest_wcc_fraction"], 0.5)


class TestBoundary28NullModel(ModularBaseTestCase):
    """Boundary cases for Feature 28: Degree-Preserving Null Model."""

    def test_b28_empty_graph_null_model_returns_empty(self):
        """Verify null model on empty graph returns identical empty graph."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.zeros((5, 5))
        G_null = swap_edges(G, num_swaps=10)
        np.testing.assert_array_equal(G, G_null)

    def test_b28_single_edge_null_model_returns_same(self):
        """Verify null model on single edge graph cannot swap and returns original edge."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.zeros((4, 4))
        G[0, 1] = G[1, 0] = 3.0
        G_null = swap_edges(G, num_swaps=10)
        np.testing.assert_array_equal(G, G_null)

    def test_b28_triangle_clique_null_model_isomorphic(self):
        """Verify triangle K_3 cannot swap without creating self-loops or multi-edges."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.ones((3, 3)) - np.eye(3)
        G_null = swap_edges(G, num_swaps=10)
        np.testing.assert_array_equal(G, G_null)

    def test_b28_weight_multiset_exact_preservation(self):
        """Verify exact multiset of positive weights is conserved in null model."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.zeros((6, 6))
        G[0, 1] = G[1, 0] = 10.0
        G[2, 3] = G[3, 2] = 20.0
        G[4, 5] = G[5, 4] = 30.0
        G_null = swap_edges(G, num_swaps=5, seed=42)
        w_orig = sorted([10.0, 20.0, 30.0])
        w_null = sorted(G_null[np.triu_indices(6, k=1)][G_null[np.triu_indices(6, k=1)] > 0])
        np.testing.assert_allclose(w_orig, w_null)

    def test_b28_zero_swaps_returns_identical_support(self):
        """Verify num_swaps = 0 returns exact input matrix."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        G = np.array([[0, 2], [2, 0]], dtype=np.float64)
        G_null = swap_edges(G, num_swaps=0)
        np.testing.assert_array_equal(G, G_null)


class TestBoundary29PopulationDiscrepancy(ModularBaseTestCase):
    """Boundary cases for Feature 29: Population Discrepancy Metric."""

    def test_b29_continuous_states_all_zero_discrepancy_zero(self):
        """Verify continuous zero states x = y = 0.0 yield d_2 = 0.0 (not 0.5)."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.zeros((10, 2))
        y = np.zeros((10, 2))
        self.assertAlmostEqual(calc_d2(x, y), 0.0, places=7)

    def test_b29_extreme_discrepancy_x_plus1_y_minus1_equals_one(self):
        """Verify complete opposite opinions across all agents yields d_2 = 1.0."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.ones((5, 1))
        y = -np.ones((5, 1))
        self.assertAlmostEqual(calc_d2(x, y), 1.0, places=7)

    def test_b29_single_agent_single_dimension_discrepancy(self):
        """Verify minimal case N = 1, D = 1 computes exact squared distance."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.array([[0.5]])
        y = np.array([[-0.5]])
        # (0.5 - (-0.5))^2 / 4 = 1.0 / 4 = 0.25
        self.assertAlmostEqual(calc_d2(x, y), 0.25, places=6)

    def test_b29_continuous_uniform_expectation_third(self):
        """Verify expectation for independent Uniform[-1, 1] states is E[d_2] = 1/6."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        rng = np.random.default_rng(42)
        N = 20_000
        x = rng.uniform(-1.0, 1.0, size=(N, 1))
        y = rng.uniform(-1.0, 1.0, size=(N, 1))
        d2 = calc_d2(x, y)
        self.assertAlmostEqual(d2, 1.0 / 6.0, delta=0.01)

    def test_b29_orthogonal_continuous_states_discrepancy(self):
        """Verify orthogonal vectors: x = [1, 0], y = [0, 1] gives d_2 = (1 + 1)/(4*1*2) = 0.25."""
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        x = np.array([[1.0, 0.0]])
        y = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(calc_d2(x, y), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
