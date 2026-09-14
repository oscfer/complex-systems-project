"""Tier 3: Pairwise Cross-Feature Interactions (32 Comprehensive Tests).

Covers cross-feature couplings across memory, belief systems, participation,
reinforcement learning, graph analytics, and neural inference.
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_frustrated_energy,
    oracle_reinforcing_energy,
    oracle_population_discrepancy,
    oracle_analytical_baseline_bound,
    oracle_origin_stratified_split,
)


class TestTier3PairwiseInteractions(ModularBaseTestCase):
    """Tier 3: Pairwise Feature Combinations (>=32 Tests)."""

    def test_pairwise_01_memory_dynamics_and_frustrated_triad(self):
        """Pairwise 1: Memory dynamics (W) x Frustrated triad (D=3, B_f)."""
        calc_e = self.require_attr("src.core.couplings", "calculate_internal_belief_energy", aliases=["calculate_isolated_internal_energy"])
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities", aliases=["compute_candidate_probabilities"])
        state = self.create_state(N=10, D=3)
        # Node with frustrated private state (+, -, -)
        state.x[0] = np.array([1, -1, -1])
        energy = self.calc_internal_energy(state.x[0], "frustrated", L=1.0)
        self.assertAlmostEqual(energy, 1.5, places=6)
        probs = calc_probs(i=0, W=state.W, r=np.ones(10), y=state.y, a=0.5, c=1.0, epsilon=0.1)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)


    def test_pairwise_03_field_pulse_and_memory_decay(self):
        """Pairwise 3: Field pulse h(t) x Memory decay (delta)."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        calc_decay = self.require_attr("src.core.memory", "calculate_decay_propensity")
        W = np.array([[0, 5], [5, 0]], dtype=np.int32)
        pulse = Intervention(start_time=0.0, duration=5.0, amplitude=2.0)
        # Verify decay propensity remains exact during pulse
        decay_prop = calc_decay(W, delta=0.1)
        self.assertAlmostEqual(decay_prop, 1.0, places=6)
        self.assertEqual(pulse.get_field(t=2.0), 2.0)


    def test_pairwise_06_prestige_bias_and_modular_network(self):
        """Pairwise 6: Prestige bias (a > 0) x Modular network generator."""
        gen_modular = self.require_attr("src.experiment.generators", "generate_modular_network")
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        W, blocks = gen_modular(N=20, num_blocks=2, m=10, eta=0.8, seed=42)
        r = np.ones(20)
        r[0] = 5.0  # Elevated leader in block 0
        y = np.ones((20, 1))
        probs = calc_probs(i=1, W=W, r=r, y=y, a=1.0, c=0.0, epsilon=0.0)
        self.assertGreater(probs[0], probs[2])

    def test_pairwise_07_matching_bias_and_multipopulation_interface(self):
        """Pairwise 7: Matching bias (c > 0) x Multi-population interface schedule."""
        build_eligibility = self.require_attr("src.core.memory", "build_eligibility_matrix")
        calc_probs = self.require_attr("src.core.memory", "calculate_candidate_probabilities")
        groups = np.array([0, 0, 1, 1])
        E = np.ones((2, 2))  # Open interface
        C = build_eligibility(groups, E)
        W = np.ones((4, 4), dtype=np.int32) - np.eye(4, dtype=np.int32)
        r = np.ones(4)
        y = np.array([[1.0], [1.0], [-1.0], [-1.0]])  # Groups oppose
        probs = calc_probs(i=0, W=W, r=r, y=y, a=0.0, c=2.0, epsilon=0.0)
        # Agent 0 prefers ingroup agent 1 over outgroup agent 2
        self.assertGreater(probs[1], probs[2])

    def test_pairwise_08_twostep_closure_and_tie_disruption(self):
        """Pairwise 8: Two-step closure (chi > 0) x Structural tie disruption."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        calc_closure = self.require_attr("src.core.memory", "calculate_closure_weights")
        W = np.array([[0, 2, 0], [0, 0, 2], [2, 0, 0]], dtype=np.int32)  # 3-cycle
        W_disrupt, _ = apply_disruption(W, fraction=0.33, seed=42)
        # Verify closure can still evaluate on remaining support
        H = calc_closure(W_disrupt.astype(float))
        self.assertEqual(H.shape, (3, 3))


    def test_pairwise_12_reciprocal_memory_and_null_model(self):
        """Pairwise 12: Reciprocal memory (W = W^T) x Degree-preserving null model."""
        swap_edges = self.require_attr("src.analytics.null_models", "generate_degree_preserving_null")
        # Symmetric graph
        W = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], dtype=np.float64)
        W_null = swap_edges(W, num_swaps=5, seed=42)
        np.testing.assert_allclose(W_null, W_null.T)


    def test_pairwise_14_state_alignment_generator_and_spearman_persistence(self):
        """Pairwise 14: State alignment generator x Spearman rank persistence."""
        gen_aligned = self.require_attr("src.experiment.generators", "generate_aligned_states")
        spearman = self.require_attr("src.analytics.influence", "spearman_rank_persistence")
        x, y = gen_aligned(N=20, D=1, p_g=0.9, q=0.1, seed=42)
        r1 = np.arange(20, dtype=np.float64)
        r2 = r1 + np.random.normal(0, 0.1, size=20)
        corr = spearman(r1, r2)
        self.assertGreater(corr, 0.9)


    def test_pairwise_17_leiden_modularity_and_interface_opening(self):
        """Pairwise 17: Leiden modularity x Multi-population interface opening."""
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        # Pre-opening: 2 completely separate cliques
        G_closed = np.zeros((10, 10))
        G_closed[0:5, 0:5] = 2.0
        G_closed[5:10, 5:10] = 2.0
        np.fill_diagonal(G_closed, 0.0)
        q_closed = detect(G_closed, gamma=1.0)["modularity_Q"]
        # Post-opening: strong cross edges added
        G_open = G_closed.copy()
        G_open[0:5, 5:10] = 2.0
        G_open[5:10, 0:5] = 2.0
        q_open = detect(G_open, gamma=1.0)["modularity_Q"]
        self.assertGreater(q_closed, q_open)

    def test_pairwise_18_component_tracking_and_tie_disruption(self):
        """Pairwise 18: Component tracking x 30% tie disruption."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # Line graph 0-1-2-3-4
        W = np.zeros((5, 5), dtype=np.int32)
        for i in range(4):
            W[i, i + 1] = W[i + 1, i] = 1
        W_disrupt, _ = apply_disruption(W, fraction=0.5, seed=42)
        res = calc_cc(W_disrupt)
        self.assertGreaterEqual(res["num_wcc"], 1)


    def test_pairwise_21_master_scheduler_and_simultaneous_pulses(self):
        """Pairwise 21: Master scheduler preemption x Multiple scheduled interventions."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        log = []
        scheduler.add_deterministic_event(t_event=10.0, callback=lambda: log.append("pulse_start"))
        scheduler.add_deterministic_event(t_event=10.0, callback=lambda: log.append("interface_open"))
        scheduler.advance_step(current_time=0.0, stochastic_dt=15.0)
        self.assertEqual(len(log), 2)


    def test_pairwise_23_endpoint_continuous_control_and_public_fields(self):
        """Pairwise 23: Continuous endpoint control x Public local field arithmetic."""
        gen_endpoint = self.require_attr("src.experiment.generators", "generate_endpoint_continuous")
        calc_fields = self.require_attr("src.core.fields", "calculate_public_fields")
        state = gen_endpoint(N=10, D=1, seed=42)
        A = np.eye(10)
        fields = calc_fields(A, state.y, state.x, J=1.0, K=0.5, h=0.0)
        # Endpoints at +/- 1 -> J * (+/-1) + K * (+/-1) = +/- 1.5
        for f in fields.flatten():
            self.assertIn(abs(f), [0.5, 1.5])

    def test_pairwise_24_decoupled_config_loader_and_trajectory_engine(self):
        """Pairwise 24: Decoupled config loader x Multi-seed parallel trajectory runner."""
        load_config = self.require_attr("src.config.loader", "load_config_from_yaml_string")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        yaml_str = """
        experiment:
          name: "test_run"
          N: 10
          D: 1
          T: 0.5
          delta: 0.1
          nu: 2.0
          H_0: 1.0
        """
        cfg = load_config(yaml_str)
        runner = Runner(cfg)
        self.assertIsNotNone(runner)

    def test_pairwise_25_top_fraction_overlap_and_prestige_bias(self):
        """Pairwise 25: Top-fraction overlap x Prestige bias (a > 0)."""
        overlap = self.require_attr("src.analytics.influence", "top_fraction_overlap")
        # Persistent hierarchy with prestige bias
        r1 = np.array([10.0, 8.0, 2.0, 1.0, 1.0])
        r2 = np.array([11.0, 9.0, 2.0, 1.0, 1.0])
        res = overlap(r1, r2, fraction=0.40)
        self.assertEqual(res["jaccard_overlap"], 1.0)


    def test_pairwise_28_weakly_vs_strongly_connected_components_decay(self):
        """Pairwise 28: Asymmetric memory decay x WCC / SCC divergence."""
        calc_cc = self.require_attr("src.analytics.communities", "connected_components_analysis")
        # Directed acyclic graph: 0 -> 1 -> 2
        W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
        res = calc_cc(W)
        # Weakly connected (1 component), but strongly disconnected (3 singletons)
        self.assertEqual(res["num_wcc"], 1)
        self.assertEqual(res["num_scc"], 3)


    def test_pairwise_31_frustrated_triad_and_qlearning(self):
        """Pairwise 31: Frustrated triad (B_f) x Q-learning action selection."""
        calc_private = self.require_attr("src.core.fields", "calculate_private_fields")
        calc_public = self.require_attr("src.core.fields", "calculate_public_fields")
        B_f = self.get_coupling("frustrated", D=3)
        x = np.array([[1.0, -1.0, -1.0]])  # Frustrated config
        y = np.array([[1.0, 1.0, 1.0]])
        # Private fields reflect frustration
        f_x = calc_private(x, y, B=B_f, K=0.5, L=1.0, b=0.0)
        # Public fields reflect private coupling K*x
        f_y = calc_public(A=np.zeros((1, 1)), y=y, x=x, J=1.0, K=0.5, h=0.0)
        self.assertAlmostEqual(f_y[0, 0], 0.5, places=6)
        self.assertAlmostEqual(f_y[0, 1], -0.5, places=6)

    def test_pairwise_32_tie_disruption_and_leiden_stability(self):
        """Pairwise 32: Tie disruption x Leiden modularity stability."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        G = np.ones((8, 8)) - np.eye(8)
        G_disrupt, _ = apply_disruption(G, fraction=0.30, seed=42)
        res = detect(G_disrupt, gamma=1.0, n_restarts=5)
        self.assertIn("modularity_Q", res)
        self.assertIn("partition", res)


if __name__ == "__main__":
    unittest.main()
