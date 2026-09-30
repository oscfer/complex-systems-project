"""Comprehensive unit and integration tests for policy optimizers.

Covers:
1. Degree-based and influence-based heuristics on static snapshots and revealed ties.
2. Candidate subset selection, quotas, and fallback allocation.
3. Greedy budget allocation enforcing k distinct-node and B exposure limits.
4. Heuristic subset search optimizer with lexicographic multi-objective ranking.
5. FeedbackTargetShiftController under strict Observation contract and noise models.
6. Dynamic target shifting, alignment shutoff, and budget reallocation.
7. Budget exhaustion, release semantics, and fallback allocations.
8. Architectural compliance (Layer 2 purity, origin immutability, 0 core physics mutation).
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.control.contracts import (
    Action,
    ActionEntry,
    ControlProblem,
    Observation,
    ObservationMode,
    RolloutResult,
    TargetType,
)
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import (
    CompletePublicObservationModel,
    SampledPublicObservationModel,
)
from src.control.optimizers import (
    FeedbackTargetShiftController,
    HeuristicSubsetSearchOptimizer,
    OptimizationResult,
    PolicyCandidate,
    ShiftStrategy,
    allocate_greedy_budget,
    compute_candidate_score,
    compute_degree_scores,
    compute_influence_scores,
    select_heuristic_candidates,
)
from src.core.state import State, StateMode
from src.experiment.baselines.ensembles import (
    ConstructedOriginSpec,
    ReferenceOrigin,
    build_constructed_origin,
)


def _dummy_rollout(
    success: bool,
    *,
    valid: bool = True,
    eligible: bool = True,
    restricted_time: float = 10.0,
    violation: float = 0.0,
    expenditure: float = 2.0,
    distinct_nodes: int = 2,
    candidate_id: str = "c",
) -> RolloutResult:
    return RolloutResult(
        problem_id="p",
        policy_id=candidate_id,
        origin_id="o",
        future_id=0,
        is_valid=valid,
        is_eligible=eligible,
        success=success,
        completion_time=restricted_time if success else None,
        restricted_time=restricted_time,
        expenditure=expenditure,
        distinct_nodes_used=distinct_nodes,
        terminal_target_violation=violation,
    )


class TestHeuristicScoringAndSelection(unittest.TestCase):
    def setUp(self):
        self.N = 4
        self.W = np.array([
            [0, 3, 0, 1],
            [0, 0, 2, 0],
            [1, 0, 0, 4],
            [0, 0, 0, 0],
        ], dtype=np.float64)

    def test_compute_degree_scores_from_dense_matrix(self):
        in_deg = compute_degree_scores(self.N, W=self.W, mode="in")
        out_deg = compute_degree_scores(self.N, W=self.W, mode="out")
        tot_deg = compute_degree_scores(self.N, W=self.W, mode="total")

        np.testing.assert_array_equal(in_deg, [1.0, 1.0, 1.0, 2.0])
        np.testing.assert_array_equal(out_deg, [2.0, 1.0, 2.0, 0.0])
        np.testing.assert_array_equal(tot_deg, in_deg + out_deg)

    def test_compute_degree_scores_from_observed_edges(self):
        edges = {(0, 1): 2, (1, 2): 1, (2, 0): 3}
        in_deg = compute_degree_scores(self.N, observed_edges=edges, mode="in")
        out_deg = compute_degree_scores(self.N, observed_edges=edges, mode="out")

        np.testing.assert_array_equal(in_deg, [1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(out_deg, [1.0, 1.0, 1.0, 0.0])

    def test_compute_influence_scores_received_attention(self):
        rec_att = compute_influence_scores(self.N, W=self.W, mode="received_attention")
        self.assertEqual(len(rec_att), self.N)
        self.assertTrue(np.all(rec_att >= 0.0))
        # Node 3 receives attention from node 0 (1/4) and node 2 (4/5) => 0.25 + 0.8 = 1.05
        self.assertAlmostEqual(rec_att[3], 0.25 + 0.8, places=6)

    def test_compute_influence_scores_attention_share(self):
        att_share = compute_influence_scores(self.N, W=self.W, mode="attention_share")
        self.assertAlmostEqual(float(np.sum(att_share)), 1.0, places=6)

    def test_compute_influence_scores_cross_group(self):
        groups = np.array([0, 0, 1, 1])
        cross = compute_influence_scores(self.N, W=self.W, mode="cross_group", group_labels=groups)
        # Ties from group 0 to group 1: (0, 3) count 1, (1, 2) count 2
        self.assertEqual(cross[3], 1.0)
        self.assertEqual(cross[2], 2.0)
        self.assertEqual(cross[0], 1.0)  # (2, 0) is group 1 -> group 0
        self.assertEqual(cross[1], 0.0)

    def test_select_heuristic_candidates_respects_roster(self):
        scores = np.array([10.0, 20.0, 30.0, 40.0])
        roster = [0, 1]  # Node 2 and 3 are not in roster
        selected = select_heuristic_candidates(scores, k=2, accessible_roster=roster)
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(selected, [1, 0])  # Node 1 has higher score than 0

    def test_select_heuristic_candidates_with_panel_quota(self):
        scores = np.array([10.0, 20.0, 30.0, 40.0])
        panel = np.array([True, True, False, False])
        # k=2, quota=0.5 -> 1 from observed {0, 1}, 1 from unobserved {2, 3}
        selected = select_heuristic_candidates(
            scores, k=2, accessible_roster=[0, 1, 2, 3], panel_mask=panel, quota=0.5, fallback_mode="sequential"
        )
        self.assertEqual(len(selected), 2)
        self.assertIn(1, selected)  # Top observed
        self.assertIn(3, selected)  # Top unobserved

    def test_select_heuristic_candidates_fallback_allocation(self):
        # Empty scores or zero scores
        scores = np.zeros(self.N)
        selected = select_heuristic_candidates(scores, k=3, accessible_roster=[0, 1, 2, 3])
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)

        # k larger than accessible roster
        selected_all = select_heuristic_candidates(scores, k=10, accessible_roster=[1, 2])
        self.assertEqual(set(selected_all), {1, 2})


class TestGreedyBudgetAllocation(unittest.TestCase):
    def test_uniform_allocation_clips_at_amplitude_cap(self):
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1],
            exposure_cap=100.0,  # Generous budget
            amplitude_cap=2.0,
            duration=5.0,
            k_limit=2,
            allocation_strategy="uniform",
        )
        self.assertEqual(alloc, {0: 2.0, 1: 2.0})

    def test_uniform_allocation_clips_at_exposure_budget(self):
        # Total budget 4.0 over duration 2.0 with 2 nodes -> max amp = 4.0 / (2 * 2.0) = 1.0
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1],
            exposure_cap=4.0,
            amplitude_cap=5.0,
            duration=2.0,
            k_limit=2,
            allocation_strategy="uniform",
        )
        self.assertAlmostEqual(alloc[0], 1.0)
        self.assertAlmostEqual(alloc[1], 1.0)
        self.assertAlmostEqual(sum(alloc.values()) * 2.0, 4.0)

    def test_greedy_priority_allocation(self):
        # Priority order: node 0, then node 1, then node 2
        # Budget = 6.0, duration = 2.0, amp_cap = 2.0
        # Node 0 gets 2.0 (cost = 4.0). Rem budget = 2.0.
        # Node 1 gets 1.0 (cost = 2.0). Rem budget = 0.0.
        # Node 2 gets 0.0 (omitted).
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1, 2],
            exposure_cap=6.0,
            amplitude_cap=2.0,
            duration=2.0,
            k_limit=3,
            allocation_strategy="greedy_priority",
        )
        self.assertAlmostEqual(alloc[0], 2.0)
        self.assertAlmostEqual(alloc[1], 1.0)
        self.assertNotIn(2, alloc)
        self.assertAlmostEqual(sum(alloc.values()) * 2.0, 6.0)

    def test_proportional_allocation(self):
        weights = {0: 3.0, 1: 1.0}
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1],
            exposure_cap=8.0,
            amplitude_cap=5.0,
            duration=2.0,
            k_limit=2,
            allocation_strategy="proportional",
            weights=weights,
        )
        # Total spending rate = 8.0 / 2.0 = 4.0
        # Node 0 gets 3/4 * 4.0 = 3.0, Node 1 gets 1/4 * 4.0 = 1.0
        self.assertAlmostEqual(alloc[0], 3.0)
        self.assertAlmostEqual(alloc[1], 1.0)

    def test_budget_exhaustion_and_edge_cases(self):
        # Zero exposure cap
        self.assertEqual(allocate_greedy_budget([0, 1], 0.0, 2.0, 5.0, 2), {})
        # Zero duration
        self.assertEqual(allocate_greedy_budget([0, 1], 10.0, 2.0, 0.0, 2), {})
        # k_limit = 0
        self.assertEqual(allocate_greedy_budget([0, 1], 10.0, 2.0, 5.0, 0), {})
        # Empty candidate nodes
        self.assertEqual(allocate_greedy_budget([], 10.0, 2.0, 5.0, 2), {})


class TestLexicographicScoring(unittest.TestCase):
    def test_feasible_tier_beats_infeasible_tier(self):
        cand1 = PolicyCandidate("c1", (0,), ((0, 1.0),), 1.0, 5.0)
        cand2 = PolicyCandidate("c2", (1,), ((1, 1.0),), 1.0, 5.0)

        # cand1 is feasible (success=True, p_hat=1.0 >= 0.8)
        score1, _ = compute_candidate_score(cand1, [_dummy_rollout(True, candidate_id="c1")], p0=0.8)
        # cand2 is infeasible (success=False, p_hat=0.0 < 0.8)
        score2, _ = compute_candidate_score(cand2, [_dummy_rollout(False, candidate_id="c2")], p0=0.8)

        self.assertEqual(score1[0], 0)  # Tier 0
        self.assertEqual(score2[0], 1)  # Tier 1
        self.assertLess(score1, score2)

    def test_feasible_comparison_prioritizes_restricted_time(self):
        cand1 = PolicyCandidate("c1", (0,), ((0, 1.0),), 1.0, 5.0)
        cand2 = PolicyCandidate("c2", (1,), ((1, 1.0),), 1.0, 5.0)

        # Both feasible (p_hat=1.0), but cand1 is faster (restricted_time=12.0 < 20.0)
        score1, _ = compute_candidate_score(cand1, [_dummy_rollout(True, restricted_time=12.0)], p0=0.8)
        score2, _ = compute_candidate_score(cand2, [_dummy_rollout(True, restricted_time=20.0)], p0=0.8)

        self.assertLess(score1, score2)

    def test_plateau_tie_breaking_by_violation(self):
        cand1 = PolicyCandidate("c1", (0,), ((0, 1.0),), 1.0, 5.0)
        cand2 = PolicyCandidate("c2", (1,), ((1, 1.0),), 1.0, 5.0)

        # Both fail (infeasible, restricted_time=H=270.0), cand1 has smaller violation d_G=0.1 < 0.8
        score1, _ = compute_candidate_score(cand1, [_dummy_rollout(False, restricted_time=270.0, violation=0.1)], p0=0.8)
        score2, _ = compute_candidate_score(cand2, [_dummy_rollout(False, restricted_time=270.0, violation=0.8)], p0=0.8)

        self.assertLess(score1, score2)


class TestHeuristicSubsetSearchOptimizer(unittest.TestCase):
    def setUp(self):
        spec = ConstructedOriginSpec(
            N=8,
            memory_draws_per_row=4,
            memory_capacity_per_row=8,
            heterogeneous=True,
            constrained_opportunities=True,
            opportunity_groups=2,
            public_positive_probability=0.8,
            random_global_sign=False,
        )
        self.origin = build_constructed_origin(0, master_seed=123, spec=spec)
        self.problem = ControlProblem(
            problem_id="test_m6_search",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=8,
            D=1,
            k=2,
            exposure_cap=16.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=tuple(range(8)),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=55,
            search_seed=55,
        )

    def test_seed_generation_is_diverse_and_deduplicated(self):
        opt = HeuristicSubsetSearchOptimizer(
            problem=self.problem,
            candidate_cap=12,
            futures_per_candidate=1,
            batch_size=4,
            search_seed=99,
        )
        seeds = opt.generate_seed_candidates(self.origin.state)
        self.assertGreaterEqual(len(seeds), 5)
        heuristic_names = {s.heuristic_name for s in seeds}
        self.assertIn("zero_control", heuristic_names)
        self.assertIn("in_degree", heuristic_names)
        self.assertIn("out_degree", heuristic_names)
        self.assertIn("received_attention", heuristic_names)

        # All target nodes must respect k=2
        for s in seeds:
            self.assertLessEqual(len(s.target_nodes), 2)
            self.assertLessEqual(s.nominal_amplitude, self.problem.amplitude_cap)
            self.assertLessEqual(s.duration, self.problem.release_deadline)

    def test_optimization_executes_batches_and_tracks_incumbent(self):
        opt = HeuristicSubsetSearchOptimizer(
            problem=self.problem,
            candidate_cap=4,
            futures_per_candidate=1,
            batch_size=2,
            search_seed=42,
        )
        result = opt.optimize(self.origin, sample_interval=5.0, adaptive_memory=False)

        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(len(result.evaluated_candidates), 4)
        self.assertEqual(result.search_state.evaluations_used, 4)
        self.assertIsNotNone(result.incumbent)
        self.assertIn(result.incumbent.candidate_id, result.candidate_scores)
        self.assertEqual(result.total_rollouts, 4)

    def test_optimizer_does_not_mutate_reference_origin(self):
        opt = HeuristicSubsetSearchOptimizer(
            problem=self.problem,
            candidate_cap=2,
            futures_per_candidate=1,
            search_seed=42,
        )
        before_x = self.origin.state.x.copy()
        before_y = self.origin.state.y.copy()
        before_W = self.origin.state.W.copy()

        opt.optimize(self.origin, adaptive_memory=False)

        np.testing.assert_array_equal(self.origin.state.x, before_x)
        np.testing.assert_array_equal(self.origin.state.y, before_y)
        np.testing.assert_array_equal(self.origin.state.W, before_W)


class TestFeedbackTargetShiftController(unittest.TestCase):
    def setUp(self):
        self.problem = ControlProblem(
            problem_id="test_m6_feedback",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=6,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=2.0,
            release_deadline=15.0,
            holding_duration=5.0,
            horizon=25.0,
            accessible_roster=(0, 1, 2, 3, 4, 5),
            observation_mode=ObservationMode.SAMPLED_PUBLIC,
            proposal_seed=101,
            search_seed=101,
        )
        self.ledger = BudgetLedger(
            N=6,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=2.0,
            release_deadline=15.0,
            accessible_roster=set(range(6)),
        )

    def _make_observation(
        self,
        t: float,
        public_vals: np.ndarray,
        panel_mask: np.ndarray,
        edges: Optional[dict] = None,
        rem_exp: Optional[float] = None,
    ) -> Observation:
        return Observation(
            time=t,
            public_states=public_vals,
            panel_mask=panel_mask,
            observed_edges=edges or {},
            accessible_roster=self.problem.accessible_roster,
            delivery_time=t,
            remaining_exposure=self.ledger.remaining_exposure if rem_exp is None else float(rem_exp),
            remaining_distinct_nodes=self.ledger.remaining_distinct_nodes,
        )

    def test_strict_observation_contract_no_private_access(self):
        controller = FeedbackTargetShiftController(nominal_amplitude=1.5, duration=5.0)
        # Initial public states all positive (+1.0) -> reversal seeks negative (-1.0)
        pub = np.ones((6, 1))
        panel = np.ones(6, dtype=bool)
        obs = self._make_observation(0.0, pub, panel)

        # Confirm private_states is None
        self.assertIsNone(obs.private_states)
        self.assertIsNone(obs.true_memory)

        controller.reset(self.problem, obs)
        action = controller.act(obs, self.ledger)
        self.assertIsInstance(action, Action)
        self.assertFalse(action.release)
        self.assertGreater(len(action.entries), 0)

        # Every actuated field should have negative sign s_* = -1.0 and nominal amplitude 1.5
        for e in action.entries:
            self.assertEqual(e.field, -1.5)

    def test_dynamic_target_shifting_when_node_aligns(self):
        controller = FeedbackTargetShiftController(
            nominal_amplitude=2.0,
            duration=10.0,
            initial_recruitment_fraction=0.5,  # recruits floor(3*0.5)=1 node initially
            alignment_threshold=0.3,  # Reaches alignment promptly
            smoothing_alpha=1.0,      # Direct belief update
            shift_strategy=ShiftStrategy.THRESHOLD,
        )
        # Initial state: all +1.0
        pub = np.ones((6, 1))
        panel = np.ones(6, dtype=bool)
        obs0 = self._make_observation(0.0, pub, panel)

        controller.reset(self.problem, obs0)
        act0 = controller.act(obs0, self.ledger)
        self.assertEqual(len(act0.entries), 1)
        recruited_first = act0.entries[0].node

        # Apply action to ledger
        self.ledger.apply_action(act0, current_time=0.0)
        self.ledger.advance_to(2.0)

        # At t=2.0, recruited_first node aligns to -1.0! Other nodes remain +1.0
        pub2 = np.ones((6, 1))
        pub2[recruited_first, 0] = -1.0
        obs2 = self._make_observation(2.0, pub2, panel)

        act2 = controller.act(obs2, self.ledger)
        # The aligned node should have its field shut off (removed from active entries)
        # And a new node should be recruited since ledger.remaining_distinct_nodes > 0!
        actuated_nodes_2 = act2.actuated_nodes()
        self.assertNotIn(recruited_first, actuated_nodes_2)
        self.assertGreater(len(actuated_nodes_2), 0)

        # Apply second action to ledger
        self.ledger.apply_action(act2, current_time=2.0)
        # Cumulative distinct nodes actuated in ledger must not exceed k=3
        self.assertLessEqual(len(self.ledger.distinct_nodes_actuated), 3)

    def test_observation_noise_resilience_binary_flips_and_nans(self):
        controller = FeedbackTargetShiftController(
            nominal_amplitude=1.0,
            duration=8.0,
            smoothing_alpha=0.5,
            alignment_threshold=0.5,
        )
        pub0 = np.ones((6, 1))
        # Mask half of the panel (NaNs)
        panel = np.array([True, True, True, False, False, False])
        pub0[~panel] = np.nan
        obs0 = self._make_observation(0.0, pub0, panel)

        controller.reset(self.problem, obs0)
        act0 = controller.act(obs0, self.ledger)
        self.ledger.apply_action(act0, current_time=0.0)
        initial_node = list(act0.actuated_nodes())[0]

        # At t=2.0, an isolated transient noise flip occurs on the actuated initial_node
        pub2 = pub0.copy()
        pub2[initial_node, 0] = -1.0  # single transient flip
        obs2 = self._make_observation(2.0, pub2, panel)

        # Smoothing alpha=0.5: belief goes from 1.0 to 0.5 * (-1.0) + 0.5 * 1.0 = 0.0
        # s_* * belief = -1.0 * 0.0 = 0.0 < threshold (0.5), so NOT prematurely shifted!
        act2 = controller.act(obs2, self.ledger)
        self.assertIn(initial_node, act2.actuated_nodes())

    def test_budget_exhaustion_clean_release(self):
        # Small budget ledger: exposure_cap=2.0, amp=2.0, duration=1.0 -> 2.0 spent in 1.0s
        small_ledger = BudgetLedger(N=6, D=1, k=2, exposure_cap=2.0, amplitude_cap=2.0, release_deadline=5.0)
        controller = FeedbackTargetShiftController(
            nominal_amplitude=2.0,
            duration=1.0,
            initial_recruitment_fraction=1.0,
        )
        pub = np.ones((6, 1))
        panel = np.ones(6, dtype=bool)
        obs = self._make_observation(0.0, pub, panel, rem_exp=2.0)

        controller.reset(self.problem, obs)
        act0 = controller.act(obs, small_ledger)
        small_ledger.apply_action(act0, current_time=0.0)

        # Advance until budget is exhausted (2.0 exposure / 2.0 spending_rate = 1.0s)
        small_ledger.advance_to(1.05)
        self.assertTrue(small_ledger.is_released)

        obs1 = self._make_observation(1.05, pub, panel, rem_exp=small_ledger.remaining_exposure)
        act1 = controller.act(obs1, small_ledger)
        self.assertTrue(act1.release)
        self.assertEqual(len(act1.entries), 0)

    def test_end_to_end_rollout_with_sampled_public_model(self):
        # Full closed-loop simulation rollout with FeedbackTargetShiftController
        spec = ConstructedOriginSpec(
            N=6,
            memory_draws_per_row=2,
            memory_capacity_per_row=4,
            public_positive_probability=0.9,
            random_global_sign=False,
        )
        origin = build_constructed_origin(0, master_seed=777, spec=spec)
        problem = ControlProblem(
            problem_id="test_m6_e2e",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=6,
            D=1,
            k=2,
            exposure_cap=8.0,
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=12.0,
            accessible_roster=(0, 1, 2, 3, 4, 5),
            observation_mode=ObservationMode.SAMPLED_PUBLIC,
            proposal_seed=88,
            search_seed=88,
        )
        controller = FeedbackTargetShiftController(
            nominal_amplitude=1.5,
            duration=4.0,
            initial_recruitment_fraction=0.5,
        )
        obs_model = SampledPublicObservationModel(
            N=6,
            panel_fraction=0.5,
            delivery_delay=1.0,
            accessible_roster=problem.accessible_roster,
            seed=42,
        )
        evaluator = TargetEvaluator.create_for_state(
            target_type=problem.target_type,
            initial_state=origin.state,
            holding_duration=problem.holding_duration,
            horizon=problem.horizon,
            release_deadline=problem.release_deadline,
            problem_id=problem.problem_id,
        )
        result, final_state, ledger = run_controlled_simulation(
            initial_state=origin.branch(),
            problem=problem,
            controller=controller,
            observation_model=obs_model,
            target_evaluator=evaluator,
            adaptive_memory=False,
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.is_eligible)
        self.assertLessEqual(result.distinct_nodes_used, 2)
        self.assertLessEqual(result.expenditure, 8.0 + 1e-10)




class TestOptimizerAdversarialChecks(unittest.TestCase):
    """Targeted adversarial tests probing edge cases, contract limits, and bug fixes."""

    def test_heuristics_self_loop_and_shape_validation(self):
        # 1. Self loops must not inflate received attention
        W_with_loops = np.array([
            [10.0, 2.0],
            [0.0, 10.0],
        ], dtype=np.float64)
        # With self loops zeroed:
        # Row 0: [0, 2] -> A[0, 1] = 1.0
        # Row 1: [0, 0] -> A[1, :] = 0.0
        # rec_att: [0.0, 1.0]
        rec = compute_influence_scores(2, W=W_with_loops, mode="received_attention")
        np.testing.assert_allclose(rec, [0.0, 1.0])

        # 2. Shape validation on W
        with self.assertRaises(ValueError):
            compute_influence_scores(2, W=np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            compute_degree_scores(2, W=np.zeros((2, 3)))

        # 3. group_labels length validation
        with self.assertRaises(ValueError):
            compute_influence_scores(2, W=np.zeros((2, 2)), mode="cross_group", group_labels=np.array([0, 1, 2]))

        # 4. panel_mask length validation
        with self.assertRaises(ValueError):
            select_heuristic_candidates(np.zeros(3), k=2, panel_mask=np.array([True, False]))

    def test_proportional_water_filling_multi_round_fairness(self):
        # 3 nodes: weights 10, 10, 1
        # duration = 1.0, amplitude_cap = 2.0, exposure_cap = 5.0
        # Ideal: node 0 and node 1 should get clipped to 2.0 (cost 2.0 each = 4.0),
        # remaining budget 1.0 goes to node 2 (amp 1.0).
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1, 2],
            exposure_cap=5.0,
            amplitude_cap=2.0,
            duration=1.0,
            k_limit=3,
            allocation_strategy="proportional",
            weights={0: 10.0, 1: 10.0, 2: 1.0},
        )
        self.assertAlmostEqual(alloc[0], 2.0)
        self.assertAlmostEqual(alloc[1], 2.0)
        self.assertAlmostEqual(alloc[2], 1.0)
        self.assertAlmostEqual(sum(alloc.values()), 5.0)

    def test_sign_inversion_mutation_distinct_and_correct(self):
        prob = ControlProblem(
            problem_id="test_inv",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        spec = ConstructedOriginSpec(N=4, memory_draws_per_row=2, memory_capacity_per_row=4)
        origin = build_constructed_origin(0, 42, spec)

        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=10, search_seed=1)
        inc = PolicyCandidate("inc", (0, 1), ((0, 1.0), (1, 1.0)), 1.0, 5.0)
        opt.seen_signatures.add(inc.signature)

        found_inverted = False
        for prop_idx in range(1, 100):
            cand = opt.mutate_candidate(inc, prop_idx, origin.state)
            if cand and "invert_sign" in cand.heuristic_name:
                found_inverted = True
                signs_dict = cand.signs_dict
                # At least one node must have -1.0
                has_neg = any(s == -1.0 for s in signs_dict.values())
                self.assertTrue(has_neg)
                self.assertNotEqual(cand.signature, inc.signature)
                break
        self.assertTrue(found_inverted)

    def test_multi_component_d3_search_and_feedback(self):
        # ControlProblem with D=3, targeting component 2
        prob = ControlProblem(
            problem_id="test_d3",
            target_type=TargetType.REVERSAL,
            target_params={"component": 2},
            N=6,
            D=3,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3, 4, 5),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=42,
            search_seed=42,
        )
        # Test Optimizer uses component 2
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=4, component=2, search_seed=42)
        self.assertEqual(opt.component, 2)

        spec = ConstructedOriginSpec(N=6, D=3, memory_draws_per_row=2, memory_capacity_per_row=4, opportunity_groups=2)
        origin = build_constructed_origin(0, 42, spec)
        seeds = opt.generate_seed_candidates(origin.state)
        for s in seeds:
            self.assertEqual(s.component, 2)

        # Test FeedbackTargetShiftController with component 2
        ctrl = FeedbackTargetShiftController(component=2)
        ledger = BudgetLedger(N=6, D=3, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=10.0)
        pub = np.zeros((6, 3))
        pub[:, 2] = 1.0  # component 2 is positive
        obs = Observation(
            time=0.0,
            public_states=pub,
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={},
            accessible_roster=(0, 1, 2, 3, 4, 5),
            delivery_time=0.0,
        )
        ctrl.reset(prob, obs)
        act = ctrl.act(obs, ledger)
        for e in act.entries:
            self.assertEqual(e.component, 2)
            self.assertEqual(e.field, -1.0)  # Reversal on positive public states

    def test_feedback_shift_already_aligned_roster_zero_expenditure(self):
        # All accessible roster nodes are already in target state at t=0
        prob = ControlProblem(
            problem_id="test_already_aligned",
            target_type=TargetType.REVERSAL,
            target_params={"target_sign": -1.0},
            N=6,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=42,
            search_seed=42,
        )
        ledger = BudgetLedger(N=6, D=1, k=3, exposure_cap=15.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl = FeedbackTargetShiftController()

        # Nodes (0, 1, 2) already have state -1.0 (aligned with target_sign -1.0)
        pub = np.array([[-1.0], [-1.0], [-1.0], [1.0], [1.0], [1.0]])
        obs = Observation(
            time=0.0,
            public_states=pub,
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={},
            accessible_roster=(0, 1, 2),
            delivery_time=0.0,
        )
        ctrl.reset(prob, obs)
        act = ctrl.act(obs, ledger)

        # Must release cleanly with 0 entries
        self.assertTrue(act.release)
        self.assertEqual(len(act.entries), 0)
        self.assertEqual(len(ctrl.actuated_distinct_nodes), 0)
        self.assertEqual(ledger.integrated_exposure, 0.0)

    def test_feedback_shift_resumes_recruitment_after_active_set_cleared(self):
        prob = ControlProblem(
            problem_id="test_resume_recruitment",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=5,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=2.0,
            release_deadline=15.0,
            holding_duration=5.0,
            horizon=25.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=5, D=1, k=3, exposure_cap=15.0, amplitude_cap=2.0, release_deadline=15.0)
        ctrl = FeedbackTargetShiftController(initial_recruitment_fraction=0.33, smoothing_alpha=1.0)

        # Initial: node 0 is unaligned (+1.0), nodes 1,2,3 are aligned (-1.0), node 4 is +2.0 (mean > 0 => s_* = -1.0)
        # Decisive positive mean (2.5 not 2.0: mean 0 is now an ambiguous tie under the F3 tie rule) => s_* = -1.0
        pub0 = np.array([[1.0], [-1.0], [-1.0], [-1.0], [2.5]])
        obs0 = Observation(time=0.0, public_states=pub0, panel_mask=np.ones(5, dtype=bool), observed_edges={}, accessible_roster=(0, 1, 2, 3), delivery_time=0.0)
        ctrl.reset(prob, obs0)
        act0 = ctrl.act(obs0, ledger)
        ledger.apply_action(act0, current_time=0.0)
        self.assertIn(0, ctrl.active_nodes)

        # t=2.0: node 0 aligns to -1.0! All accessible nodes are now aligned.
        pub2 = np.array([[-1.0], [-1.0], [-1.0], [-1.0], [2.0]])
        obs2 = Observation(time=2.0, public_states=pub2, panel_mask=np.ones(5, dtype=bool), observed_edges={}, accessible_roster=(0, 1, 2, 3), delivery_time=2.0)
        ledger.advance_to(2.0)
        act2 = ctrl.act(obs2, ledger)
        ledger.apply_action(act2, current_time=2.0)
        self.assertEqual(len(ctrl.active_nodes), 0)

        # t=4.0: Node 2 becomes unaligned (+1.0)! Controller must recruit node 2!
        pub4 = np.array([[-1.0], [-1.0], [1.0], [-1.0], [2.0]])
        obs4 = Observation(time=4.0, public_states=pub4, panel_mask=np.ones(5, dtype=bool), observed_edges={}, accessible_roster=(0, 1, 2, 3), delivery_time=4.0)
        ledger.advance_to(4.0)
        act4 = ctrl.act(obs4, ledger)
        self.assertIn(2, ctrl.active_nodes)
        self.assertGreater(len(act4.entries), 0)
        self.assertEqual(act4.entries[0].node, 2)

    def test_feedback_shift_extreme_noise_and_90_percent_missingness(self):
        N = 20
        prob = ControlProblem(
            problem_id="test_extreme_noise",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=N,
            D=1,
            k=4,
            exposure_cap=20.0,
            amplitude_cap=2.0,
            release_deadline=15.0,
            holding_duration=5.0,
            horizon=25.0,
            accessible_roster=tuple(range(N)),
            observation_mode=ObservationMode.SAMPLED_PUBLIC,
            proposal_seed=42,
            search_seed=42,
        )
        obs_model = SampledPublicObservationModel(
            N=N,
            panel_fraction=0.10,  # 90% missingness!
            binary_flip_prob=0.40,  # 40% noise flip!
            seed=123,
        )
        ctrl = FeedbackTargetShiftController(smoothing_alpha=0.3, alignment_threshold=0.6, duration=8.0)
        ledger = BudgetLedger(N=N, D=1, k=4, exposure_cap=20.0, amplitude_cap=2.0, release_deadline=15.0)

        state = State(N=N, D=1, x=np.ones((N, 1)), y=np.ones((N, 1)), W=np.zeros((N, N), dtype=np.uint32), state_mode=StateMode.BINARY, n=N, Q=1)
        obs0 = obs_model.sample_observation(state, 0.0, ledger.remaining_exposure, ledger.remaining_distinct_nodes)
        ctrl.reset(prob, obs0)

        t = 0.0
        for _ in range(5):
            obs = obs_model.sample_observation(state, t, ledger.remaining_exposure, ledger.remaining_distinct_nodes)
            act = ctrl.act(obs, ledger)
            self.assertIsInstance(act, Action)
            # Actuation must respect amplitude cap
            for e in act.entries:
                self.assertLessEqual(abs(e.field), prob.amplitude_cap + 1e-12)
                self.assertIn(e.node, prob.accessible_roster)
            ledger.apply_action(act, current_time=t)
            t += 1.0
            ledger.advance_to(t)

        self.assertLessEqual(len(ledger.distinct_nodes_actuated), prob.k)
        self.assertLessEqual(ledger.integrated_exposure, prob.exposure_cap + 1e-9)


class TestOptimizerAdversarialChecksExtended(unittest.TestCase):
    """Deep adversarial tests verifying Round 2 defects and edge case guarantees."""

    def test_policy_candidate_hashable_and_set_compatible(self):
        cand1 = PolicyCandidate("c1", (0, 1), ((0, 1.0), (1, -1.0)), 1.0, 5.0, metadata={"source": "test"})
        cand2 = PolicyCandidate("c2", (1, 2), ((1, 1.0), (2, -1.0)), 1.0, 5.0, metadata={"source": "test2"})
        h1 = hash(cand1)
        self.assertIsInstance(h1, int)
        cand_set = {cand1, cand2}
        self.assertEqual(len(cand_set), 2)
        cand_map = {cand1: "first", cand2: "second"}
        self.assertEqual(cand_map[cand1], "first")

    def test_mutate_candidate_short_deadlines_and_low_amplitude_cap(self):
        prob = ControlProblem(
            problem_id="test_short_deadline",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=2,
            exposure_cap=5.0,
            amplitude_cap=0.3,
            release_deadline=0.7,
            holding_duration=0.5,
            horizon=2.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=20, search_seed=1)
        inc = PolicyCandidate("inc", (0,), ((0, 1.0),), 0.3, 0.7)
        state = State(N=4, D=1, state_mode=StateMode.BINARY, x=np.ones((4, 1)), y=np.ones((4, 1)), W=np.zeros((4, 4), dtype=np.uint32), n=4, Q=1)

        for prop_idx in range(1, 51):
            cand = opt.mutate_candidate(inc, prop_idx, state)
            if cand is not None:
                self.assertLessEqual(cand.nominal_amplitude, 0.3 + 1e-12)
                self.assertLessEqual(cand.duration, 0.7 + 1e-12)

    def test_mutate_candidate_from_zero_controller_produces_active_candidate(self):
        prob = ControlProblem(
            problem_id="test_zero_mutation",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=10, search_seed=1)
        zero_inc = PolicyCandidate("seed_zero", (), (), 0.0, 0.0)
        state = State(N=4, D=1, state_mode=StateMode.BINARY, x=np.ones((4, 1)), y=np.ones((4, 1)), W=np.zeros((4, 4), dtype=np.uint32), n=4, Q=1)

        opt.rng = np.random.default_rng(2)
        cand = opt.mutate_candidate(zero_inc, 1, state)
        self.assertIsNotNone(cand)
        self.assertGreater(cand.nominal_amplitude, 0.0)
        self.assertGreater(cand.duration, 0.0)
        ctrl = cand.to_controller()
        from src.control.controllers import OpenLoopPulseController
        self.assertIsInstance(ctrl, OpenLoopPulseController)

    def test_determine_target_sign_respects_target_params(self):
        state = State(N=4, D=1, state_mode=StateMode.BINARY, x=np.ones((4, 1)), y=np.ones((4, 1)), W=np.zeros((4, 4), dtype=np.uint32), n=4, Q=1)
        prob = ControlProblem(
            problem_id="test_target_sign",
            target_type=TargetType.REVERSAL,
            target_params={"s_star": 1.0},
            N=4,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=10, search_seed=1)
        sign = opt._determine_target_sign(state)
        self.assertEqual(sign, 1.0)

        prob2 = ControlProblem(
            problem_id="test_target_sign_2",
            target_type=TargetType.REVERSAL,
            target_params={"target_sign": 1.0},
            N=4,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt2 = HeuristicSubsetSearchOptimizer(prob2, candidate_cap=10, search_seed=1)
        self.assertEqual(opt2._determine_target_sign(state), 1.0)

    def test_feedback_shift_relapsed_node_reactivation_at_zero_slot_cost(self):
        prob = ControlProblem(
            problem_id="test_relapse_reactivation",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=3,
            D=1,
            k=1,
            exposure_cap=15.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=15.0,
            accessible_roster=(0, 1),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=3, D=1, k=1, exposure_cap=15.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl = FeedbackTargetShiftController(smoothing_alpha=1.0)

        obs0 = Observation(0.0, np.array([[1.0], [-1.0], [2.0]]), np.ones(3, dtype=bool), {}, (0, 1), delivery_time=0.0)
        ctrl.reset(prob, obs0)
        act0 = ctrl.act(obs0, ledger)
        ledger.apply_action(act0, 0.0)
        self.assertIn(0, ctrl.active_nodes)
        self.assertEqual(ledger.remaining_distinct_nodes, 0)

        obs2 = Observation(2.0, np.array([[-1.0], [-1.0], [2.0]]), np.ones(3, dtype=bool), {}, (0, 1), delivery_time=2.0)
        ledger.advance_to(2.0)
        act2 = ctrl.act(obs2, ledger)
        ledger.apply_action(act2, 2.0)
        self.assertEqual(len(ctrl.active_nodes), 0)

        obs4 = Observation(4.0, np.array([[1.0], [-1.0], [2.0]]), np.ones(3, dtype=bool), {}, (0, 1), delivery_time=4.0)
        ledger.advance_to(4.0)
        act4 = ctrl.act(obs4, ledger)
        self.assertIn(0, ctrl.active_nodes)
        self.assertEqual(len(act4.entries), 1)
        self.assertEqual(act4.entries[0].node, 0)
        valid, msg = ledger.validate_action(act4, current_time=4.0)
        self.assertTrue(valid, msg)

    def test_shift_strategy_greedy_marginal_replaces_most_aligned_node(self):
        prob = ControlProblem(
            problem_id="test_greedy_marginal",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=3,
            exposure_cap=20.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=15.0,
            accessible_roster=(0, 1, 2),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=4, D=1, k=3, exposure_cap=20.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl = FeedbackTargetShiftController(
            initial_recruitment_fraction=0.67,
            shift_strategy=ShiftStrategy.GREEDY_MARGINAL,
            smoothing_alpha=1.0,
        )

        obs0 = Observation(0.0, np.array([[1.0], [1.0], [1.0], [1.0]]), np.ones(4, dtype=bool), {}, (0, 1, 2), delivery_time=0.0)
        ctrl.reset(prob, obs0)
        ctrl.act(obs0, ledger)
        ctrl.active_nodes = {0, 1}
        ctrl.actuated_distinct_nodes = {0, 1}
        ledger.distinct_nodes_actuated = {0, 1}

        obs2 = Observation(2.0, np.array([[-0.3], [0.8], [1.0], [1.0]]), np.ones(4, dtype=bool), {}, (0, 1, 2), delivery_time=2.0)
        ledger.advance_to(2.0)
        act2 = ctrl.act(obs2, ledger)

        self.assertNotIn(0, ctrl.active_nodes)
        self.assertIn(2, ctrl.active_nodes)
        self.assertIn(1, ctrl.active_nodes)

    def test_shift_strategy_periodic_rotation(self):
        prob = ControlProblem(
            problem_id="test_periodic",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=3,
            exposure_cap=20.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=15.0,
            accessible_roster=(0, 1, 2),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=4, D=1, k=3, exposure_cap=20.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl = FeedbackTargetShiftController(
            initial_recruitment_fraction=0.67,
            shift_strategy=ShiftStrategy.PERIODIC,
            shift_period=2.0,
            smoothing_alpha=1.0,
        )

        obs0 = Observation(0.0, np.array([[1.0], [1.0], [1.0], [1.0]]), np.ones(4, dtype=bool), {}, (0, 1, 2), delivery_time=0.0)
        ctrl.reset(prob, obs0)
        ctrl.act(obs0, ledger)
        ctrl.active_nodes = {0, 1}
        ctrl.actuated_distinct_nodes = {0, 1}
        ledger.distinct_nodes_actuated = {0, 1}
        ctrl.last_shift_time = 0.0

        obs2 = Observation(2.5, np.array([[0.2], [0.9], [1.0], [1.0]]), np.ones(4, dtype=bool), {}, (0, 1, 2), delivery_time=2.5)
        ledger.advance_to(2.5)
        act2 = ctrl.act(obs2, ledger)
        self.assertNotIn(0, ctrl.active_nodes)
        self.assertIn(2, ctrl.active_nodes)

    def test_select_heuristic_candidates_deduplicates_roster(self):
        scores = np.array([10.0, 5.0, 1.0])
        selected = select_heuristic_candidates(scores, k=2, accessible_roster=[0, 0, 1])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(selected, [0, 1])

    def test_optimizer_populates_search_state_candidate_registry_and_rollouts(self):
        spec = ConstructedOriginSpec(N=4, memory_draws_per_row=2, memory_capacity_per_row=4)
        origin = build_constructed_origin(0, 123, spec)
        prob = ControlProblem(
            problem_id="test_registry",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=10.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=2, futures_per_candidate=1)
        res = opt.optimize(origin, adaptive_memory=False)
        self.assertGreaterEqual(len(res.search_state.candidate_registry), 2)
        self.assertGreaterEqual(len(res.search_state.completed_evaluations), 2)
        for cid, cand in res.search_state.candidate_registry.items():
            self.assertIsInstance(cand, PolicyCandidate)
            self.assertEqual(cid, cand.candidate_id)

    def test_feedback_controller_parameter_validation(self):
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(nominal_amplitude=-1.0)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(duration=0.0)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(initial_recruitment_fraction=-0.1)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(alignment_threshold=1.5)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(smoothing_alpha=-0.5)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(quota=1.2)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(component=-1)


class TestOptimizerAdversarialChecksFurther(unittest.TestCase):
    """Deep adversarial tests verifying Round 3 defects, edge cases, and robustness."""

    def test_optimizer_multiple_runs_deterministic_and_no_index_error(self):
        """Verify repeated calls to optimize() on the same optimizer instance do not crash with IndexError and are deterministic."""
        prob = ControlProblem(
            problem_id="test_repeat",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=2,
            D=1,
            k=1,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=10.0,
            accessible_roster=(0,),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        state = State(N=2, D=1, state_mode=StateMode.BINARY, x=np.ones((2, 1)), y=np.ones((2, 1)), W=np.zeros((2, 2), dtype=np.uint32), n=2, Q=1)
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=10, futures_per_candidate=1)

        res1 = opt.optimize(state)
        res2 = opt.optimize(state)

        self.assertEqual(len(res1.evaluated_candidates), 10)
        self.assertEqual(len(res2.evaluated_candidates), 10)
        self.assertEqual(res1.incumbent.candidate_id, res2.incumbent.candidate_id)
        self.assertEqual(res1.incumbent_score, res2.incumbent_score)

    def test_feedback_shift_controller_respects_quota(self):
        """Verify FeedbackTargetShiftController respects quota between observed and unobserved accessible nodes."""
        N = 4
        prob = ControlProblem(
            problem_id="test_quota_respect",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=N,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=15.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.SAMPLED_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=10.0)

        # 0 and 1 are observed, 2 and 3 are unobserved
        panel = np.array([True, True, False, False])
        pub = np.array([[1.0], [1.0], [np.nan], [np.nan]])
        obs = Observation(
            time=0.0,
            public_states=pub,
            panel_mask=panel,
            observed_edges={},
            accessible_roster=(0, 1, 2, 3),
            delivery_time=0.0,
        )

        # 1. quota=0.5: should recruit 1 from observed {0, 1}, 1 from unobserved {2, 3}
        ctrl50 = FeedbackTargetShiftController(initial_recruitment_fraction=1.0, quota=0.5)
        ctrl50.reset(prob, obs)
        act50 = ctrl50.act(obs, ledger)
        actuated50 = act50.actuated_nodes()
        self.assertEqual(len(actuated50), 2)
        self.assertEqual(len(actuated50 & {0, 1}), 1)
        self.assertEqual(len(actuated50 & {2, 3}), 1)

        # 2. quota=1.0: should recruit all from observed {0, 1}
        ledger100 = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl100 = FeedbackTargetShiftController(initial_recruitment_fraction=1.0, quota=1.0)
        ctrl100.reset(prob, obs)
        act100 = ctrl100.act(obs, ledger100)
        actuated100 = act100.actuated_nodes()
        self.assertEqual(actuated100, {0, 1})

        # 3. quota=0.0: should recruit all from unobserved {2, 3}
        ledger0 = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl0 = FeedbackTargetShiftController(initial_recruitment_fraction=1.0, quota=0.0)
        ctrl0.reset(prob, obs)
        act0 = ctrl0.act(obs, ledger0)
        actuated0 = act0.actuated_nodes()
        self.assertEqual(actuated0, {2, 3})

    def test_select_heuristic_candidates_validation(self):
        """Verify select_heuristic_candidates validates quota and fallback_mode."""
        scores = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            select_heuristic_candidates(scores, k=2, quota=-0.1)
        with self.assertRaises(ValueError):
            select_heuristic_candidates(scores, k=2, quota=1.5)
        with self.assertRaises(ValueError):
            select_heuristic_candidates(scores, k=2, fallback_mode="invalid_fallback")

    def test_mutation_uniform_reset_clips_amplitude_to_budget(self):
        """Verify mutation_uniform_reset (proposal_idx % 4 == 0) clips amplitude against exposure budget."""
        prob = ControlProblem(
            problem_id="test_uniform_reset_budget",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            D=1,
            k=2,
            exposure_cap=2.0,  # Tight exposure cap!
            amplitude_cap=4.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2, 3),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=10, search_seed=1)
        inc = PolicyCandidate("inc", (0,), ((0, 1.0),), 1.0, 5.0)
        state = State(N=4, D=1, state_mode=StateMode.BINARY, x=np.ones((4, 1)), y=np.ones((4, 1)), W=np.zeros((4, 4), dtype=np.uint32), n=4, Q=1)

        # proposal_idx=4 triggers mutation_uniform_reset
        cand = opt.mutate_candidate(inc, 4, state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.heuristic_name, "mutation_uniform_reset")
        total_potential_exposure = cand.nominal_amplitude * len(cand.target_nodes) * cand.duration
        self.assertLessEqual(total_potential_exposure, prob.exposure_cap + 1e-12)

    def test_proportional_budget_allocation_does_not_leak_to_zero_weight_nodes(self):
        """Verify proportional allocation does not reallocate excess budget to nodes with explicit 0.0 weight."""
        alloc = allocate_greedy_budget(
            candidate_nodes=[0, 1],
            exposure_cap=10.0,
            amplitude_cap=2.0,
            duration=1.0,
            k_limit=2,
            allocation_strategy="proportional",
            weights={0: 1.0, 1: 0.0},
        )
        self.assertIn(0, alloc)
        self.assertAlmostEqual(alloc[0], 2.0)
        # Node 1 has explicit weight 0.0, so it must receive 0 field
        self.assertNotIn(1, alloc)

    def test_feedback_controller_shift_period_and_component_validation(self):
        """Verify FeedbackTargetShiftController validates shift_period and component relative to problem.D."""
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(shift_period=-1.0)
        with self.assertRaises(ValueError):
            FeedbackTargetShiftController(shift_period=0.0)

        prob_d1 = ControlProblem(
            problem_id="test_d1",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=2,
            D=1,
            k=1,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=10.0,
            accessible_roster=(0, 1),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
        )
        obs = Observation(0.0, np.ones((2, 1)), np.ones(2, dtype=bool), {}, (0, 1))
        ctrl = FeedbackTargetShiftController(component=2)
        with self.assertRaises(ValueError):
            ctrl.reset(prob_d1, obs)

    def test_heuristic_search_optimizer_component_validation(self):
        """Verify HeuristicSubsetSearchOptimizer raises ValueError if component >= problem.D."""
        prob_d1 = ControlProblem(
            problem_id="test_d1_opt",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=2,
            D=1,
            k=1,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=10.0,
            accessible_roster=(0, 1),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
        )
        with self.assertRaises(ValueError):
            HeuristicSubsetSearchOptimizer(prob_d1, component=3)

    def test_greedy_marginal_structural_target(self):
        """Verify greedy marginal strategy on structural target swaps based on influence scores rather than belief sign."""
        prob = ControlProblem(
            problem_id="test_greedy_marginal_structural",
            target_type=TargetType.DECENTRALIZATION,
            target_params={},
            N=4,
            D=1,
            k=3,
            exposure_cap=20.0,
            amplitude_cap=2.0,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=15.0,
            accessible_roster=(0, 1, 2),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        ledger = BudgetLedger(N=4, D=1, k=3, exposure_cap=20.0, amplitude_cap=2.0, release_deadline=10.0)
        ctrl = FeedbackTargetShiftController(
            initial_recruitment_fraction=0.67,
            shift_strategy=ShiftStrategy.GREEDY_MARGINAL,
            smoothing_alpha=1.0,
        )

        obs0 = Observation(0.0, np.zeros((4, 1)), np.ones(4, dtype=bool), {}, (0, 1, 2), delivery_time=0.0)
        ctrl.reset(prob, obs0)
        ctrl.act(obs0, ledger)
        ctrl.active_nodes = {0, 1}
        ctrl.actuated_distinct_nodes = {0, 1}
        ledger.distinct_nodes_actuated = {0, 1}

        # Observed edges: node 2 receives high attention, node 0 receives 0 attention
        # (3, 2): 10 counts, (3, 1): 5 counts
        edges = {(3, 2): 10, (3, 1): 5}
        obs2 = Observation(2.0, np.zeros((4, 1)), np.ones(4, dtype=bool), edges, (0, 1, 2), delivery_time=2.0)
        ledger.advance_to(2.0)
        act2 = ctrl.act(obs2, ledger)

        # Node 0 (lowest influence) should be swapped out for node 2 (highest influence)
        self.assertNotIn(0, ctrl.active_nodes)
        self.assertIn(2, ctrl.active_nodes)

    def test_policy_candidate_signature_precision(self):
        """Verify PolicyCandidate signature rounds sign floats to avoid micro-precision deduplication failure."""
        cand1 = PolicyCandidate("c1", (0,), ((0, 1.0),), 1.0, 5.0)
        cand2 = PolicyCandidate("c2", (0,), ((0, 1.000000000001),), 1.0, 5.0)
        self.assertEqual(cand1.signature, cand2.signature)

    def test_search_state_incumbent_id_populated_on_fallback(self):
        """Verify search_state.incumbent_policy_id is set to fallback_zero when search fallback triggers."""
        prob = ControlProblem(
            problem_id="test_fallback_id",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=2,
            D=1,
            k=1,
            exposure_cap=0.0,  # 0 budget causes zero-controller
            amplitude_cap=2.0,
            release_deadline=5.0,
            holding_duration=3.0,
            horizon=10.0,
            accessible_roster=(0, 1),
            observation_mode=ObservationMode.COMPLETE_PUBLIC,
            proposal_seed=1,
            search_seed=1,
        )
        opt = HeuristicSubsetSearchOptimizer(prob, candidate_cap=1)
        state = State(N=2, D=1, state_mode=StateMode.BINARY, x=np.ones((2, 1)), y=np.ones((2, 1)), W=np.zeros((2, 2), dtype=np.uint32), n=2, Q=1)
        res = opt.optimize(state)
        self.assertIsNotNone(res.search_state.incumbent_policy_id)
        self.assertIsNotNone(res.search_state.incumbent_score)
        self.assertEqual(res.search_state.incumbent_policy_id, res.incumbent.candidate_id)

if __name__ == "__main__":
    unittest.main()
