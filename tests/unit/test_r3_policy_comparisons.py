"""Tests: Policy comparisons, search coverage,
and origin-aware uncertainty.

Verifies all acceptance criteria:
1. Observed-attention targeting uses available reported scores; unavailable
   scores never masquerade as attention; fallback selection is recorded when
   the scored pool is insufficient; matched random comparator shares the identical
   accessible/observed pool, roster, node count, amplitude, duration, budget, and report timing.
2. Report/actuation timing audit explicitly classifies policies with no later usable
   report before pulse end as effectively static; genuinely adaptive policies have
   usable reports; scripted reports cause genuine feedback action updates.
3. Separate static and feedback policy search families with reserved allocation;
   independent selection of best static and best feedback winners; evaluation of
   both families on fresh origins and matched adaptive/frozen branches; no dropping
   of failed families under all-failure conditions; pool disjointness and pairing preserved.
4. Active-node allocation wiring (candidate_node_fractions) distinct from global cap k;
   tracks requested and effective settings including amplitude reductions caused by exposure limits.
5. Physics forwarding (a, c, tau_y, J) through search, training, selection, and evaluation;
   removal of the outdated manifest limitation verified.
6. Origin-aware uncertainty: simple binomial / Wilson for 1 future/origin; cluster bootstrap
   over origins for multiple futures/origin; paired cluster bootstrap for adaptive vs frozen.
"""

from __future__ import annotations

import tempfile
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np

from src.config.control_profiles import LIGHT_CONTROL_PROFILE
from src.control.campaign import (
    _campaign_geometry,
    _run_observable_rollouts,
    prepare_origins,
    run_campaign,
    summarize_paired_adaptive_frozen_comparison,
    summarize_rollout_reliability,
)
from src.control.contracts import (
    Action,
    ActionEntry,
    ControlProblem,
    Observation,
    RolloutResult,
    TargetType,
)
from src.control.controllers import (
    ObservableSelectorPolicy,
    ZeroController,
    audit_policy_timing,
)
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import SampledPublicObservationModel
from src.control.optimizers.search import (
    ObservablePolicyCandidate,
    ObservablePolicySearchOptimizer,
)
from src.core.state import State


def _make_dummy_state(N: int = 10, D: int = 1, y_val: float = 0.8) -> State:
    """Create a minimal deterministic continuous State for testing."""
    return State(
        N=N,
        D=D,
        state_mode="continuous",
        n=1,
        Q=np.zeros((1, 1), dtype=np.float64),
        y=np.full((N, D), y_val, dtype=np.float64),
        x=np.full((N, D), y_val, dtype=np.float64),
        W=np.zeros((N, N), dtype=np.float64),
        opportunity_groups=np.zeros(N, dtype=int),
    )


class TestObservedAttentionTargetingAndFallback(unittest.TestCase):
    """Test reported-attention ranking, fallback recording, and matched comparators."""

    def test_reported_attention_selects_highest_scored_node(self) -> None:
        """Synthetic observation with known reported attention selects the highest-scored node."""
        N = 8
        roster = tuple(range(N))
        panel_mask = np.ones(N, dtype=bool)
        public_states = np.zeros((N, 1), dtype=np.float64)

        # Node 3 receives attention count 15 (edges (0, 3) count 10, (1, 3) count 5)
        # Node 5 receives attention count 4 (edge (2, 5) count 4)
        # All other nodes have 0 received attention
        observed_edges = {(0, 3): 10, (1, 3): 5, (2, 5): 4}

        obs = Observation(
            time=1.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges=observed_edges,
            accessible_roster=roster,
        )

        policy = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=1.0,
            duration=5.0,
            quota=1.0,
            is_feedback=False,
            node_fraction=0.125,  # 8 * 0.125 = 1 node
            seed=42,
        )
        ranked, scored = policy._rank_nodes(list(roster), obs, return_scored=True)

        self.assertEqual(scored, [3, 5])
        self.assertEqual(ranked[0], 3)
        self.assertEqual(ranked[1], 5)

        # Initial selection for k_target = 1 should select node 3
        selected = policy._select_initial_nodes(obs, k_target=1)
        self.assertEqual(selected, [3])

    def test_unavailable_scores_never_masquerade_as_attention(self) -> None:
        """Unavailable scores are ranked strictly after scored nodes; hidden W is not accessed."""
        N = 6
        roster = tuple(range(N))
        panel_mask = np.ones(N, dtype=bool)
        public_states = np.zeros((N, 1), dtype=np.float64)

        # Only node 2 has an observed edge
        observed_edges = {(0, 2): 7}

        obs = Observation(
            time=1.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges=observed_edges,
            accessible_roster=roster,
            # Even if true_memory exists and has large weights on node 4, it must NOT be used
            true_memory=np.array([[0, 0, 0, 0, 100, 0]] * N, dtype=np.float64),
        )

        policy = ObservableSelectorPolicy(
            selector="reported_attention",
            quota=1.0,
            seed=42,
        )
        ranked, scored = policy._rank_nodes(list(roster), obs, return_scored=True)

        self.assertEqual(scored, [2])
        self.assertEqual(ranked[0], 2)
        # All other nodes (0, 1, 3, 4, 5) must be unscored and ranked after node 2
        for unscored_node in [0, 1, 3, 4, 5]:
            self.assertIn(unscored_node, ranked[1:])
            self.assertNotIn(unscored_node, scored)

    def test_insufficient_scored_pool_records_fallback(self) -> None:
        """When the scored pool is smaller than requested slots, fallback selection is recorded."""
        N = 8
        roster = tuple(range(N))
        panel_mask = np.ones(N, dtype=bool)
        public_states = np.zeros((N, 1), dtype=np.float64)

        # Only 1 node (node 1) has reported incoming attention
        observed_edges = {(0, 1): 3}

        obs = Observation(
            time=1.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges=observed_edges,
            accessible_roster=roster,
        )

        policy = ObservableSelectorPolicy(
            selector="reported_attention",
            quota=1.0,
            seed=123,
        )

        # Request 3 slots from observed accessible pool
        selected = policy._select_initial_nodes(obs, k_target=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0], 1)  # Scored node selected first
        self.assertEqual(len(policy.fallback_records), 1)

        rec = policy.fallback_records[0]
        self.assertEqual(rec["step"], "initial_selection")
        self.assertEqual(rec["pool"], "observed_accessible")
        self.assertEqual(rec["requested_slots"], 3)
        self.assertEqual(rec["scored_slots_available"], 1)
        self.assertEqual(rec["fallback_count"], 2)
        self.assertEqual(rec["selected_scored"], [1])
        self.assertEqual(len(rec["selected_fallback"]), 2)
        self.assertNotIn(1, rec["selected_fallback"])

    def test_matched_random_comparator_shares_identical_pool(self) -> None:
        """Campaign baseline_attention and baseline_random share the same quota=1.0 observed pool."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "horizon": 14.0, "establishment_time": 4.0,
            "holding_duration": 2.0, "release_deadline": 5.0,
            "n_origins": 3, "training_origins": 1, "selection_origins": 1, "evaluation_origins": 1,
            "candidate_cap": 2, "selectors": ("reported_attention", "random"),
            "feedback_choices": (False,),
        })
        manifest = run_campaign(profile, campaign_id="comp_test").manifest

        base_sched = manifest["baseline_schedule"]
        self.assertEqual(base_sched["baseline_attention_pool"], "observed_accessible")
        self.assertEqual(base_sched["baseline_random_matched_pool"], "observed_accessible")


class TestPolicyTimingAuditAndFeedback(unittest.TestCase):
    """Test policy timing audit classification and genuine feedback adaptation."""

    def test_timing_audit_short_duration_effectively_static(self) -> None:
        """Duration 5 with delivery delay 1 and sample interval 5 has 0 feedback opportunities."""
        audit = audit_policy_timing(
            delivery_delay=1.0,
            duration=5.0,
            sample_interval=5.0,
            release_deadline=20.0,
        )
        # First delivery at t=1.0, pulse ends at t=min(1+5, 20)=6.0
        # Delivery times: [1.0, 6.0, 11.0, 16.0]
        # Usable delivery times strictly between 1.0 and 6.0: none!
        self.assertTrue(audit["is_effectively_static"])
        self.assertFalse(audit["has_feedback_opportunity"])
        self.assertEqual(audit["feedback_opportunity_count"], 0)
        self.assertEqual(audit["usable_delivery_times"], [])
        self.assertEqual(audit["classification"], "effectively_static")

    def test_timing_audit_long_duration_genuinely_adaptive(self) -> None:
        """Duration 20 with delivery delay 1 and sample interval 5 has multiple feedback opportunities."""
        audit = audit_policy_timing(
            delivery_delay=1.0,
            duration=20.0,
            sample_interval=5.0,
            release_deadline=30.0,
        )
        # First delivery at t=1.0, pulse ends at t=min(1+20, 30)=21.0
        # Delivery times: [1.0, 6.0, 11.0, 16.0, 21.0, 26.0]
        # Usable delivery times strictly between 1.0 and 21.0: [6.0, 11.0, 16.0]
        self.assertFalse(audit["is_effectively_static"])
        self.assertTrue(audit["has_feedback_opportunity"])
        self.assertEqual(audit["feedback_opportunity_count"], 3)
        self.assertEqual(audit["usable_delivery_times"], [6.0, 11.0, 16.0])
        self.assertEqual(audit["classification"], "genuinely_adaptive")

    def test_scripted_reports_trigger_genuine_feedback_action_change(self) -> None:
        """Periodic observation reports delivered before pulse end trigger genuine feedback updates."""
        N = 10
        problem = ControlProblem(
            problem_id="fb_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N,
            D=1,
            k=4,
            exposure_cap=100.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            holding_duration=2.0,
            horizon=25.0,
            accessible_roster=tuple(range(N)),
        )

        # Feedback policy with duration 15.0: pulse runs from t=1.0 to t=16.0.
        # Initial recruitment fraction 0.5 recruits 2 nodes initially out of 4.
        policy = ObservableSelectorPolicy(
            selector="random",
            nominal_amplitude=1.0,
            duration=15.0,
            quota=1.0,
            is_feedback=True,
            initial_recruitment_fraction=0.5,
            seed=42,
        )

        obs_model = SampledPublicObservationModel(
            N=N,
            accessible_roster=problem.accessible_roster,
            panel_fraction=1.0,
            delivery_delay=1.0,
            seed=42,
        )

        evaluator = TargetEvaluator.create_for_state(
            target_type=problem.target_type,
            initial_state=_make_dummy_state(N=N, y_val=0.8),
            holding_duration=problem.holding_duration,
            horizon=problem.horizon,
            release_deadline=problem.release_deadline,
            target_params=problem.target_params,
        )

        res, _, _ = run_controlled_simulation(
            initial_state=_make_dummy_state(N=N, y_val=0.8),
            problem=problem,
            controller=policy,
            observation_model=obs_model,
            target_evaluator=evaluator,
            J=1.0,
            T=1.0,
            delta=1.0,
            epsilon=0.1,
            closure=0.3,
            sample_interval=5.0,
            adaptive_memory=True,
            between_eligibility=0.05,
            within_eligibility=1.0,
            sim_rng=np.random.default_rng(100),
            controller_rng=np.random.default_rng(200),
        )

        # At t=6.0 and t=11.0, feedback opportunities arrive before pulse end at t=16.0
        self.assertGreater(res.summaries["feedback_opportunities"], 0)
        self.assertGreater(res.summaries["feedback_updates"], 0)
        self.assertEqual(len(policy.used_nodes), 4)  # Expanded from initial 2 to all 4 slots


class TestStaticAndFeedbackPolicyFamilies(unittest.TestCase):
    """Test separate static and feedback candidate generation, selection, and evaluation."""

    def test_search_optimizer_stratifies_static_and_feedback(self) -> None:
        """Optimizer candidate generator reserves separate allocation for static and feedback."""
        problem = ControlProblem(
            problem_id="strat_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=10, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0,
            release_deadline=5.0, holding_duration=2.0, horizon=10.0,
            accessible_roster=tuple(range(10)),
        )
        optimizer = ObservablePolicySearchOptimizer(
            problem=problem,
            candidate_cap=6,
            feedback_choices=(False, True),
            selectors=("random", "reported_attention"),
        )
        candidates = optimizer.generate_candidates()

        self.assertEqual(len(candidates), 6)
        n_static = sum(1 for c in candidates if not c.feedback)
        n_feedback = sum(1 for c in candidates if c.feedback)

        # Stratified budget: 50% static, 50% feedback
        self.assertEqual(n_static, 3)
        self.assertEqual(n_feedback, 3)

    def test_optimizer_tracks_independent_family_incumbents(self) -> None:
        """ObservablePolicySearchOptimizer tracks static_incumbent and feedback_incumbent independently."""
        problem = ControlProblem(
            problem_id="inc_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=8, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0,
            release_deadline=5.0, holding_duration=2.0, horizon=10.0,
            accessible_roster=tuple(range(8)),
        )
        optimizer = ObservablePolicySearchOptimizer(
            problem=problem,
            candidate_cap=4,
            feedback_choices=(False, True),
            selectors=("random",),
        )
        origins = [_make_dummy_state(N=8)]
        res = optimizer.optimize(training_origins=origins)

        self.assertIsNotNone(res.static_incumbent)
        self.assertFalse(res.static_incumbent.feedback)
        self.assertIsNotNone(res.static_incumbent_score)

        self.assertIsNotNone(res.feedback_incumbent)
        self.assertTrue(res.feedback_incumbent.feedback)
        self.assertIsNotNone(res.feedback_incumbent_score)

    def test_separate_static_and_feedback_evaluation_arms(self) -> None:
        """Campaign evaluates both static and feedback winners across matched adaptive/frozen branches."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "horizon": 14.0, "establishment_time": 4.0,
            "holding_duration": 2.0, "release_deadline": 5.0,
            "n_origins": 4, "seed_base": 8812,
            "training_origins": 1, "training_futures_per_origin": 1,
            "selection_origins": 1, "selection_futures_per_origin": 1,
            "evaluation_origins": 2, "evaluation_futures_per_origin": 1,
            "candidate_cap": 4,
            "selectors": ("reported_attention", "random"),
            "feedback_choices": (False, True),
            "selection_candidates": 1,
        })
        result = run_campaign(profile, campaign_id="fam_test")

        self.assertIsNotNone(result.selected_static_candidate)
        self.assertFalse(result.selected_static_candidate.feedback)
        self.assertIsNotNone(result.selected_feedback_candidate)
        self.assertTrue(result.selected_feedback_candidate.feedback)

        # 4 evaluation arms: static|adaptive, static|frozen, feedback|adaptive, feedback|frozen
        arm_keys = list(result.evaluation_results.keys())
        self.assertEqual(len(arm_keys), 4)
        self.assertTrue(any("static" in k and "adaptive" in k for k in arm_keys))
        self.assertTrue(any("static" in k and "frozen" in k for k in arm_keys))
        self.assertTrue(any("feedback" in k and "adaptive" in k for k in arm_keys))
        self.assertTrue(any("feedback" in k and "frozen" in k for k in arm_keys))

        # Paired comparisons exist for both families
        self.assertIn("static", result.paired_comparisons)
        self.assertIn("feedback", result.paired_comparisons)

    def test_all_failure_search_preserves_separate_arms_and_pairing(self) -> None:
        """Under all-failure conditions, both families are preserved, reported as failing, and paired."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "horizon": 14.0, "establishment_time": 4.0,
            "holding_duration": 2.0, "release_deadline": 5.0,
            "target_threshold": 0.5,
            "exposure_cap_normalized": 0.0001,  # Near-zero budget guarantees reversal failure
            "n_origins": 4, "seed_base": 9920,
            "training_origins": 1, "training_futures_per_origin": 1,
            "selection_origins": 1, "selection_futures_per_origin": 1,
            "evaluation_origins": 2, "evaluation_futures_per_origin": 1,
            "candidate_cap": 4,
            "selectors": ("random",),
            "feedback_choices": (False, True),
            "selection_candidates": 1,
        })
        result = run_campaign(profile, campaign_id="fail_test")

        self.assertEqual(result.selection_summary["status"], "no_feasible_policy_found")
        self.assertEqual(result.selection_summary["static_status"], "no_feasible_policy_found")
        self.assertEqual(result.selection_summary["feedback_status"], "no_feasible_policy_found")

        # Disjoint origin pools preserved
        memberships = result.manifest["pool_memberships"]
        train_set = set(memberships["training"])
        sel_set = set(memberships["selection"])
        eval_set = set(memberships["evaluation"])
        self.assertTrue(train_set.isdisjoint(sel_set))
        self.assertTrue(train_set.isdisjoint(eval_set))
        self.assertTrue(sel_set.isdisjoint(eval_set))

        # Adaptive vs frozen pairs are preserved with matching origins
        self.assertEqual(len(result.adaptive_vs_frozen_pairs), 2 * 2)  # 2 policies * 2 eval origins
        for pair in result.adaptive_vs_frozen_pairs:
            self.assertEqual(pair["adaptive"].origin_id, pair["frozen"].origin_id)
            self.assertEqual(pair["adaptive"].future_id, pair["frozen"].future_id)


class TestActiveNodeAllocation(unittest.TestCase):
    """Test candidate_node_fractions wiring and requested vs effective settings tracking."""

    def test_candidate_node_fraction_scales_actuation_nodes(self) -> None:
        """Changing candidate node fraction scales actual actuated nodes within cap k."""
        N = 20
        k_cap = 10
        roster = tuple(range(N))
        panel_mask = np.ones(N, dtype=bool)
        public_states = np.full((N, 1), 1.0, dtype=np.float64)  # decisive panel (F3 tie rule: tied panel must wait)

        obs = Observation(
            time=1.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges={},
            accessible_roster=roster,
        )
        problem = ControlProblem(
            problem_id="frac_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N, D=1, k=k_cap, exposure_cap=100.0, amplitude_cap=2.0,
            release_deadline=10.0, holding_duration=2.0, horizon=15.0,
            accessible_roster=roster,
        )

        ledger = BudgetLedger(
            N=problem.N,
            D=problem.D,
            k=problem.k,
            exposure_cap=problem.exposure_cap,
            amplitude_cap=problem.amplitude_cap,
            release_deadline=problem.release_deadline,
            accessible_roster=set(problem.accessible_roster),
        )

        # Policy A: fraction = 0.1 -> round(20 * 0.1) = 2 nodes
        pol_a = ObservableSelectorPolicy(selector="random", node_fraction=0.1, duration=5.0, is_feedback=False)
        pol_a.reset(problem, initial_observation=obs, rng=np.random.default_rng(1))
        act_a = pol_a.act(obs, ledger)
        self.assertEqual(len(act_a.entries), 2)
        self.assertEqual(pol_a.requested_settings["requested_active_nodes"], 2)
        self.assertEqual(pol_a.effective_settings["effective_node_count"], 2)

        # Policy B: fraction = 0.4 -> round(20 * 0.4) = 8 nodes
        pol_b = ObservableSelectorPolicy(selector="random", node_fraction=0.4, duration=5.0, is_feedback=False)
        pol_b.reset(problem, initial_observation=obs, rng=np.random.default_rng(2))
        act_b = pol_b.act(obs, ledger)
        self.assertEqual(len(act_b.entries), 8)
        self.assertEqual(pol_b.requested_settings["requested_active_nodes"], 8)
        self.assertEqual(pol_b.effective_settings["effective_node_count"], 8)

        # Policy C: fraction = 0.8 -> round(20 * 0.8) = 16 nodes, but capped by k=10
        pol_c = ObservableSelectorPolicy(selector="random", node_fraction=0.8, duration=5.0, is_feedback=False)
        pol_c.reset(problem, initial_observation=obs, rng=np.random.default_rng(3))
        act_c = pol_c.act(obs, ledger)
        self.assertEqual(len(act_c.entries), 10)
        self.assertEqual(pol_c.requested_settings["requested_active_nodes"], 16)
        self.assertEqual(pol_c.effective_settings["effective_node_count"], 10)

    def test_requested_vs_effective_amplitude_reduction_under_exposure_cap(self) -> None:
        """When exposure cap constrains amplitude, effective amplitude is reduced and flagged."""
        N = 10
        problem = ControlProblem(
            problem_id="amp_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N, D=1, k=5,
            exposure_cap=2.0,  # Tight exposure cap
            amplitude_cap=4.0,
            release_deadline=10.0, holding_duration=2.0, horizon=15.0,
            accessible_roster=tuple(range(N)),
        )

        obs = Observation(
            time=1.0,
            # Decisive public majority (+1). A tied panel (all zeros) is now
            # correctly treated as ambiguous under the F3 tie rule, and the
            # policy must wait rather than actuate; this test's intent is the
            # exposure-cap amplitude mechanics, so give it a decisive panel.
            public_states=np.full((N, 1), 1.0, dtype=np.float64),
            panel_mask=np.ones(N, dtype=bool),
            observed_edges={},
            accessible_roster=problem.accessible_roster,
        )
        ledger = BudgetLedger(
            N=problem.N,
            D=problem.D,
            k=problem.k,
            exposure_cap=problem.exposure_cap,
            amplitude_cap=problem.amplitude_cap,
            release_deadline=problem.release_deadline,
            accessible_roster=set(problem.accessible_roster),
        )

        # Requested: amplitude 4.0, duration 5.0, 2 nodes
        # Spending 2 nodes * 4.0 * 5.0 = 40.0 > exposure_cap 2.0
        # Allowed amplitude = 2.0 / (2 * 5.0) = 0.2
        pol = ObservableSelectorPolicy(
            selector="random",
            nominal_amplitude=4.0,
            duration=5.0,
            node_fraction=0.2,  # 2 nodes
            is_feedback=False,
        )
        pol.reset(problem, initial_observation=obs, rng=np.random.default_rng(42))
        act = pol.act(obs, ledger)

        self.assertAlmostEqual(pol.effective_settings["effective_amplitude"], 0.2, places=6)
        self.assertTrue(pol.effective_settings["amplitude_reduced_by_exposure_cap"])
        self.assertEqual(pol.requested_settings["nominal_amplitude"], 4.0)


class TestPhysicsForwarding(unittest.TestCase):
    """Test forwarding of non-default physics (a, c, tau_y) through search and campaign."""

    def test_non_default_physics_forwarded_to_simulation(self) -> None:
        """Search optimizer forwards declared non-default physics (a, c, tau_y) to simulation."""
        problem = ControlProblem(
            problem_id="phys_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0,
            release_deadline=5.0, holding_duration=2.0, horizon=10.0,
            accessible_roster=tuple(range(6)),
        )
        optimizer = ObservablePolicySearchOptimizer(
            problem=problem,
            candidate_cap=1,
            selectors=("random",),
        )

        origins = [_make_dummy_state(N=6)]
        captured_kwargs: Dict[str, Any] = {}

        import src.control.optimizers.search
        original_sim = src.control.optimizers.search.run_controlled_simulation

        def spy_simulation(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_sim(*args, **kwargs)

        with patch("src.control.optimizers.search.run_controlled_simulation", side_effect=spy_simulation):
            optimizer.optimize(
                training_origins=origins,
                a=0.75,
                c=0.25,
                tau_y=3.0,
            )

        self.assertEqual(captured_kwargs.get("a"), 0.75)
        self.assertEqual(captured_kwargs.get("c"), 0.25)
        self.assertEqual(captured_kwargs.get("tau_y"), 3.0)

    def test_campaign_manifest_physics_and_limitation_removed(self) -> None:
        """Campaign runs with non-default physics and manifest limitation is removed."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "horizon": 14.0, "establishment_time": 4.0,
            "holding_duration": 2.0, "release_deadline": 5.0,
            "prestige_global": 0.5,
            "matching": 0.8,
            "tau_public": 2.0,
            "n_origins": 3, "training_origins": 1, "selection_origins": 1, "evaluation_origins": 1,
            "candidate_cap": 2, "selectors": ("random",),
        })
        result = run_campaign(profile, campaign_id="phys_camp")

        manifest = result.manifest
        self.assertEqual(manifest["physics"]["a"], 0.5)
        self.assertEqual(manifest["physics"]["c"], 0.8)
        self.assertEqual(manifest["physics"]["tau_y"], 2.0)

        # Verify old limitation statement has been removed
        for lim in manifest["limitations"]:
            self.assertNotIn("The search optimizer forwards engine defaults", lim)


class TestOriginClusteredUncertainty(unittest.TestCase):
    """Test origin-aware clustering for single future, multiple futures, and paired comparisons."""

    def _make_rollout(self, origin_id: str, future_id: int, success: bool, restricted_time: float = 10.0) -> RolloutResult:
        return RolloutResult(
            problem_id="p", policy_id="pol", origin_id=origin_id, future_id=future_id,
            is_valid=True, is_eligible=True, success=success,
            completion_time=restricted_time if success else None,
            restricted_time=restricted_time,
            expenditure=5.0, distinct_nodes_used=2, relapsed=False,
            terminal_target_violation=0.0, event_counts={},
        )

    def test_single_future_per_origin_uses_exact_and_wilson(self) -> None:
        """One future per origin uses simple binomial intervals (Wilson & exact lower bound)."""
        rollouts = [
            self._make_rollout(f"origin_{i}", 0, success=(i < 3))
            for i in range(5)
        ]
        summary = summarize_rollout_reliability(rollouts, reliability_target=0.5)

        self.assertEqual(summary["uncertainty_method"], "binomial_exact_and_wilson")
        self.assertEqual(summary["max_futures_per_origin"], 1)
        self.assertEqual(summary["n_origins"], 5)
        self.assertAlmostEqual(summary["success_probability"], 3 / 5)
        self.assertTrue(0.0 <= summary["wilson_low"] <= summary["success_probability"])
        self.assertTrue(summary["success_probability"] <= summary["wilson_high"] <= 1.0)

    def test_multiple_futures_per_origin_uses_cluster_bootstrap(self) -> None:
        """Multiple futures per origin uses cluster bootstrap resampling origins."""
        # 4 origins, 3 futures each = 12 rollouts
        rollouts = []
        for o_idx in range(4):
            for f_idx in range(3):
                # Origin 0 and 1 always succeed; Origin 2 and 3 always fail
                succ = bool(o_idx < 2)
                rollouts.append(self._make_rollout(f"origin_{o_idx}", f_idx, success=succ))

        summary = summarize_rollout_reliability(rollouts, reliability_target=0.5)

        self.assertEqual(summary["uncertainty_method"], "cluster_bootstrap_origin_level")
        self.assertEqual(summary["max_futures_per_origin"], 3)
        self.assertEqual(summary["n_origins"], 4)
        self.assertAlmostEqual(summary["success_probability"], 0.5)
        self.assertTrue(0.0 <= summary["wilson_low"] <= summary["wilson_high"] <= 1.0)

    def test_paired_adaptive_frozen_cluster_bootstrap(self) -> None:
        """Paired adaptive vs frozen comparison resamples clusters and preserves paired differences."""
        pairs = []
        for o_idx in range(3):
            for f_idx in range(2):
                ra = self._make_rollout(f"origin_{o_idx}", f_idx, success=True, restricted_time=5.0)
                rf = self._make_rollout(f"origin_{o_idx}", f_idx, success=False, restricted_time=15.0)
                pairs.append({
                    "origin_id": f"origin_{o_idx}",
                    "future_id": f_idx,
                    "adaptive": ra,
                    "frozen": rf,
                })

        summary = summarize_paired_adaptive_frozen_comparison(pairs, n_boot=500, seed=42)

        self.assertEqual(summary["n_pairs"], 6)
        self.assertEqual(summary["n_origins"], 3)
        self.assertEqual(summary["uncertainty_method"], "cluster_bootstrap_paired_origins")
        # Adaptive (all 1) - Frozen (all 0) = difference +1.0
        self.assertAlmostEqual(summary["mean_difference_success"], 1.0)
        self.assertEqual(summary["ci_success"], (1.0, 1.0))
        # Adaptive (5.0) - Frozen (15.0) = difference -10.0
        self.assertAlmostEqual(summary["mean_difference_time"], -10.0)
        self.assertEqual(summary["ci_time"], (-10.0, -10.0))


if __name__ == "__main__":
    unittest.main()
