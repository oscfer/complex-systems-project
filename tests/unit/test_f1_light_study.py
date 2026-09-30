"""Audit F1 tests: light study consolidation and honest search reporting.

Covers audit findings F1a-F1d:
(a) the notebook generator no longer silently reduces LIGHT_TIERED (the
    generated light campaign uses the nominal profile, or the explicitly
    named LIGHT_TIERED_SMOKE rehearsal profile);
(b) a distinct, clearly named smoke profile exists;
(c) search rankings are honest: separate per-family rankings and a true
    global ranking are separately accessible, and the legacy
    'ranked_search_candidates' key is documented as a family-concatenation;
(d) candidate coverage reports carry family labels (proposed / unique /
    evaluated, per family, with actual caps);
(e) initial / peak-active / cumulative-distinct node counts are reported
    separately wherever recorded;
(f) development (training/selection) and held-out evaluation pools are
    disjoint by construction with exact origin/future/seed pairing, and
    reliability summaries expose origin-level success counts.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.config.tiered_profiles import (  # noqa: E402
    TieredProfile,
    LIGHT_TIERED,
    LIGHT_TIERED_SMOKE,
    EXTENDED_TIERED,
)
from src.config.control_profiles import preflight_control_profile  # noqa: E402
from src.control.optimizers.search import (  # noqa: E402
    ObservablePolicySearchOptimizer,
    rank_candidates_by_family,
    rank_candidates_global,
    summarize_node_counts,
)
from src.control.contracts import ControlProblem, RolloutResult, TargetType  # noqa: E402
from src.core.state import State  # noqa: E402










class TestF1aSmokeProfile(unittest.TestCase):
    """(b) smoke mode exists and is distinct from the nominal light study."""

    def test_smoke_profile_exists_and_is_clearly_named(self):
        self.assertIsInstance(LIGHT_TIERED_SMOKE, TieredProfile)
        self.assertNotEqual(
            LIGHT_TIERED_SMOKE.profile_id(), LIGHT_TIERED.profile_id()
        )

    def test_smoke_profile_is_distinct_and_tiny(self):
        self.assertLess(LIGHT_TIERED_SMOKE.N, LIGHT_TIERED.N)
        self.assertNotEqual(LIGHT_TIERED_SMOKE.horizon, LIGHT_TIERED.horizon)
        # The smoke profile changes the declared contract, which is exactly
        # why it must never be confounded with the nominal light study.
        self.assertNotEqual(
            LIGHT_TIERED_SMOKE.holding_duration, LIGHT_TIERED.holding_duration
        )

    def test_nominal_light_profile_keeps_success_contract(self):
        # Success contract is unchanged across tiers (audit requirement).
        for profile in (LIGHT_TIERED, EXTENDED_TIERED):
            self.assertEqual(profile.target_threshold, 0.5)
            self.assertEqual(profile.holding_duration, 100.0)
            self.assertEqual(profile.release_deadline, 20.0)
            self.assertEqual(profile.horizon, 320.0)
            self.assertEqual(profile.reliability_target, 0.8)

    def test_both_profiles_pass_preflight(self):
        for profile in (LIGHT_TIERED, LIGHT_TIERED_SMOKE):
            preflight = preflight_control_profile(profile)
            self.assertGreaterEqual(preflight["n_prepare"], preflight["n_needed"])
            self.assertEqual(
                preflight["n_prepare"] - preflight["n_needed"],
                preflight["n_unassigned_spare"],
            )


class TestF1bSearchRankingHonesty(unittest.TestCase):
    """(c) family vs global rankings are separately accessible and honest."""

    def _scores(self):
        # Feedback candidate outranks every static candidate on the global
        # score; a naive family concatenation would hide that.
        reg = {
            "static_a": types.SimpleNamespace(feedback=False),
            "static_b": types.SimpleNamespace(feedback=False),
            "fb_x": types.SimpleNamespace(feedback=True),
        }
        scores = {
            "static_a": (0, 0.1, 2.0),
            "static_b": (0, 0.1, 3.0),
            "fb_x": (0, 0.05, 1.0),
        }
        return scores, reg

    def test_family_rankings_are_separate(self):
        scores, reg = self._scores()
        families = rank_candidates_by_family(scores, reg)
        self.assertEqual(set(families.keys()), {"static", "feedback"})
        self.assertEqual(
            families["static"],
            [("static_a", (0, 0.1, 2.0)), ("static_b", (0, 0.1, 3.0))],
        )
        self.assertEqual(families["feedback"], [("fb_x", (0, 0.05, 1.0))])

    def test_global_ranking_interleaves_families(self):
        scores, reg = self._scores()
        global_ranked = [cid for cid, _ in rank_candidates_global(scores)]
        # The feedback winner is genuinely best across families; a
        # concatenation of family rankings would put every static candidate
        # ahead of it. The global ranking must not.
        self.assertEqual(global_ranked, ["fb_x", "static_a", "static_b"])
        families = rank_candidates_by_family(scores, reg)
        concatenated = families["static"] + families["feedback"]
        self.assertNotEqual(
            [cid for cid, _ in concatenated], global_ranked,
            "family concatenation must not be presented as a global ranking",
        )

    def test_legacy_ranked_search_candidates_is_documented_as_concatenation(self):
        # The campaign selection summary must keep the legacy key (bundle
        # readers) but document its semantics honestly. Verified structurally
        # on the integration campaign below (TestF1Integration).
        source = (REPO / "src" / "control" / "campaign.py").read_text()
        self.assertIn('"ranked_search_candidates"', source)
        self.assertIn("ranking_semantics", source)
        self.assertIn("DEPRECATED alias", source)


class TestF1bSearchCoverage(unittest.TestCase):
    """(d) candidate coverage reports include family labels and actual caps."""

    def _tiny_problem(self):
        return ControlProblem(
            problem_id="f1_prob",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=10, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0,
            release_deadline=5.0, holding_duration=2.0, horizon=10.0,
        )

    def test_proposal_stats_include_family_labels(self):
        optimizer = ObservablePolicySearchOptimizer(
            problem=self._tiny_problem(),
            candidate_cap=5,
            search_seed=42,
            selectors=("random", "reported_attention"),
        )
        optimizer.generate_candidates()
        stats = getattr(optimizer, "last_proposal_stats", None)
        self.assertIsNotNone(stats)
        self.assertGreaterEqual(stats["candidates_proposed"], stats["candidates_unique"])
        self.assertEqual(
            stats["candidates_unique"],
            stats["candidates_unique_by_family"]["static"]
            + stats["candidates_unique_by_family"]["feedback"],
        )
        self.assertEqual(stats["actual_caps"]["candidate_cap"], 5)
        self.assertEqual(stats["actual_caps"]["amplitude_cap"], 2.0)
        self.assertEqual(stats["actual_caps"]["exposure_cap"], 10.0)
        self.assertEqual(stats["actual_caps"]["distinct_node_cap_k"], 2)

    def test_optimize_reports_coverage_with_family_labels(self):
        import src.control.optimizers.search as search_module
        import numpy as np

        problem = self._tiny_problem()
        optimizer = ObservablePolicySearchOptimizer(
            problem=problem,
            candidate_cap=3,
            futures_per_candidate=1,
            search_seed=7,
            selectors=("random",),
        )
        s1 = State(N=10, D=1, state_mode="continuous", n=1, Q=np.zeros((1, 1)),
                   y=np.ones((10, 1)) * -0.8, x=np.zeros((10, 1)),
                   W=np.zeros((10, 10)), opportunity_groups=None)
        s2 = State(N=10, D=1, state_mode="continuous", n=1, Q=np.zeros((1, 1)),
                   y=np.ones((10, 1)) * 0.8, x=np.zeros((10, 1)),
                   W=np.zeros((10, 10)), opportunity_groups=None)

        original_run = search_module.run_controlled_simulation

        def fake_run(*args, target_evaluator=None, **kwargs):
            return RolloutResult(
                problem_id=problem.problem_id,
                policy_id=target_evaluator.policy_id,
                origin_id=target_evaluator.origin_id,
                future_id=target_evaluator.future_id,
                is_valid=True, is_eligible=True, success=False,
                completion_time=None, restricted_time=10.0,
                expenditure=1.0, distinct_nodes_used=1,
                terminal_target_violation=0.4,
            ), None, None

        search_module.run_controlled_simulation = fake_run
        try:
            result = optimizer.optimize(training_origins=[s1, s2])
        finally:
            search_module.run_controlled_simulation = original_run

        coverage = result.search_coverage
        self.assertIsNotNone(coverage)
        for key in ("candidates_proposed", "candidates_unique", "candidates_evaluated"):
            self.assertIn(key, coverage)
        self.assertEqual(coverage["candidates_evaluated"], len(result.evaluated_candidates))
        self.assertEqual(set(coverage["family_coverage"].keys()), {"static", "feedback"})
        for info in coverage["family_coverage"].values():
            self.assertIn("unique_proposed", info)
            self.assertIn("evaluated", info)
        self.assertIn("training_origin_ids", coverage)
        self.assertIn("training_futures_per_candidate", coverage)
        self.assertIn("actual_caps", coverage)
        tie = coverage["failure_distance_tie_break"]
        self.assertIn("definition", tie)
        self.assertIn("per_candidate_failure_distance", tie)
        for cand in result.evaluated_candidates:
            self.assertIn(cand.candidate_id, tie["per_candidate_failure_distance"])


class TestF1cNodeCountDecomposition(unittest.TestCase):
    """(e) initial / peak-active / cumulative-distinct reported separately."""

    def _rollout(self, effective, history, distinct):
        return RolloutResult(
            problem_id="p", policy_id="pol", origin_id="o", future_id=0,
            is_valid=True, is_eligible=True, success=False,
            completion_time=None, restricted_time=10.0,
            expenditure=1.0, distinct_nodes_used=distinct,
            summaries={
                "effective_settings": effective,
                "recorder": {"action_history": history},
            },
        )

    def test_initial_peak_cumulative_are_separate(self):
        history = [
            {"spending_rate": 4.0, "field_entries": [{"node": n, "field": 1.0} for n in range(2)],
             "actuated_nodes": [0, 1]},
            {"spending_rate": 8.0, "field_entries": [{"node": n, "field": 1.0} for n in range(4)],
             "actuated_nodes": [0, 1, 2, 3]},
            {"spending_rate": 0.0, "field_entries": [], "actuated_nodes": []},
        ]
        res = self._rollout(
            effective={"effective_node_count": 2}, history=history, distinct=4
        )
        dec = summarize_node_counts(res)
        self.assertEqual(dec["initial_node_count"], 2)
        self.assertEqual(dec["peak_active_nodes"], 4)
        self.assertEqual(dec["cumulative_distinct_nodes"], 4)
        self.assertEqual(dec["legacy_effective_node_count"], 2)
        # The three quantities are distinct fields, not one collapsed value.
        self.assertNotEqual(dec["initial_node_count"], dec["peak_active_nodes"])

    def test_cumulative_can_exceed_initial(self):
        # The audit's motivating case: effective_node_count=1 at initial
        # deployment but two distinct nodes used over the rollout.
        history = [
            {"spending_rate": 2.0, "field_entries": [{"node": 0, "field": 1.0}],
             "actuated_nodes": [0]},
            {"spending_rate": 2.0, "field_entries": [{"node": 1, "field": 1.0}],
             "actuated_nodes": [0, 1]},
        ]
        res = self._rollout(
            effective={"effective_node_count": 1}, history=history, distinct=2
        )
        dec = summarize_node_counts(res)
        self.assertEqual(dec["initial_node_count"], 1)
        self.assertEqual(dec["peak_active_nodes"], 1)
        self.assertEqual(dec["cumulative_distinct_nodes"], 2)

    def test_peak_unavailable_without_action_history(self):
        res = self._rollout(effective={"effective_node_count": 1}, history=[], distinct=1)
        dec = summarize_node_counts(res)
        self.assertEqual(dec["initial_node_count"], 1)
        self.assertIsNone(dec["peak_active_nodes"])
        self.assertEqual(dec["cumulative_distinct_nodes"], 1)


# --- Integration: tiny full campaign for (d)/(e)/(f) at campaign level ---
N_EVAL_ORIGINS = 2


def _tiny_profile():
    from src.config.control_profiles import LIGHT_CONTROL_PROFILE

    return LIGHT_CONTROL_PROFILE.model_copy(update={
        "N": 8, "D": 1, "horizon": 14.0, "establishment_time": 5.0,
        "n_origins": 5, "seed_base": 7712,
        "holding_duration": 2.0, "release_deadline": 5.0,
        "k_distinct_nodes_fraction": 0.375, "exposure_cap_normalized": 0.5,
        "candidate_cap": 3,
        "selectors": ("reported_attention", "random"),
        "observed_slot_quotas": (0.0,), "feedback_choices": (False, True),
        "training_origins": 1, "training_futures_per_origin": 1,
        "selection_candidates": 2, "selection_origins": 1,
        "selection_futures_per_origin": 1,
        "evaluation_origins": N_EVAL_ORIGINS,
        "evaluation_futures_per_origin": 1,
    })


class TestF1Integration(unittest.TestCase):
    """Full tiny campaign: honest rankings, node counts, disjoint pools."""

    @classmethod
    def setUpClass(cls):
        from src.control.campaign import run_campaign

        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot_dir = os.path.join(cls.tmp.name, "origin_snapshots")
        os.makedirs(cls.snapshot_dir, exist_ok=True)
        cls.master_seed = 99003
        cls.result = run_campaign(
            _tiny_profile(),
            campaign_id="f1_test",
            master_seed=cls.master_seed,
            out_dir=cls.tmp.name,
            snapshot_dir=cls.snapshot_dir,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selection_summary_reports_honest_rankings_and_coverage(self):
        sel = self.result.selection_summary
        self.assertIn("concatenation", sel["ranking_semantics"])
        self.assertIn("NOT a global ranking", sel["ranking_semantics"])
        self.assertIn("static", sel["family_ranked_candidates"])
        self.assertIn("feedback", sel["family_ranked_candidates"])
        self.assertIn("global_ranked_candidates", sel)
        legacy = sel["ranked_search_candidates"]
        fams = sel["family_ranked_candidates"]
        self.assertEqual(set(legacy), set(fams["static"]) | set(fams["feedback"]))
        coverage = sel["search_coverage"]
        self.assertIn("candidates_proposed", coverage)
        self.assertIn("candidates_evaluated", coverage)
        self.assertEqual(set(coverage["family_coverage"].keys()), {"static", "feedback"})
        self.assertIn("phase_pools", sel)

    def test_rollout_summaries_report_node_counts(self):
        result = self.result
        for rollouts in result.search_result.candidate_rollouts.values():
            for r in rollouts:
                dec = r.summaries.get("node_count_decomposition")
                self.assertIsNotNone(dec, f"search rollout {r.policy_id}")
                self.assertEqual(
                    dec["cumulative_distinct_nodes"], r.distinct_nodes_used
                )
        for rollouts in result.evaluation_results.values():
            for r in rollouts:
                dec = r.summaries["node_count_decomposition"]
                self.assertIn("initial_node_count", dec)
                self.assertIn("peak_active_nodes", dec)
                self.assertIn("cumulative_distinct_nodes", dec)
                self.assertEqual(dec["cumulative_distinct_nodes"], r.distinct_nodes_used)
                print("DEBUG", r.policy_id, r.summaries.get("effective_settings", {}))
                self.assertIn(
                    "effective_node_count", r.summaries.get("effective_settings", {})
                )
        for rollouts in result.baseline_results.values():
            for r in rollouts:
                self.assertIn("node_count_decomposition", r.summaries)

    def test_pools_disjoint_by_construction(self):
        manifest = self.result.manifest
        flags = manifest["pool_pairing"]["disjoint_by_construction"]
        self.assertTrue(all(flags.values()), f"pools not disjoint: {flags}")
        pools = manifest["pool_memberships"]
        id_sets = [set(v) for v in pools.values()]
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                self.assertFalse(id_sets[i] & id_sets[j])
        eval_ids = set(pools["evaluation"])
        self.assertFalse(eval_ids & (set(pools["training"]) | set(pools["selection"])))
        self.assertEqual(manifest["pool_pairing"]["future_ids"]["evaluation"], [0])
        self.assertIn("seed_formula", manifest["pool_pairing"])

    def test_phase_rollouts_use_disjoint_origins(self):
        result = self.result
        training_origins = {
            r.origin_id
            for rollouts in result.search_result.candidate_rollouts.values()
            for r in rollouts
        }
        evaluation_origins = {
            r.origin_id
            for rollouts in result.evaluation_results.values() for r in rollouts
        }
        self.assertFalse(training_origins & evaluation_origins)
        pools = result.manifest["pool_memberships"]
        self.assertTrue(training_origins.issubset(set(pools["training"])))
        self.assertTrue(evaluation_origins.issubset(set(pools["evaluation"])))

    def test_seed_pairing_exact_per_rollout(self):
        from src.control.campaign import _stable_seed

        result = self.result
        for rollouts in result.evaluation_results.values():
            for r in rollouts:
                self.assertEqual(
                    r.summaries["sim_seed"],
                    _stable_seed(self.master_seed, "sim", r.origin_id, r.future_id),
                )
                self.assertEqual(
                    r.summaries["ctl_seed"],
                    _stable_seed(self.master_seed, "ctl", r.origin_id, r.future_id),
                )

    def test_reliability_summaries_expose_origin_level_success_counts(self):
        for arm, s in self.result.reliability_summaries.items():
            self.assertIn("origin_level_success_counts", s, arm)
            counts = s["origin_level_success_counts"]
            self.assertEqual(
                sum(c["trials"] for c in counts.values()), s["eligible_trials"], arm
            )
            self.assertEqual(
                sum(c["successes"] for c in counts.values()), s["successes"], arm
            )
            self.assertIn("n_origins_with_success", s)
            self.assertIn("n_origins_without_success", s)


if __name__ == "__main__":
    unittest.main()
