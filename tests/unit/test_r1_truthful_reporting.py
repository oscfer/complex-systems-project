"""R1 targeted tests: truthful preparation, configuration, and outcome reporting.

Deterministic fixtures only — no Colab rerun, no hour-long campaign:

1. Rejection accounting: the historical Colab1 pattern (8 accepted origins with
   indices ending at 74) yields 8/75, not 8/8.
2. Profile preflight: both default profiles and generated effective profiles
   pass cheaply (no simulation); invalid `model_copy(update=...)` profiles are
   rejected before a runner can be built.
3. Preparation attempt records from a tiny deterministic preparation run.
4. All-failure summaries report no feasible policy, failure-penalty times, and
   conditional relapse N/A.
5. Explicit outcome statuses and the selection tie-break reason.
"""

import os
import tempfile
import unittest

from src.config.control_profiles import (
    EXTENDED_CONTROL_PROFILE,
    LIGHT_CONTROL_PROFILE,
    preflight_control_profile,
)
from src.control.contracts import RolloutResult


def _rollout(success, restricted_time=None, relapsed=False, origin="o1", future=0):
    return RolloutResult(
        problem_id="p", policy_id="pol", origin_id=origin, future_id=future,
        is_valid=True, is_eligible=True, success=success,
        completion_time=(1.0 if success else None),
        restricted_time=(1.0 if success else 320.0) if restricted_time is None
        else restricted_time,
        expenditure=1.0, distinct_nodes_used=2, relapsed=relapsed,
        terminal_target_violation=0.1, event_counts={},
    )


class TestPreparationAttemptAccounting(unittest.TestCase):
    """Deterministic rejection accounting without running any simulation."""

    def test_historical_yield_is_8_of_75(self):
        """Colab1 accepted indices 31,34,36,43,54,59,70,74 => 8 accepted / 75 attempts."""
        from src.control.campaign import preparation_attempt_summary

        accepted_indices = {31, 34, 36, 43, 54, 59, 70, 74}
        attempts = [
            {"index": i, "seed": i * 1000, "accepted": i in accepted_indices,
             "reason": "eligible" if i in accepted_indices else "ineligible",
             "m_y0": 0.9, "m_x0": 0.9}
            for i in range(75)  # indices end at 74
        ]
        summary = preparation_attempt_summary(attempts)
        self.assertEqual(summary["n_attempted"], 75)
        self.assertEqual(summary["n_accepted"], 8)
        self.assertEqual(summary["n_rejected"], 67)
        self.assertAlmostEqual(summary["eligibility_yield"], 8.0 / 75.0)
        self.assertEqual(summary["eligibility_yield_denominator"], "attempted")

    def test_empty_attempts(self):
        from src.control.campaign import preparation_attempt_summary

        summary = preparation_attempt_summary([])
        self.assertEqual(summary["n_attempted"], 0)
        self.assertEqual(summary["eligibility_yield"], 0.0)


class TestProfilePreflight(unittest.TestCase):
    """Cheap preflight of both profiles and of edited (model_copy) profiles."""

    def test_default_profiles_pass_without_simulation(self):
        geo_light = preflight_control_profile(LIGHT_CONTROL_PROFILE)
        geo_ext = preflight_control_profile(EXTENDED_CONTROL_PROFILE)
        # Light: 12 prepared >= 2+2+4 (4 retained spares are not assigned).
        self.assertEqual(geo_light["n_needed"], 8)
        self.assertEqual(geo_light["n_prepare"], 12)
        self.assertEqual(geo_light["n_unassigned_spare"], 4)
        self.assertGreaterEqual(geo_light["k"], 1)
        self.assertGreaterEqual(geo_light["accessible_roster_size"], 1)
        # Extended allocation is now internally consistent: 54 prepared for
        # 12+12+30 (the historical 30-prepared profile is rejected below).
        self.assertEqual(geo_ext["n_prepare"], 54)
        self.assertEqual(geo_ext["n_needed"], 54)
        self.assertEqual(geo_ext["n_unassigned_spare"], 0)

    def test_copied_profile_bypassing_validation_is_caught(self):
        # Historical defect: extended profile with only 30 prepared origins.
        broken = EXTENDED_CONTROL_PROFILE.model_copy(update={"n_origins": 30})
        with self.assertRaises(ValueError) as ctx:
            preflight_control_profile(broken)
        self.assertIn("internally inconsistent allocation", str(ctx.exception))

    def test_infeasible_holding_is_caught(self):
        # horizon <= release_deadline is exactly what the Pydantic validator
        # would catch; model_copy bypasses it, so preflight must catch it.
        broken = LIGHT_CONTROL_PROFILE.model_copy(update={"horizon": 10.0})
        with self.assertRaises(ValueError) as ctx:
            preflight_control_profile(broken)
        self.assertIn("holding infeasible", str(ctx.exception))

    def test_negative_count_is_caught(self):
        broken = LIGHT_CONTROL_PROFILE.model_copy(update={"training_origins": 0})
        with self.assertRaises(ValueError) as ctx:
            preflight_control_profile(broken)
        self.assertIn("must be >= 1", str(ctx.exception))

    def test_node_cap_overrun_is_caught(self):
        broken = LIGHT_CONTROL_PROFILE.model_copy(
            update={"k_distinct_nodes_fraction": 1.5}
        )
        with self.assertRaises(ValueError) as ctx:
            preflight_control_profile(broken)
        self.assertIn("k=", str(ctx.exception))

    def test_legitimate_short_non_feedback_pulse_not_rejected(self):
        # A duration far below the delivery delay is operationally static but
        # must NOT be rejected by preflight (no false positive).
        short = LIGHT_CONTROL_PROFILE.model_copy(update={"durations_tau_public": (0.1,)})
        geo = preflight_control_profile(short)  # must not raise
        self.assertIn("amplitude_cap", geo)


class TestTinyPreparationRecords(unittest.TestCase):
    """Deterministic tiny preparation: every attempt recorded with reasons."""

    @classmethod
    def setUpClass(cls):
        from src.control.campaign import prepare_origins

        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot_dir = os.path.join(cls.tmp.name, "snapshots")
        os.makedirs(cls.snapshot_dir, exist_ok=True)
        cls.profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "D": 1, "horizon": 14.0, "establishment_time": 5.0,
            "n_origins": 5, "seed_base": 7710,
            "holding_duration": 2.0, "release_deadline": 5.0,
            "training_origins": 1, "selection_origins": 1,
            "evaluation_origins": 2,
        })
        cls.prep = prepare_origins(cls.profile, snapshot_dir=cls.snapshot_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts_are_consistent(self):
        s = self.prep.summary
        self.assertEqual(s["n_attempted"], len(s["attempts"]))
        self.assertEqual(s["n_accepted"] + s["n_rejected"], s["n_attempted"])
        self.assertEqual(s["n_accepted"], s["n_prepared"])
        self.assertEqual(s["n_prepared"], 5)
        self.assertEqual(s["n_retained"], 5)
        self.assertEqual(s["n_assigned"], 4)
        self.assertAlmostEqual(
            s["eligibility_yield"], s["n_accepted"] / s["n_attempted"]
        )
        # The yield denominator is attempted, not accepted (8/8-style bug gone).
        self.assertLessEqual(s["eligibility_yield"], 1.0)
        self.assertGreater(s["n_attempted"], 0)

    def test_every_attempt_has_required_fields(self):
        for rec in self.prep.summary["attempts"]:
            for key in ("index", "seed", "accepted", "reason", "m_y0", "m_x0"):
                self.assertIn(key, rec)
            if rec["accepted"]:
                self.assertTrue(rec["origin_id"])
            else:
                self.assertIsNone(rec["origin_id"])
                self.assertNotEqual(rec["reason"], "eligible")

    def test_pool_membership_declared_and_disjoint(self):
        pools = self.prep.summary["pool_membership"]
        self.assertEqual(len(pools["training"]), 1)
        self.assertEqual(len(pools["selection"]), 1)
        self.assertEqual(len(pools["evaluation"]), 2)
        ids = pools["training"] + pools["selection"] + pools["evaluation"]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(ids), set(self.prep.summary["origin_ids"][:4]))
        # Accepted attempt records carry their pool assignment.
        pools_by_id = {}
        for rec in self.prep.summary["attempts"]:
            if rec["accepted"]:
                pools_by_id[rec["origin_id"]] = rec["pool"]
        for oid, pool in pools_by_id.items():
            expected = (
                "training" if oid in pools["training"]
                else "selection" if oid in pools["selection"]
                else "evaluation" if oid in pools["evaluation"]
                else None  # retained spare beyond the declared pools
            )
            self.assertEqual(pool, expected)


class TestOutcomeStatusesAndTimes(unittest.TestCase):
    """All-failure and mixed summaries: statuses, times, conditional relapse."""

    HORIZON = 320.0

    def test_all_failure_reports_no_feasible_policy(self):
        from src.control.campaign import summarize_rollout_reliability

        rollouts = [
            RolloutResult(
                problem_id="p", policy_id="obs_cand_235", origin_id=f"o{i}",
                future_id=0, is_valid=True, is_eligible=True, success=False,
                completion_time=None, restricted_time=self.HORIZON,
                expenditure=120.0, distinct_nodes_used=3,
                relapsed=False, terminal_target_violation=0.5, event_counts={},
            )
            for i in range(4)
        ]
        s = summarize_rollout_reliability(rollouts, reliability_target=0.8)
        self.assertEqual(s["status"], "no_feasible_policy_found")
        self.assertEqual(s["eligible_trials"], 4)
        self.assertEqual(s["successes"], 0)
        self.assertEqual(s["failure_fraction"], 1.0)
        self.assertEqual(s["mean_restricted_time"], self.HORIZON)
        self.assertEqual(s["restricted_time_interpretation"], "failure_penalty_horizon")
        self.assertIsNone(s["mean_restricted_time_on_successes"])
        self.assertFalse(s["success_conditional_completion_time_available"])
        self.assertFalse(s["relapse_conditional_available"])
        self.assertIsNone(s["relapse_rate_conditional_on_success"])
        self.assertEqual(s["relapse_numerator"], 0)
        self.assertEqual(s["relapse_conditional_denominator"], 0)

    def test_mixed_outcomes_conditional_relapse(self):
        from src.control.campaign import summarize_rollout_reliability

        rollouts = (
            [_rollout(True, origin=f"o{i}") for i in range(3)]
            + [_rollout(False, origin=f"o{i}") for i in range(3, 5)]
        )
        rollouts[0].relapsed = True
        rollouts[1].relapsed = True
        rollouts[1].restricted_time = 7.0
        rollouts[2].restricted_time = 9.0
        s = summarize_rollout_reliability(rollouts, reliability_target=0.8)
        self.assertEqual(s["successes"], 3)
        self.assertEqual(s["failure_fraction"], 0.4)
        self.assertEqual(s["status"], "best_failing_candidate")
        self.assertEqual(s["restricted_time_interpretation"],
                         "mixed_completion_and_failure_penalty")
        self.assertAlmostEqual(s["mean_restricted_time_on_successes"],
                               (1.0 + 7.0 + 9.0) / 3.0)
        self.assertEqual(s["relapse_conditional_numerator"], 2)
        self.assertEqual(s["relapse_conditional_denominator"], 3)
        self.assertAlmostEqual(s["relapse_rate_conditional_on_success"], 2.0 / 3.0)

    def test_reliability_certified_requires_lower_bound(self):
        from src.control.campaign import summarize_rollout_reliability

        # 20/20 successes: the exact one-sided lower bound exceeds p0=0.8.
        rollouts = [_rollout(True, origin=f"o{i}") for i in range(20)]
        s = summarize_rollout_reliability(rollouts, reliability_target=0.8)
        self.assertEqual(s["status"], "reliability_certified")
        self.assertEqual(s["restricted_time_interpretation"], "successful_completions")

        # 8/10 successes: point estimate above p0 but the bound is not.
        rollouts = (
            [_rollout(True, origin=f"o{i}") for i in range(8)]
            + [_rollout(False, origin=f"o{i}") for i in range(8, 10)]
        )
        s = summarize_rollout_reliability(rollouts, reliability_target=0.8)
        self.assertEqual(s["status"], "estimated_feasible")

    def test_no_eligible_trials_status(self):
        from src.control.campaign import summarize_rollout_reliability

        s = summarize_rollout_reliability([], reliability_target=0.8)
        self.assertEqual(s["status"], "no_eligible_trials")


class TestSelectionTieBreakReason(unittest.TestCase):
    """The tie-break reason reproduces the historical cost tie-break pattern."""

    def test_cost_tie_break_exposed(self):
        from src.control.campaign import describe_selection_tie_break

        # All finalists fail identically on success/time/violation; exposure decides.
        best = (1, -0.0, 320.0, 0.5, 100.0, 3.0, "obs_cand_235")
        other = (1, -0.0, 320.0, 0.5, 110.0, 3.0, "obs_cand_009")
        tb = describe_selection_tie_break(best, other)
        self.assertEqual(tb["decided_by"], "mean_exposure")
        self.assertIn("success_probability", tb["tied_components"])
        self.assertIn("mean_restricted_time", tb["tied_components"])
        self.assertEqual(tb["tier"], "infeasible")
        self.assertIn("mean_exposure", tb["description"])

    def test_feasible_time_decides(self):
        from src.control.campaign import describe_selection_tie_break

        best = (0, 5.0, -0.9, 0.1, 50.0, 3.0, "a")
        other = (0, 6.0, -0.9, 0.4, 40.0, 2.0, "b")
        tb = describe_selection_tie_break(best, other)
        self.assertEqual(tb["decided_by"], "mean_restricted_time")
        self.assertEqual(tb["tier"], "feasible")

    def test_tier_decides_and_single_candidate(self):
        from src.control.campaign import describe_selection_tie_break

        tb = describe_selection_tie_break((0, 5.0), (1, -0.5))
        self.assertEqual(tb["decided_by"], "tier")
        tb = describe_selection_tie_break((0, 5.0), None)
        self.assertEqual(tb["decided_by"], "single_candidate")


if __name__ == "__main__":
    unittest.main()