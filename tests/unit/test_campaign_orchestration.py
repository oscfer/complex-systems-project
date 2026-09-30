"""Targeted tests: campaign orchestration and export bundles.

Validates the full light-allocation campaign pipeline at a tiny scale:
1. End-to-end run produces the declared structure (baselines, selected policy,
   adaptive/frozen evaluation, pairs, manifest).
2. Adaptive/frozen pairing groups identical fresh origins with matched seeds,
   and the frozen branch genuinely freezes W (no encounter or decay events).
3. export_control_bundle writes rollouts.csv, adaptive_frozen_pairs.csv, and a
   manifest.json whose row counts match the executed rollouts.
4. Reliability summaries count only eligible trials with bounds in [0, 1].
"""

import csv
import json
import os
import tempfile
import unittest

from src.config.control_profiles import LIGHT_CONTROL_PROFILE

N_EVAL_ORIGINS = 2
N_EVAL_FUTURES = 1


def _tiny_profile():
    # Light allocation at tiny physics; n_origins >= training+selection+evaluation.
    return LIGHT_CONTROL_PROFILE.model_copy(update={
        "N": 8, "D": 1, "horizon": 14.0, "establishment_time": 5.0,
        "n_origins": 5, "seed_base": 7710,
        "holding_duration": 2.0, "release_deadline": 5.0,
        "k_distinct_nodes_fraction": 0.375, "exposure_cap_normalized": 0.5,
        "candidate_cap": 3,
        "selectors": ("reported_attention", "random"),
        "observed_slot_quotas": (0.0,), "feedback_choices": (False,),
        "training_origins": 1, "training_futures_per_origin": 1,
        "selection_candidates": 2, "selection_origins": 1,
        "selection_futures_per_origin": 1,
        "evaluation_origins": N_EVAL_ORIGINS,
        "evaluation_futures_per_origin": N_EVAL_FUTURES,
    })


class TestCampaignOrchestration(unittest.TestCase):
    """full pipeline, pairing invariants, and export bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.control.campaign import run_campaign

        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot_dir = os.path.join(cls.tmp.name, "origin_snapshots")
        os.makedirs(cls.snapshot_dir, exist_ok=True)
        cls.result = run_campaign(
            _tiny_profile(),
            campaign_id="m4_test",
            master_seed=99001,
            out_dir=cls.tmp.name,
            snapshot_dir=cls.snapshot_dir,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_declared_structure(self) -> None:
        result = self.__class__.result
        # Baselines: zero, random, observed-attention on fresh origins.
        self.assertEqual(
            set(result.baseline_results.keys()),
            {"baseline_zero", "baseline_random", "baseline_attention"},
        )
        n_expected = N_EVAL_ORIGINS * N_EVAL_FUTURES
        for rollouts in result.baseline_results.values():
            self.assertEqual(len(rollouts), n_expected)
        # Selected policy evaluated on adaptive AND frozen branches.
        self.assertEqual(len(result.evaluation_results), 2)
        self.assertTrue(all(
            k.endswith("|frozen") or k.endswith("|adaptive")
            for k in result.evaluation_results
        ))
        # Selection declared a candidate consistent with the search registry.
        selected_id = result.selection_summary["selected_candidate_id"]
        self.assertIn(selected_id, result.search_result.search_state.candidate_registry)
        self.assertEqual(result.selected_candidate.candidate_id, selected_id)

    def test_preparation_summary_and_snapshots(self) -> None:
        result = self.__class__.result
        prep = result.preparation_summary
        self.assertEqual(prep["n_prepared"], 5)
        self.assertGreaterEqual(prep["n_eligible"], 0)
        self.assertLessEqual(prep["eligibility_yield"], 1.0)
        self.assertEqual(prep["saved_snapshots"], 5)
        # Snapshots saved once per origin, loadable.
        from src.control.campaign import load_origin_snapshot

        files = sorted(os.listdir(self.snapshot_dir))
        self.assertEqual(len(files), 5)
        state = load_origin_snapshot(os.path.join(self.snapshot_dir, files[0]))
        self.assertEqual(state.N, 8)

    def test_adaptive_frozen_pairing(self) -> None:
        result = self.__class__.result
        pairs = result.adaptive_vs_frozen_pairs
        self.assertEqual(len(pairs), N_EVAL_ORIGINS * N_EVAL_FUTURES)
        for pair in pairs:
            ra, rf = pair["adaptive"], pair["frozen"]
            self.assertIsNotNone(ra)
            # Identical fresh origin and matched seeds across branches.
            self.assertEqual(ra.origin_id, rf.origin_id)
            self.assertEqual(ra.future_id, rf.future_id)
            self.assertEqual(ra.summaries["sim_seed"], rf.summaries["sim_seed"])
            self.assertEqual(ra.summaries["ctl_seed"], rf.summaries["ctl_seed"])
            self.assertEqual(ra.policy_id, rf.policy_id)
            # Frozen branch genuinely frozen: no encounter or decay events.
            self.assertEqual(rf.event_counts.get("encounters", 0), 0)
            self.assertEqual(rf.event_counts.get("memory_decays", 0), 0)
            # Adaptive branch keeps memory dynamics alive.
            self.assertTrue(
                ra.event_counts.get("encounters", 0) > 0
                or ra.event_counts.get("memory_decays", 0) > 0
            )

    def test_reliability_summaries_structure(self) -> None:
        result = self.__class__.result
        summaries = result.reliability_summaries
        # Baselines + selected policy on both branches.
        self.assertIn("baseline_zero|adaptive", summaries)
        self.assertTrue(any(k.endswith("|frozen") for k in summaries))
        n_expected = N_EVAL_ORIGINS * N_EVAL_FUTURES
        for s in summaries.values():
            self.assertGreaterEqual(s["success_probability"], 0.0)
            self.assertLessEqual(s["success_probability"], 1.0)
            self.assertLessEqual(s["successes"], s["eligible_trials"])
            if s["eligible_trials"] > 0:
                self.assertLessEqual(s["wilson_low"], s["wilson_high"])
            self.assertLessEqual(s["total_rollouts"], n_expected)

    def test_uncontrolled_persistence_check_present(self) -> None:
        result = self.__class__.result
        check = result.manifest["uncontrolled_persistence"]
        # R1/R2: the unsupported `stable` claim is removed; original-state
        # persistence is measured in R2.
        self.assertNotIn("stable", check)
        self.assertTrue(check["original_state_persistence_measured"])
        self.assertIn("interpretation", check)
        self.assertEqual(check["n_rollouts"], N_EVAL_ORIGINS * N_EVAL_FUTURES)

    def test_export_control_bundle(self) -> None:
        from src.io.results import export_control_bundle

        result = self.__class__.result
        bundle = export_control_bundle(result, self.tmp.name)
        self.assertTrue(os.path.isdir(bundle))
        manifest_path = os.path.join(bundle, "manifest.json")
        rollouts_path = os.path.join(bundle, "rollout_results.csv")
        pairs_path = os.path.join(bundle, "adaptive_frozen_pairs.csv")
        schedules_path = os.path.join(bundle, "policy_schedules.json")
        search_path = os.path.join(bundle, "search_summary.json")
        for p in (manifest_path, rollouts_path, pairs_path, schedules_path, search_path):
            self.assertTrue(os.path.isfile(p), p)

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertIn("sha256", manifest)
        self.assertEqual(manifest["campaign_id"], "m4_test")
        self.assertIn("problem", manifest)
        self.assertTrue(manifest["problem"]["model_hash"])
        self.assertIn("spec", manifest)
        self.assertIn("reliability_summaries", manifest)

        with open(rollouts_path) as f:
            rows = list(csv.DictReader(f))
        expected_rows = (
            sum(len(r) for r in result.baseline_results.values())
            + sum(len(r) for r in result.evaluation_results.values())
        )
        self.assertEqual(len(rows), expected_rows)

        with open(pairs_path) as f:
            pair_rows = list(csv.DictReader(f))
        self.assertEqual(len(pair_rows), len(result.adaptive_vs_frozen_pairs))

    def test_bundle_rejects_wrong_type(self) -> None:
        from src.io.results import export_control_bundle

        with self.assertRaises(TypeError):
            export_control_bundle("not-a-result", self.tmp.name)


if __name__ == "__main__":
    unittest.main()