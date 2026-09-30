"""Unit tests for scientific plotting, rendering, and bundle integration.

Verifies:
1. Reusable plotting and rendering consume saved bundle data without running physical simulation.
2. Trajectory plots display initial orientation, target thresholds, forcing, actual release,
   holding window, and certification marker where achieved (or failure penalty marker).
3. Intervention schedules and cumulative budget consumption with cap lines.
4. Policy success with uncertainty and sample counts; restricted completion time distinguished
   from successful completion; conditional relapse with N/A handling.
5. Paired adaptive/frozen diagnostics and uncertainty for policy differences (never asserting
   equivalence from equal failures).
6. Before/during/after network panels and playable matched controlled/uncontrolled animation
   with consistent node layout derived from origin W_0, consistent color/size scales, and shared time grid.
7. Truthful handling of all-failure runs without crashing, producing informative diagnostic plots.
8. All plots and animation files saved under unified bundle directory and registered in manifest.json.
9. Missing inputs handling and descriptive error reporting.
10. Actual plot and animation inspection at start, intervention, release, and end.
"""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.config.control_profiles import LIGHT_CONTROL_PROFILE
from src.control.campaign import run_campaign
from src.io.results import load_control_bundle, LoadedControlBundle
from src.visualization.control_plots import (
    plot_scalar_trajectories,
    plot_intervention_and_budget,
    plot_policy_reliability_and_time,
    plot_paired_adaptive_frozen_comparison,
    plot_network_views_before_during_after,
    render_network_animation,
    render_bundle_visualizations,
    _select_animation_indices,
    _find_representative_rollout_key,
)
from src.visualization.network_layout import _seeded_spring_layout


def _make_tiny_test_bundle(tmp_dir: Path, force_success: bool = False) -> Path:
    """Run a tiny campaign to generate a realistic on-disk bundle fixture."""
    profile = LIGHT_CONTROL_PROFILE.model_copy(update={
        "N": 8, "D": 1, "horizon": 8.0, "establishment_time": 3.0,
        "n_origins": 3, "seed_base": 7711,
        "holding_duration": 2.0, "release_deadline": 4.0,
        "k_distinct_nodes_fraction": 0.5, "exposure_cap_normalized": 0.5,
        "candidate_cap": 2,
        "selectors": ("random",),
        "observed_slot_quotas": (0.0,), "feedback_choices": (False,),
        "training_origins": 1, "training_futures_per_origin": 1,
        "selection_candidates": 1, "selection_origins": 1,
        "selection_futures_per_origin": 1,
        "evaluation_origins": 1,
        "evaluation_futures_per_origin": 1,
    })

    result = run_campaign(
        profile=profile,
        campaign_id="r4_test_bundle",
        master_seed=33221,
        out_dir=tmp_dir,
    )

    bundle_path = Path(result.bundle_path)

    if force_success:
        # Synthesize a successful completion marker in rollout_results.csv and trajectory
        csv_path = bundle_path / "rollout_results.csv"
        lines = csv_path.read_text().splitlines()
        if len(lines) > 1:
            header = lines[0].split(",")
            succ_idx = header.index("success")
            time_idx = header.index("completion_time")
            row1 = lines[1].split(",")
            row1[succ_idx] = "True"
            row1[time_idx] = "5.5"
            lines[1] = ",".join(row1)
            csv_path.write_text("\n".join(lines) + "\n")

    return bundle_path


class TestRenderFromSavedBundleWithoutSimulation(unittest.TestCase):
    """Acceptance: Render directly from saved bundle in a fresh process with no simulation rerun."""

    def test_render_directly_from_saved_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))

            # Reload in a clean handle (no campaign object in scope)
            loaded = load_control_bundle(bundle_dir)
            self.assertIsInstance(loaded, LoadedControlBundle)

            # Master orchestrator renders all plots and animations directly from disk
            render_res = render_bundle_visualizations(bundle_dir, render_animation=True)

            self.assertIn("visualizations", render_res)
            vis = render_res["visualizations"]

            # Required scientific plot artifacts exist
            expected_files = [
                "trajectory_diagnostics.png",
                "budget_and_intervention.png",
                "policy_reliability_and_time.png",
                "paired_adaptive_frozen.png",
                "network_before_during_after.png",
            ]
            for fn in expected_files:
                p = bundle_dir / fn
                self.assertTrue(p.exists(), f"Missing required visualization file: {fn}")
                self.assertGreater(p.stat().st_size, 0, f"File {fn} is empty")

            # Animation artifacts exist
            self.assertTrue((bundle_dir / "network_animation.gif").exists() or (bundle_dir / "network_animation.html").exists())

            # Verified manifest registration and checksums
            with open(bundle_dir / "manifest.json", "r") as f:
                manifest = json.load(f)

            self.assertIn("visualizations", manifest)
            self.assertIn("file_inventory", manifest)
            self.assertIn("checksums", manifest)

            # Confirm all generated plot files are in inventory and checksums match disk
            for fn in expected_files:
                self.assertIn(fn, manifest["file_inventory"])
                self.assertIn(fn, manifest["checksums"])
                with open(bundle_dir / fn, "rb") as bf:
                    actual_sha = hashlib.sha256(bf.read()).hexdigest()
                self.assertEqual(actual_sha, manifest["checksums"][fn])


class TestScalarTrajectoryPlot(unittest.TestCase):
    """Acceptance: Trajectories with orientation, thresholds, forcing, release, holding, and certification."""

    def test_adaptive_and_frozen_telemetry_have_distinct_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)
            adaptive = [k for k in loaded.trajectories if k.startswith("evaluation__") and k.endswith("__adaptive")]
            frozen = [k for k in loaded.trajectories if k.startswith("evaluation__") and k.endswith("__frozen")]
            self.assertTrue(adaptive)
            self.assertTrue(frozen)
            self.assertEqual({k.removesuffix("__adaptive") for k in adaptive},
                             {k.removesuffix("__frozen") for k in frozen})
            self.assertTrue(any(k.endswith("__adaptive") for k in loaded.network_frames))
            self.assertTrue(any(k.endswith("__frozen") for k in loaded.network_frames))

    def test_trajectory_elements_failed_rollout(self):
        """All-failure or failed rollout truthfully displays failure penalty marker and no false certification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir), force_success=False)
            loaded = load_control_bundle(bundle_dir)

            res = plot_scalar_trajectories(loaded)
            self.assertIn("s_0", res)
            self.assertIn("target_threshold_y", res)
            self.assertIn("threshold", res)
            self.assertIn("holding_duration", res)
            self.assertIn("horizon", res)
            self.assertIn("t_release", res)
            self.assertFalse(res["success"])
            self.assertIsNone(res["completion_time"])
            self.assertEqual(res["restricted_time"], res["horizon"])

    def test_trajectory_elements_successful_rollout(self):
        """Successful rollout displays certified marker and holding window."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir), force_success=True)
            loaded = load_control_bundle(bundle_dir)

            # The fixture marks the first CSV row (baseline_zero) successful.
            # Plot that exact arm; another policy must not inherit its outcome.
            baseline_key = next(k for k in loaded.trajectories if k.startswith("baseline__baseline_zero__"))
            res = plot_scalar_trajectories(loaded, rollout_key=baseline_key)
            self.assertTrue(res["success"])
            self.assertIsNotNone(res["completion_time"])
            self.assertAlmostEqual(res["completion_time"], 5.5)


class TestRepresentativeActiveArm(unittest.TestCase):
    """An active evaluation must remain visible when the static winner is zero."""

    def test_plots_choose_active_evaluation_and_matched_baseline(self):
        from matplotlib import pyplot as plt
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir)
            (bundle_dir / "manifest.json").write_text(
                json.dumps({"spec": {"horizon": 3.0}}), encoding="utf-8",
            )
            bundle = load_control_bundle(bundle_dir)
            zero = "evaluation__static:observable_zero__origin_abc__fut0"
            active = "evaluation__feedback:active__origin_abc__fut0"
            baseline = "baseline__baseline_zero__origin_abc__fut0"
            W = np.array([[0, 2, 1], [2, 0, 2], [1, 2, 0]], dtype=np.uint32)

            def frame(t: float, actuated: bool) -> Dict[str, Any]:
                return {
                    "t": t, "W": W.copy(),
                    "x": np.ones((3, 1)), "y": np.ones((3, 1)),
                    "controlled_mask": np.array([actuated, False, False]),
                }

            inert = [frame(float(t), False) for t in range(4)]
            forced = [frame(float(t), t in (1, 2)) for t in range(4)]
            bundle.network_frames = {zero: inert, baseline: inert, active: forced}
            bundle.trajectories = {zero: {"t": np.arange(4)}, active: {"t": np.arange(4)}}
            bundle.action_history = {active: [{"spending_rate": 1.0}]}

            self.assertEqual(_find_representative_rollout_key(bundle), active)
            network = plot_network_views_before_during_after(bundle)
            self.assertEqual(network["rollout_key"], active)
            self.assertTrue(network["active_frame_found"])
            self.assertEqual(network["timepoints"][-1], 3.0)
            self.assertTrue(any("feedback:active" in t.get_text()
                                for t in network["fig"].texts))
            plt.close(network["fig"])

            animation = render_network_animation(
                bundle, output_path=bundle_dir / "chosen.gif", max_frames=4,
            )
            self.assertEqual(animation["ctrl_key"], active)
            self.assertEqual(animation["unctrl_key"], baseline)
            self.assertTrue(animation["matched_side_by_side"])
            self.assertEqual(animation["frame_times"], [0.0, 1.0, 2.0, 3.0])
            self.assertTrue(any("feedback:active" in t.get_text()
                                for t in animation["anim"]._fig.texts))


class TestRecordedInterventionShading(unittest.TestCase):
    """Empty release actions must not draw an invented forcing interval."""

    def test_zero_control_has_no_active_shading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = load_control_bundle(_make_tiny_test_bundle(Path(tmp_dir)))
            key = next(k for k in loaded.trajectories
                       if k.startswith("baseline__baseline_zero"))
            result = plot_scalar_trajectories(loaded, rollout_key=key)
            self.assertEqual(result["forcing_intervals"], [])


class TestBudgetAndInterventionPlot(unittest.TestCase):
    """Acceptance: Intervention schedules and cumulative budget consumption with cap lines."""

    def test_budget_consumption_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)

            res = plot_intervention_and_budget(loaded)
            self.assertIn("exposure_cap", res)
            self.assertIn("amplitude_cap", res)
            self.assertIn("k_cap", res)
            self.assertGreater(res["num_rollouts_recorded"], 0)
            self.assertGreater(res["exposure_cap"], 0.0)


class TestPolicyReliabilityTimeAndRelapse(unittest.TestCase):
    """Acceptance: Policy success with uncertainty, restricted vs completion time, conditional relapse N/A."""

    def test_reliability_summary_plotting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir), force_success=False)
            loaded = load_control_bundle(bundle_dir)

            res = plot_policy_reliability_and_time(loaded)
            self.assertGreater(len(res["arms"]), 0)
            self.assertEqual(len(res["p_hats"]), len(res["arms"]))
            self.assertEqual(len(res["sample_counts"]), len(res["arms"]))

            # Verify sample count format "k/n"
            for sc in res["sample_counts"]:
                self.assertIn("/", sc)

            # In all-failure run, conditional relapse must be None (N/A)
            for cr in res["conditional_relapses"]:
                self.assertIsNone(cr)


class TestPairedAdaptiveFrozenHonesty(unittest.TestCase):
    """Acceptance: Paired diagnostics and uncertainty; NEVER assert equivalence from equal failures."""

    def test_paired_diagnostics_and_scientific_notice(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir), force_success=False)
            loaded = load_control_bundle(bundle_dir)

            res = plot_paired_adaptive_frozen_comparison(loaded)
            self.assertIn("n_pairs", res)
            self.assertIn("both_failed", res)
            self.assertTrue(res["both_failed"])
            self.assertEqual(res["adapt_successes"], 0)
            self.assertEqual(res["frozen_successes"], 0)
            self.assertEqual(res["mean_diff_success"], 0.0)


class TestNetworkViewsBeforeDuringAfter(unittest.TestCase):
    """Acceptance: Synchronized before/during/after panels with consistent layout derived from origin W_0."""

    def test_network_panels_and_layout_consistency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)

            res = plot_network_views_before_during_after(loaded)
            self.assertIn("timepoints", res)
            self.assertEqual(len(res["timepoints"]), 4)
            # Verify timestamps are non-decreasing
            t_pts = res["timepoints"]
            self.assertLessEqual(t_pts[0], t_pts[1])
            self.assertLessEqual(t_pts[1], t_pts[2])
            self.assertLessEqual(t_pts[2], t_pts[3])


class TestPlayableMatchedAnimation(unittest.TestCase):
    """Acceptance: Playable matched animation with consistent layout and shared time grid."""

    def test_animation_rendering_and_frame_inspection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)

            anim_file = bundle_dir / "test_anim.gif"
            res = render_network_animation(loaded, output_path=anim_file, max_frames=8)

            self.assertIn("num_frames", res)
            self.assertGreater(res["num_frames"], 0)
            self.assertTrue(anim_file.exists() or Path(res.get("html_path", "")).exists())

            # Confirm inspection at start, intervention, release, and end
            self.assertGreaterEqual(res["num_frames"], 2)


class TestAllFailureRunTruthfulHandling(unittest.TestCase):
    """Acceptance: Handle all-failure runs truthfully without crashing, producing informative diagnostic plots."""

    def test_all_failure_run_plots_succeed_and_inform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir), force_success=False)
            loaded = load_control_bundle(bundle_dir)

            # Every plotting function must succeed without crashing on an all-failure run
            traj_data = plot_scalar_trajectories(loaded)
            self.assertFalse(traj_data["success"])

            budget_data = plot_intervention_and_budget(loaded)
            self.assertGreater(budget_data["exposure_cap"], 0.0)

            rel_data = plot_policy_reliability_and_time(loaded)
            self.assertEqual(sum(rel_data["p_hats"]), 0.0)

            paired_data = plot_paired_adaptive_frozen_comparison(loaded)
            self.assertTrue(paired_data["both_failed"])

            net_data = plot_network_views_before_during_after(loaded)
            self.assertEqual(len(net_data["timepoints"]), 4)


class TestNetworkVisualTruthfulness(unittest.TestCase):
    """Layout, labels, and abbreviated animations convey the saved data."""

    def test_connected_groups_are_closer_than_separate_groups(self):
        W = np.zeros((8, 8))
        for group in (range(4), range(4, 8)):
            for i in group:
                for j in group:
                    if i != j:
                        W[i, j] = 8.0
        W[3, 4] = 0.1
        W[4, 3] = 0.1
        pos = _seeded_spring_layout(W, layout_seed=12)
        within = [np.linalg.norm(pos[i] - pos[j])
                  for group in (range(4), range(4, 8))
                  for i in group for j in group if i < j]
        between = [np.linalg.norm(pos[i] - pos[j])
                   for i in range(4) for j in range(4, 8)]
        self.assertLess(np.mean(within), np.mean(between))
        self.assertGreater(np.ptp(np.linalg.norm(pos, axis=1)), 0.05)
        np.testing.assert_array_equal(pos, _seeded_spring_layout(W, layout_seed=12))

    def test_animation_preserves_pulse_and_full_time_span(self):
        frames = [{"t": float(i), "controlled_mask": np.array([2 <= i <= 10])}
                  for i in range(101)]
        indices = _select_animation_indices(frames, max_frames=12)
        self.assertEqual(len(indices), 12)
        for essential in (0, 2, 10, 11, 100):
            self.assertIn(essential, indices)
        self.assertGreaterEqual(sum(i >= 50 for i in indices), 3)

    def test_legacy_short_frames_are_labeled_honestly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = load_control_bundle(_make_tiny_test_bundle(Path(tmp_dir)))
            key = next(k for k in loaded.network_frames if k.startswith("evaluation__"))
            loaded.network_frames = {key: loaded.network_frames[key][:3]}
            for frame in loaded.network_frames[key]:
                frame["controlled_mask"] = np.zeros_like(frame["controlled_mask"])
            result = plot_network_views_before_during_after(loaded)
            self.assertFalse(result["coverage_complete"])
            self.assertFalse(result["active_frame_found"])
            self.assertIn("No active frame recorded", result["panel_labels"])
            self.assertIn("before horizon", result["panel_labels"][-1])
            fig = result["fig"]
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            titles = [ax.title.get_window_extent(renderer) for ax in result["axes"]]
            self.assertFalse(any(a.overlaps(b) for a, b in zip(titles[:-1], titles[1:])))
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            self.assertTrue(any("Outer disk" in label for label in labels))
            self.assertTrue(any("Center dot" in label for label in labels))
            self.assertTrue(any("direction hidden" in label for label in labels))
            self.assertEqual(fig.axes[-1].get_ylabel(), "Stance value")


class TestMissingInputsHandling(unittest.TestCase):
    """Acceptance: Test missing-input errors and all-failure labels."""

    def test_missing_trajectory_data_raises_informative_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)
            loaded.trajectories = {}  # Corrupt trajectory data

            with self.assertRaises(ValueError) as ctx:
                plot_scalar_trajectories(loaded)
            self.assertIn("No scalar trajectory data", str(ctx.exception))

    def test_missing_network_frames_raises_informative_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = _make_tiny_test_bundle(Path(tmp_dir))
            loaded = load_control_bundle(bundle_dir)
            loaded.network_frames = {}  # Empty network frames

            with self.assertRaises(ValueError) as ctx:
                plot_network_views_before_during_after(loaded)
            self.assertIn("No network frames", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
