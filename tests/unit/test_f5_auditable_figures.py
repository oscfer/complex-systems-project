"""Unit tests for audit finding F5: auditable figures and executable-source provenance.

Verifies:
1. Network panel stance counts (red/blue counts and percentages, both belief
   layers) are computed directly from saved x/y arrays and match them exactly.
2. The directed normalized-attention view uses exactly the row-normalized
   attention matrix A_ij = W_ij / sum_k W_ik from src.analytics.influence.
3. The node-size key discloses the additive area scaling
   display_area = 75 + 300*r_j/r_max (monotonic, not proportional) and the
   edge key discloses that lines use raw W_ij+W_ji (remembered interaction
   volume, not directed influence).
4. Fixed positions are labeled as a nonphysical rendering choice.
5. The shown recorded arm is labeled with an explicit selection reason
   (recorded active evaluation arm vs overall selected policy).
6. Executable-source provenance (git_head, source_digest, runtime_versions) is
   written on new bundle manifests while legacy manifests without these fields
   still load unchanged.
7. Layout-seed changes move only the drawing geometry; simulation outputs
   (saved W/y/x arrays) stay exactly unchanged.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analytics.influence import calculate_attention_matrix
from src.config.control_profiles import LIGHT_CONTROL_PROFILE
from src.control.campaign import run_campaign
from src.io.results import (
    load_control_bundle,
    update_bundle_manifest,
    _source_digest,
    _git_head,
)
from src.visualization.control_plots import (
    plot_network_views_before_during_after,
    _stance_counts,
    _top_directed_edges,
    _active_node_counts,
)


def _make_tiny_test_bundle(tmp_dir: Path) -> Path:
    """Run a tiny deterministic campaign to generate an on-disk bundle fixture."""
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
        campaign_id="f5_test_bundle",
        master_seed=33221,
        out_dir=tmp_dir,
    )
    return Path(result.bundle_path)


class TestFigureCountsMatchSavedArrays(unittest.TestCase):
    """Acceptance: displayed counts match saved arrays exactly."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle_path = _make_tiny_test_bundle(Path(cls._tmp.name))
        cls.bundle = load_control_bundle(cls.bundle_path)
        cls.plot = plot_network_views_before_during_after(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_stance_counts_helper_matches_saved_arrays_exactly(self):
        for frame in next(iter(self.bundle.network_frames.values())):
            y = np.asarray(frame["y"], dtype=np.float64).reshape(-1)
            x = np.asarray(frame["x"], dtype=np.float64).reshape(-1)
            cy = _stance_counts(y)
            cx = _stance_counts(x)
            self.assertEqual(cy["red"], int(np.sum(y > 0)))
            self.assertEqual(cy["blue"], int(np.sum(y < 0)))
            self.assertEqual(cx["red"], int(np.sum(x > 0)))
            self.assertEqual(cx["blue"], int(np.sum(x < 0)))
            self.assertEqual(cy["red"] + cy["blue"] + cy["neutral"], y.size)
            if y.size:
                self.assertEqual(cy["red_pct"], 100.0 * cy["red"] / y.size)
                self.assertEqual(cy["blue_pct"], 100.0 * cy["blue"] / y.size)

    def test_panel_counts_equal_recomputed_saved_arrays(self):
        key = self.plot["rollout_key"]
        frames = self.bundle.network_frames[key]
        for entry, idx in zip(self.plot["stance_counts"], self.plot["frame_indices"]):
            frame = frames[idx]
            self.assertEqual(entry["t"], float(frame["t"]))
            y = np.asarray(frame["y"], dtype=np.float64).reshape(-1)
            x = np.asarray(frame["x"], dtype=np.float64).reshape(-1)
            self.assertEqual(
                entry["public"],
                _stance_counts(y),
                f"public counts at t={entry['t']} differ from saved y array",
            )
            self.assertEqual(
                entry["private"],
                _stance_counts(x),
                f"private counts at t={entry['t']} differ from saved x array",
            )

    def test_count_lines_rendered_on_panel_titles(self):
        first_title = self.plot["axes"][0].get_title()
        self.assertIn("Public:", first_title)
        self.assertIn("Private:", first_title)
        public = self.plot["stance_counts"][0]["public"]
        self.assertIn(f"{public['red']} red", first_title)
        private = self.plot["stance_counts"][0]["private"]
        self.assertIn(f"{private['blue']} blue", first_title)

    def test_active_node_counts_reported_separately(self):
        key = self.plot["rollout_key"]
        frames = self.bundle.network_frames[key]
        expected = _active_node_counts(frames)
        self.assertEqual(self.plot["active_node_counts"], expected)
        masks = [np.asarray(f.get("controlled_mask", []), dtype=bool).reshape(-1)
                 for f in frames]
        if masks and masks[0].size:
            self.assertEqual(expected["initial"], int(np.sum(masks[0])))
            self.assertEqual(expected["peak"], max(int(np.sum(m)) for m in masks))
            union = set().union(*(set(np.flatnonzero(m).tolist()) for m in masks))
            self.assertEqual(expected["cumulative_distinct"], len(union))


class TestDirectedAttentionView(unittest.TestCase):
    """Acceptance: directed values match row-normalized A."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle = load_control_bundle(_make_tiny_test_bundle(Path(cls._tmp.name)))
        cls.plot = plot_network_views_before_during_after(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_directed_matrix_equals_row_normalized_attention(self):
        key = self.plot["rollout_key"]
        W0 = np.asarray(self.bundle.network_frames[key][0]["W"], dtype=np.float64)
        A_expected, _ = calculate_attention_matrix(W0)
        np.testing.assert_array_equal(self.plot["directed_attention_matrix"], A_expected)
        active_rows = A_expected.sum(axis=1) > 0
        np.testing.assert_allclose(
            A_expected.sum(axis=1)[active_rows], 1.0, atol=1e-12)

    def test_top_directed_edge_values_are_exact_A_entries(self):
        key = self.plot["rollout_key"]
        W0 = np.asarray(self.bundle.network_frames[key][0]["W"], dtype=np.float64)
        A, top = _top_directed_edges(W0, k=12)
        for i, j, value in top:
            self.assertEqual(value, float(A[i, j]))
        values = [v for _, _, v in top]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLessEqual(len(top), 12)

    def test_directed_panel_declares_row_normalization_and_filtering(self):
        ax_dir = self.plot["axes"][-1]
        self.assertIn("Directed normalized attention", ax_dir.get_title())
        self.assertIn("Row-normalized", ax_dir.get_xlabel())
        self.assertIn("omitted", ax_dir.get_xlabel())


class TestFigureKeyDisclosures(unittest.TestCase):
    """Acceptance: size/edge/geometry/selection keys are disclosed on the figure."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle = load_control_bundle(_make_tiny_test_bundle(Path(cls._tmp.name)))
        cls.plot = plot_network_views_before_during_after(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _legend_text(self) -> str:
        return "\n".join(t.get_text() for t in self.plot["fig"].legends[0].get_texts())

    def test_node_size_key_discloses_additive_area_scaling(self):
        self.assertIn("75 + 300", self._legend_text())
        self.assertIn("monotonic, not proportional", self._legend_text())

    def test_edge_key_discloses_undirected_memory_volume(self):
        text = self._legend_text()
        self.assertIn("W_{ij}+W_{ji}", text)
        self.assertIn("remembered interaction volume", text)
        self.assertIn("not directed influence", text)

    def test_fixed_positions_disclosed_as_nonphysical(self):
        texts = [t.get_text() for t in self.plot["fig"].texts]
        geometry = [t for t in texts if "nonphysical rendering choice" in t]
        self.assertTrue(geometry)
        self.assertIn("not geographic distance", geometry[0])

    def test_stance_colors_distinguished_from_trajectory_curve_colors(self):
        self.assertIn("trajectory-plot curve colors do not encode stance",
                      self._legend_text())

    def test_selection_reason_labels_shown_arm_and_selected_policy(self):
        lines = self.plot["selection_reason"]
        joined = "\n".join(lines)
        self.assertTrue(any("Shown arm =" in line for line in lines))
        self.assertTrue(any("Overall selected policy =" in line for line in lines))
        sel = (self.bundle.manifest or {}).get("selection_summary", {})
        selected = sel.get("selected_candidate_id")
        if selected:
            self.assertIn(selected, joined)


class TestProvenanceFields(unittest.TestCase):
    """Acceptance: provenance on new manifest writes; legacy reads keep working."""

    def _legacy_bundle(self, base: Path) -> Path:
        """Minimal legacy-style bundle directory without the new fields."""
        bundle = base / "legacy_bundle"
        bundle.mkdir(parents=True)
        manifest = {
            "campaign_id": "legacy",
            "python": "3.0.0",
            "numpy": "1.0.0",
            "file_inventory": [],
            "checksums": {},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        (bundle / "rollout_results.csv").write_text("group,policy_id\n")
        return bundle

    def test_new_manifest_write_contains_provenance_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._legacy_bundle(Path(tmp))
            manifest = update_bundle_manifest(bundle)
            self.assertIn("git_head", manifest)
            self.assertIn("source_digest", manifest)
            self.assertIn("runtime_versions", manifest)
            for lib in ("python", "numpy", "networkx", "matplotlib"):
                self.assertIn(lib, manifest["runtime_versions"])
            self.assertTrue(len(manifest["git_head"]) == 40)
            self.assertTrue(len(manifest["source_digest"]) == 64)

    def test_legacy_manifest_without_new_keys_loads_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._legacy_bundle(Path(tmp))
            loaded = load_control_bundle(bundle)
            self.assertEqual(loaded.manifest.get("campaign_id"), "legacy")
            self.assertNotIn("git_head", loaded.manifest)
            self.assertNotIn("source_digest", loaded.manifest)
            self.assertNotIn("runtime_versions", loaded.manifest)
            self.assertEqual(loaded.work_units, {})

    def test_source_digest_is_deterministic_and_git_head_valid(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(_source_digest(root), _source_digest(root))
        head = _git_head(root)
        self.assertTrue(head is None or len(head) == 40)


class TestLayoutSeedDoesNotChangeOutputs(unittest.TestCase):
    """Acceptance: layout-seed changes leave simulation outputs unchanged."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle_path = _make_tiny_test_bundle(Path(cls._tmp.name))
        cls.bundle = load_control_bundle(cls.bundle_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_saved_arrays_bit_identical_for_different_layout_seeds(self):
        key = next(iter(self.bundle.network_frames))
        before = {
            "W": [np.array(f["W"], copy=True) for f in self.bundle.network_frames[key]],
            "y": [np.array(f["y"], copy=True) for f in self.bundle.network_frames[key]],
            "x": [np.array(f["x"], copy=True) for f in self.bundle.network_frames[key]],
        }
        plot_a = plot_network_views_before_during_after(self.bundle, layout_seed=42)
        plot_b = plot_network_views_before_during_after(self.bundle, layout_seed=7)
        for i, frame in enumerate(self.bundle.network_frames[key]):
            np.testing.assert_array_equal(frame["W"], before["W"][i])
            np.testing.assert_array_equal(frame["y"], before["y"][i])
            np.testing.assert_array_equal(frame["x"], before["x"][i])
        # Geometry moves with the seed; outputs and derived numbers do not.
        self.assertGreater(
            float(np.max(np.abs(plot_a["positions"] - plot_b["positions"]))), 1e-6)
        np.testing.assert_array_equal(plot_a["directed_attention_matrix"],
                                      plot_b["directed_attention_matrix"])
        self.assertEqual(plot_a["stance_counts"], plot_b["stance_counts"])
        self.assertEqual(plot_a["active_node_counts"], plot_b["active_node_counts"])
        self.assertEqual(plot_a["selection_reason"], plot_b["selection_reason"])


if __name__ == "__main__":
    unittest.main()