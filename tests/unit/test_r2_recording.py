"""Unit tests for Non-invasive simulation recording, action history,
retention metric, and unified campaign bundle export/reload.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.config.control_profiles import LIGHT_CONTROL_PROFILE
from src.control.campaign import prepare_origins, run_campaign
from src.control.contracts import (
    Action,
    ActionEntry,
    ControlProblem,
    RolloutResult,
    TargetType,
)
from src.control.controllers import ObservableSelectorPolicy, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.recording import (
    ActionRecord,
    NetworkFrame,
    RecorderConfig,
    RetentionResult,
    RolloutRecorder,
    ScalarTrajectoryPoint,
    evaluate_original_state_retention,
)
from src.core.state import State
from src.io.results import (
    LoadedControlBundle,
    export_control_bundle,
    load_control_bundle,
    make_campaign_bundle_dir,
)


def _make_test_origin_and_problem(seed: int = 12345):
    """Create a minimal deterministic origin and control problem for unit testing."""
    profile = LIGHT_CONTROL_PROFILE.model_copy(update={
        "N": 8,
        "D": 1,
        "horizon": 10.0,
        "establishment_time": 4.0,
        "holding_duration": 2.0,
        "release_deadline": 5.0,
        "target_threshold": 0.5,
        "seed_base": seed,
        "n_origins": 3,
        "training_origins": 1,
        "selection_origins": 1,
        "evaluation_origins": 1,
    })
    prep = prepare_origins(profile, progress=False)
    origin = prep.evaluation[0]
    problem = ControlProblem(
        problem_id="test_problem",
        target_type=TargetType.REVERSAL,
        target_params={"target_sign": 1.0, "threshold": 0.5},
        N=8,
        D=1,
        k=4,
        exposure_cap=50.0,
        amplitude_cap=4.0,
        release_deadline=5.0,
        holding_duration=2.0,
        horizon=10.0,
        accessible_roster=(0, 1, 2, 3),
        proposal_seed=seed + 1,
        search_seed=seed + 2,
    )
    physics = {
        "J": float(profile.J),
        "T": float(profile.temperature),
        "delta": float(profile.delta),
        "epsilon": float(profile.exploration),
        "closure": float(profile.closure),
        "a": float(profile.prestige_global),
        "c": float(profile.matching),
        "tau_y": float(profile.tau_public),
        "sample_interval": float(profile.sample_interval_tau_public) * float(profile.tau_public),
    }
    obs_kwargs = {
        "panel_fraction": float(profile.panel_fraction),
        "edge_reveal_prob": float(profile.edge_reveal_prob),
        "binary_flip_prob": float(profile.binary_flip_prob),
        "delivery_delay": float(profile.delivery_delay_tau_public) * float(profile.tau_public),
    }
    return origin, problem, physics, obs_kwargs


def _run_sim(
    origin,
    problem,
    controller,
    physics,
    obs_kwargs,
    sim_seed: int,
    ctl_seed: int,
    dynamics: str = "adaptive",
    recorder: Optional[RolloutRecorder] = None,
):
    from src.control.observation import SampledPublicObservationModel
    state = origin.state.copy()
    obs_model = SampledPublicObservationModel(
        N=problem.N,
        panel_fraction=obs_kwargs["panel_fraction"],
        edge_reveal_prob=obs_kwargs["edge_reveal_prob"],
        binary_flip_prob=obs_kwargs["binary_flip_prob"],
        accessible_roster=problem.accessible_roster,
        group_labels=state.opportunity_groups,
        rng=np.random.default_rng(ctl_seed),
        delivery_delay=obs_kwargs["delivery_delay"],
    )
    return run_controlled_simulation(
        initial_state=state,
        problem=problem,
        controller=controller,
        observation_model=obs_model,
        J=physics["J"],
        T=physics["T"],
        delta=physics["delta"],
        epsilon=physics["epsilon"],
        closure=physics["closure"],
        sample_interval=physics["sample_interval"],
        adaptive_memory=(dynamics == "adaptive"),
        between_eligibility=origin.between_eligibility,
        within_eligibility=origin.within_eligibility,
        a=physics["a"],
        c=physics["c"],
        tau_y=physics["tau_y"],
        sim_rng=np.random.default_rng(sim_seed),
        controller_rng=np.random.default_rng(ctl_seed),
        recorder=recorder,
    )


class TestStochasticInvariance(unittest.TestCase):
    """Acceptance: Recorder off vs on and varying cadences produce identical outcomes bit-for-bit."""

    def test_stochastic_invariance_recorder_off_vs_on(self):
        """Simulations with recorder=None, cadence=1.0, and cadence=0.25 produce identical states."""
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=42001)

        policy = ObservableSelectorPolicy(
            selector="random",
            nominal_amplitude=2.0,
            duration=4.0,
            quota=0.0,
            is_feedback=False,
            component=0,
        )

        sim_seed = 99881
        ctl_seed = 77661

        # Run 1: recorder = None
        res_off, state_off, ledger_off = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=sim_seed,
            ctl_seed=ctl_seed,
            dynamics="adaptive",
            recorder=None,
        )

        # Run 2: recorder on, cadence = 1.0
        rec_1 = RolloutRecorder(RecorderConfig(cadence=1.0, record_network_frames=True))
        res_1, state_1, ledger_1 = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=sim_seed,
            ctl_seed=ctl_seed,
            dynamics="adaptive",
            recorder=rec_1,
        )

        # Run 3: recorder on, cadence = 0.25 (different sampling interval)
        rec_025 = RolloutRecorder(RecorderConfig(cadence=0.25, record_network_frames=True))
        res_025, state_025, ledger_025 = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=sim_seed,
            ctl_seed=ctl_seed,
            dynamics="adaptive",
            recorder=rec_025,
        )

        # Exact bit-for-bit state preservation
        np.testing.assert_array_equal(state_off.x, state_1.x, err_msg="state.x diverged between off and cadence=1.0")
        np.testing.assert_array_equal(state_off.y, state_1.y, err_msg="state.y diverged between off and cadence=1.0")
        np.testing.assert_array_equal(state_off.W, state_1.W, err_msg="state.W diverged between off and cadence=1.0")

        np.testing.assert_array_equal(state_off.x, state_025.x, err_msg="state.x diverged between off and cadence=0.25")
        np.testing.assert_array_equal(state_off.y, state_025.y, err_msg="state.y diverged between off and cadence=0.25")
        np.testing.assert_array_equal(state_off.W, state_025.W, err_msg="state.W diverged between off and cadence=0.25")

        # Exact ledger expenditure and distinct nodes preservation
        self.assertEqual(ledger_off.integrated_exposure, ledger_1.integrated_exposure)
        self.assertEqual(ledger_off.integrated_exposure, ledger_025.integrated_exposure)
        self.assertEqual(ledger_off.distinct_nodes_actuated, ledger_1.distinct_nodes_actuated)
        self.assertEqual(ledger_off.distinct_nodes_actuated, ledger_025.distinct_nodes_actuated)

        # Exact target outcomes and event counts preservation
        self.assertEqual(res_off.success, res_1.success)
        self.assertEqual(res_off.success, res_025.success)
        self.assertEqual(res_off.event_counts, res_1.event_counts)
        self.assertEqual(res_off.event_counts, res_025.event_counts)


class TestSpendingVsLedgerAgreement(unittest.TestCase):
    """Acceptance: Recorded integrated spending agrees with ledger expenditure within numerical tolerance."""

    def test_integrated_spending_agrees_with_ledger(self):
        """Recorded cumulative exposure and rate integral match ledger expenditure within 1e-6."""
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=33112)

        policy = ObservableSelectorPolicy(
            selector="random",
            nominal_amplitude=2.5,
            duration=3.0,
            quota=0.0,
            is_feedback=False,
            component=0,
        )

        rec = RolloutRecorder(RecorderConfig(cadence=0.5, record_action_history=True))
        res, state, ledger = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=55441,
            ctl_seed=66331,
            dynamics="adaptive",
            recorder=rec,
        )

        # Positive spending occurred
        self.assertGreater(ledger.integrated_exposure, 0.0)

        # 1. Terminal recorded exposure agrees with ledger exposure
        rec_spending = rec.integrated_spending()
        self.assertAlmostEqual(rec_spending, ledger.integrated_exposure, delta=1e-6)

        # 2. Last scalar point exposure agrees with ledger exposure
        self.assertAlmostEqual(rec.scalar_points[-1].exposure, ledger.integrated_exposure, delta=1e-6)

        # 3. Monotonicity of cumulative exposure
        for i in range(len(rec.scalar_points) - 1):
            p0 = rec.scalar_points[i]
            p1 = rec.scalar_points[i + 1]
            self.assertGreaterEqual(p1.exposure, p0.exposure - 1e-12)

        # 4. Left Riemann sum over spending rate matches ledger when duration aligns with cadence
        rate_integral = rec.integrate_spending_rate(method="left")
        self.assertAlmostEqual(rate_integral, ledger.integrated_exposure, delta=1e-6)

    def test_recorder_deep_copies_arrays(self):
        """Mutating live simulation arrays after rollout does not corrupt recorded frames."""
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=22110)
        rec = RolloutRecorder(RecorderConfig(cadence=1.0, record_network_frames=True))
        res, state, ledger = _run_sim(
            origin=origin,
            problem=problem,
            controller=ZeroController(),
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=11223,
            ctl_seed=33445,
            dynamics="adaptive",
            recorder=rec,
        )

        self.assertGreater(len(rec.network_frames), 0)
        first_frame = rec.network_frames[0]
        x_saved = first_frame.x.copy()
        y_saved = first_frame.y.copy()
        W_saved = first_frame.W.copy()

        # Mutate live simulation state
        state.x.fill(999.0)
        state.y.fill(-888.0)
        state.W.fill(777.0)

        # Confirm recorded frame was deep-copied and remained completely unaffected
        np.testing.assert_array_equal(first_frame.x, x_saved)
        np.testing.assert_array_equal(first_frame.y, y_saved)
        np.testing.assert_array_equal(first_frame.W, W_saved)


class TestActionHistoryRecording(unittest.TestCase):
    """Acceptance: Save actual action history with decisions, parameters, clampings, and shutoffs."""

    def test_action_history_fields_and_clamping(self):
        """Action history captures initial actuation, requested vs effective end, and shutoff."""
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=12121)

        # Small exposure cap to trigger budget clamping
        problem_clamped = ControlProblem(
            problem_id="clamped_problem",
            target_type=TargetType.REVERSAL,
            target_params={"target_sign": 1.0, "threshold": 0.5},
            N=8,
            D=1,
            k=4,
            exposure_cap=3.0,  # Tight budget: 3.0 / (2.0 * 2 nodes = 4.0 rate) = 0.75 duration max
            amplitude_cap=4.0,
            release_deadline=5.0,
            holding_duration=2.0,
            horizon=10.0,
            accessible_roster=(0, 1, 2, 3),
            proposal_seed=123,
            search_seed=456,
        )

        class ClampedPulsePolicy:
            def __init__(self, amplitude: float = 2.0, duration: float = 4.0):
                self.amplitude = amplitude
                self.duration = duration
                self.acted = False

            def reset(self, problem, initial_obs=None, rng=None):
                self.acted = False

            def act(self, obs, ledger):
                if self.acted or ledger.is_released:
                    return Action(entries=(), release=True)
                self.acted = True
                t = ledger.current_time
                return Action(
                    entries=(
                        ActionEntry(node=0, component=0, field=self.amplitude),
                        ActionEntry(node=1, component=0, field=self.amplitude),
                    ),
                    intended_end_time=t + self.duration,
                    release=False,
                )

        policy = ClampedPulsePolicy(amplitude=2.0, duration=4.0)

        rec = RolloutRecorder(RecorderConfig(cadence=1.0, record_action_history=True))
        res, state, ledger = _run_sim(
            origin=origin,
            problem=problem_clamped,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=9911,
            ctl_seed=8822,
            dynamics="adaptive",
            recorder=rec,
        )

        self.assertGreater(len(rec.action_history), 0)
        t0_action = rec.action_history[0]

        # Verify action record attributes: delivered at t=1.0 from report sampled at t=0.0
        self.assertEqual(t0_action.decision_time, 1.0)
        self.assertEqual(t0_action.report_sample_time, 0.0)
        self.assertGreater(len(t0_action.actuated_nodes), 0)
        self.assertGreater(len(t0_action.field_entries), 0)
        self.assertEqual(t0_action.requested_end, 5.0)

        # Budget clamping verified: 1.0 + 3.0 / 4.0 = 1.75 < 5.0
        self.assertTrue(t0_action.budget_clamped)
        self.assertLess(t0_action.effective_end, t0_action.requested_end)
        self.assertAlmostEqual(t0_action.effective_end, 1.75, delta=1e-3)

        # Shutoff or release record exists
        event_types = [a.shutoff_or_release for a in rec.action_history]
        self.assertTrue(any(et in ("field_shutoff", "release_deadline", "budget_exhaustion") for et in event_types))


class TestFrozenFramesAndMatchedTimestamps(unittest.TestCase):
    """Acceptance: Frozen frames have unchanged W; matched frames share origin and physical timestamps."""

    def test_frozen_frames_constant_W_and_matched_timestamps(self):
        """Frozen frames retain initial W, and matched adaptive/frozen frames share timestamps."""
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=77114)
        policy = ZeroController()

        cadence = 0.5
        rec_adapt = RolloutRecorder(RecorderConfig(cadence=cadence, record_network_frames=True))
        rec_froz = RolloutRecorder(RecorderConfig(cadence=cadence, record_network_frames=True))

        sim_seed = 44556
        ctl_seed = 11223

        # Adaptive rollout
        res_a, state_a, _ = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=sim_seed,
            ctl_seed=ctl_seed,
            dynamics="adaptive",
            recorder=rec_adapt,
        )

        # Frozen rollout
        res_f, state_f, _ = _run_sim(
            origin=origin,
            problem=problem,
            controller=policy,
            physics=physics,
            obs_kwargs=obs_kwargs,
            sim_seed=sim_seed,
            ctl_seed=ctl_seed,
            dynamics="frozen",
            recorder=rec_froz,
        )

        # 1. Frozen frames have unchanged W across all time points
        initial_W = origin.state.W
        for frame in rec_froz.network_frames:
            np.testing.assert_array_equal(
                frame.W,
                initial_W,
                err_msg=f"Frozen frame at t={frame.t} had modified W!",
            )

        # 2. Matched frames share physical timestamps exactly
        times_adapt = [round(f.t, 6) for f in rec_adapt.network_frames]
        times_froz = [round(f.t, 6) for f in rec_froz.network_frames]
        self.assertEqual(times_adapt, times_froz)
        self.assertGreater(len(times_adapt), 1)

        # 3. Initial frames share exact origin
        np.testing.assert_array_equal(rec_adapt.network_frames[0].x, rec_froz.network_frames[0].x)
        np.testing.assert_array_equal(rec_adapt.network_frames[0].y, rec_froz.network_frames[0].y)
        np.testing.assert_array_equal(rec_adapt.network_frames[0].W, rec_froz.network_frames[0].W)


class TestBoundedNetworkFrameCoverage(unittest.TestCase):
    """Long runs retain the actuation period and the actual final network."""

    def test_frames_span_intervention_release_and_horizon(self):
        origin, problem, physics, obs_kwargs = _make_test_origin_and_problem(seed=66201)
        problem = replace(problem, horizon=120.0, release_deadline=20.0,
                          holding_duration=30.0)
        policy = ObservableSelectorPolicy(
            selector="random", nominal_amplitude=2.0, duration=15.0,
            quota=0.0, is_feedback=False, component=0,
        )
        recorder = RolloutRecorder(RecorderConfig(
            cadence=1.0, record_network_frames=True, max_network_frames=12,
        ))
        _run_sim(origin, problem, policy, physics, obs_kwargs,
                 sim_seed=19281, ctl_seed=19282, recorder=recorder)
        frames = recorder.network_frames
        times = [frame.t for frame in frames]
        self.assertLessEqual(len(frames), 12)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 120.0)
        self.assertEqual(times, sorted(set(times)))
        self.assertTrue(any(0 < frame.t < 20 and frame.controlled_mask.any()
                            for frame in frames))
        self.assertTrue(any(20 < frame.t < 120 for frame in frames))


class TestOriginalStateRetention(unittest.TestCase):
    """Acceptance: Define original-state retention separately from reversal.
    A trajectory losing alignment without reversing is certified as nonpersistent.
    """

    def test_retention_fixture_nonpersistent_without_reversal(self):
        """Trajectory relaxes below threshold without reversing sign; reported as nonpersistent."""
        # Initial positive basin: m_y(0) = 0.8, m_x(0) = 0.8 (s_0 = +1.0)
        # Relaxes to 0.2 at t=5.0: 0.2 > 0 (no sign reversal!), but 0.2 < threshold 0.5.
        synthetic_trajectory = [
            {"t": 0.0, "m_y": 0.8, "m_x": 0.8},
            {"t": 1.0, "m_y": 0.75, "m_x": 0.75},
            {"t": 2.0, "m_y": 0.65, "m_x": 0.65},
            {"t": 3.0, "m_y": 0.55, "m_x": 0.55},
            {"t": 4.0, "m_y": 0.50, "m_x": 0.50},
            {"t": 5.0, "m_y": 0.20, "m_x": 0.20},  # Lost retention here!
            {"t": 6.0, "m_y": 0.15, "m_x": 0.15},
            {"t": 7.0, "m_y": 0.10, "m_x": 0.10},
        ]

        ret = evaluate_original_state_retention(
            synthetic_trajectory,
            threshold=0.5,
            window=(0.0, 7.0),
        )

        self.assertFalse(ret.is_retained, "Trajectory should NOT be retained!")
        self.assertEqual(ret.initial_sign, 1.0)
        self.assertEqual(ret.threshold, 0.5)
        self.assertEqual(ret.loss_time, 5.0)
        self.assertAlmostEqual(ret.min_alignment, 0.10, delta=1e-6)
        self.assertAlmostEqual(ret.max_excursion, 0.40, delta=1e-6)  # 0.5 - 0.10 = 0.40
        self.assertGreater(ret.loss_duration, 0.0)

    def test_retention_fixture_sustained_persistence(self):
        """Trajectory maintains orientation above threshold; certified as retained."""
        synthetic_trajectory = [
            {"t": 0.0, "m_y": 0.8, "m_x": 0.8},
            {"t": 1.0, "m_y": 0.78, "m_x": 0.77},
            {"t": 2.0, "m_y": 0.70, "m_x": 0.72},
            {"t": 3.0, "m_y": 0.65, "m_x": 0.66},
            {"t": 4.0, "m_y": 0.60, "m_x": 0.62},
            {"t": 5.0, "m_y": 0.63, "m_x": 0.65},
        ]

        ret = evaluate_original_state_retention(
            synthetic_trajectory,
            threshold=0.5,
            window=(0.0, 5.0),
        )

        self.assertTrue(ret.is_retained, "Trajectory should be certified as retained!")
        self.assertEqual(ret.initial_sign, 1.0)
        self.assertIsNone(ret.loss_time)
        self.assertEqual(ret.loss_duration, 0.0)
        self.assertEqual(ret.max_excursion, 0.0)
        self.assertAlmostEqual(ret.min_alignment, 0.60, delta=1e-6)

    def test_retention_negative_basin_orientation(self):
        """Negative initial basin orientation s_0 = -1.0 correctly evaluates retention."""
        # Initial negative basin: m_y(0) = -0.85, m_x(0) = -0.80
        # Retention condition: (-1) * m_y >= 0.5 <=> m_y <= -0.5
        synthetic_trajectory = [
            {"t": 0.0, "m_y": -0.85, "m_x": -0.80},
            {"t": 1.0, "m_y": -0.70, "m_x": -0.70},
            {"t": 2.0, "m_y": -0.30, "m_x": -0.60},  # m_y drifted to -0.30 > -0.50 (lost!)
        ]

        ret = evaluate_original_state_retention(
            synthetic_trajectory,
            threshold=0.5,
        )

        self.assertFalse(ret.is_retained)
        self.assertEqual(ret.initial_sign, -1.0)
        self.assertEqual(ret.loss_time, 2.0)
        self.assertAlmostEqual(ret.min_m_y_alignment, 0.30, delta=1e-6)
        self.assertAlmostEqual(ret.max_excursion, 0.20, delta=1e-6)  # 0.50 - 0.30 = 0.20


class TestUnifiedBundleExportAndReload(unittest.TestCase):
    """Acceptance: Create bundle directory once, re-export without simulation, reload and reconstruct plots."""

    def test_bundle_creation_and_reexport(self):
        """Single bundle directory created once, re-export updates cleanly without subdirectories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = make_campaign_bundle_dir(tmp_dir, "test_camp")
            self.assertTrue(bundle_dir.exists())
            self.assertTrue((bundle_dir / "snapshots").exists())

            # Create mock profile and run mini campaign
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "horizon": 10.0, "establishment_time": 4.0,
                "n_origins": 3, "seed_base": 8801,
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
                campaign_id="test_camp",
                master_seed=44001,
                out_dir=tmp_dir,
            )

            # Unified bundle directory was reused
            self.assertEqual(Path(result.bundle_path).resolve(), bundle_dir.resolve())

            # Required bundle files exist
            expected_files = [
                "manifest.json",
                "accepted_snapshots.json",
                "preparation_attempts.json",
                "trajectories.npz",
                "diagnostic_trajectories.json",
                "action_history.json",
                "network_frames.npz",
                "retention_summary.json",
                "rollout_results.csv",
                "adaptive_frozen_pairs.csv",
            ]
            for fn in expected_files:
                self.assertTrue((bundle_dir / fn).exists(), f"Missing bundle file: {fn}")

            # Verify manifest schema and file inventory
            with open(bundle_dir / "manifest.json", "r") as f:
                manifest = json.load(f)

            self.assertIn("file_inventory", manifest)
            self.assertIn("checksums", manifest)
            self.assertIn("uncontrolled_persistence", manifest)
            self.assertEqual(manifest["uncontrolled_persistence"]["status"], "measured")

            # Checksums match disk contents
            for rel_path, expected_hash in manifest["checksums"].items():
                p = bundle_dir / rel_path
                self.assertTrue(p.exists())
                with open(p, "rb") as bf:
                    actual_hash = hashlib.sha256(bf.read()).hexdigest()
                self.assertEqual(actual_hash, expected_hash, f"Hash mismatch for {rel_path}")

            # Re-export to same directory does not create a new directory or re-run
            reexported_dir = export_control_bundle(result, out_dir=bundle_dir)
            self.assertEqual(Path(reexported_dir).resolve(), bundle_dir.resolve())
            self.assertFalse((bundle_dir / "bundle").exists())

    def test_bundle_reload_and_plot_data_reconstruction(self):
        """Reload bundle in fresh process and reconstruct plot data without live simulation object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "horizon": 8.0, "establishment_time": 3.0,
                "n_origins": 3, "seed_base": 9901,
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
                campaign_id="reload_camp",
                master_seed=55112,
                out_dir=tmp_dir,
            )

            bundle_path = Path(result.bundle_path)

            # Reload in a totally isolated handle
            loaded = load_control_bundle(bundle_path)

            self.assertIsInstance(loaded, LoadedControlBundle)
            self.assertGreater(len(loaded.trajectories), 0)
            self.assertGreater(len(loaded.rollouts), 0)
            self.assertGreater(len(loaded.diagnostic_summary), 0)
            self.assertIn("status", loaded.retention_summary.get("uncontrolled_persistence", {}))

            # Reconstruct plot data
            plot_data = loaded.reconstruct_plot_data()

            self.assertIn("trajectories", plot_data)
            self.assertIn("budget_consumption", plot_data)
            self.assertIn("policy_comparisons", plot_data)
            self.assertIn("network_frames", plot_data)
            self.assertIn("retention", plot_data)
            self.assertIn("manifest", plot_data)

            # Verify reconstructed trajectory points have required fields
            for rk, pts in plot_data["trajectories"].items():
                self.assertGreater(len(pts), 0)
                first = pts[0]
                for req_key in ("t", "m_y", "m_x", "distance", "exposure", "spending_rate", "hold_progress", "predicate", "is_released"):
                    self.assertIn(req_key, first, f"Missing key {req_key} in reconstructed trajectory")

            # Verify network frames were reconstructed
            self.assertGreater(len(plot_data["network_frames"]), 0)
            for rk, frames in plot_data["network_frames"].items():
                self.assertGreater(len(frames), 0)
                f0 = frames[0]
                self.assertIn("t", f0)
                self.assertIn("W", f0)
                self.assertIn("x", f0)
                self.assertIn("y", f0)
                self.assertIn("controlled_mask", f0)


if __name__ == "__main__":
    unittest.main()
