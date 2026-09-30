"""Unit tests for Resumption, Checkpointing, Work Registry, and Exploratory Grid.

Verifies:
1. Controlled rollout checkpointing & mid-rollout continuation:
   - Resumed controlled rollout matches uninterrupted seeded rollout bit-for-bit.
   - Exact continuation with pending observation deliveries.
   - Exact continuation during active holding windows.
2. Checkpoint & registry configuration compatibility and corruption rejection:
   - Incompatible problem parameters reject with IncompatibleConfigurationError.
   - Corrupted or tampered files reject with CorruptedCheckpointError.
3. Campaign work registry & interruption resumption:
   - Interrupted preparation reuses accepted snapshots without duplicate work.
   - Resumed campaign reloads completed rollouts without re-executing simulations.
4. Phase timings & workload projection:
   - Explicit phase timings tracked and recorded across all phases.
   - Preparation progress accurately tracked.
   - Runtime projection computed with empirical uncertainty (95% CI).
5. Exploratory diagnostic response grid & 2D maps:
   - Declared axes, disjoint seeds, and strict compute budget cap enforcement.
   - Unevaluated cells marked clearly without invented smoothing.
   - Trajectory failure classification across physical mechanisms.
   - 2D response map rendering directly from saved JSON.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.config.control_profiles import LIGHT_CONTROL_PROFILE
from src.control.campaign import (
    ControlCampaignSpec,
    build_campaign_spec,
    prepare_origins,
    run_campaign,
)
from src.control.checkpointing import (
    ControlledRolloutCheckpoint,
    CorruptedCheckpointError,
    IncompatibleConfigurationError,
    capture_rollout_checkpoint,
    load_controlled_checkpoint,
    restore_rollout_state,
    save_controlled_checkpoint,
)
from src.control.contracts import Action, ControlProblem, RolloutResult, TargetType
from src.control.controllers import ObservableSelectorPolicy, OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.grid import (
    ExploratoryGridConfig,
    ExploratoryGridResult,
    GridAxis,
    classify_rollout_failure_mechanism,
    run_exploratory_response_grid,
)
from src.control.observation import SampledPublicObservationModel
from src.control.recording import ActionRecord, RecorderConfig, RolloutRecorder, ScalarTrajectoryPoint
from src.control.registry import (
    CampaignWorkRegistry,
    deserialize_rollout_result,
    serialize_rollout_result,
)
from src.core.state import State, StateMode
from src.visualization.control_plots import plot_diagnostic_response_maps


def _make_test_state(N: int = 8, D: int = 1, seed: int = 12345) -> State:
    """Create a deterministic test state with initialized beliefs and memory ties."""
    rng = np.random.default_rng(seed)
    y = np.full((N, D), -0.8, dtype=np.float64)
    x = np.full((N, D), -0.7, dtype=np.float64)
    W = np.zeros((N, N), dtype=np.uint32)
    for i in range(N):
        j = (i + 1) % N
        W[i, j] = 5
        W[j, i] = 5
    return State(
        N=N,
        D=D,
        state_mode=StateMode.CONTINUOUS,
        x=x,
        y=y,
        W=W,
        n=np.zeros(N, dtype=np.uint8),
        Q=np.zeros((N, D, 2), dtype=np.float64),
        t=0.0,
        opportunity_groups=np.zeros(N, dtype=int),
        K=rng.uniform(0.8, 1.2, size=N),
        tau_x=rng.uniform(0.5, 1.5, size=N),
        nu=rng.uniform(0.1, 0.3, size=N),
        g=rng.uniform(0.05, 0.15, size=N),
        b=np.zeros((N, D), dtype=np.float64),
    )


def _make_test_problem(N: int = 8, D: int = 1, horizon: float = 10.0) -> ControlProblem:
    """Create a minimal control problem for unit testing."""
    return ControlProblem(
        problem_id="test_prob",
        target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.3},
        N=N,
        D=D,
        k=4,
        exposure_cap=20.0,
        amplitude_cap=5.0,
        release_deadline=horizon * 0.8,
        holding_duration=2.0,
        horizon=horizon,
        accessible_roster=tuple(range(N)),
        proposal_seed=111,
        search_seed=222,
        config_version="control-campaign-m4-1.0",
    )


class TestControlledRolloutCheckpointingAndContinuation(unittest.TestCase):
    """Verifies mid-rollout checkpoint capture, exact continuation, and bit-for-bit parity."""

    def test_mid_rollout_checkpoint_exact_match_uninterrupted(self):
        """Resumed controlled rollout matches uninterrupted seeded rollout bit-for-bit."""
        problem = _make_test_problem(N=8, horizon=8.0)
        initial_state = _make_test_state(N=8, seed=42)

        ctl = OpenLoopPulseController(
            target_nodes=(0, 1, 2),
            signs={0: 1.0, 1: 1.0, 2: 1.0},
            nominal_amplitude=2.5,
            duration=3.0,
        )

        obs_model_full = SampledPublicObservationModel(
            N=8,
            accessible_roster=problem.accessible_roster,
            rng=np.random.default_rng(888),
            panel_fraction=0.5,
            edge_reveal_prob=0.5,
            delivery_delay=0.5,
        )
        eval_full = TargetEvaluator.create_for_state(
            target_type=problem.target_type,
            initial_state=initial_state,
            holding_duration=problem.holding_duration,
            horizon=problem.horizon,
            release_deadline=problem.release_deadline,
            target_params=problem.target_params,
        )
        rec_full = RolloutRecorder(RecorderConfig(cadence=0.5, record_scalar_trajectories=True))

        # 1. Run uninterrupted rollout
        res_full, final_full, ledger_full = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=obs_model_full,
            target_evaluator=eval_full,
            sim_rng=np.random.default_rng(999),
            controller_rng=np.random.default_rng(777),
            recorder=rec_full,
        )

        # 2. Run interrupted rollout: stop at checkpoint t = 3.5
        obs_model_part = SampledPublicObservationModel(
            N=8,
            accessible_roster=problem.accessible_roster,
            rng=np.random.default_rng(888),
            panel_fraction=0.5,
            edge_reveal_prob=0.5,
            delivery_delay=0.5,
        )
        eval_part = TargetEvaluator.create_for_state(
            target_type=problem.target_type,
            initial_state=initial_state,
            holding_duration=problem.holding_duration,
            horizon=problem.horizon,
            release_deadline=problem.release_deadline,
            target_params=problem.target_params,
        )
        rec_part = RolloutRecorder(RecorderConfig(cadence=0.5, record_scalar_trajectories=True))

        res_part, _, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=obs_model_part,
            target_evaluator=eval_part,
            sim_rng=np.random.default_rng(999),
            controller_rng=np.random.default_rng(777),
            recorder=rec_part,
            checkpoint_at=3.5,
            stop_at_checkpoint=True,
        )
        self.assertTrue(res_part.is_checkpoint_interrupted)
        self.assertIsNotNone(res_part.checkpoint)

        # 3. Resume from checkpoint to horizon
        res_resumed, final_resumed, ledger_resumed = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            resume_checkpoint=res_part.checkpoint,
        )

        # 4. Check bit-for-bit parity
        np.testing.assert_array_equal(final_full.x, final_resumed.x, err_msg="State x mismatch after continuation")
        np.testing.assert_array_equal(final_full.y, final_resumed.y, err_msg="State y mismatch after continuation")
        np.testing.assert_array_equal(final_full.W, final_resumed.W, err_msg="Matrix W mismatch after continuation")
        self.assertEqual(res_full.event_counts, res_resumed.event_counts)
        self.assertAlmostEqual(res_full.expenditure, res_resumed.expenditure, places=9)
        self.assertEqual(res_full.distinct_nodes_used, res_resumed.distinct_nodes_used)
        self.assertEqual(res_full.success, res_resumed.success)

        # Verify trajectory scalar points match
        self.assertEqual(len(res_full.recorder.scalar_points), len(res_resumed.recorder.scalar_points))
        for p_f, p_r in zip(res_full.recorder.scalar_points, res_resumed.recorder.scalar_points):
            self.assertAlmostEqual(p_f.t, p_r.t, places=9)
            self.assertAlmostEqual(p_f.m_x, p_r.m_x, places=9)
            self.assertAlmostEqual(p_f.m_y, p_r.m_y, places=9)

    def test_checkpoint_continuation_with_pending_observation_delivery(self):
        """Continuation with pending observation deliveries matches uninterrupted rollout exactly."""
        problem = _make_test_problem(N=8, horizon=8.0)
        initial_state = _make_test_state(N=8, seed=42)

        ctl = OpenLoopPulseController(
            target_nodes=(0, 1),
            signs={0: 1.0, 1: 1.0},
            nominal_amplitude=2.0,
            duration=4.0,
        )

        # Uninterrupted run with delivery delay
        obs_1 = SampledPublicObservationModel(
            N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(1234),
            panel_fraction=0.5, edge_reveal_prob=0.5, delivery_delay=2.0,
        )
        res_full, final_full, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=obs_1,
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 8.0, 6.0, problem.target_params),
            sample_interval=1.0,
            sim_rng=np.random.default_rng(555),
            controller_rng=np.random.default_rng(777),
        )

        # Interrupted run stopping at t=1.5 (while observations scheduled for t=2.0 and t=3.0 are pending)
        obs_2 = SampledPublicObservationModel(
            N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(1234),
            panel_fraction=0.5, edge_reveal_prob=0.5, delivery_delay=2.0,
        )
        res_part, _, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=obs_2,
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 8.0, 6.0, problem.target_params),
            sample_interval=1.0,
            sim_rng=np.random.default_rng(555),
            controller_rng=np.random.default_rng(777),
            checkpoint_at=1.5,
            stop_at_checkpoint=True,
        )
        self.assertIsNotNone(res_part.checkpoint)
        self.assertGreater(len(res_part.checkpoint.observation_model_state["pending_deliveries"]), 0)

        # Resumed run from checkpoint
        res_resumed, final_resumed, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            resume_checkpoint=res_part.checkpoint,
        )

        np.testing.assert_array_equal(final_full.y, final_resumed.y)
        self.assertEqual(res_full.event_counts, res_resumed.event_counts)

    def test_checkpoint_continuation_active_holding_window(self):
        """Continuation during active holding window retains certified success and bit-for-bit parity."""
        problem = _make_test_problem(N=8, horizon=6.0)
        problem = dataclasses.replace(
            problem,
            k=8,
            exposure_cap=1000.0,
            amplitude_cap=30.0,
            release_deadline=2.0,
            holding_duration=2.0,
        )
        initial_state = _make_test_state(N=8, seed=42)
        initial_state.nu = np.full(8, 10.0)
        initial_state.tau_x = np.full(8, 0.1)

        # Pulse drives state into target basin before release deadline
        ctl = OpenLoopPulseController(
            target_nodes=tuple(range(8)),
            signs={i: 1.0 for i in range(8)},
            nominal_amplitude=20.0,
            duration=2.0,
        )

        res_full, final_full, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=SampledPublicObservationModel(N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(123), delivery_delay=0.0),
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 6.0, 2.0, problem.target_params),
            sim_rng=np.random.default_rng(333),
            controller_rng=np.random.default_rng(444),
        )

        # Checkpoint at t=2.5 when holding is active
        res_part, _, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=SampledPublicObservationModel(N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(123), delivery_delay=0.0),
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 6.0, 2.0, problem.target_params),
            sim_rng=np.random.default_rng(333),
            controller_rng=np.random.default_rng(444),
            checkpoint_at=2.5,
            stop_at_checkpoint=True,
        )
        self.assertIsNotNone(res_part.checkpoint)
        # Evaluator holding state captured
        eval_state = res_part.checkpoint.target_evaluator_state
        self.assertTrue(eval_state["in_holding"] or eval_state["holding_start_time"] is not None or eval_state["certified"])

        # Continuation
        res_resumed, final_resumed, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            resume_checkpoint=res_part.checkpoint,
        )

        self.assertEqual(res_full.success, res_resumed.success)
        self.assertEqual(res_full.event_counts, res_resumed.event_counts)
        np.testing.assert_array_equal(final_full.y, final_resumed.y)


class TestConfigCompatibilityAndCorruptionRejection(unittest.TestCase):
    """Verifies that corrupted checkpoints and incompatible configurations are strictly rejected."""

    def test_checkpoint_rejects_incompatible_problem_configuration(self):
        """Loading a checkpoint against an incompatible problem configuration raises IncompatibleConfigurationError."""
        problem = _make_test_problem(N=8, horizon=8.0)
        initial_state = _make_test_state(N=8, seed=42)
        ctl = ZeroController()

        res, _, _ = run_controlled_simulation(
            initial_state=initial_state,
            problem=problem,
            controller=ctl,
            observation_model=SampledPublicObservationModel(N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(1)),
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 8.0, 6.0, problem.target_params),
            sim_rng=np.random.default_rng(10),
            controller_rng=np.random.default_rng(20),
            checkpoint_at=2.0,
            stop_at_checkpoint=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "checkpoint.json"
            save_controlled_checkpoint(res.checkpoint, chk_path)

            # Incompatible problem: different N
            incompatible_prob = _make_test_problem(N=12, horizon=8.0)
            with self.assertRaises(IncompatibleConfigurationError):
                load_controlled_checkpoint(chk_path, expected_problem=incompatible_prob)

            # Incompatible problem: different holding duration
            incompatible_holding = dataclasses.replace(problem, holding_duration=5.0)
            with self.assertRaises(IncompatibleConfigurationError):
                load_controlled_checkpoint(chk_path, expected_problem=incompatible_holding)

    def test_checkpoint_rejects_corrupted_payload_or_hash(self):
        """Tampered checkpoint content or checksum mismatch raises CorruptedCheckpointError."""
        problem = _make_test_problem(N=8, horizon=8.0)
        initial_state = _make_test_state(N=8, seed=42)
        ctl = ZeroController()

        res, _, _ = run_controlled_simulation(
            initial_state=initial_state,
            problem=problem,
            controller=ctl,
            observation_model=SampledPublicObservationModel(N=8, accessible_roster=problem.accessible_roster, rng=np.random.default_rng(1)),
            target_evaluator=TargetEvaluator.create_for_state(problem.target_type, initial_state, 2.0, 8.0, 6.0, problem.target_params),
            sim_rng=np.random.default_rng(10),
            controller_rng=np.random.default_rng(20),
            checkpoint_at=2.0,
            stop_at_checkpoint=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "checkpoint.json"
            save_controlled_checkpoint(res.checkpoint, chk_path)

            # Tamper with file on disk
            with open(chk_path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
            # Modify payload while keeping old sha256
            envelope["payload"]["sim_time"] = 999.0
            with open(chk_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f)

            with self.assertRaises(CorruptedCheckpointError):
                load_controlled_checkpoint(chk_path, expected_problem=problem)

    def test_registry_rejects_incompatible_spec_or_profile_hash(self):
        """Loading a campaign registry with mismatched spec or profile hash raises IncompatibleConfigurationError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "bundle"
            bundle_dir.mkdir()

            spec_1 = {"target_type": "reversal"}
            prof_1 = {"seed_base": 1234}
            hash_spec = CampaignWorkRegistry.compute_spec_hash(spec_1)
            hash_prof = CampaignWorkRegistry.compute_profile_hash(prof_1)

            reg = CampaignWorkRegistry(bundle_dir, spec_hash=hash_spec, profile_hash=hash_prof)
            reg.save()

            # Compatible spec/profile loads fine
            reg_loaded = CampaignWorkRegistry.load_or_create(
                bundle_dir=bundle_dir,
                spec=spec_1,
                profile=prof_1,
                verify_checksums=False,
            )
            self.assertIsNotNone(reg_loaded)

            # Incompatible spec raises IncompatibleConfigurationError
            with self.assertRaises(IncompatibleConfigurationError):
                CampaignWorkRegistry.load_or_create(
                    bundle_dir=bundle_dir,
                    spec={"target_type": "polarization_maintenance"},  # Different spec
                    profile=prof_1,
                    verify_checksums=True,
                )

            # Incompatible profile raises IncompatibleConfigurationError
            with self.assertRaises(IncompatibleConfigurationError):
                CampaignWorkRegistry.load_or_create(
                    bundle_dir=bundle_dir,
                    spec=spec_1,
                    profile={"seed_base": 999},  # Non-matching profile
                    verify_checksums=True,
                )

    def test_registry_detects_corrupted_work_unit_file(self):
        """Corrupted work unit file raises CorruptedCheckpointError when retrieved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "bundle"
            bundle_dir.mkdir()
            reg = CampaignWorkRegistry(bundle_dir, spec_hash="", profile_hash="")

            res = RolloutResult(
                problem_id="p1", policy_id="pol1", origin_id="o1", future_id=0,
                is_valid=True, is_eligible=True, success=True, completion_time=3.5,
                restricted_time=3.5, expenditure=1.2, distinct_nodes_used=2,
            )
            reg.record_rollout("baselines", "pol1", "o1", 0, "adaptive", res, elapsed_seconds=0.1)

            # Tamper with the saved work unit JSON file
            unit_file = bundle_dir / "work_units/baselines__pol1__o1__fut0__adaptive.json"
            self.assertTrue(unit_file.exists())
            with open(unit_file, "r") as f:
                data = json.load(f)
            data["payload"]["expenditure"] = 9999.0  # Corrupt without updating sha256
            with open(unit_file, "w") as f:
                json.dump(data, f)

            # Clear memory cache to force reloading from disk
            reg._cached_results.clear()
            with self.assertRaises(CorruptedCheckpointError):
                reg.get_rollout("baselines", "pol1", "o1", 0, "adaptive")


class TestCampaignWorkRegistryAndInterruptionResumption(unittest.TestCase):
    """Verifies campaign resumption across preparation, baselines, and search without work duplication."""

    def test_preparation_interruption_and_snapshot_reuse(self):
        """Interrupted origin preparation reuses previously accepted snapshots without re-running realism."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            snap_dir = os.path.join(tmp_dir, "snapshots")
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "n_origins": 4,
                "training_origins": 2, "selection_origins": 1, "evaluation_origins": 1,
                "seed_base": 7711,
            })

            # 1. First run: prepare 3 origins (simulate interruption before 4th origin)
            profile_part = profile.model_copy(update={"n_origins": 3, "training_origins": 1, "selection_origins": 1, "evaluation_origins": 1})
            reg_dir = Path(tmp_dir) / "bundle"
            reg = CampaignWorkRegistry(reg_dir)
            prep_1 = prepare_origins(profile_part, snapshot_dir=snap_dir, registry=reg)
            self.assertEqual(len(prep_1.prepared), 3)
            self.assertEqual(reg.preparation_progress["completed_units"], 3)

            # Check snapshot files on disk
            snaps_created = [f for f in os.listdir(snap_dir) if f.endswith(".npz")]
            self.assertEqual(len(snaps_created), 3)

            # 2. Resumed run: prepare all 4 origins
            prep_2 = prepare_origins(profile, snapshot_dir=snap_dir, registry=reg)
            self.assertEqual(len(prep_2.prepared), 4)

            # The first three origins must match the previously prepared ones exactly
            self.assertEqual(prep_1.prepared[0].origin_id, prep_2.prepared[0].origin_id)
            self.assertEqual(prep_1.prepared[1].origin_id, prep_2.prepared[1].origin_id)
            self.assertEqual(prep_1.prepared[2].origin_id, prep_2.prepared[2].origin_id)

    def test_snapshot_reuse_requires_matching_preparation_configuration(self):
        """A different population cannot silently reuse a same-index snapshot."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "n_origins": 3, "training_origins": 1, "selection_origins": 1,
                "evaluation_origins": 1, "seed_base": 7711,
            })
            calls = []

            def fake_builder(current_profile, origin_seed):
                calls.append((current_profile.N, origin_seed))
                state = _make_test_state(N=current_profile.N, seed=origin_seed)
                return SimpleNamespace(run=lambda horizon: SimpleNamespace(final_state=state))

            with patch("src.config.profiles.build_realism_runner", side_effect=fake_builder):
                first = prepare_origins(profile, snapshot_dir=tmp_dir)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(first.prepared), 3)
                again = prepare_origins(profile, snapshot_dir=tmp_dir)
                self.assertEqual(len(calls), 3)
                self.assertEqual(first.prepared[0].origin_id, again.prepared[0].origin_id)

                changed = profile.model_copy(update={"N": 10})
                other = prepare_origins(changed, snapshot_dir=tmp_dir)
                self.assertEqual(len(calls), 6)
                self.assertTrue(all(origin.state.N == 10 for origin in other.origins))
                self.assertEqual(len(list(Path(tmp_dir).glob("origin_*.npz"))), 6)

    def test_tampered_snapshot_metadata_is_rejected(self):
        """A file at a compatible name still needs matching embedded provenance."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "n_origins": 3, "training_origins": 1, "selection_origins": 1,
                "evaluation_origins": 1, "seed_base": 7711,
            })

            def fake_builder(current_profile, origin_seed):
                state = _make_test_state(N=current_profile.N, seed=origin_seed)
                return SimpleNamespace(run=lambda horizon: SimpleNamespace(final_state=state))

            with patch("src.config.profiles.build_realism_runner", side_effect=fake_builder):
                prepare_origins(profile, snapshot_dir=tmp_dir)
                path = sorted(Path(tmp_dir).glob("origin_*.npz"))[0]
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                payload["preparation_fingerprint"] = np.array("tampered")
                np.savez_compressed(path, **payload)
                with self.assertRaisesRegex(ValueError, "incompatible preparation"):
                    prepare_origins(profile, snapshot_dir=tmp_dir)

    def test_between_rollout_campaign_resumption_no_duplicate_work(self):
        """Resumed campaign reloads completed rollouts without re-executing simulations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "results")
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "n_origins": 3, "training_origins": 1, "selection_origins": 1, "evaluation_origins": 1,
                "candidate_cap": 2, "evaluation_futures_per_origin": 1, "seed_base": 9922,
            })

            # Run 1: Run full campaign, persisting to bundle
            res_1 = run_campaign(profile, campaign_id="resump_test", out_dir=out_dir, master_seed=12345, resume=True)
            self.assertIsNotNone(res_1.work_registry)
            initial_units_count = len(res_1.work_registry.units)
            self.assertGreater(initial_units_count, 0)

            # Run 2: Resume campaign on the same bundle directory
            res_2 = run_campaign(profile, campaign_id="resump_test", out_dir=out_dir, master_seed=12345, resume=True)
            self.assertIsNotNone(res_2.work_registry)

            # Number of registered units must be identical (no duplicates added)
            self.assertEqual(len(res_2.work_registry.units), initial_units_count)

            # Verification that selected policy and evaluation results match exactly
            self.assertEqual(res_1.selected_candidate.candidate_id, res_2.selected_candidate.candidate_id)
            for arm in res_1.baseline_results:
                self.assertEqual(
                    [r.success for r in res_1.baseline_results[arm]],
                    [r.success for r in res_2.baseline_results[arm]],
                )


class TestPhaseTimingsAndWorkloadProjection(unittest.TestCase):
    """Verifies explicit phase timing tracking and workload projection with empirical uncertainty."""

    def test_phase_timings_recorded_and_persisted(self):
        """All campaign phases record explicit wall-clock timings and completed work counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "n_origins": 3, "training_origins": 1, "selection_origins": 1, "evaluation_origins": 1,
                "candidate_cap": 2, "evaluation_futures_per_origin": 1, "seed_base": 5511,
            })
            result = run_campaign(profile, campaign_id="timings_test", out_dir=tmp_dir, master_seed=4321, resume=True)

            self.assertIn("phase_timings", result.manifest["workload_tracking"])
            timings = result.manifest["workload_tracking"]["phase_timings"]

            # Required phases must all be tracked
            for phase in ["preparation", "baselines", "search", "selection", "evaluation", "export"]:
                self.assertIn(phase, timings)
                self.assertGreaterEqual(timings[phase]["wall_clock_seconds"], 0.0)
                self.assertEqual(timings[phase]["status"], "completed")

    def test_preparation_acceptance_progress_tracking(self):
        """Preparation origin acceptance progress tracks attempts, accepts, rejects, and remaining count."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = CampaignWorkRegistry(Path(tmp_dir))
            reg.record_preparation_progress(attempts_made=10, accepted_count=3, rejected_count=7, remaining_units=2)

            self.assertEqual(reg.preparation_progress["attempts_made"], 10)
            self.assertEqual(reg.preparation_progress["accepted_count"], 3)
            self.assertEqual(reg.preparation_progress["rejected_count"], 7)
            self.assertAlmostEqual(reg.preparation_progress["acceptance_rate"], 0.3)
            self.assertEqual(reg.preparation_progress["remaining_units"], 2)

    def test_runtime_projection_with_empirical_uncertainty(self):
        """Workload runtime projection reports mean, standard error, and 95% confidence intervals."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = CampaignWorkRegistry(Path(tmp_dir))
            # Simulate 5 measured rollouts: [2.0, 2.5, 3.0, 2.8, 2.2]
            for dur in [2.0, 2.5, 3.0, 2.8, 2.2]:
                res = RolloutResult("p", "pol", "o", 0, True, True, True, 2.0, 2.0, 1.0, 1)
                reg.record_rollout("baselines", "pol", "o", len(reg.unit_runtimes), "adaptive", res, dur)

            # Project remaining for 20 total planned rollouts (15 remaining)
            projection = reg.compute_runtime_projection(total_planned_rollouts=20)

            self.assertEqual(projection["measured_completed_rollouts"], 5)
            self.assertEqual(projection["remaining_planned_rollouts"], 15)
            self.assertAlmostEqual(projection["mean_rollout_seconds"], 2.5, places=5)
            ci = projection["projected_remaining_ci_95"]
            self.assertIsNotNone(ci)
            self.assertLess(ci[0], projection["projected_remaining_seconds"])
            self.assertGreater(ci[1], projection["projected_remaining_seconds"])
            self.assertIn("no unsupported promises", projection["disclaimer"])


class TestExploratoryDiagnosticResponseGrid(unittest.TestCase):
    """Verifies exploratory response grid, compute cap, failure categorization, and 2D response map rendering."""

    def test_exploratory_grid_predeclared_axes_and_disjoint_origins(self):
        """Exploratory grid declares axes and executes rollouts on dedicated exploratory origins."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
            "holding_duration": 2.0, "release_deadline": 4.0,
            "seed_base": 8800,
        })
        cfg = ExploratoryGridConfig(
            grid_id="test_grid",
            axis_x=GridAxis(name="node_fraction", label="Fraction", values=(0.1, 0.25), parameter_target="policy"),
            axis_y=GridAxis(name="exposure_cap", label="Exposure Cap", values=(5.0, 10.0), parameter_target="problem"),
            n_exploratory_origins=1,
            futures_per_origin=1,
            compute_budget_cap=10,
        )

        res = run_exploratory_response_grid(profile, cfg)
        self.assertEqual(len(res.cells), 4)
        self.assertEqual(res.total_rollouts_executed, 4)
        self.assertFalse(res.compute_budget_exhausted)
        for cell in res.cells:
            self.assertTrue(cell.is_evaluated)
            self.assertEqual(cell.status, "evaluated")
            self.assertIsNotNone(cell.success_probability)
            self.assertIsNotNone(cell.mean_restricted_time)

    def test_compute_budget_cap_enforcement_and_unevaluated_cells(self):
        """Exhausting compute budget cap stops execution and flags remaining cells as unevaluated."""
        profile = LIGHT_CONTROL_PROFILE.model_copy(update={
            "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
            "holding_duration": 2.0, "release_deadline": 4.0,
            "seed_base": 8800,
        })
        # 3x3 grid = 9 cells, but cap = 2 rollouts
        cfg = ExploratoryGridConfig(
            grid_id="capped_grid",
            axis_x=GridAxis(name="node_fraction", label="Fraction", values=(0.1, 0.25, 0.5), parameter_target="policy"),
            axis_y=GridAxis(name="exposure_cap", label="Exposure Cap", values=(2.0, 5.0, 10.0), parameter_target="problem"),
            n_exploratory_origins=1,
            futures_per_origin=1,
            compute_budget_cap=2,
        )

        res = run_exploratory_response_grid(profile, cfg)
        self.assertEqual(res.total_rollouts_executed, 2)
        self.assertTrue(res.compute_budget_exhausted)

        evaluated_cells = [c for c in res.cells if c.is_evaluated]
        unevaluated_cells = [c for c in res.cells if not c.is_evaluated]
        self.assertEqual(len(evaluated_cells), 2)
        self.assertEqual(len(unevaluated_cells), 7)
        for un in unevaluated_cells:
            self.assertEqual(un.status, "unevaluated_budget_cap")
            self.assertIsNone(un.success_probability)
            self.assertIsNone(un.mean_restricted_time)

    def test_trajectory_failure_classification_categories(self):
        """Classifier reports observable patterns without claiming a physical cause."""
        # 1. Success
        succ_res = RolloutResult("p", "pol", "o", 0, True, True, True, 2.0, 2.0, 1.0, 1)
        self.assertEqual(classify_rollout_failure_mechanism(succ_res), "success")

        # 2. Budget clipping
        clamped_rec = RolloutRecorder()
        clamped_rec.action_history.append(ActionRecord(
            decision_time=1.0, report_sample_time=1.0, actuated_nodes=[0, 1], field_entries=[{"node": 0, "component": 0, "amplitude": 1.0}],
            requested_end=5.0, effective_end=2.0, budget_clamped=True, shutoff_or_release="budget_exhaustion",
            reason="budget_exhausted", spending_rate=1.0, cumulative_exposure=10.0,
        ))
        clamped_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=0.0, m_x=-0.7, m_y=-0.8, m_x_all=[-0.7], m_y_all=[-0.8],
            target_direction=1.0, distance_to_target=1.1, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=1.0, distinct_nodes_used=2,
        ))
        clip_res = RolloutResult("p", "pol", "o", 0, True, True, False, None, 10.0, 10.0, 2, recorder=clamped_rec)
        self.assertEqual(classify_rollout_failure_mechanism(clip_res), "budget_limited")

        # 3. Weak joint population shift (delta < 0.10)
        local_rec = RolloutRecorder()
        local_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=0.0, m_x=-0.7, m_y=-0.8, m_x_all=[-0.7], m_y_all=[-0.8],
            target_direction=1.0, distance_to_target=1.1, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        local_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=5.0, m_x=-0.69, m_y=-0.78, m_x_all=[-0.69], m_y_all=[-0.78],
            target_direction=1.0, distance_to_target=1.08, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        local_res = RolloutResult("p", "pol", "o", 0, True, True, False, None, 10.0, 1.0, 2, recorder=local_rec)
        self.assertEqual(classify_rollout_failure_mechanism(local_res), "weak_joint_shift")
        no_trajectory = RolloutResult("p", "pol", "o", 0, True, True, False, None, 10.0, 1.0, 2)
        self.assertEqual(classify_rollout_failure_mechanism(no_trajectory), "unclassified")

        # Public means alone can move markedly while private means stay fixed.
        public_only = RolloutRecorder()
        public_only.scalar_points.extend([
            local_rec.scalar_points[0],
            ScalarTrajectoryPoint(
                t=5.0, m_x=-0.7, m_y=-0.1, m_x_all=[-0.7], m_y_all=[-0.1],
                target_direction=1.0, distance_to_target=1.2, target_predicate=False,
                hold_progress=0.0, actual_release=False, exposure=1.0,
                spending_rate=0.0, distinct_nodes_used=2,
            ),
        ])
        public_only_res = RolloutResult(
            "p", "pol", "o", 0, True, True, False, None, 10.0, 1.0, 2,
            recorder=public_only,
        )
        self.assertEqual(classify_rollout_failure_mechanism(public_only_res), "weak_joint_shift")

        # 4. Partial joint population shift (delta >= 0.10, target never reached)
        prop_rec = RolloutRecorder()
        prop_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=0.0, m_x=-0.7, m_y=-0.8, m_x_all=[-0.7], m_y_all=[-0.8],
            target_direction=1.0, distance_to_target=1.1, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        prop_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=5.0, m_x=-0.2, m_y=-0.4, m_x_all=[-0.2], m_y_all=[-0.4],
            target_direction=1.0, distance_to_target=0.7, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        prop_res = RolloutResult("p", "pol", "o", 0, True, True, False, None, 10.0, 1.0, 2, recorder=prop_rec)
        self.assertEqual(classify_rollout_failure_mechanism(prop_res), "partial_joint_shift")

        # 5. Target reached or holding started, but not sustained
        pers_rec = RolloutRecorder()
        pers_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=0.0, m_x=-0.7, m_y=-0.8, m_x_all=[-0.7], m_y_all=[-0.8],
            target_direction=1.0, distance_to_target=1.1, target_predicate=False,
            hold_progress=0.0, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        pers_rec.scalar_points.append(ScalarTrajectoryPoint(
            t=3.0, m_x=0.6, m_y=0.7, m_x_all=[0.6], m_y_all=[0.7],
            target_direction=1.0, distance_to_target=0.0, target_predicate=True,
            hold_progress=1.5, actual_release=False, exposure=1.0, spending_rate=0.0, distinct_nodes_used=2,
        ))
        pers_res = RolloutResult("p", "pol", "o", 0, True, True, False, None, 10.0, 1.0, 2, recorder=pers_rec)
        self.assertEqual(classify_rollout_failure_mechanism(pers_res), "target_not_sustained")

    def test_2d_response_maps_rendering_and_unevaluated_hatching(self):
        """2D response maps render all panels cleanly from saved disk JSON without live simulation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = LIGHT_CONTROL_PROFILE.model_copy(update={
                "N": 8, "D": 1, "establishment_time": 2.0, "horizon": 8.0,
                "holding_duration": 2.0, "release_deadline": 4.0,
                "seed_base": 8800,
            })
            cfg = ExploratoryGridConfig(
                grid_id="viz_grid",
                axis_x=GridAxis(name="node_fraction", label="Candidate Fraction", values=(0.1, 0.25), parameter_target="policy"),
                axis_y=GridAxis(name="exposure_cap", label="Exposure Cap", values=(5.0, 10.0), parameter_target="problem"),
                n_exploratory_origins=1,
                futures_per_origin=1,
                compute_budget_cap=2,  # 2 evaluated, 2 unevaluated
            )
            res = run_exploratory_response_grid(profile, cfg)
            json_path = Path(tmp_dir) / "diagnostic_grid_data.json"
            res.save_json(json_path)

            img_path = Path(tmp_dir) / "diagnostic_response_maps.png"
            plot_diagnostic_response_maps(json_path, save_path=img_path)

            self.assertTrue(img_path.is_file())
            self.assertGreater(img_path.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
