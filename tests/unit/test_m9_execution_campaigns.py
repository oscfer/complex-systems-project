"""Comprehensive unit test suite for Execution Campaigns & Notebooks.

Tests:
1. TrajectoryRunner: Single-trajectory simulation, config loading, and bit-for-bit restart reproducibility.
2. Checkpointing: save_checkpoint, load_checkpoint, branch_checkpoint, PRNG state restoration, corrupt data rejection.
3. Resource Profiles: 'lightweight'/'small', 'standard', 'extended', and parameter validity.
4. CampaignRunner: Multiprocessing parallel execution, checkpoint persistence on disk, resumability (zero re-execution),
   and incremental growth (executing only missing work).
5. Visualization Suite: Plot data verification for all 7 R4 plotting requirements.
6. Stage0TimingCampaign: Setup vs kernel time benchmarking and event throughput.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.config.schema import SimulationConfig
from src.campaigns.stage0_timing import Stage0TimingCampaign
from src.core.state import State, StateMode
from src.experiment.interventions import TieDisruptionIntervention
from src.analytics.influence import calculate_influence_metrics, spearman_rank_persistence
from src.execution.campaign import (
    CampaignResult,
    CampaignRunner,
    ResourceProfile,
    get_profile,
)
from src.execution.runner import (
    Trajectory,
    TrajectoryRunner,
    branch_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.visualization.plotting import (
    animate_network_temporal,
    plot_community_fragmentation,
    plot_coupled_beliefs_response,
    plot_influence_metrics,
    plot_null_model_comparison,
    plot_styled_network,
    plot_wilson_confidence_bands,
)


class TestTrajectoryRunnerAndCheckpoints(unittest.TestCase):
    """Unit tests for single-trajectory execution and checkpointing."""

    def test_trajectory_runner_from_config(self):
        cfg = SimulationConfig(N=10, D=1, T=0.4, delta=0.1, nu=2.0, seed=123)
        runner = TrajectoryRunner(cfg)
        self.assertEqual(runner.N, 10)
        self.assertEqual(runner.D, 1)
        self.assertEqual(runner.T, 0.4)
        traj = runner.run(horizon=2.0)
        self.assertIsInstance(traj, Trajectory)
        self.assertGreater(traj.final_state.t, 0.0)
        self.assertIn("timeseries", traj.data)

    def test_bit_identical_restart_reproducibility(self):
        """Bit-for-bit restart equivalence between continuous and checkpointed execution."""
        seed = 45678
        # Run 1: Continuous run to 4.0
        r1 = TrajectoryRunner(N=12, D=1, seed=seed, T=0.3, delta=0.1, nu=1.5)
        traj1 = r1.run(horizon=4.0)

        # Run 2: Run to 2.0, save checkpoint, restore, run to 4.0
        r2 = TrajectoryRunner(N=12, D=1, seed=seed, T=0.3, delta=0.1, nu=1.5)
        _ = r2.run(horizon=2.0)
        chk = r2.get_checkpoint()

        r3 = TrajectoryRunner.from_checkpoint(chk)
        traj2 = r3.run(horizon=4.0)

        # Assert exact bit-for-bit equality
        self.assertEqual(traj1.final_state.t, traj2.final_state.t)
        np.testing.assert_array_equal(traj1.final_state.x, traj2.final_state.x)
        np.testing.assert_array_equal(traj1.final_state.y, traj2.final_state.y)
        np.testing.assert_array_equal(traj1.final_state.W, traj2.final_state.W)

    def test_save_and_load_checkpoint_roundtrip_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "checkpoint.json"
            rng = np.random.default_rng(999)
            _ = rng.random()
            state = State.create_unbiased_emergence(N=8, D=1, state_mode=StateMode.BINARY, rng=rng)
            state.W[0, 1] = 5

            events = [{"type": "field_pulse", "start_time": 3.0}]
            save_checkpoint(
                filepath=chk_path,
                state=state,
                rng=rng,
                t=2.5,
                scheduled_events=events,
                seed=999,
            )
            self.assertTrue(chk_path.is_file())

            loaded = load_checkpoint(chk_path)
            self.assertEqual(loaded["t"], 2.5)
            self.assertEqual(loaded["seed"], 999)
            self.assertEqual(len(loaded["scheduled_events"]), 1)
            self.assertEqual(loaded["scheduled_events"][0]["type"], "field_pulse")

            # Verify PRNG continuation matches original rng next draw
            self.assertIn("rng", loaded)
            self.assertEqual(rng.random(), loaded["rng"].random())

    def test_load_checkpoint_corrupt_data_raises_error(self):
        with self.assertRaises(ValueError):
            load_checkpoint("corrupted_invalid_data_not_json")

    def test_branch_checkpoint_seed_isolation(self):
        chk = {"seed": 500, "t": 10.0, "state": {}}
        b1 = branch_checkpoint(chk, arm="control", seed_offset=1)
        b2 = branch_checkpoint(chk, arm="treatment", seed_offset=2)

        self.assertEqual(b1["arm"], "control")
        self.assertEqual(b2["arm"], "treatment")
        self.assertNotEqual(b1["seed"], b2["seed"])
        b1_repeat = branch_checkpoint(chk, arm="control", seed_offset=1)
        self.assertEqual(b1["seed"], b1_repeat["seed"])
        self.assertNotEqual(b1["seed"], b2["seed"])
        # Verify distinct generators
        self.assertNotEqual(b1["rng"].random(), b2["rng"].random())

    def test_exact_horizon_and_restart_between_events(self):
        continuous = TrajectoryRunner(N=2, seed=1, nu=0.0, delta=0.0, tau_x=1000.0, tau_y=1000.0)
        final_continuous = continuous.run(200.0).final_state

        interrupted = TrajectoryRunner(N=2, seed=1, nu=0.0, delta=0.0, tau_x=1000.0, tau_y=1000.0)
        at_boundary = interrupted.run(0.1)
        self.assertEqual(at_boundary.final_state.t, 0.1)
        resumed = TrajectoryRunner.from_checkpoint(interrupted.get_checkpoint()).run(200.0).final_state

        self.assertEqual(resumed.t, 200.0)
        np.testing.assert_array_equal(final_continuous.x, resumed.x)
        np.testing.assert_array_equal(final_continuous.y, resumed.y)
        np.testing.assert_array_equal(final_continuous.W, resumed.W)

    def test_checkpoint_restores_nondefault_parameters_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            runner = TrajectoryRunner(
                N=5, seed=9, K=3.0, L=4.0, b=2.0,
                enable_q_learning=True, q_eta=0.9, q_temperature=0.2,
            )
            runner.run(0.2)
            runner.get_checkpoint(path)
            restored = TrajectoryRunner.from_checkpoint(path)
            self.assertEqual(restored.K, 3.0)
            self.assertEqual(restored.L, 4.0)
            self.assertEqual(restored.b, 2.0)
            self.assertEqual(restored.q_params.eta, 0.9)
            self.assertEqual(restored.q_params.T_Q, 0.2)

            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["t"] = 999.0
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_participation_channel_and_tie_disruption_stay_synchronized(self):
        W = np.ones((4, 4), dtype=np.uint32)
        np.fill_diagonal(W, 0)
        disruption = TieDisruptionIntervention(disruption_time=0.0, fraction=1.0)
        runner = TrajectoryRunner(W=W, D=1, tau_n=0.01, interventions=[disruption], seed=7)
        trajectory = runner.run(0.1)
        self.assertGreater(runner.scheduler.stats.participation_updates, 0)
        np.testing.assert_array_equal(trajectory.final_state.n, runner.transition_tracker.n)
        np.testing.assert_array_equal(trajectory.final_state.W, runner.memory_mgr.W)
        self.assertEqual(int(runner.scheduler.R_forget), 0)


class TestResourceProfiles(unittest.TestCase):
    """Unit tests for declarative resource profiles."""

    def test_get_predefined_profiles(self):
        p_light = get_profile("lightweight")
        p_small = get_profile("small")
        p_std = get_profile("standard")
        p_ext = get_profile("extended")

        self.assertEqual(p_light.name, "lightweight")
        self.assertEqual(p_small.N, p_light.N)
        self.assertEqual(p_std.N, 30)
        self.assertEqual(p_ext.D, 3)
        self.assertGreater(p_ext.num_realizations, p_std.num_realizations)

    def test_get_unknown_profile_raises_error(self):
        with self.assertRaises(ValueError):
            get_profile("nonexistent_profile_scale")


class TestCampaignRunnerResumability(unittest.TestCase):
    """Unit tests verifying campaign execution, disk checkpoints, and resumability."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out_dir = Path(self.tmpdir) / "campaign_run"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_campaign_execution_checkpoints_and_resumability(self):
        """Proof of resumability: pass 1 executes work, pass 2 skips all existing checkpoints."""
        runner1 = CampaignRunner(
            profile="lightweight",
            num_realizations=2,
            master_seed=101,
            output_dir=self.out_dir,
            campaign_id="test_camp",
            max_workers=2,
        )
        res1 = runner1.run()
        self.assertEqual(res1.total_realizations, 2)
        self.assertEqual(res1.executed_count, 2)
        self.assertEqual(res1.skipped_count, 0)
        self.assertEqual(len(res1.realizations), 2)

        # Verify checkpoints written to disk
        chk_dir = self.out_dir / "checkpoints"
        self.assertTrue(chk_dir.is_dir())
        chk_files = list(chk_dir.glob("*.json"))
        self.assertEqual(len(chk_files), 2)

        # Verify manifest
        manifest_file = self.out_dir / "manifest.json"
        self.assertTrue(manifest_file.is_file())

        # Second Pass: Re-running exact campaign must execute 0 runs and skip 2
        runner2 = CampaignRunner(
            profile="lightweight",
            num_realizations=2,
            master_seed=101,
            output_dir=self.out_dir,
            campaign_id="test_camp",
            max_workers=2,
        )
        res2 = runner2.run()
        self.assertEqual(res2.executed_count, 0)
        self.assertEqual(res2.skipped_count, 2)
        self.assertEqual(res2.total_realizations, 2)
        self.assertTrue(all(r.is_cached for r in res2.realizations))

    def test_campaign_incremental_growth(self):
        """Increasing realization count executes only the missing incremental work."""
        # Initial run: 2 realizations
        c1 = CampaignRunner(
            profile="lightweight",
            num_realizations=2,
            master_seed=202,
            output_dir=self.out_dir,
            campaign_id="test_inc",
            max_workers=2,
        )
        r1 = c1.run()
        self.assertEqual(r1.executed_count, 2)

        # Expanded run: 4 realizations
        c2 = CampaignRunner(
            profile="lightweight",
            num_realizations=4,
            master_seed=202,
            output_dir=self.out_dir,
            campaign_id="test_inc",
            max_workers=2,
        )
        r2 = c2.run()
        self.assertEqual(r2.total_realizations, 4)
        self.assertEqual(r2.skipped_count, 2)
        self.assertEqual(r2.executed_count, 2)

    def test_seed_assignment_independence(self):
        """Verify realization seed assignments are deterministic and independent of scheduling."""
        c = CampaignRunner(
            profile="lightweight",
            num_realizations=3,
            master_seed=777,
            output_dir=self.out_dir,
            campaign_id="test_seeds",
        )
        res = c.run()
        seeds = [r.seed for r in res.realizations]
        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(set(seeds)), 3) # strictly unique

    def test_changed_definition_does_not_reuse_stale_results(self):
        first = CampaignRunner(
            profile="lightweight", num_realizations=1, master_seed=1,
            output_dir=self.out_dir, campaign_id="same", max_workers=1,
            N=3, horizon=0.05, T=0.1,
        ).run()
        second = CampaignRunner(
            profile="lightweight", num_realizations=1, master_seed=999,
            output_dir=self.out_dir, campaign_id="same", max_workers=1,
            N=8, horizon=0.2, T=0.9,
        ).run()
        self.assertEqual(second.executed_count, 1)
        self.assertEqual(second.skipped_count, 0)
        self.assertNotEqual(first.realizations[0].seed, second.realizations[0].seed)
        self.assertEqual(second.realizations[0].horizon, 0.2)
        self.assertEqual(second.realizations[0].final_time, 0.2)

    def test_incomplete_parseable_result_is_reexecuted(self):
        runner = CampaignRunner(
            profile="lightweight", num_realizations=1, master_seed=22,
            output_dir=self.out_dir, campaign_id="corrupt", max_workers=1,
        )
        initial = runner.run()
        Path(initial.realizations[0].checkpoint_path).write_text("{}", encoding="utf-8")
        repaired = runner.run()
        self.assertEqual(repaired.executed_count, 1)
        self.assertEqual(repaired.skipped_count, 0)


class TestVisualizationSuite(unittest.TestCase):
    """Unit tests validating data structures and logic for all 7 R4 plotting requirements."""

    def test_plot_styled_network(self):
        W = np.array([[0, 3, 0], [2, 0, 1], [0, 0, 0]], dtype=np.uint32)
        beliefs = np.array([[-0.5], [0.8], [0.0]])
        data = plot_styled_network(W, beliefs)
        self.assertEqual(data["N"], 3)
        self.assertEqual(data["edge_count"], 3)
        self.assertEqual(len(data["node_sizes"]), 3)
        self.assertGreater(data["node_sizes"][1], data["node_sizes"][2]) # node 1 received more attention
        self.assertFalse(np.isnan(data["C_N"]))

    def test_animate_network_temporal(self):
        timeseries = [
            {"t": 0.0, "m_y": [0.0], "total_memory": 0, "W": [[1]], "y": [1]},
            {"t": 1.0, "m_y": [0.3], "total_memory": 10, "W": [[1]], "y": [1]},
            {"t": 2.0, "m_y": [0.7], "total_memory": 25, "W": [[1]], "y": [1]},
        ]
        data = animate_network_temporal(timeseries)
        self.assertEqual(data["num_frames"], 3)

    def test_plot_influence_metrics(self):
        timeseries = [
            {"t": 0.0, "C_N": 0.05, "N_eff": 5.0},
            {"t": 2.0, "C_N": 0.35, "N_eff": 2.1},
        ]
        data = plot_influence_metrics(timeseries)
        self.assertEqual(len(data["times"]), 2)
        self.assertEqual(data["C_N"][1], 0.35)
        self.assertEqual(data["N_eff"][1], 2.1)

        W = np.array([[0, 2], [1, 0]], dtype=np.uint32)
        telemetry = plot_influence_metrics([{"t": 1.0, "W": W}])
        expected = calculate_influence_metrics(W)
        self.assertAlmostEqual(telemetry["C_N"][0], expected["C_N"])
        self.assertAlmostEqual(telemetry["N_eff"][0], expected["effective_leaders"])

    def test_plot_community_fragmentation(self):
        W = np.zeros((10, 10), dtype=np.uint32)
        W[0:5, 0:5] = 2
        np.fill_diagonal(W, 0)
        W[5:10, 5:10] = 3
        np.fill_diagonal(W, 0)

        data = plot_community_fragmentation(W, gammas=[0.5, 1.0, 2.0])
        self.assertIn("gamma_0.5", data["leiden_resolutions"])
        self.assertIn("gamma_1.0", data["leiden_resolutions"])
        self.assertIn("gamma_2.0", data["leiden_resolutions"])
        self.assertGreaterEqual(data["num_wcc"], 1)

    def test_plot_null_model_comparison(self):
        W = np.zeros((8, 8), dtype=np.uint32)
        W[0:4, 0:4] = 2
        np.fill_diagonal(W, 0)
        W[4:8, 4:8] = 2
        np.fill_diagonal(W, 0)

        data = plot_null_model_comparison(W, num_swaps=20, seed=42)
        self.assertIn("empirical_Q", data)
        self.assertIn("null_Q", data)
        self.assertIn("modularity_excess", data)
        self.assertEqual(data["projected_degree_sequence"], data["null_degree_sequence"])
        empirical_weights = sorted(data["empirical_graph"][np.triu_indices(8, 1)].tolist())
        null_weights = sorted(data["null_graph"][np.triu_indices(8, 1)].tolist())
        self.assertEqual(empirical_weights, null_weights)

    def test_plot_coupled_beliefs_response(self):
        timeseries = [
            {"t": 0.0, "m_y": [0.1, -0.1, 0.0]},
            {"t": 5.0, "m_y": [-0.8, -0.2, 0.1]},
        ]
        data = plot_coupled_beliefs_response(timeseries, perturbed_component=0)
        self.assertEqual(data["perturbed_component"], 0)
        self.assertEqual(len(data["components"][0]), 2)
        self.assertEqual(data["components"][0][1], -0.8)

    def test_plot_wilson_confidence_bands(self):
        trials = [50, 50]
        successes = [45, 20]
        data = plot_wilson_confidence_bands(trials, successes, confidence=0.95)
        self.assertEqual(len(data["point_estimates"]), 2)
        self.assertAlmostEqual(data["point_estimates"][0], 0.90)
        self.assertAlmostEqual(data["point_estimates"][1], 0.40)
        # Wilson interval strictly strictly encapsulates point estimates
        self.assertLess(data["lower_bounds"][0], 0.90)
        self.assertGreater(data["upper_bounds"][0], 0.90)
        self.assertLess(data["lower_bounds"][1], 0.40)
        self.assertGreater(data["upper_bounds"][1], 0.40)

    def test_spearman_nonperfect_rank_correlation(self):
        value = spearman_rank_persistence(np.array([1, 2, 3]), np.array([1, 3, 2]))
        self.assertAlmostEqual(value, 0.5)


class TestStage0TimingCampaign(unittest.TestCase):
    """Unit tests for Stage 0 timing campaign."""

    def test_stage0_timing_campaign_execution(self):
        campaign = Stage0TimingCampaign(N=16, horizon=2.0, num_runs=1, seed=42)
        res = campaign.execute()
        self.assertIn("setup_time", res)
        self.assertIn("kernel_time", res)
        self.assertIn("total_events", res)
        self.assertIn("events_per_sec", res)
        self.assertGreater(res["total_events"], 0)
        self.assertGreaterEqual(res["setup_time"], 0.0)
        self.assertGreaterEqual(res["kernel_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
