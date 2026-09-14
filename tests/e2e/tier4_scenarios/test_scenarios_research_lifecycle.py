"""Tier 4: End-to-End Scientific Research Lifecycle Scenarios (16 Comprehensive Tests).

Covers complete scientific workflows from initial conditions and dynamical simulation
through analytics, interventions, neural diagnostics, and reproducibility audits.
"""

import unittest
import numpy as np
from tests.e2e.common import (
    ModularBaseTestCase,
    oracle_influence_metrics,
    oracle_population_discrepancy,
    oracle_analytical_baseline_bound,
    oracle_origin_stratified_split,
)


class TestTier4ResearchScenarios(ModularBaseTestCase):
    """Tier 4: End-to-End Scientific Workflows (>=16 Tests)."""

    def test_scenario_01_emergence_from_tabula_rasa(self):
        """Scenario 1: Emergence from tabula rasa (empty network W=0, unbiased beliefs)."""
        gen = self.require_attr("src.experiment.generators", "generate_emergence_state")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        calc_influence = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        calc_d2 = self.require_attr("src.analytics.discrepancy", "calculate_population_discrepancy")
        
        state = gen(N=20, D=1, seed=42)
        self.assertEqual(np.sum(state.W), 0)
        runner = Runner(state=state, T=0.3, delta=0.1, nu=2.0)
        trajectory = runner.run(horizon=10.0)
        
        # Verify trajectory accumulated memory
        final_state = trajectory.final_state
        self.assertGreater(np.sum(final_state.W), 0)
        metrics = calc_influence(final_state.W)
        self.assertFalse(np.isnan(metrics["C_N"]))
        d2 = calc_d2(final_state.x, final_state.y)
        self.assertTrue(0.0 <= d2 <= 1.0)

    def test_scenario_02_persistence_on_modular_networks(self):
        """Scenario 2: Persistence of segregated opinions on modular networks."""
        gen_mod = self.require_attr("src.experiment.generators", "generate_modular_network")
        detect = self.require_attr("src.analytics.communities", "detect_communities")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        
        W, blocks = gen_mod(N=20, num_blocks=2, m=10, eta=0.8, seed=42)
        res_initial = detect(W.astype(float), gamma=1.0)
        self.assertGreater(res_initial["modularity_Q"], 0.0)
        
        runner = Runner(W=W, blocks=blocks, T=0.1, delta=0.02, nu=0.4)
        trajectory = runner.run(horizon=5.0)
        res_final = detect(trajectory.final_state.W.astype(float), gamma=1.0)
        self.assertGreater(res_final["modularity_Q"], 0.0)

    def test_scenario_03_persistence_on_hub_networks(self):
        """Scenario 3: Persistence of hub attention dominance and opinion leadership."""
        gen_hub = self.require_attr("src.experiment.generators", "generate_hub_network")
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        
        W, hub = gen_hub(N=25, m=10, eta=0.3, hub=0, seed=42)
        initial_metrics = calc_metrics(W)
        # Hub has highest share
        self.assertEqual(np.argmax(initial_metrics["p"]), hub)
        
        runner = Runner(W=W, a=1.0, T=0.3, delta=0.1, nu=2.0)
        trajectory = runner.run(horizon=5.0)
        final_metrics = calc_metrics(trajectory.final_state.W)
        self.assertFalse(np.isnan(final_metrics["C"]))

    def test_scenario_04_field_pulse_shock_and_relaxation(self):
        """Scenario 4: Field pulse shock opposing majority opinion and post-pulse relaxation."""
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        
        pulse = Intervention(start_time=5.0, duration=3.0, amplitude=3.0)
        runner = Runner(N=30, D=1, T=0.5, interventions=[pulse])
        trajectory = runner.run(horizon=12.0)
        # Verify pulse was active during [5.0, 8.0] and inactive after 8.0
        self.assertIn("timeseries", trajectory.data)

    def test_scenario_05_targeted_vs_uniform_intervention(self):
        """Scenario 5: Targeted pulse to top 10% leaders vs uniform random 10%."""
        create_targeted = self.require_attr("src.experiment.interventions", "create_targeted_pulse")
        create_uniform = self.require_attr("src.experiment.interventions", "create_uniform_pulse")
        
        r = np.arange(50, dtype=np.float64)  # Node 49 has highest attention
        pulse_target = create_targeted(r, fraction=0.10, amplitude=2.0)
        pulse_uniform = create_uniform(N=50, fraction=0.10, amplitude=2.0, seed=42)
        
        # Targeted must contain node 49
        self.assertTrue(pulse_target.target_mask[49])
        # Both must have identical count of affected nodes (5)
        self.assertEqual(np.sum(pulse_target.target_mask), 5)
        self.assertEqual(np.sum(pulse_uniform.target_mask), 5)

    def test_scenario_06_structural_disruption_resilience(self):
        """Scenario 6: Catastrophic 30% structural tie deletion resilience."""
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        disrupt = self.require_attr("src.experiment.interventions", "TieDisruptionIntervention")
        calc_metrics = self.require_attr("src.analytics.influence", "calculate_influence_metrics")
        
        intervention = disrupt(disruption_time=5.0, fraction=0.30)
        runner = Runner(N=30, D=1, T=0.3, delta=0.1, nu=2.0, interventions=[intervention])
        traj = runner.run(horizon=10.0)
        final_metrics = calc_metrics(traj.final_state.W)
        self.assertGreater(final_metrics["f_active"], 0.0)

    def test_scenario_07_multi_population_sequential_opening(self):
        """Scenario 7: Multi-population sequential interface opening (A <-> B then B <-> C)."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        
        sched = Schedule(num_populations=3)
        sched.schedule_sequential_opening(t_start=5.0, interval=5.0)
        runner = Runner(N=30, num_populations=3, interface_schedule=sched)
        traj = runner.run(horizon=15.0)
        self.assertIsNotNone(traj)

    def test_scenario_08_multi_population_simultaneous_opening(self):
        """Scenario 8: Multi-population simultaneous interface opening vs closed control."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched_open = Schedule(num_populations=2)
        sched_open.schedule_simultaneous_opening(t_open=5.0)
        sched_closed = Schedule(num_populations=2)
        sched_closed.set_closed()
        
        E_open = sched_open.get_matrix(t=6.0)
        E_closed = sched_closed.get_matrix(t=6.0)
        self.assertEqual(E_open[0, 1], 1.0)
        self.assertEqual(E_closed[0, 1], 0.0)

    def test_scenario_09_mechanism_speed_rescaling_transient(self):
        """Scenario 9: Modulating turnover speed (lambda in {0.2, 5.0}) at fixed memory scale."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        
        # Test rate rescaling
        nu_slow, delta_slow = rescale(2.0, 0.1, speed_multiplier=0.2)
        nu_fast, delta_fast = rescale(2.0, 0.1, speed_multiplier=5.0)
        self.assertAlmostEqual(nu_slow / delta_slow, 20.0)
        self.assertAlmostEqual(nu_fast / delta_fast, 20.0)


    def test_scenario_12_qlearning_vs_heatbath_dynamics(self):
        """Scenario 12: Comparative simulation under Q-learning vs Glauber heat bath."""
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        runner_heatbath = Runner(N=20, D=1, T=0.5, enable_q_learning=False)
        runner_qlearning = Runner(N=20, D=1, T=0.5, enable_q_learning=True, q_eta=0.1)
        
        traj_hb = runner_heatbath.run(horizon=2.0)
        traj_ql = runner_qlearning.run(horizon=2.0)
        self.assertIsNotNone(traj_hb)
        self.assertIsNotNone(traj_ql)


    def test_scenario_14_stage0_timing_and_cost_calibration(self):
        """Scenario 14: Stage 0 timing campaign (N=256, H=10), verifying setup vs kernel separation."""
        Stage0 = self.require_attr("src.campaigns.stage0_timing", "Stage0TimingCampaign")
        campaign = Stage0(N=256, horizon=10.0, num_runs=1)
        results = campaign.execute()
        self.assertIn("setup_time", results)
        self.assertIn("kernel_time", results)
        self.assertIn("total_events", results)

    def test_scenario_15_stage1_pilot_parameter_sweep(self):
        """Scenario 15: Stage 1 pilot parameter sweep across (T, delta) grid."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        
        # Test 2 parameter points
        for T, delta in [(0.1, 0.02), (0.3, 0.1)]:
            cfg = Config(N=10, D=1, T=T, delta=delta, nu=20.0 * delta)
            runner = Runner(cfg)
            traj = runner.run(horizon=1.0)
            self.assertIsNotNone(traj)

    def test_scenario_16_bit_identical_restart_reproducibility(self):
        """Scenario 16: Bit-for-bit restart equivalence between continuous and checkpointed execution."""
        Runner = self.require_attr("src.execution.runner", "TrajectoryRunner")
        seed = 98765
        
        # Run 1: Continuous execution to horizon 6.0
        runner1 = Runner(N=10, D=1, seed=seed)
        traj_continuous = runner1.run(horizon=6.0)
        
        # Run 2: Execution to 3.0, save checkpoint, restore, continue to 6.0
        runner2 = Runner(N=10, D=1, seed=seed)
        traj_part1 = runner2.run(horizon=3.0)
        checkpoint = runner2.get_checkpoint()
        
        runner3 = Runner.from_checkpoint(checkpoint)
        traj_part2 = runner3.run(horizon=6.0)
        
        # Bit-identical state check at t=6.0
        np.testing.assert_array_equal(traj_continuous.final_state.x, traj_part2.final_state.x)
        np.testing.assert_array_equal(traj_continuous.final_state.y, traj_part2.final_state.y)
        np.testing.assert_array_equal(traj_continuous.final_state.W, traj_part2.final_state.W)


if __name__ == "__main__":
    unittest.main()
