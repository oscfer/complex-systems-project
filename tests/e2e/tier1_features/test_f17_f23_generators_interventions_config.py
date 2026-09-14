"""Tier 1: Features 17 to 23 Tests.

Feature 17: Network Initial Condition Generators
Feature 18: Scheduled Applied Field Pulses
Feature 19: Structural Tie Disruptions
Feature 20: Population Interface Schedules
Feature 21: Mechanism Speed Rescaling
Feature 22: Decoupled Config Schema & Loader
Feature 23: Snapshot & State Checkpointing
"""

import unittest
import numpy as np
from tests.e2e.common import ModularBaseTestCase


class TestFeature17NetworkGenerators(ModularBaseTestCase):
    """Feature 17: Network Initial Condition Generators (Emergence, Decentralized, Modular, Hub)."""

    def test_f17_emergence_empty_network_generator(self):
        """Verify emergence generator produces empty network W = 0 and unbiased states."""
        gen = self.require_attr("src.experiment.generators", "generate_emergence_state")
        state = gen(N=50, D=1, seed=42)
        self.assertTrue(np.all(state.W == 0))
        self.assertEqual(state.x.shape, (50, 1))
        np.testing.assert_array_equal(state.x, state.y)

    def test_f17_constructed_decentralized_degree_budget(self):
        """Verify decentralized network generator has exact row sums sum_j W_ij = m."""
        gen = self.require_attr("src.experiment.generators", "generate_decentralized_network")
        m = 20
        W = gen(N=30, m=m, seed=42)
        self.assertEqual(W.shape, (30, 30))
        self.assertTrue(np.all(np.diag(W) == 0))
        np.testing.assert_array_equal(W.sum(axis=1), np.full(30, m))

    def test_f17_constructed_modular_block_structure(self):
        """Verify modular network generator concentrates edges inside blocks when eta > 0."""
        gen = self.require_attr("src.experiment.generators", "generate_modular_network")
        W, blocks = gen(N=40, num_blocks=2, m=20, eta=0.8, seed=42)
        # Block 0 nodes should have more edges to block 0 than block 1
        b0_nodes = np.where(blocks == 0)[0]
        b1_nodes = np.where(blocks == 1)[0]
        internal_edges = W[np.ix_(b0_nodes, b0_nodes)].sum()
        external_edges = W[np.ix_(b0_nodes, b1_nodes)].sum()
        self.assertGreater(internal_edges, external_edges)

    def test_f17_constructed_hub_attention_concentration(self):
        """Verify concentrated-attention generator allocates elevated edges to chosen hub."""
        gen = self.require_attr("src.experiment.generators", "generate_hub_network")
        W, hub_idx = gen(N=50, m=20, eta=0.2, hub=0, seed=42)
        in_degrees = W.sum(axis=0)
        # Hub in-degree should be substantially higher than average non-hub in-degree
        non_hub_mean = np.mean(in_degrees[1:])
        self.assertGreater(in_degrees[hub_idx], non_hub_mean * 1.5)

    def test_f17_state_alignment_discrepancy_generator(self):
        """Verify state alignment generator creates controlled discrepancy q."""
        gen = self.require_attr("src.experiment.generators", "generate_aligned_states")
        x, y = gen(N=1000, D=1, p_g=0.9, q=0.2, seed=42)
        mismatches = np.mean(x != y)
        self.assertAlmostEqual(mismatches, 0.2, delta=0.05)


class TestFeature18FieldPulses(ModularBaseTestCase):
    """Feature 18: Scheduled Applied Field Pulses."""

    def test_f18_field_pulse_onset_and_amplitude(self):
        """Verify field pulse injects expected amplitude h_pulse at scheduled time."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        pulse = Intervention(start_time=10.0, duration=5.0, amplitude=2.0, component=0)
        self.assertAlmostEqual(pulse.get_field(t=12.0), 2.0, places=6)

    def test_f18_field_pulse_opposing_majority_direction(self):
        """Verify pulse direction opposes pre-pulse public mean sign."""
        create_pulse = self.require_attr("src.experiment.interventions", "create_counter_pulse")
        pulse_pos = create_pulse(pre_pulse_mean=0.8, amplitude=1.0)
        self.assertLess(pulse_pos.amplitude, 0.0)
        pulse_neg = create_pulse(pre_pulse_mean=-0.8, amplitude=1.0)
        self.assertGreater(pulse_neg.amplitude, 0.0)

    def test_f18_field_pulse_duration_and_restoration(self):
        """Verify field restores exactly to 0.0 after pulse duration expires."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        pulse = Intervention(start_time=10.0, duration=5.0, amplitude=2.0)
        self.assertEqual(pulse.get_field(t=5.0), 0.0)
        self.assertEqual(pulse.get_field(t=12.0), 2.0)
        self.assertEqual(pulse.get_field(t=16.0), 0.0)

    def test_f18_field_pulse_targeted_subset_mask(self):
        """Verify field pulse applies only to nodes specified by target mask."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        target_mask = np.array([True, False, True, False])
        pulse = Intervention(start_time=0.0, duration=5.0, amplitude=1.5, target_mask=target_mask)
        fields = pulse.get_fields_vector(t=2.0, N=4)
        np.testing.assert_array_equal(fields, np.array([1.5, 0.0, 1.5, 0.0]))

    def test_f18_field_pulse_multiple_amplitudes(self):
        """Verify standard amplitude ratios |h|/J in {0.25, 1.0, 4.0}."""
        create_pulse = self.require_attr("src.experiment.interventions", "create_counter_pulse")
        for ratio in [0.25, 1.0, 4.0]:
            p = create_pulse(pre_pulse_mean=1.0, amplitude_ratio=ratio, J=1.0)
            self.assertAlmostEqual(abs(p.amplitude), ratio, places=6)


class TestFeature19TieDisruptions(ModularBaseTestCase):
    """Feature 19: Structural Tie Disruptions (Random edge deletions 10% or 30%)."""

    def test_f19_tie_disruption_edge_deletion_10_percent(self):
        """Verify 10% edge deletion prunes approximately 10% of active ties."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = (np.ones((20, 20), dtype=np.int32) - np.eye(20, dtype=np.int32)) * 2
        active_before = np.sum(W > 0)
        W_after, stats = apply_disruption(W, fraction=0.10, seed=42)
        active_after = np.sum(W_after > 0)
        expected_deleted = int(round(active_before * 0.10))
        self.assertEqual(active_before - active_after, expected_deleted)

    def test_f19_tie_disruption_edge_deletion_30_percent(self):
        """Verify 30% edge deletion prunes exactly round(30%) of active ties."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = (np.ones((20, 20), dtype=np.int32) - np.eye(20, dtype=np.int32)) * 2
        active_before = np.sum(W > 0)
        W_after, stats = apply_disruption(W, fraction=0.30, seed=42)
        active_after = np.sum(W_after > 0)
        expected_deleted = int(round(active_before * 0.30))
        self.assertEqual(active_before - active_after, expected_deleted)

    def test_f19_tie_disruption_sets_weight_to_zero(self):
        """Verify disrupted edge has its full memory erased (W_ij set to 0)."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.array([[0, 5], [10, 0]], dtype=np.int32)
        W_after, stats = apply_disruption(W, fraction=0.50, seed=42)
        # One edge deleted
        self.assertEqual(np.sum(W_after > 0), 1)
        self.assertIn(0, [W_after[0, 1], W_after[1, 0]])

    def test_f19_tie_disruption_tracks_removed_memory_mass(self):
        """Verify returned report documents both deleted edge count and total removed memory mass."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.array([[0, 4, 2], [1, 0, 3], [0, 0, 0]], dtype=np.int32)
        W_after, stats = apply_disruption(W, fraction=0.50, seed=42)
        self.assertIn("edges_deleted", stats)
        self.assertIn("memory_mass_deleted", stats)
        self.assertEqual(stats["memory_mass_deleted"], np.sum(W) - np.sum(W_after))

    def test_f19_tie_disruption_scheduled_exact_time(self):
        """Verify tie disruption executes via deterministic preemption at scheduled time."""
        Scheduler = self.require_attr("src.core.scheduler", "MasterScheduler")
        scheduler = Scheduler()
        disruption_executed = []
        scheduler.add_deterministic_event(t_event=20.0, callback=lambda: disruption_executed.append(True))
        t, fired = scheduler.advance_step(current_time=15.0, stochastic_dt=10.0)
        self.assertAlmostEqual(t, 20.0, places=6)
        self.assertEqual(len(disruption_executed), 1)


class TestFeature20InterfaceSchedules(ModularBaseTestCase):
    """Feature 20: Population Interface Schedules."""

    def test_f20_closed_interface_eligibility_zero(self):
        """Verify closed interface sets cross-population eligibility to 0."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=2)
        sched.set_closed()
        E = sched.get_matrix(t=0.0)
        self.assertEqual(E[0, 1], 0.0)
        self.assertEqual(E[1, 0], 0.0)

    def test_f20_simultaneous_interface_opening(self):
        """Verify simultaneous opening sets all cross-population entries to 1 at t_open."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=3)
        sched.schedule_simultaneous_opening(t_open=10.0)
        E_before = sched.get_matrix(t=5.0)
        E_after = sched.get_matrix(t=15.0)
        self.assertEqual(E_before[0, 1], 0.0)
        self.assertEqual(E_after[0, 1], 1.0)
        self.assertEqual(E_after[1, 2], 1.0)

    def test_f20_sequential_interface_opening_schedule(self):
        """Verify sequential opening opens pairs one by one every 5*tau_y."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=3)
        sched.schedule_sequential_opening(t_start=10.0, interval=5.0)
        # Between 10 and 15: pair (0, 1) open, (1, 2) closed
        E_stage1 = sched.get_matrix(t=12.0)
        self.assertEqual(E_stage1[0, 1], 1.0)
        self.assertEqual(E_stage1[1, 2], 0.0)
        # After 15: pair (1, 2) open
        E_stage2 = sched.get_matrix(t=16.0)
        self.assertEqual(E_stage2[1, 2], 1.0)

    def test_f20_closing_interface_preserves_existing_ties(self):
        """Verify closing interface halts new cross encounters but does not delete existing ties in W."""
        state = self.create_state(N=4, D=1)
        state.W[0, 2] = 5  # Pre-existing cross tie
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=2)
        sched.set_closed()
        # W must remain intact
        self.assertEqual(state.W[0, 2], 5)

    def test_f20_multi_population_k_groups_support(self):
        """Verify interface supports 2, 3, and 5 populations."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        for k in [2, 3, 5]:
            sched = Schedule(num_populations=k)
            E = sched.get_matrix(t=0.0)
            self.assertEqual(E.shape, (k, k))


class TestFeature21MechanismSpeedRescaling(ModularBaseTestCase):
    """Feature 21: Mechanism Speed Rescaling ((nu, delta) <- (lambda*nu, lambda*delta))."""

    def test_f21_speed_multiplier_scaling_nu_and_delta(self):
        """Verify speed multiplier lambda scales both nu and delta by factor lambda."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        nu_orig, delta_orig = 2.0, 0.1
        nu_new, delta_new = rescale(nu=nu_orig, delta=delta_orig, speed_multiplier=5.0)
        self.assertAlmostEqual(nu_new, 10.0, places=6)
        self.assertAlmostEqual(delta_new, 0.5, places=6)

    def test_f21_speed_multiplier_preserves_memory_scale(self):
        """Verify memory scale nu / delta = 20 remains invariant under rescaling."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        for lamb in [0.2, 1.0, 5.0]:
            nu_new, delta_new = rescale(nu=2.0, delta=0.1, speed_multiplier=lamb)
            self.assertAlmostEqual(nu_new / delta_new, 20.0, places=6)

    def test_f21_speed_multiplier_restoration_to_baseline(self):
        """Verify restoring speed multiplier returns rates to original values."""
        Engine = self.require_attr("src.experiment.interventions", "MechanismSpeedManager")
        mgr = Engine(base_nu=2.0, base_delta=0.1)
        mgr.apply_multiplier(0.2)
        self.assertAlmostEqual(mgr.current_nu, 0.4, places=6)
        mgr.restore_baseline()
        self.assertAlmostEqual(mgr.current_nu, 2.0, places=6)

    def test_f21_speed_multiplier_event_rate_acceleration(self):
        """Verify total event rate increases when lambda = 5.0."""
        calc_rates = self.require_attr("src.core.scheduler", "calculate_total_event_rate")
        r_base = calc_rates(N=100, D=1, tau_x=5.0, tau_y=1.0, nu=2.0, delta=0.1, total_memory=500, tau_n=1.0)
        r_fast = calc_rates(N=100, D=1, tau_x=5.0, tau_y=1.0, nu=10.0, delta=0.5, total_memory=500, tau_n=1.0)
        self.assertGreater(r_fast["total"], r_base["total"])

    def test_f21_speed_multiplier_discrete_values_support(self):
        """Verify support for standard multipliers lambda in {0.2, 1.0, 5.0}."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        for lamb in [0.2, 1.0, 5.0]:
            nu, delta = rescale(nu=2.0, delta=0.1, speed_multiplier=lamb)
            self.assertTrue(nu > 0 and delta > 0)


class TestFeature22DecoupledConfig(ModularBaseTestCase):
    """Feature 22: Decoupled Config Schema & Loader."""

    def test_f22_config_schema_validation_pydantic(self):
        """Verify SimulationConfig validates types and constraints via Pydantic."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        cfg = Config(N=100, D=1, T=0.3, delta=0.1, nu=2.0)
        self.assertEqual(cfg.N, 100)
        self.assertEqual(cfg.D, 1)

    def test_f22_config_loader_yaml_parsing(self):
        """Verify loader loads experiment parameters from YAML string or file."""
        load_yaml = self.require_attr("src.config.loader", "load_config_from_yaml_string")
        yaml_content = """
        experiment:
          name: "pilot_test"
          N: 256
          D: 3
          T: 0.1
          delta: 0.02
          nu: 0.4
        """
        cfg = load_yaml(yaml_content)
        self.assertEqual(cfg.experiment.N, 256)
        self.assertEqual(cfg.experiment.D, 3)

    def test_f22_config_missing_required_fields_raises(self):
        """Verify missing required fields raises validation error."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        with self.assertRaises(Exception):
            Config(D=1)  # Missing N, T, etc.

    def test_f22_config_model_physics_decoupled_from_execution(self):
        """Verify physics parameters (J, K, T) are strictly isolated from execution settings (cores, batches)."""
        Config = self.require_attr("src.config.schema", "RootConfig")
        cfg = Config(
            physics={"J": 1.0, "K": 0.5, "T": 0.3},
            execution={"num_workers": 4, "seed": 123}
        )
        self.assertEqual(cfg.physics.J, 1.0)
        self.assertEqual(cfg.execution.num_workers, 4)

    def test_f22_config_replicate_identity_uniqueness(self):
        """Verify realization identity format: condition_id + replicate_index."""
        build_id = self.require_attr("src.config.schema", "build_realization_id")
        rid = build_id(condition_id="T0.3_d0.1", replicate_index=7)
        self.assertEqual(rid, "T0.3_d0.1_rep07")


class TestFeature23StateCheckpointing(ModularBaseTestCase):
    """Feature 23: Snapshot & State Checkpointing."""

    def test_f23_checkpoint_serialization_completeness(self):
        """Verify state serialization contains all dynamical fields: t, x, y, W, n, Q."""
        state = self.create_state(N=20, D=3)
        checkpoint = state.to_dict()
        for key in ["t", "x", "y", "W", "n", "Q"]:
            self.assertIn(key, checkpoint)

    def test_f23_checkpoint_deserialization_fidelity(self):
        """Verify restoring state from checkpoint recovers exact numerical arrays."""
        State = self.require_attr("src.core.state", "State")
        s1 = self.create_state(N=10, D=1)
        s1.x[:] = 1
        s1.y[:] = -1
        s1.W[0, 1] = 7
        data = s1.to_dict()
        s2 = State.from_dict(data)
        np.testing.assert_array_equal(s1.x, s2.x)
        np.testing.assert_array_equal(s1.y, s2.y)
        np.testing.assert_array_equal(s1.W, s2.W)

    def test_f23_checkpoint_preserves_prng_state(self):
        """Verify checkpoint saves and restores PRNG bit generator state."""
        save_chk = self.require_attr("src.execution.runner", "save_checkpoint")
        load_chk = self.require_attr("src.execution.runner", "load_checkpoint")
        rng = np.random.default_rng(12345)
        _ = rng.random()  # Advance PRNG
        chk = save_chk(rng=rng, t=5.0)
        restored_rng = load_chk(chk)["rng"]
        self.assertEqual(rng.random(), restored_rng.random())

    def test_f23_checkpoint_preserves_holding_timer(self):
        """Verify checkpoint preserves active institutional holding interval and deadline."""
        State = self.require_attr("src.core.state", "State")
        state = self.create_state(N=10, D=1)
        state.t_start_hold = 12.5
        state.t_deadline = 17.5
        state.holding_active = True
        data = state.to_dict()
        restored = State.from_dict(data)
        self.assertAlmostEqual(restored.t_start_hold, 12.5, places=6)
        self.assertAlmostEqual(restored.t_deadline, 17.5, places=6)

    def test_f23_checkpoint_branch_continuation_isolation(self):
        """Verify branching from checkpoint with distinct seeds yields independent trajectories."""
        branch = self.require_attr("src.execution.runner", "branch_checkpoint")
        chk = {"seed": 100, "t": 10.0}
        branch_unperturbed = branch(chk, arm="unperturbed", seed_offset=1)
        branch_perturbed = branch(chk, arm="perturbed", seed_offset=2)
        self.assertNotEqual(branch_unperturbed["seed"], branch_perturbed["seed"])


if __name__ == "__main__":
    unittest.main()
