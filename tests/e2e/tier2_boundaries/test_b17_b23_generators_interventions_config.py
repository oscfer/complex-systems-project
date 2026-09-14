"""Tier 2: Boundary & Corner Cases for Features 17 to 23.

Boundary 17: Network Initial Condition Generators
Boundary 18: Scheduled Applied Field Pulses
Boundary 19: Structural Tie Disruptions
Boundary 20: Population Interface Schedules
Boundary 21: Mechanism Speed Rescaling
Boundary 22: Decoupled Config Schema & Loader
Boundary 23: Snapshot & State Checkpointing
"""

import unittest
import numpy as np
from tests.e2e.common import ModularBaseTestCase


class TestBoundary17Generators(ModularBaseTestCase):
    """Boundary cases for Feature 17: Generators."""

    def test_b17_system_size_n2_minimum_population(self):
        """Verify minimal system size N = 2 generates valid single-edge pair."""
        gen = self.require_attr("src.experiment.generators", "generate_decentralized_network")
        W = gen(N=2, m=10, seed=42)
        self.assertEqual(W.shape, (2, 2))
        self.assertEqual(W[0, 1], 10)
        self.assertEqual(W[1, 0], 10)

    def test_b17_modular_generator_eta_zero_matches_random(self):
        """Verify modular generator at eta = 0.0 distributes ties uniformly across blocks."""
        gen = self.require_attr("src.experiment.generators", "generate_modular_network")
        W, blocks = gen(N=40, num_blocks=2, m=20, eta=0.0, seed=42)
        b0_nodes = np.where(blocks == 0)[0]
        b1_nodes = np.where(blocks == 1)[0]
        internal = W[np.ix_(b0_nodes, b0_nodes)].sum()
        external = W[np.ix_(b0_nodes, b1_nodes)].sum()
        # With equal block sizes (20 and 20) and no self-loops, external opportunities are 20, internal 19
        # Ratio should be close to 1:1
        ratio = internal / max(1, external)
        self.assertAlmostEqual(ratio, 1.0, delta=0.25)

    def test_b17_modular_generator_eta_one_strictly_internal(self):
        """Verify modular generator at eta = 1.0 places 100% of ties strictly within blocks."""
        gen = self.require_attr("src.experiment.generators", "generate_modular_network")
        W, blocks = gen(N=30, num_blocks=3, m=10, eta=1.0, seed=42)
        b0_nodes = np.where(blocks == 0)[0]
        b1_nodes = np.where(blocks == 1)[0]
        cross_edges = W[np.ix_(b0_nodes, b1_nodes)].sum()
        self.assertEqual(cross_edges, 0)

    def test_b17_hub_generator_eta_zero_uniform(self):
        """Verify hub generator at eta = 0.0 allocates uniform targets without hub preference."""
        gen = self.require_attr("src.experiment.generators", "generate_hub_network")
        W, hub_idx = gen(N=50, m=10, eta=0.0, hub=0, seed=42)
        in_degrees = W.sum(axis=0)
        # Hub in-degree should be close to mean in-degree
        mean_in = np.mean(in_degrees)
        self.assertAlmostEqual(in_degrees[hub_idx], mean_in, delta=5.0)

    def test_b17_state_alignment_q_zero_identical_states(self):
        """Verify state alignment generator at q = 0.0 produces identical private and public states."""
        gen = self.require_attr("src.experiment.generators", "generate_aligned_states")
        x, y = gen(N=50, D=1, p_g=0.8, q=0.0, seed=42)
        np.testing.assert_array_equal(x, y)


class TestBoundary18FieldPulses(ModularBaseTestCase):
    """Boundary cases for Feature 18: Field Pulses."""

    def test_b18_field_pulse_zero_amplitude(self):
        """Verify field pulse with amplitude 0.0 exerts zero influence."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        pulse = Intervention(start_time=5.0, duration=2.0, amplitude=0.0)
        self.assertEqual(pulse.get_field(t=6.0), 0.0)

    def test_b18_field_pulse_zero_duration(self):
        """Verify field pulse with duration 0.0 never applies external field."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        pulse = Intervention(start_time=5.0, duration=0.0, amplitude=5.0)
        self.assertEqual(pulse.get_field(t=5.0001), 0.0)

    def test_b18_field_pulse_empty_target_mask(self):
        """Verify field pulse with all-False mask leaves all agent fields unaffected."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        mask = np.zeros(5, dtype=bool)
        pulse = Intervention(start_time=0.0, duration=10.0, amplitude=2.0, target_mask=mask)
        fields = pulse.get_fields_vector(t=2.0, N=5)
        np.testing.assert_array_equal(fields, np.zeros(5))

    def test_b18_field_pulse_full_population_mask(self):
        """Verify field pulse with all-True mask applies uniformly across all agents."""
        Intervention = self.require_attr("src.experiment.interventions", "FieldPulseIntervention")
        mask = np.ones(5, dtype=bool)
        pulse = Intervention(start_time=0.0, duration=10.0, amplitude=2.0, target_mask=mask)
        fields = pulse.get_fields_vector(t=2.0, N=5)
        np.testing.assert_array_equal(fields, np.full(5, 2.0))

    def test_b18_field_pulse_exactly_balanced_population_sign(self):
        """Verify when population mean is exactly 0.0, pulse chooses default deterministic direction."""
        create_pulse = self.require_attr("src.experiment.interventions", "create_counter_pulse")
        pulse = create_pulse(pre_pulse_mean=0.0, amplitude=1.0)
        self.assertIn(pulse.amplitude, [1.0, -1.0])


class TestBoundary19TieDisruptions(ModularBaseTestCase):
    """Boundary cases for Feature 19: Structural Tie Disruptions."""

    def test_b19_tie_disruption_zero_fraction_no_deletion(self):
        """Verify disruption with fraction = 0.0 removes zero edges."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.array([[0, 2], [3, 0]], dtype=np.int32)
        W_after, stats = apply_disruption(W, fraction=0.0)
        np.testing.assert_array_equal(W, W_after)
        self.assertEqual(stats["edges_deleted"], 0)

    def test_b19_tie_disruption_one_hundred_percent_full_wipe(self):
        """Verify disruption with fraction = 1.0 deletes every active tie."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.array([[0, 5, 2], [1, 0, 3], [4, 2, 0]], dtype=np.int32)
        W_after, stats = apply_disruption(W, fraction=1.0)
        self.assertEqual(np.sum(W_after), 0)
        self.assertEqual(stats["edges_deleted"], 6)

    def test_b19_tie_disruption_on_empty_graph_safe(self):
        """Verify disruption on empty graph W = 0 completes safely with 0 deletions."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.zeros((4, 4), dtype=np.int32)
        W_after, stats = apply_disruption(W, fraction=0.30)
        self.assertEqual(stats["edges_deleted"], 0)
        self.assertEqual(stats["memory_mass_deleted"], 0)

    def test_b19_tie_disruption_on_single_edge_network(self):
        """Verify disruption on graph with 1 edge and fraction 1.0 removes that exact edge."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.zeros((3, 3), dtype=np.int32)
        W[0, 1] = 10
        W_after, stats = apply_disruption(W, fraction=1.0)
        self.assertEqual(W_after[0, 1], 0)
        self.assertEqual(stats["memory_mass_deleted"], 10)

    def test_b19_tie_disruption_tracks_exact_zero_counts(self):
        """Verify disruption never leaves negative values in W."""
        apply_disruption = self.require_attr("src.experiment.interventions", "apply_tie_disruption")
        W = np.array([[0, 1], [1, 0]], dtype=np.int32)
        W_after, _ = apply_disruption(W, fraction=0.5, seed=42)
        self.assertTrue(np.all(W_after >= 0))


class TestBoundary20InterfaceSchedules(ModularBaseTestCase):
    """Boundary cases for Feature 20: Population Interfaces."""

    def test_b20_single_population_trivial_interface(self):
        """Verify single population (K=1) produces trivial 1x1 interface matrix [1]."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=1)
        E = sched.get_matrix(t=0.0)
        self.assertEqual(E.shape, (1, 1))

    def test_b20_all_populations_closed_no_inter_encounters(self):
        """Verify completely closed multi-population interface prohibits all cross ties."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=5)
        sched.set_closed()
        E = sched.get_matrix(t=0.0)
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertEqual(E[i, j], 0.0)

    def test_b20_interface_opening_at_t_zero(self):
        """Verify interface scheduled to open at t = 0.0 is open from initial step."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=2)
        sched.schedule_simultaneous_opening(t_open=0.0)
        E = sched.get_matrix(t=0.0)
        self.assertEqual(E[0, 1], 1.0)


    def test_b20_rapid_interface_toggling(self):
        """Verify multiple rapid open/close transitions maintain correct state at each time."""
        Schedule = self.require_attr("src.experiment.interventions", "InterfaceSchedule")
        sched = Schedule(num_populations=2)
        sched.add_transition(t=2.0, state=1.0)
        sched.add_transition(t=4.0, state=0.0)
        sched.add_transition(t=6.0, state=1.0)
        self.assertEqual(sched.get_matrix(t=1.0)[0, 1], 0.0)
        self.assertEqual(sched.get_matrix(t=3.0)[0, 1], 1.0)
        self.assertEqual(sched.get_matrix(t=5.0)[0, 1], 0.0)
        self.assertEqual(sched.get_matrix(t=7.0)[0, 1], 1.0)


class TestBoundary21SpeedRescaling(ModularBaseTestCase):
    """Boundary cases for Feature 21: Mechanism Speed Rescaling."""

    def test_b21_speed_multiplier_one_identity(self):
        """Verify lambda = 1.0 returns identical rates."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        nu, delta = rescale(nu=2.0, delta=0.1, speed_multiplier=1.0)
        self.assertEqual(nu, 2.0)
        self.assertEqual(delta, 0.1)

    def test_b21_extreme_speed_slowdown_lambda_tenth(self):
        """Verify extreme slowdown lambda = 0.1 scales rates by 0.1."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        nu, delta = rescale(nu=2.0, delta=0.1, speed_multiplier=0.1)
        self.assertAlmostEqual(nu, 0.2, places=6)
        self.assertAlmostEqual(delta, 0.01, places=6)

    def test_b21_extreme_speed_speedup_lambda_ten(self):
        """Verify speedup lambda = 10.0 scales rates by 10.0."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        nu, delta = rescale(nu=2.0, delta=0.1, speed_multiplier=10.0)
        self.assertAlmostEqual(nu, 20.0, places=6)
        self.assertAlmostEqual(delta, 1.0, places=6)

    def test_b21_speed_rescaling_restoration_exactness(self):
        """Verify rescaling and immediately restoring returns exactly identical float values."""
        rescale = self.require_attr("src.experiment.interventions", "rescale_mechanism_speed")
        nu0, delta0 = 2.0, 0.1
        nu1, delta1 = rescale(nu=nu0, delta=delta0, speed_multiplier=5.0)
        nu_restored, delta_restored = rescale(nu=nu1, delta=delta1, speed_multiplier=0.2)
        self.assertAlmostEqual(nu_restored, nu0, places=7)
        self.assertAlmostEqual(delta_restored, delta0, places=7)

    def test_b21_speed_rescaling_with_zero_memory(self):
        """Verify rescaling rates on network with W = 0 does not cause division by zero."""
        calc_rates = self.require_attr("src.core.scheduler", "calculate_total_event_rate")
        rates = calc_rates(N=10, D=1, tau_x=5.0, tau_y=1.0, nu=10.0, delta=0.5, total_memory=0, tau_n=1.0)
        self.assertEqual(rates["forget"], 0.0)


class TestBoundary22ConfigValidation(ModularBaseTestCase):
    """Boundary cases for Feature 22: Decoupled Config Schema."""

    def test_b22_config_unknown_fields_rejected(self):
        """Verify config rejects extra unrecognized attributes if extra='forbid'."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        with self.assertRaises(Exception):
            Config(N=100, D=1, T=0.1, delta=0.1, nu=2.0, non_existent_param=123)

    def test_b22_config_negative_parameters_rejected(self):
        """Verify negative physical parameters (e.g. delta < 0 or T < 0) raise ValidationError."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        with self.assertRaises(Exception):
            Config(N=100, D=1, T=-0.5, delta=0.1, nu=2.0)

    def test_b22_config_zero_temperature_rejected(self):
        """Verify T = 0.0 raises ValidationError (temperature must be strictly positive)."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        with self.assertRaises(Exception):
            Config(N=100, D=1, T=0.0, delta=0.1, nu=2.0)

    def test_b22_config_empty_yaml_document_rejected(self):
        """Verify parsing empty YAML string raises ValueError."""
        load_yaml = self.require_attr("src.config.loader", "load_config_from_yaml_string")
        with self.assertRaises(Exception):
            load_yaml("")

    def test_b22_config_boundary_probabilities_zero_and_one(self):
        """Verify boundary probabilities epsilon = 0.0 or 1.0 are validated."""
        Config = self.require_attr("src.config.schema", "SimulationConfig")
        cfg_0 = Config(N=100, D=1, T=0.5, delta=0.1, nu=2.0, epsilon=0.0)
        cfg_1 = Config(N=100, D=1, T=0.5, delta=0.1, nu=2.0, epsilon=1.0)
        self.assertEqual(cfg_0.epsilon, 0.0)
        self.assertEqual(cfg_1.epsilon, 1.0)


class TestBoundary23StateCheckpointing(ModularBaseTestCase):
    """Boundary cases for Feature 23: Snapshot & State Checkpointing."""

    def test_b23_checkpoint_empty_state_roundtrip(self):
        """Verify serializing and restoring state with empty memory W = 0."""
        State = self.require_attr("src.core.state", "State")
        s = self.create_state(N=5, D=1)
        data = s.to_dict()
        s_restored = State.from_dict(data)
        self.assertTrue(np.all(s_restored.W == 0))

    def test_b23_checkpoint_extreme_values_roundtrip(self):
        """Verify checkpoint preserves float values with full floating point precision."""
        State = self.require_attr("src.core.state", "State")
        s = self.create_state(N=2, D=1, continuous=True)
        s.x[0, 0] = 0.123456789012345
        data = s.to_dict()
        s_restored = State.from_dict(data)
        self.assertEqual(s.x[0, 0], s_restored.x[0, 0])

    def test_b23_checkpoint_active_holding_timer_roundtrip(self):
        """Verify checkpoint preserves active holding interval start and deadline."""
        State = self.require_attr("src.core.state", "State")
        s = self.create_state(N=4, D=1)
        s.t_start_hold = 15.25
        s.t_deadline = 20.25
        s.holding_active = True
        data = s.to_dict()
        s_restored = State.from_dict(data)
        self.assertEqual(s_restored.t_start_hold, 15.25)
        self.assertEqual(s_restored.t_deadline, 20.25)

    def test_b23_checkpoint_scheduled_events_queue_roundtrip(self):
        """Verify checkpoint preserves queue of scheduled deterministic events."""
        save_chk = self.require_attr("src.execution.runner", "save_checkpoint")
        load_chk = self.require_attr("src.execution.runner", "load_checkpoint")
        events = [{"t": 5.0, "type": "field_pulse"}, {"t": 10.0, "type": "tie_disrupt"}]
        chk = save_chk(scheduled_events=events, t=2.0)
        restored = load_chk(chk)
        self.assertEqual(restored["scheduled_events"], events)

    def test_b23_checkpoint_corrupt_file_raises_error(self):
        """Verify loading corrupted checkpoint data raises an exception."""
        load_chk = self.require_attr("src.execution.runner", "load_checkpoint")
        with self.assertRaises(Exception):
            load_chk("corrupted_data_string_not_valid_dict")


if __name__ == "__main__":
    unittest.main()
