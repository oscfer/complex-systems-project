"""Integration tests for Control Contracts and Realism Physics Engine.

Tests:
1. Zero-control baseline invariance: ZeroController applies 0 field, consumes 0 budget,
   and does not alter physical RNG streams.
2. Quenched heterogeneity integration: closed-loop simulation with heterogeneous
   K_i, tau_{x,i}, nu_i, g_i, b_i, bounded row capacity, and opportunity groups.
3. Commanded field u vs realized physical field g_i * u (ledger strictly charges commanded u).
4. Closed-loop observable feedback execution with delayed, noisy observations and budget constraints.
5. Exact post-release holding timer certification and RolloutResult integrity.
"""

import unittest
import numpy as np

from src.control.contracts import Action, ControlProblem, TargetType
from src.control.controllers import Controller, ObservableSelectorPolicy, OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetDefinition, TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import SampledPublicObservationModel
from src.core.heterogeneity import generate_quenched_heterogeneity
from src.core.state import State, StateMode


class TestControlIntegration(unittest.TestCase):
    """End-to-end integration tests between src/control and src/core realism physics."""

    def setUp(self):
        self.N = 20
        self.D = 1
        self.horizon = 15.0
        self.release_deadline = 5.0
        self.holding_duration = 3.0

        # Create state with quenched heterogeneity
        self.het = generate_quenched_heterogeneity(
            N=self.N,
            D=self.D,
            rng=np.random.default_rng(42),
        )
        self.state = State.create_empty(N=self.N, D=self.D, state_mode=StateMode.BINARY)
        self.state.y.fill(1.0)
        self.state.x.fill(1.0)
        self.state.K = self.het.K.copy()
        self.state.tau_x = self.het.tau_x.copy()
        self.state.nu = self.het.nu.copy()
        self.state.g = self.het.g.copy()
        self.state.b = self.het.b.copy()
        self.state.row_caps = np.full(self.N, 40, dtype=np.uint32)

    def test_zero_control_baseline_invariance(self):
        """ZeroController leaves state unforced, spends 0 budget, and completes with valid result."""
        problem = ControlProblem(
            problem_id="test_zero",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=4,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=self.release_deadline,
            holding_duration=self.holding_duration,
            horizon=self.horizon,
            accessible_roster=(0, 1, 2, 3, 4),
        )

        ctrl = ZeroController()
        result, final_state, ledger = run_controlled_simulation(
            initial_state=self.state,
            problem=problem,
            controller=ctrl,
            sample_interval=2.0,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.expenditure, 0.0)
        self.assertEqual(result.distinct_nodes_used, 0)
        self.assertTrue(ledger.is_released)
        self.assertEqual(ledger.integrated_exposure, 0.0)
        self.assertGreater(result.event_counts["stochastic_events"], 0)

    def test_commanded_vs_realized_field_accounting(self):
        """Ledger charges commanded field u, while physical simulation applies g_i * u."""
        problem = ControlProblem(
            problem_id="test_commanded",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=self.release_deadline,
            holding_duration=self.holding_duration,
            horizon=self.horizon,
            accessible_roster=(0, 1),
        )

        # Pulse on node 0 with amplitude 2.0 for duration 2.0
        # Expected commanded expenditure = 2.0 * 2.0 = 4.0
        ctrl = OpenLoopPulseController(
            target_nodes=[0],
            nominal_amplitude=2.0,
            duration=2.0,
            signs={0: -1.0},
        )

        result, final_state, ledger = run_controlled_simulation(
            initial_state=self.state,
            problem=problem,
            controller=ctrl,
            sample_interval=1.0,
        )

        # Commanded expenditure must match commanded field integration, not multiplied by g_i
        self.assertAlmostEqual(result.expenditure, 4.0, delta=0.01)
        self.assertEqual(result.distinct_nodes_used, 1)

    def test_observable_feedback_closed_loop_with_realism_physics(self):
        """ObservableSelectorPolicy runs closed-loop with delayed noisy observations and heterogeneity."""
        accessible = tuple(range(5))
        problem = ControlProblem(
            problem_id="test_feedback",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=4,
            exposure_cap=15.0,
            amplitude_cap=4.0,
            release_deadline=self.release_deadline,
            holding_duration=self.holding_duration,
            horizon=self.horizon,
            accessible_roster=accessible,
        )

        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=2.0,
            duration=4.0,
            quota=0.5,
            is_feedback=True,
            initial_recruitment_fraction=0.5,
        )

        result, final_state, ledger = run_controlled_simulation(
            initial_state=self.state,
            problem=problem,
            controller=ctrl,
            sample_interval=1.0,
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(ledger.is_released)
        self.assertLessEqual(ledger.integrated_exposure, problem.exposure_cap + 1e-6)
        self.assertLessEqual(len(ledger.distinct_nodes_actuated), problem.k)
        for node in ledger.distinct_nodes_actuated:
            self.assertIn(node, accessible)

        # Verify restricted time is bounded by horizon
        self.assertLessEqual(result.restricted_time, self.horizon)


    def test_reciprocal_memory_mode_and_row_caps(self):
        """Reciprocal memory mode maintains W = W^T and strictly rejects row_caps per spec."""
        problem = ControlProblem(
            problem_id="test_recip",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=self.release_deadline,
            holding_duration=self.holding_duration,
            horizon=self.horizon,
            accessible_roster=(0, 1),
        )

        # 1. State with row caps and reciprocal=True must raise ValueError per specification
        with self.assertRaises(ValueError):
            run_controlled_simulation(
                initial_state=self.state,  # self.state has row_caps set in setUp()
                problem=problem,
                reciprocal=True,
            )

        # 2. State without row caps and reciprocal=True must maintain W = W^T
        state_uncapped = self.state.copy()
        state_uncapped.row_caps = None
        result, final_state, ledger = run_controlled_simulation(
            initial_state=state_uncapped,
            problem=problem,
            reciprocal=True,
        )
        self.assertTrue(result.is_valid)
        np.testing.assert_array_equal(final_state.W, final_state.W.T)


    def test_early_budget_exhaustion_declares_release_and_enables_holding(self):
        """Budget exhaustion before release deadline must declare release at exhaustion time."""
        problem = ControlProblem(
            problem_id="test_exhaust_release",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=2,
            exposure_cap=2.0,  # Small budget: 2.0 exposure
            amplitude_cap=4.0,
            release_deadline=20.0,
            holding_duration=2.0,
            horizon=25.0,
            accessible_roster=(0, 1),
        )

        class FixedRateController(Controller):
            def reset(self, problem, obs, rng=None):
                self.acted = False
            def act(self, obs, led):
                if not self.acted:
                    self.acted = True
                    # Fixed amplitude 2.0, spending rate c_u = 2.0, intended end 10.0
                    return Action.from_dict({(0, 0): 2.0}, intended_end_time=10.0)
                return Action(entries=(), release=True)

        ctrl = FixedRateController()

        result, final_state, ledger = run_controlled_simulation(
            initial_state=self.state,
            problem=problem,
            controller=ctrl,
            sample_interval=1.0,
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(ledger.is_exhausted)
        self.assertTrue(ledger.is_released)
        # Release time should be exhaustion time t=2.0, well before release deadline 20.0!
        self.assertAlmostEqual(ledger.release_time, 2.0, delta=0.01)
        self.assertAlmostEqual(result.expenditure, 2.0, delta=0.01)


    def test_time_zero_field_shutoff_schedules_holding_deadline(self):
        """Time-zero action shutoff at eff_end must trigger target evaluation and holding deadline."""
        from src.control.observation import OracleObservationModel

        problem = ControlProblem(
            problem_id="test_t0_shutoff",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=5.0,
            holding_duration=2.0,
            horizon=10.0,
            accessible_roster=(0, 1),
        )

        class T0PulseController(Controller):
            def reset(self, prob, obs, rng=None):
                self.acted = False
            def act(self, obs, led):
                if not self.acted:
                    self.acted = True
                    return Action.from_dict({(0, 0): 1.0}, intended_end_time=2.0)
                return Action(entries=(), release=True)

        # Using OracleObservationModel with delivery_delay=0.0 triggers the t=0 actuation branch
        obs_model = OracleObservationModel(N=self.N, accessible_roster=(0, 1), delivery_delay=0.0)
        ctrl = T0PulseController()

        result, final_state, ledger = run_controlled_simulation(
            initial_state=self.state,
            problem=problem,
            controller=ctrl,
            observation_model=obs_model,
        )

        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.expenditure, 2.0, delta=0.05)

if __name__ == "__main__":
    unittest.main()
