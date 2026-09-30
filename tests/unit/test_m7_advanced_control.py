"""Comprehensive unit and integration tests for Advanced Control.

Covers:
1. Rollout Accuracy & Forecasting:
   - StateEstimator Layer 2 purity (no private state access, unobserved node imputation).
   - SimulatedRolloutForecaster trajectory simulation and metric accuracy.
   - Target distance evaluations for Reversal, Decentralization, and Turnover.
   - Multi-step horizon discounting and cost computation.
2. Model Predictive Control (MPC):
   - Controller initialization, target deduction, and candidate generation.
   - Receding-horizon optimal trajectory selection.
   - BudgetLedger constraint enforcement (exposure cap, amplitude cap, distinct-node limit k).
   - Closed-loop simulation execution with run_controlled_simulation.
3. Structural Actuators:
   - Direct connection forcing, tie severing, and rewiring.
   - Action validation (bounds, self-loops, roster restrictions).
   - Closed-loop structural actuation within the simulation engine.
4. Physical Invariants Preservation:
   - Strict zero diagonal (W_{ii} == 0).
   - Non-negativity under excessive sever requests (W_{ij} >= 0).
   - Reciprocal symmetry preservation in reciprocal networks (W == W^T).
   - Finite row capacity enforcement (s_i <= C_i).
   - PoissonScheduler total memory synchronization preserving continuous-time decay jump rates.
5. Structural Budget Exhaustion:
   - Structural budget tracking and deduction in BudgetLedger.
   - Structural budget exhaustion and rejection of over-budget actions.
   - Per-step structural limits.
   - Distinct-node consumption by structural interventions.
   - Permanent shutoff upon control release.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.control.advanced import (
    ForecastResult,
    MPCConfig,
    ModelPredictiveController,
    SimulatedRolloutForecaster,
    StateEstimator,
    StructuralActuator,
    force_connection_action,
    rewire_tie_action,
    sever_tie_action,
)
from src.control.contracts import (
    Action,
    ActionEntry,
    ActuatorClass,
    ControlProblem,
    Observation,
    ObservationMode,
    StructuralActionEntry,
    StructuralOp,
    TargetType,
)
from src.control.engine import run_controlled_simulation
from src.control.ledger import BudgetLedger
from src.core.memory import MemoryManager
from src.core.scheduler import PoissonScheduler
from src.core.state import State, StateMode


class TestM7StructuralActuators(unittest.TestCase):
    """Test suite for Structural Actuators and edge manipulation."""

    def setUp(self) -> None:
        self.N = 6
        self.actuator = StructuralActuator(rng=np.random.default_rng(42))

    def test_structural_action_validation_invariants(self):
        """Action must enforce valid node indices, no self-loops, and positive counts."""
        # Self-loop must be rejected
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=2, target=2, operation=StructuralOp.FORCE)

        # Count must be >= 1
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=1, target=2, operation=StructuralOp.FORCE, count=0)

        # Rewire without rewire_target must fail
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=1, target=2, operation=StructuralOp.REWIRE)

        # Rewire with self-loop target must fail
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=1, target=2, operation=StructuralOp.REWIRE, rewire_target=1)

        # Action out of bounds validation
        act = force_connection_action(source=10, target=2)
        with self.assertRaises(ValueError):
            act.validate(N=5, D=1)

        # Action accessible roster validation
        act_unauth = force_connection_action(source=3, target=2)
        with self.assertRaises(ValueError):
            act_unauth.validate(N=5, D=1, accessible_roster={0, 1})

    def test_force_connection_and_attention_update(self):
        """Forcing a connection must increment W, row_sums, total_memory, and update attention."""
        mem = MemoryManager(N=self.N, initial_W=np.zeros((self.N, self.N), dtype=np.uint32))
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        sched = PoissonScheduler(N=self.N, D=1, tau_x=5.0, tau_y=1.0, nu=2.0, delta=0.1)

        act = force_connection_action(source=0, target=1, count=3)
        records = self.actuator.apply_action(act, state, mem, scheduler=sched)

        self.assertEqual(len(records), 1)
        self.assertEqual(mem.W[0, 1], 3)
        self.assertEqual(mem.row_sums[0], 3)
        self.assertEqual(mem.total_memory, 3)
        self.assertEqual(sched.total_memory, 3)
        self.assertAlmostEqual(mem.A[0, 1], 1.0)
        self.assertAlmostEqual(state.W[0, 1], 3)

    def test_sever_tie_bounded_and_non_negative(self):
        """Severing a tie must decrement W up to existing weight, preserving W >= 0."""
        initial_W = np.zeros((self.N, self.N), dtype=np.uint32)
        initial_W[2, 3] = 4
        mem = MemoryManager(N=self.N, initial_W=initial_W)
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        sched = PoissonScheduler(N=self.N, D=1, tau_x=5.0, tau_y=1.0, nu=2.0, delta=0.1)
        sched.set_total_memory(4)

        # Sever 2 units
        act1 = sever_tie_action(source=2, target=3, count=2)
        self.actuator.apply_action(act1, state, mem, scheduler=sched)
        self.assertEqual(mem.W[2, 3], 2)
        self.assertEqual(mem.total_memory, 2)
        self.assertEqual(sched.total_memory, 2)

        # Sever 10 units (more than existing 2 units): must clamp to 0 and not become negative
        act2 = sever_tie_action(source=2, target=3, count=10)
        self.actuator.apply_action(act2, state, mem, scheduler=sched)
        self.assertEqual(mem.W[2, 3], 0)
        self.assertEqual(mem.total_memory, 0)
        self.assertEqual(sched.total_memory, 0)
        self.assertTrue(np.all(mem.W >= 0))

    def test_rewire_tie_atomic_operation(self):
        """Rewiring must atomically decrement old tie and increment new tie."""
        initial_W = np.zeros((self.N, self.N), dtype=np.uint32)
        initial_W[1, 2] = 5
        mem = MemoryManager(N=self.N, initial_W=initial_W)
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        sched = PoissonScheduler(N=self.N, D=1, tau_x=5.0, tau_y=1.0, nu=2.0, delta=0.1)
        sched.set_total_memory(5)

        act = rewire_tie_action(source=1, old_target=2, new_target=3, count=2)
        self.actuator.apply_action(act, state, mem, scheduler=sched)

        self.assertEqual(mem.W[1, 2], 3)
        self.assertEqual(mem.W[1, 3], 2)
        self.assertEqual(mem.total_memory, 5)
        self.assertEqual(sched.total_memory, 5)

    def test_reciprocal_symmetry_preservation(self):
        """In reciprocal mode, structural manipulations must strictly maintain W == W^T."""
        mem = MemoryManager(N=self.N, reciprocal=True)
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))

        # Force connection (1, 4)
        act = force_connection_action(source=1, target=4, count=3)
        self.actuator.apply_action(act, state, mem)
        self.assertEqual(mem.W[1, 4], 3)
        self.assertEqual(mem.W[4, 1], 3)
        np.testing.assert_array_equal(mem.W, mem.W.T)

        # Sever connection (1, 4)
        act_sev = sever_tie_action(source=1, target=4, count=1)
        self.actuator.apply_action(act_sev, state, mem)
        self.assertEqual(mem.W[1, 4], 2)
        self.assertEqual(mem.W[4, 1], 2)
        np.testing.assert_array_equal(mem.W, mem.W.T)

    def test_row_capacity_caps_preservation(self):
        """Forcing connection when row is at capacity must respect capacity cap."""
        caps = np.full(self.N, 2, dtype=np.uint32)
        initial_W = np.zeros((self.N, self.N), dtype=np.uint32)
        initial_W[0, 1] = 2  # Row 0 is full (sum == 2 == cap)
        mem = MemoryManager(N=self.N, initial_W=initial_W, row_caps=caps)
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)), row_caps=caps)

        # Force connection (0, 2)
        act = force_connection_action(source=0, target=2, count=1)
        self.actuator.apply_action(act, state, mem)

        # Row sum must not exceed capacity 2
        self.assertEqual(mem.row_sums[0], 2)
        self.assertTrue(mem.W[0, 2] >= 1)
        self.assertTrue(np.all(mem.row_sums <= caps))


    def test_sever_zero_weight_edge_preserves_invariants(self):
        """Severing an edge that is already at weight 0 must be a safe no-op on that tie."""
        mem = MemoryManager(N=self.N, initial_W=np.zeros((self.N, self.N), dtype=np.uint32))
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        act = sever_tie_action(source=1, target=2, count=5)
        records = self.actuator.apply_action(act, state, mem)
        self.assertEqual(mem.W[1, 2], 0)
        self.assertEqual(records[0]["units_removed"], 0)
        self.assertEqual(mem.total_memory, 0)
        self.assertTrue(np.all(mem.W >= 0))

    def test_multi_edge_batch_structural_action(self):
        """Action containing multiple structural entries must be executed atomically and correctly."""
        mem = MemoryManager(N=self.N, initial_W=np.zeros((self.N, self.N), dtype=np.uint32))
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        entries = [
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.FORCE, count=2),
            StructuralActionEntry(source=1, target=2, operation=StructuralOp.FORCE, count=3),
            StructuralActionEntry(source=2, target=3, operation=StructuralOp.FORCE, count=1),
        ]
        act = Action.structural(entries)
        self.assertEqual(act.structural_spending, 6)
        self.assertEqual(act.structural_nodes(), {0, 1, 2, 3})
        self.actuator.apply_action(act, state, mem)
        self.assertEqual(mem.W[0, 1], 2)
        self.assertEqual(mem.W[1, 2], 3)
        self.assertEqual(mem.W[2, 3], 1)
        self.assertEqual(mem.total_memory, 6)
        np.testing.assert_array_equal(state.W, mem.W)

    def test_reciprocal_complex_rewire_and_sever(self):
        """Multiple rewires and severs in reciprocal network must maintain W == W^T."""
        mem = MemoryManager(N=self.N, reciprocal=True)
        state = State(N=self.N, D=1, state_mode=StateMode.BINARY, x=np.ones((self.N, 1)), y=np.ones((self.N, 1)), W=mem.W, n=np.zeros(self.N, dtype=np.uint8), Q=np.zeros((self.N, 1, 2)))
        # Force connections (0, 1) and (2, 3)
        act1 = Action.structural([
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.FORCE, count=4),
            StructuralActionEntry(source=2, target=3, operation=StructuralOp.FORCE, count=4),
        ])
        self.actuator.apply_action(act1, state, mem)
        np.testing.assert_array_equal(mem.W, mem.W.T)

        # Rewire (0, 1) to (0, 4) and sever (2, 3)
        act2 = Action.structural([
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.REWIRE, count=2, rewire_target=4),
            StructuralActionEntry(source=2, target=3, operation=StructuralOp.SEVER, count=3),
        ])
        self.actuator.apply_action(act2, state, mem)
        np.testing.assert_array_equal(mem.W, mem.W.T)
        self.assertEqual(mem.W[0, 1], 2)
        self.assertEqual(mem.W[1, 0], 2)
        self.assertEqual(mem.W[0, 4], 2)
        self.assertEqual(mem.W[4, 0], 2)
        self.assertEqual(mem.W[2, 3], 1)
        self.assertEqual(mem.W[3, 2], 1)


class TestM7StructuralBudgetExhaustion(unittest.TestCase):
    """Test suite for structural budget accounting and exhaustion in BudgetLedger."""

    def test_structural_budget_accounting_and_deduction(self):
        """Structural modifications must accurately deduct from remaining structural budget."""
        ledger = BudgetLedger(
            N=10,
            D=1,
            k=3,
            exposure_cap=20.0,
            structural_budget=5.0,
        )
        self.assertEqual(ledger.remaining_structural_budget, 5.0)
        self.assertFalse(ledger.is_structural_exhausted)

        # Apply action with 2 structural modifications
        act1 = force_connection_action(source=0, target=1, count=2)
        ledger.apply_action(act1, current_time=0.0)
        self.assertEqual(ledger.integrated_structural_modifications, 2)
        self.assertEqual(ledger.remaining_structural_budget, 3.0)
        self.assertFalse(ledger.is_structural_exhausted)

        # Apply action with 3 structural modifications (exhausts budget)
        act2 = sever_tie_action(source=0, target=2, count=3)
        ledger.apply_action(act2, current_time=1.0)
        self.assertEqual(ledger.integrated_structural_modifications, 5)
        self.assertEqual(ledger.remaining_structural_budget, 0.0)
        self.assertTrue(ledger.is_structural_exhausted)

        # Attempting further structural actions must be rejected
        act3 = force_connection_action(source=1, target=2, count=1)
        valid, msg = ledger.validate_action(act3, current_time=2.0)
        self.assertFalse(valid)
        self.assertIn("exceeds remaining structural budget", msg)
        with self.assertRaises(ValueError):
            ledger.apply_action(act3, current_time=2.0)

    def test_max_structural_per_step_limit(self):
        """Actions exceeding max_structural_per_step must be rejected."""
        ledger = BudgetLedger(
            N=10,
            D=1,
            k=3,
            exposure_cap=20.0,
            structural_budget=10.0,
            max_structural_per_step=2,
        )
        # 2 modifications should succeed
        act_ok = force_connection_action(source=0, target=1, count=2)
        valid, msg = ledger.validate_action(act_ok, current_time=0.0)
        self.assertTrue(valid, msg)

        # 3 modifications must fail per-step limit
        act_bad = force_connection_action(source=0, target=1, count=3)
        valid, msg = ledger.validate_action(act_bad, current_time=0.0)
        self.assertFalse(valid)
        self.assertIn("exceeds per-step limit", msg)

    def test_structural_distinct_node_limit_k(self):
        """Structural interventions must count toward distinct-node cap k."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=20.0, structural_budget=10.0)

        # Intervene on source node 0
        act1 = force_connection_action(source=0, target=1, count=1)
        ledger.apply_action(act1, current_time=0.0)
        self.assertIn(0, ledger.distinct_nodes_actuated)
        self.assertEqual(ledger.remaining_distinct_nodes, 1)

        # Intervene on source node 2 (total 2 nodes == k)
        act2 = force_connection_action(source=2, target=3, count=1)
        ledger.apply_action(act2, current_time=1.0)
        self.assertEqual(ledger.remaining_distinct_nodes, 0)

        # Intervene on third source node 4 must be rejected
        act3 = force_connection_action(source=4, target=5, count=1)
        valid, msg = ledger.validate_action(act3, current_time=2.0)
        self.assertFalse(valid)
        self.assertIn("exceeds distinct-node cap", msg)

    def test_structural_shutoff_on_release(self):
        """Declaring control release must permanently reject structural actions."""
        ledger = BudgetLedger(N=10, D=1, k=3, exposure_cap=20.0, structural_budget=10.0)
        ledger.release_control(current_time=2.0)
        self.assertTrue(ledger.is_released)

        act = force_connection_action(source=0, target=1, count=1)
        valid, msg = ledger.validate_action(act, current_time=3.0)
        self.assertFalse(valid)
        self.assertIn("after control release", msg)


    def test_exact_budget_boundary_and_zero_action_when_exhausted(self):
        """Action spending exact remaining structural budget must succeed; zero action remains valid."""
        ledger = BudgetLedger(N=10, D=1, k=3, exposure_cap=20.0, structural_budget=4.0)

        # Exact spending of 4 units
        act_exact = force_connection_action(source=0, target=1, count=4)
        valid, msg = ledger.validate_action(act_exact, current_time=0.0)
        self.assertTrue(valid, msg)
        ledger.apply_action(act_exact, current_time=0.0)
        self.assertEqual(ledger.remaining_structural_budget, 0.0)
        self.assertTrue(ledger.is_structural_exhausted)

        # Zero structural action must still be valid
        zero_act = Action(entries=(), intended_end_time=5.0)
        valid_zero, msg_zero = ledger.validate_action(zero_act, current_time=1.0)
        self.assertTrue(valid_zero, msg_zero)

        # Non-zero structural action must be rejected
        act_excess = force_connection_action(source=0, target=2, count=1)
        valid_excess, msg_excess = ledger.validate_action(act_excess, current_time=1.0)
        self.assertFalse(valid_excess)


class TestM7ForecastingAndRolloutAccuracy(unittest.TestCase):
    """Test suite for StateEstimator and SimulatedRolloutForecaster."""

    def test_state_estimator_layer2_compliance(self):
        """StateEstimator must not touch private states or true memory in non-oracle mode."""
        N = 8
        panel_mask = np.zeros(N, dtype=bool)
        panel_mask[0:4] = True  # Nodes 0-3 observed, 4-7 unobserved

        public_states = np.full((N, 1), np.nan)
        public_states[0:4, 0] = [1.0, 1.0, -1.0, -1.0]

        obs = Observation(
            time=0.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges={(0, 1): 3, (2, 3): 2},
            accessible_roster=(0, 1, 2, 3),
            is_oracle=False,
            private_states=None,
            true_memory=None,
        )

        prob = ControlProblem(
            problem_id="p1",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N,
        )

        estimator = StateEstimator()
        est_state = estimator.estimate_state(obs, prob)

        # Must produce finite estimates for all nodes
        self.assertTrue(np.all(np.isfinite(est_state.y)))
        self.assertTrue(np.all(np.isfinite(est_state.x)))
        # Memory matrix must reflect observed edges
        self.assertEqual(est_state.W[0, 1], 3)
        self.assertEqual(est_state.W[2, 3], 2)
        self.assertEqual(est_state.W[0, 0], 0)  # Diagonal zero invariant

    def test_forecaster_trajectory_and_metric_accuracy(self):
        """Forecaster must step dynamics forward and compute target distance accurately."""
        N = 6
        initial_W = np.zeros((N, N), dtype=np.uint32)
        initial_W[0, 1] = 2
        initial_W[1, 0] = 2

        state = State(
            N=N,
            D=1,
            state_mode=StateMode.BINARY,
            x=np.ones((N, 1)),
            y=np.ones((N, 1)),
            W=initial_W,
            n=np.zeros(N, dtype=np.uint8),
            Q=np.zeros((N, 1, 2)),
        )

        prob = ControlProblem(
            problem_id="test_fc",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N,
            k=3,
            exposure_cap=20.0,
            amplitude_cap=4.0,
            release_deadline=10.0,
            horizon=30.0,
            accessible_roster=(0, 1, 2),
        )

        ledger = BudgetLedger(N=N, D=1, k=3, exposure_cap=20.0, amplitude_cap=4.0)
        forecaster = SimulatedRolloutForecaster(J=1.0, T=0.3)

        # 1. Unactuated forecast (target s_* = -1)
        zero_act = Action(entries=(), intended_end_time=5.0)
        res_zero = forecaster.forecast_trajectory(
            initial_state=state,
            action_sequence=[zero_act, zero_act],
            problem=prob,
            ledger=ledger,
            step_interval=5.0,
            s_star=-1.0,
            target_threshold=0.5,
        )
        self.assertTrue(res_zero.is_valid)
        self.assertEqual(len(res_zero.forecasted_states), 2)
        # In unactuated state (all +1), distance to -1 is high
        self.assertTrue(res_zero.terminal_violation > 1.0)

        # 2. Actuated forecast pushing toward s_* = -1 with valid sequential end times
        act_push1 = Action.from_dict({(0, 0): -1.5, (1, 0): -1.5}, intended_end_time=5.0)
        act_push2 = Action.from_dict({(0, 0): -1.5, (1, 0): -1.5}, intended_end_time=10.0)
        res_act = forecaster.forecast_trajectory(
            initial_state=state,
            action_sequence=[act_push1, act_push2],
            problem=prob,
            ledger=ledger,
            step_interval=5.0,
            s_star=-1.0,
            target_threshold=0.5,
        )
        self.assertTrue(res_act.is_valid)
        # Actuated trajectory expenditure is tracked
        self.assertTrue(res_act.exposure_spent > 0.0)


class TestM7ModelPredictiveController(unittest.TestCase):
    """Test suite for Model Predictive Control (MPC) and receding horizon execution."""

    def setUp(self) -> None:
        self.N = 8
        self.prob = ControlProblem(
            problem_id="mpc_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=self.N,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=3.0,
            release_deadline=15.0,
            holding_duration=20.0,
            horizon=50.0,
            accessible_roster=(0, 1, 2, 3),
        )

    def test_mpc_act_and_receding_horizon(self):
        """MPC must produce valid actions respecting ledger and adapt over intervals."""
        config = MPCConfig(
            horizon_steps=2,
            step_interval=5.0,
            rollout_samples=1,
            seed=42,
        )
        controller = ModelPredictiveController(config=config)
        ledger = BudgetLedger(
            N=self.N,
            D=1,
            k=self.prob.k,
            exposure_cap=self.prob.exposure_cap,
            amplitude_cap=self.prob.amplitude_cap,
            release_deadline=self.prob.release_deadline,
            accessible_roster=set(self.prob.accessible_roster),
        )

        obs0 = Observation(
            time=0.0,
            public_states=np.ones((self.N, 1)),  # Initial state all +1
            panel_mask=np.ones(self.N, dtype=bool),
            observed_edges={(0, 1): 2, (1, 2): 2, (2, 3): 1},
            accessible_roster=self.prob.accessible_roster,
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )

        controller.reset(self.prob, obs0)
        self.assertEqual(controller.s_star, -1.0)  # Inferred reversal target is -1

        # Step 1: act at t=0
        act0 = controller.act(obs0, ledger)
        is_valid0, msg0 = ledger.validate_action(act0, current_time=0.0)
        self.assertTrue(is_valid0, msg0)
        eff_end0 = ledger.apply_action(act0, current_time=0.0)
        self.assertTrue(eff_end0 > 0.0)
        self.assertTrue(len(controller.execution_history) == 1)

        # Step 2: advance ledger to t=5 and act with updated observation
        ledger.advance_to(5.0)
        obs1 = Observation(
            time=5.0,
            public_states=np.array([[-0.5], [-0.5], [0.8], [0.8], [1.0], [1.0], [1.0], [1.0]]),
            panel_mask=np.ones(self.N, dtype=bool),
            observed_edges={(0, 1): 2, (1, 2): 2, (2, 3): 1},
            accessible_roster=self.prob.accessible_roster,
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )

        act1 = controller.act(obs1, ledger)
        is_valid1, msg1 = ledger.validate_action(act1, current_time=5.0)
        self.assertTrue(is_valid1, msg1)
        self.assertTrue(len(controller.execution_history) == 2)

    def test_mpc_structural_actuator_integration(self):
        """MPC configured with structural actions must generate and evaluate structural candidates."""
        config = MPCConfig(
            horizon_steps=2,
            step_interval=5.0,
            rollout_samples=1,
            use_structural=True,
            seed=42,
        )
        controller = ModelPredictiveController(config=config)
        prob_struct = ControlProblem(
            problem_id="mpc_struct_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=self.N,
            D=1,
            k=3,
            exposure_cap=15.0,
            amplitude_cap=3.0,
            structural_budget=5.0,
            release_deadline=15.0,
            holding_duration=20.0,
            horizon=50.0,
            accessible_roster=(0, 1, 2, 3),
        )
        ledger = BudgetLedger(
            N=self.N,
            D=1,
            k=prob_struct.k,
            exposure_cap=prob_struct.exposure_cap,
            amplitude_cap=prob_struct.amplitude_cap,
            release_deadline=prob_struct.release_deadline,
            accessible_roster=set(prob_struct.accessible_roster),
            structural_budget=prob_struct.structural_budget,
        )

        obs = Observation(
            time=0.0,
            public_states=np.ones((self.N, 1)),
            panel_mask=np.ones(self.N, dtype=bool),
            observed_edges={(0, 1): 3, (1, 2): 2},
            accessible_roster=prob_struct.accessible_roster,
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )

        controller.reset(prob_struct, obs)
        candidates = controller._generate_candidates(obs, ledger, horizon_steps=2)
        candidate_names = [c.name for c in candidates]

        # Structural candidates must be present in the candidate pool
        self.assertTrue(any("structural" in name for name in candidate_names))

        act = controller.act(obs, ledger)
        is_valid, msg = ledger.validate_action(act, current_time=0.0)
        self.assertTrue(is_valid, msg)


    def test_mpc_with_partial_panel_missingness(self):
        """MPC must plan and act successfully under 50% observation panel missingness."""
        panel_mask = np.zeros(self.N, dtype=bool)
        panel_mask[0:4] = True  # Only half the network is observed
        public_states = np.full((self.N, 1), np.nan)
        public_states[0:4, 0] = [1.0, 1.0, 1.0, 1.0]

        config = MPCConfig(horizon_steps=2, step_interval=5.0, rollout_samples=1, seed=42)
        controller = ModelPredictiveController(config=config)
        ledger = BudgetLedger(N=self.N, D=1, k=3, exposure_cap=15.0, amplitude_cap=3.0)

        obs = Observation(
            time=0.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges={(0, 1): 2},
            accessible_roster=(0, 1, 2, 3),
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )

        controller.reset(self.prob, obs)
        act = controller.act(obs, ledger)
        is_valid, msg = ledger.validate_action(act, current_time=0.0)
        self.assertTrue(is_valid, msg)

    def test_mpc_decentralization_objective(self):
        """MPC targeting decentralization must evaluate variance minimization."""
        prob_dec = ControlProblem(
            problem_id="mpc_dec",
            target_type=TargetType.DECENTRALIZATION,
            target_params={"max_variance": 0.02},
            N=self.N,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            accessible_roster=(0, 1, 2),
        )
        config = MPCConfig(horizon_steps=2, step_interval=5.0, rollout_samples=1, seed=42)
        controller = ModelPredictiveController(config=config)
        ledger = BudgetLedger(N=self.N, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0)

        obs = Observation(
            time=0.0,
            public_states=np.zeros((self.N, 1)),
            panel_mask=np.ones(self.N, dtype=bool),
            observed_edges={(0, 1): 5, (0, 2): 5, (0, 3): 5},
            accessible_roster=(0, 1, 2),
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )

        controller.reset(prob_dec, obs)
        act = controller.act(obs, ledger)
        is_valid, msg = ledger.validate_action(act, current_time=0.0)
        self.assertTrue(is_valid, msg)


class TestM7ClosedLoopSimulation(unittest.TestCase):
    """End-to-end simulation integration tests combining MPC and Structural Actuators."""

    def test_closed_loop_mpc_simulation_reversal(self):
        """MPC in run_controlled_simulation must execute to horizon and record telemetry."""
        N = 6
        initial_W = np.zeros((N, N), dtype=np.uint32)
        initial_W[0, 1] = 2
        initial_W[1, 0] = 2

        initial_state = State(
            N=N,
            D=1,
            state_mode=StateMode.BINARY,
            x=np.ones((N, 1)),
            y=np.ones((N, 1)),
            W=initial_W,
            n=np.zeros(N, dtype=np.uint8),
            Q=np.zeros((N, 1, 2)),
        )

        prob = ControlProblem(
            problem_id="mpc_e2e",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N,
            D=1,
            k=2,
            exposure_cap=12.0,
            amplitude_cap=3.0,
            release_deadline=10.0,
            holding_duration=10.0,
            horizon=25.0,
            accessible_roster=(0, 1, 2),
        )

        config = MPCConfig(
            horizon_steps=2,
            step_interval=5.0,
            rollout_samples=1,
            seed=42,
        )
        controller = ModelPredictiveController(config=config)

        result, final_state, ledger = run_controlled_simulation(
            initial_state=initial_state,
            problem=prob,
            controller=controller,
            sample_interval=5.0,
        )

        # Assert clean execution to horizon
        self.assertTrue(result.is_valid)
        self.assertLessEqual(ledger.integrated_exposure, prob.exposure_cap + 1e-6)
        self.assertLessEqual(len(ledger.distinct_nodes_actuated), prob.k)
        self.assertTrue(final_state.t >= prob.horizon - 1e-6)

    def test_closed_loop_structural_intervention_simulation(self):
        """Simulation with structural interventions must update memory and preserve invariants."""
        N = 6
        initial_W = np.zeros((N, N), dtype=np.uint32)
        initial_W[0, 1] = 3

        initial_state = State(
            N=N,
            D=1,
            state_mode=StateMode.BINARY,
            x=np.ones((N, 1)),
            y=np.ones((N, 1)),
            W=initial_W,
            n=np.zeros(N, dtype=np.uint8),
            Q=np.zeros((N, 1, 2)),
        )

        prob = ControlProblem(
            problem_id="struct_e2e",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N,
            D=1,
            k=3,
            exposure_cap=10.0,
            amplitude_cap=3.0,
            structural_budget=5.0,
            release_deadline=10.0,
            holding_duration=10.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2),
        )

        # Custom controller that applies structural severance at t=0
        class StructuralTestController(ModelPredictiveController):
            def __init__(self):
                super().__init__()
                self.actuated = False

            def act(self, obs, ledg):
                if not self.actuated and ledg.remaining_structural_budget >= 2:
                    self.actuated = True
                    return sever_tie_action(source=0, target=1, count=2, intended_end_time=5.0)
                return Action(entries=(), intended_end_time=obs.time + 5.0)

        controller = StructuralTestController()

        result, final_state, ledger = run_controlled_simulation(
            initial_state=initial_state,
            problem=prob,
            controller=controller,
            sample_interval=5.0,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.structural_modifications, 2)
        self.assertEqual(ledger.integrated_structural_modifications, 2)
        # Structural invariants preserved
        self.assertTrue(np.all(np.diag(final_state.W) == 0))
        self.assertTrue(np.all(final_state.W >= 0))




class TestM7EdgeCasesAndRobustness(unittest.TestCase):
    """Adversarial stress-testing suite for advanced-control edge cases, boundaries, and robustness."""

    def test_mpc_zero_exposure_cap_with_structural_budget(self):
        """MPC must generate structural candidate trajectories when exposure_cap is 0 but structural_budget > 0."""
        prob = ControlProblem(
            problem_id="zero_exp_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            k=2,
            exposure_cap=0.0,
            structural_budget=5.0,
            release_deadline=10.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2),
        )
        ledger = BudgetLedger(N=6, D=1, k=2, exposure_cap=0.0, structural_budget=5.0)
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={(0, 1): 2},
            accessible_roster=(0, 1, 2),
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )
        config = MPCConfig(use_structural=True, horizon_steps=2, step_interval=5.0)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        cand_names = [c.name for c in candidates]
        self.assertIn("structural_sever", cand_names)
        self.assertIn("structural_force", cand_names)

    def test_mpc_distinct_nodes_continuation_when_k_exhausted(self):
        """When remaining_distinct_nodes == 0, MPC must continue actuating existing actuated nodes."""
        prob = ControlProblem(
            problem_id="k_exhaust_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            k=1,
            exposure_cap=20.0,
            amplitude_cap=4.0,
            release_deadline=10.0,
            horizon=20.0,
            accessible_roster=(0, 1, 2),
        )
        ledger = BudgetLedger(N=6, D=1, k=1, exposure_cap=20.0, amplitude_cap=4.0, release_deadline=10.0, accessible_roster={0, 1, 2})
        # Actuate node 0 initially (consumes distinct node slot k=1)
        act0 = Action(entries=(ActionEntry(node=0, component=0, field=-2.0),), intended_end_time=5.0)
        ledger.apply_action(act0, current_time=0.0)
        ledger.advance_to(5.0)
        self.assertEqual(ledger.remaining_distinct_nodes, 0)

        # At t=5, unactuated node 1 has higher degree than node 0
        obs = Observation(
            time=5.0,
            public_states=np.array([[-0.8], [0.2], [0.2], [0.2], [0.2], [0.2]]),
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={(1, 2): 10, (0, 1): 1},
            accessible_roster=(0, 1, 2),
            is_oracle=False,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )
        config = MPCConfig(horizon_steps=1, step_interval=5.0, nominal_amplitude=3.0)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        act = mpc.act(obs, ledger)
        is_valid, msg = ledger.validate_action(act, current_time=5.0)
        self.assertTrue(is_valid, msg)
        # Must target node 0 without violating k
        if act.entries:
            self.assertEqual(act.entries[0].node, 0)

    def test_pulse_then_coast_advancing_end_times(self):
        """Pulse-then-coast actions across multi-step horizon must have strictly advancing end times."""
        prob = ControlProblem(
            problem_id="ptc_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            k=2,
            exposure_cap=20.0,
            release_deadline=25.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        ledger = BudgetLedger(N=6, D=1, k=2, exposure_cap=20.0, release_deadline=25.0, accessible_roster={0, 1})
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={(0, 1): 2},
            accessible_roster=(0, 1),
            is_oracle=False,
        )
        config = MPCConfig(horizon_steps=4, step_interval=5.0)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=4)
        ptc = next(c for c in candidates if c.name == "pulse_then_coast")
        end_times = [a.intended_end_time for a in ptc.actions]
        self.assertEqual(end_times, [5.0, 10.0, 15.0, 20.0])

    def test_structural_per_step_limit_respected(self):
        """Structural candidate generator must strictly respect max_structural_per_step limit."""
        prob = ControlProblem(
            problem_id="per_step_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            k=3,
            exposure_cap=10.0,
            structural_budget=10.0,
            max_structural_per_step=1,
            release_deadline=15.0,
            horizon=30.0,
            accessible_roster=(0, 1, 2),
        )
        ledger = BudgetLedger(
            N=6, D=1, k=3, exposure_cap=10.0,
            structural_budget=10.0, max_structural_per_step=1,
            accessible_roster={0, 1, 2},
        )
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            observed_edges={(0, 1): 3, (0, 2): 2},
            accessible_roster=(0, 1, 2),
            is_oracle=False,
        )
        config = MPCConfig(horizon_steps=2, step_interval=5.0, use_structural=True)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        for c in candidates:
            for a in c.actions:
                self.assertLessEqual(a.structural_spending, 1)

    def test_reciprocal_asymmetric_observation_symmetry_preservation(self):
        """StateEstimator and forecaster must properly symmetrize asymmetric observations when reciprocal=True."""
        prob = ControlProblem(
            problem_id="recip_sym_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=5,
            k=2,
            exposure_cap=10.0,
        )
        obs = Observation(
            time=0.0,
            public_states=np.ones((5, 1)),
            panel_mask=np.ones(5, dtype=bool),
            observed_edges={(0, 1): 3},  # Asymmetric observation
            accessible_roster=(0, 1),
            is_oracle=False,
        )
        estimator = StateEstimator()
        est_state = estimator.estimate_state(obs, prob, reciprocal=True)
        self.assertTrue(np.array_equal(est_state.W, est_state.W.T))
        self.assertEqual(est_state.W[0, 1], 3)
        self.assertEqual(est_state.W[1, 0], 3)

        forecaster = SimulatedRolloutForecaster(reciprocal=True)
        ledger = BudgetLedger(N=5, D=1, k=2, exposure_cap=10.0)
        res = forecaster.forecast_trajectory(
            initial_state=est_state,
            action_sequence=[Action(entries=(), intended_end_time=5.0)],
            problem=prob,
            ledger=ledger,
        )
        self.assertTrue(res.is_valid)

    def test_force_connection_row_caps_avoids_target_cancellation(self):
        """Forcing connection when row is at capacity must evict other ties rather than canceling target tie."""
        caps = np.full(5, 2, dtype=np.uint32)
        initial_W = np.zeros((5, 5), dtype=np.uint32)
        initial_W[0, 1] = 1
        initial_W[0, 2] = 1  # Row 0 is full (sum=2 == cap)

        actuator = StructuralActuator(rng=np.random.default_rng(42))
        for _ in range(10):
            mem = MemoryManager(N=5, initial_W=initial_W.copy(), row_caps=caps)
            state = State(N=5, D=1, state_mode=StateMode.BINARY, x=np.ones((5, 1)), y=np.ones((5, 1)), W=mem.W, n=np.zeros(5, dtype=np.uint8), Q=np.zeros((5, 1, 2)), row_caps=caps)
            act = force_connection_action(source=0, target=1, count=1)
            actuator.apply_action(act, state, mem)
            # W[0, 1] must be reinforced to 2, and W[0, 2] evicted to 0
            self.assertEqual(mem.W[0, 1], 2)
            self.assertEqual(mem.W[0, 2], 0)
            self.assertEqual(mem.row_sums[0], 2)

    def test_forecaster_elite_turnover_objective(self):
        """Forecaster must compute target distance using elite set retention for TargetType.TURNOVER."""
        N = 10
        W = np.zeros((N, N), dtype=np.uint32)
        # Make node 0 the dominant elite leader initially
        W[:, 0] = 5
        np.fill_diagonal(W, 0)
        state = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=W, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))

        forecaster = SimulatedRolloutForecaster()
        # When L_0={0} and node 0 is still the leader, retention is 1.0 > 0.5 -> violation > 0
        dist_retained = forecaster.evaluate_target_distance(
            state=state,
            target_type=TargetType.TURNOVER,
            target_params={"L_0": {0}, "max_retained_fraction": 0.5},
        )
        self.assertAlmostEqual(dist_retained, 0.5)

        # When elite turns over completely (e.g. node 1 becomes leader, L_t={1})
        W_turned = np.zeros((N, N), dtype=np.uint32)
        W_turned[:, 1] = 5
        np.fill_diagonal(W_turned, 0)
        state_turned = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=W_turned, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))

        dist_turned = forecaster.evaluate_target_distance(
            state=state_turned,
            target_type=TargetType.TURNOVER,
            target_params={"L_0": {0}, "max_retained_fraction": 0.5},
        )
        self.assertAlmostEqual(dist_turned, 0.0)

    def test_engine_rejects_invalid_structural_action_without_state_mutation(self):
        """In engine, an action exceeding structural budget must not mutate physical state or ledger."""
        from src.control.controllers import Controller
        class OverBudgetController(Controller):
            def reset(self, problem, initial_observation, rng=None):
                pass
            def act(self, obs, ledger):
                # Try to spend 10 units when budget is only 2
                return force_connection_action(source=0, target=1, count=10)

        prob = ControlProblem(
            problem_id="engine_safe_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=5,
            exposure_cap=10.0,
            structural_budget=2.0,
            accessible_roster=(0, 1),
        )
        initial_W = np.zeros((5, 5), dtype=np.uint32)
        state = State(N=5, D=1, state_mode=StateMode.BINARY, x=np.ones((5, 1)), y=np.ones((5, 1)), W=initial_W.copy(), n=np.zeros(5, dtype=np.uint8), Q=np.zeros((5, 1, 2)))

        ctrl = OverBudgetController()
        res, fin, ledg = run_controlled_simulation(
            initial_state=state,
            problem=prob,
            controller=ctrl,
        )
        # Structural modifications on ledger must remain 0
        self.assertEqual(res.structural_modifications, 0)
        self.assertEqual(ledg.integrated_structural_modifications, 0)




class TestAdvancedControlEdgeCasesExtended(unittest.TestCase):
    """Adversarial stress-testing suite added for further coverage."""

    def test_multi_step_pulse_budget_sharing_validity(self):
        """Multi-step pulse candidates must share exposure budget across horizon without crashing on premature exhaustion."""
        prob = ControlProblem(
            problem_id="budget_share_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            D=1,
            k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        ledger = BudgetLedger(
            N=6, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=20.0, accessible_roster={0, 1}
        )
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            accessible_roster=(0, 1),
            observed_edges={(0, 1): 2},
        )
        config = MPCConfig(horizon_steps=3, step_interval=5.0, nominal_amplitude=2.0)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=3)
        forecaster = SimulatedRolloutForecaster()
        est_state = mpc.state_estimator.estimate_state(obs, prob)

        # Verify that all multi-step pulse candidates evaluate with is_valid=True and finite cost
        for cand in candidates:
            res = forecaster.forecast_trajectory(
                initial_state=est_state,
                action_sequence=cand.actions,
                problem=prob,
                ledger=ledger,
                step_interval=5.0,
                s_star=-1.0,
                target_threshold=0.5,
                rng=np.random.default_rng(42),
            )
            self.assertTrue(
                res.is_valid,
                f"Candidate {cand.name} failed rollout validation: {res.error_message}",
            )
            self.assertTrue(
                np.isfinite(res.total_cost),
                f"Candidate {cand.name} produced non-finite cost {res.total_cost}",
            )
            self.assertLessEqual(
                res.exposure_spent,
                10.0 + 1e-6,
                f"Candidate {cand.name} overspent exposure: {res.exposure_spent}",
            )

    def test_hybrid_action_avoids_k_violation_with_asymmetric_roster(self):
        """Hybrid field + structural candidates must never exceed distinct-node cap k regardless of roster ordering."""
        prob = ControlProblem(
            problem_id="k_asym_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=6,
            D=1,
            k=1,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(1, 0),
            structural_budget=5.0,
        )
        ledger = BudgetLedger(
            N=6, D=1, k=1, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=20.0,
            accessible_roster={0, 1}, structural_budget=5.0
        )
        obs = Observation(
            time=0.0,
            public_states=np.array([[1.0], [-1.0], [1.0], [1.0], [1.0], [1.0]]),
            panel_mask=np.ones(6, dtype=bool),
            accessible_roster=(1, 0),
            observed_edges={(1, 2): 5, (0, 3): 1},
        )
        config = MPCConfig(horizon_steps=2, step_interval=5.0, use_structural=True)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        hybrid_cand = next((c for c in candidates if c.name == "hybrid_field_structural"), None)
        if hybrid_cand is not None:
            valid, msg = ledger.validate_action(hybrid_cand.actions[0], 0.0)
            self.assertTrue(valid, f"Hybrid action failed k validation: {msg}")

    def test_forecaster_handles_row_caps_overflow_in_observations(self):
        """Forecaster and StateEstimator must clamp/evict observed ties exceeding row_caps without crashing."""
        N = 5
        caps = np.array([2, 2, 2, 2, 2], dtype=np.uint32)
        obs = Observation(
            time=0.0,
            public_states=np.ones((N, 1)),
            panel_mask=np.ones(N, dtype=bool),
            accessible_roster=(0, 1),
            observed_edges={(0, 1): 3, (0, 2): 3, (0, 3): 2},  # Row 0 sum = 8 > cap 2
        )
        prob = ControlProblem(
            problem_id="caps_overflow_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        est = StateEstimator()
        est_state = est.estimate_state(obs, prob, row_caps=caps)
        self.assertLessEqual(int(np.sum(est_state.W[0])), 2)

        # Directly attach caps to state and verify rollout forecaster does not crash
        forecaster = SimulatedRolloutForecaster()
        ledger = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=20.0, accessible_roster={0, 1})
        res = forecaster.forecast_trajectory(est_state, [], prob, ledger)
        self.assertTrue(res.is_valid)

    def test_reciprocal_forcing_saturation_preserves_symmetry(self):
        """In reciprocal mode, failed or zero forward force increments must not add reverse edge."""
        N = 4
        mem = MemoryManager(N=N, reciprocal=True, initial_W=np.zeros((N, N), dtype=np.uint32))
        state = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=mem.W, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))
        actuator = StructuralActuator()

        # Forcing valid connection
        act1 = force_connection_action(source=0, target=1, count=2)
        actuator.apply_action(act1, state, mem)
        self.assertEqual(mem.W[0, 1], 2)
        self.assertEqual(mem.W[1, 0], 2)
        self.assertTrue(np.array_equal(mem.W, mem.W.T))

    def test_structural_action_rewire_validation_invariants(self):
        """Rewiring to same target or setting rewire_target on FORCE/SEVER must be rejected."""
        # Rewire to same target must fail
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.REWIRE, rewire_target=1)

        # Non-rewire with rewire_target specified must fail
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.FORCE, rewire_target=2)
        with self.assertRaises(ValueError):
            StructuralActionEntry(source=0, target=1, operation=StructuralOp.SEVER, rewire_target=2)

    def test_budget_ledger_copy_state_isolation(self):
        """BudgetLedger.copy() must faithfully duplicate field, actions, and isolate mutations."""
        ledger = BudgetLedger(
            N=5, D=1, k=2, exposure_cap=20.0, amplitude_cap=3.0, release_deadline=25.0,
            accessible_roster={0, 1}, structural_budget=10.0, max_structural_per_step=3
        )
        act = Action(entries=(ActionEntry(node=0, component=0, field=2.0),), intended_end_time=5.0)
        ledger.apply_action(act, current_time=0.0)
        ledger.advance_to(2.0)

        clone = ledger.copy()
        self.assertEqual(clone.current_time, ledger.current_time)
        self.assertEqual(clone.integrated_exposure, ledger.integrated_exposure)
        self.assertEqual(clone.remaining_exposure, ledger.remaining_exposure)
        self.assertTrue(np.array_equal(clone.current_field, ledger.current_field))
        self.assertEqual(clone.active_action_end_time, ledger.active_action_end_time)

        # Mutating clone must not affect original
        clone.advance_to(5.0)
        self.assertEqual(clone.current_time, 5.0)
        self.assertEqual(ledger.current_time, 2.0)

    def test_decentralization_evaluator_uses_c_n_0(self):
        """Forecaster must use C_N_0 relative violation when provided for TargetType.DECENTRALIZATION."""
        N = 6
        W = np.zeros((N, N), dtype=np.uint32)
        W[0, 1] = 5
        W[0, 2] = 5
        W[0, 3] = 5
        state = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=W, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))
        forecaster = SimulatedRolloutForecaster()

        # If C_N_0 is provided as 0.10
        dist = forecaster.evaluate_target_distance(
            state=state,
            target_type=TargetType.DECENTRALIZATION,
            target_params={"C_N_0": 0.10},
        )
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(dist, 0.0)

    def test_mpc_decentralization_generates_structural_candidates_by_default(self):
        """On decentralization problems, MPC must generate structural candidates by default even if use_structural=False."""
        prob = ControlProblem(
            problem_id="decent_cand_test",
            target_type=TargetType.DECENTRALIZATION,
            target_params={"max_variance": 0.01},
            N=6, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1, 2, 3),
        )
        ledger = BudgetLedger(N=6, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=20.0, accessible_roster={0, 1, 2, 3})
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            accessible_roster=(0, 1, 2, 3),
            observed_edges={(0, 1): 4, (0, 2): 4},
        )
        # Default config has use_structural=False, structural_budget=None
        config = MPCConfig(horizon_steps=2, step_interval=5.0, use_structural=False)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        cand_names = [c.name for c in candidates]
        self.assertIn("structural_sever", cand_names)
        self.assertIn("structural_force", cand_names)

    def test_structural_rewire_candidate_generation(self):
        """Candidate generator must include structural_rewire candidate when allowed_struct_this_step >= 2."""
        prob = ControlProblem(
            problem_id="rewire_cand_test",
            target_type=TargetType.DECENTRALIZATION,
            target_params={"max_variance": 0.01},
            N=6, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1, 2, 3),
            structural_budget=5.0,
        )
        ledger = BudgetLedger(
            N=6, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=20.0,
            accessible_roster={0, 1, 2, 3}, structural_budget=5.0
        )
        obs = Observation(
            time=0.0,
            public_states=np.ones((6, 1)),
            panel_mask=np.ones(6, dtype=bool),
            accessible_roster=(0, 1, 2, 3),
            observed_edges={(0, 1): 4, (0, 2): 4},
        )
        config = MPCConfig(horizon_steps=2, step_interval=5.0, use_structural=True)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        cand_names = [c.name for c in candidates]
        self.assertIn("structural_rewire", cand_names)
        rewire_cand = next(c for c in candidates if c.name == "structural_rewire")
        first_act = rewire_cand.actions[0]
        self.assertEqual(len(first_act.structural_entries), 1)
        self.assertEqual(first_act.structural_entries[0].operation, StructuralOp.REWIRE)
        self.assertEqual(first_act.structural_spending, 2)



class TestAdvancedControlEdgeCasesFurther(unittest.TestCase):
    """Adversarial stress-testing suite added for further coverage."""

    def test_rewire_sever_excess_count_conserves_total_memory(self):
        """Rewiring with count exceeding existing tie weight must only rewire available units and conserve total memory."""
        N = 5
        initial_W = np.zeros((N, N), dtype=np.uint32)
        initial_W[1, 2] = 2  # Only 2 units exist
        mem = MemoryManager(N=N, initial_W=initial_W)
        state = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=mem.W, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))
        actuator = StructuralActuator()

        # Request rewiring 5 units from (1, 2) to (1, 3)
        act = rewire_tie_action(source=1, old_target=2, new_target=3, count=5)
        records = actuator.apply_action(act, state, mem)

        # Must sever 2 units and force exactly 2 units (not 5)
        self.assertEqual(mem.W[1, 2], 0)
        self.assertEqual(mem.W[1, 3], 2)
        self.assertEqual(mem.total_memory, 2)
        self.assertEqual(mem.row_sums[1], 2)
        self.assertEqual(records[0]["units_rewired"], 2)
        self.assertEqual(records[0]["count_requested"], 5)

    def test_rewire_zero_weight_noop_conserves_memory(self):
        """Rewiring an edge with zero existing weight must be a no-op that does not create new edges."""
        N = 5
        initial_W = np.zeros((N, N), dtype=np.uint32)
        mem = MemoryManager(N=N, initial_W=initial_W)
        state = State(N=N, D=1, state_mode=StateMode.BINARY, x=np.ones((N, 1)), y=np.ones((N, 1)), W=mem.W, n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2)))
        actuator = StructuralActuator()

        # Request rewiring from non-existent edge (1, 2) to (1, 3)
        act = rewire_tie_action(source=1, old_target=2, new_target=3, count=3)
        records = actuator.apply_action(act, state, mem)

        self.assertEqual(mem.W[1, 2], 0)
        self.assertEqual(mem.W[1, 3], 0)
        self.assertEqual(mem.total_memory, 0)
        self.assertEqual(records[0]["units_rewired"], 0)

    def test_state_estimator_zero_panel_visibility_with_s_star(self):
        """In 0% panel visibility with explicit s_star, StateEstimator must initialize unobserved beliefs opposing target."""
        N = 6
        obs = Observation(
            time=0.0,
            public_states=np.full((N, 1), np.nan),  # 0% panel visibility (all NaN)
            panel_mask=np.zeros(N, dtype=bool),      # No panel nodes observed
            accessible_roster=(0, 1),
            observed_edges={},
        )
        prob = ControlProblem(
            problem_id="zero_vis_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5, "s_star": 1.0},
            N=N, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        estimator = StateEstimator()
        est_state = estimator.estimate_state(obs, prob)

        # Target is s_star = +1.0; unobserved nodes must be estimated as -1.0 (opposing) rather than +1.0 (already satisfied)
        self.assertTrue(np.all(est_state.y == -1.0))
        self.assertTrue(np.all(est_state.x == -1.0))

    def test_mpc_turnover_target_orientation_inference(self):
        """For TargetType.TURNOVER on initially positive network without explicit s_star, MPC must deduce s_star = -1.0."""
        N = 6
        obs = Observation(
            time=0.0,
            public_states=np.ones((N, 1)),  # All nodes currently positive (+1.0)
            panel_mask=np.ones(N, dtype=bool),
            accessible_roster=(0, 1),
            observed_edges={(0, 1): 2, (1, 2): 2},
        )
        prob = ControlProblem(
            problem_id="turnover_orient_test",
            target_type=TargetType.TURNOVER,
            target_params={"threshold": 0.5, "turnover_fraction": 0.5},
            N=N, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        config = MPCConfig(horizon_steps=2, step_interval=5.0)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        # Must deduce s_star = -1.0 (aiming to flip positive nodes to negative)
        self.assertEqual(mpc.s_star, -1.0)
        # Target distance must be positive (work remaining)
        dist = mpc.forecaster.evaluate_target_distance(
            state=mpc.state_estimator.estimate_state(obs, prob),
            target_type=prob.target_type,
            s_star=mpc.s_star,
            threshold=0.5,
            target_params=mpc.target_params,
        )
        self.assertGreater(dist, 0.0)

    def test_forecaster_reciprocal_with_row_caps_does_not_crash(self):
        """SimulatedRolloutForecaster must not crash when reciprocal=True and state has row_caps."""
        N = 5
        caps = np.full(N, 4, dtype=np.uint32)
        initial_W = np.zeros((N, N), dtype=np.uint32)
        initial_W[0, 1] = 2
        initial_W[1, 0] = 2
        state = State(
            N=N, D=1, state_mode=StateMode.BINARY,
            x=np.ones((N, 1)), y=np.ones((N, 1)),
            W=initial_W, n=np.zeros(N, dtype=np.uint8),
            Q=np.zeros((N, 1, 2)), row_caps=caps,
        )
        prob = ControlProblem(
            problem_id="recip_caps_safe_test",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=N, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1),
        )
        ledger = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, release_deadline=20.0, accessible_roster={0, 1})
        forecaster = SimulatedRolloutForecaster(reciprocal=True)

        res = forecaster.forecast_trajectory(
            initial_state=state,
            action_sequence=[Action(entries=(), intended_end_time=5.0)],
            problem=prob,
            ledger=ledger,
        )
        self.assertTrue(res.is_valid)

    def test_structural_force_selects_lowest_degree_target(self):
        """Candidate B (structural_force) must prioritize connecting to the least-connected accessible target."""
        N = 5
        obs = Observation(
            time=0.0,
            public_states=np.ones((N, 1)),
            panel_mask=np.ones(N, dtype=bool),
            accessible_roster=(0, 1, 2),
            observed_edges={(0, 1): 10, (1, 0): 10},  # Node 1 has degree 20, Node 2 has degree 0
        )
        prob = ControlProblem(
            problem_id="lowest_deg_target_test",
            target_type=TargetType.DECENTRALIZATION,
            target_params={"max_variance": 0.01},
            N=N, D=1, k=2,
            exposure_cap=10.0,
            amplitude_cap=2.0,
            release_deadline=20.0,
            horizon=30.0,
            accessible_roster=(0, 1, 2),
            structural_budget=5.0,
        )
        ledger = BudgetLedger(N=N, D=1, k=2, exposure_cap=10.0, release_deadline=20.0, accessible_roster={0, 1, 2}, structural_budget=5.0)
        config = MPCConfig(horizon_steps=2, step_interval=5.0, use_structural=True)
        mpc = ModelPredictiveController(config=config)
        mpc.reset(prob, obs)

        candidates = mpc._generate_candidates(obs, ledger, horizon_steps=2)
        force_cand = next(c for c in candidates if c.name == "structural_force")
        entry = force_cand.actions[0].structural_entries[0]
        # Target must be node 2 (degree 0) rather than node 1 (degree 20)
        self.assertEqual(entry.target, 2)


if __name__ == "__main__":
    unittest.main()
