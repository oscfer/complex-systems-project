"""Unit tests for Target Evaluator and Exact Holding Clocks.

Tests:
1. Reversal target eligibility checks, target sign determination, and condition evaluation.
2. Reversal terminal target violation d_G computation.
3. Decentralization target eligibility, active fraction, WCC connectivity, and target condition.
4. Decentralization terminal target violation d_G computation.
5. Exact 5 reference trace evaluations matching validate_control_plan.py:
   - Pre-release entry: target true before release does not start timer.
   - Interruption and re-entry: violating event resets timer; re-entry restarts it.
   - Conservative tie convention: violating event exactly at deadline prevents completion.
   - Horizon censoring: incomplete hold at horizon H is censored (success=False).
   - Completion exactly at horizon: completion certified at H.
6. Subsequent relapse detection after certification.
"""

import unittest
import numpy as np

from src.control.contracts import TargetType
from src.control.evaluator import (
    TargetDefinition,
    TargetEvaluator,
    evaluate_holding_trace,
)
from src.core.state import State, StateMode


class TestTargetEvaluator(unittest.TestCase):
    """Test suite for TargetEvaluator, predicates, and exact holding timers."""

    def test_reversal_eligibility_and_conditions(self):
        """Reversal target checks initial m_y(0), m_x(0), and simultaneous alignment."""
        # 1. Eligible positive state: m_y = 0.8, m_x = 0.8 -> target sign s_* = -1
        state_pos = State.create_empty(N=10, D=1, state_mode=StateMode.BINARY)
        state_pos.y.fill(1.0)
        state_pos.x.fill(1.0)

        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.REVERSAL,
            initial_state=state_pos,
            holding_duration=50.0,
            horizon=270.0,
            release_deadline=20.0,
        )
        self.assertTrue(evaluator.defn.is_eligible)
        self.assertEqual(evaluator.defn.s_star, -1.0)
        # Initially condition is False because target is negative beliefs
        self.assertFalse(evaluator.evaluate_target_condition(state_pos))

        # 2. Ineligible state: m_y = 0.2 < 0.5
        state_weak = State.create_empty(N=10, D=1, state_mode=StateMode.BINARY)
        state_weak.y.fill(1.0)
        state_weak.y[0:4] = -1.0  # mean = (6 - 4)/10 = 0.2
        eval_weak = TargetEvaluator.create_for_state(
            target_type=TargetType.REVERSAL,
            initial_state=state_weak,
        )
        self.assertFalse(eval_weak.defn.is_eligible)
        self.assertIn("0.2000 < 0.5", eval_weak.defn.ineligibility_reason)

        # 3. Successful reversal condition: both x and y have mean <= -0.5 (since s_* = -1)
        state_reversed = State.create_empty(N=10, D=1, state_mode=StateMode.BINARY)
        state_reversed.y.fill(-1.0)
        state_reversed.x.fill(-1.0)
        self.assertTrue(evaluator.evaluate_target_condition(state_reversed))

        # Partial reversal: only y is reversed, x is not -> False
        state_partial = State.create_empty(N=10, D=1, state_mode=StateMode.BINARY)
        state_partial.y.fill(-1.0)
        state_partial.x.fill(1.0)
        self.assertFalse(evaluator.evaluate_target_condition(state_partial))

    def test_reversal_terminal_target_violation(self):
        """d_G = max(0, 0.5 - s_* * m_x, 0.5 - s_* * m_y)."""
        state = State.create_empty(N=10, D=1, state_mode=StateMode.BINARY)
        state.y.fill(1.0)
        state.x.fill(1.0)
        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.REVERSAL,
            initial_state=state,
        )
        # s_* = -1.0. If m_x = 0.0, m_y = -0.2:
        # 0.5 - (-1.0 * 0.0) = 0.5. 0.5 - (-1.0 * -0.2) = 0.5 - 0.2 = 0.3.
        # d_G = max(0, 0.5, 0.3) = 0.5.
        state.x.fill(0.0)
        state.y.fill(-0.2)
        d_G = evaluator.compute_terminal_target_violation(state)
        self.assertAlmostEqual(d_G, 0.5, places=6)

    def test_five_hand_calculated_trace_checks(self):
        """Verify the 5 exact reference trace cases matching validate_control_plan.py."""
        cases = [
            ([(0, True)], 1, 2, 10, 3),
            ([(0, True), (2, False), (3, True)], 1, 2, 10, 5),
            ([(0, True), (3, False)], 1, 2, 10, None),
            ([(0, False), (9, True)], 0, 2, 10, None),
            ([(0, True)], 0, 2, 2, 2),
        ]
        for trace, release, duration, horizon, expected in cases:
            res = evaluate_holding_trace(trace, release, duration, horizon)
            self.assertEqual(res, expected, f"Failed on case: trace={trace}, release={release}")

    def test_evaluator_holding_clock_interruption_and_relapse(self):
        """TargetEvaluator holding clock resets on interruption and flags relapse."""
        defn = TargetDefinition(
            target_type=TargetType.REVERSAL,
            holding_duration=2.0,
            horizon=10.0,
            release_deadline=1.0,
            s_star=-1.0,
            is_eligible=True,
        )
        evaluator = TargetEvaluator(target_definition=defn)

        state_true = State.create_empty(N=2, D=1, state_mode=StateMode.BINARY)
        state_true.x.fill(-1.0)
        state_true.y.fill(-1.0)

        state_false = State.create_empty(N=2, D=1, state_mode=StateMode.BINARY)
        state_false.x.fill(1.0)
        state_false.y.fill(1.0)

        # Before release (t=0.5): state is true, but holding timer must NOT start
        evaluator.process_state_update(state_true, current_time=0.5, is_released=False)
        self.assertIsNone(evaluator.t_start_hold)

        # At release (t=1.0): holding timer starts
        evaluator.process_state_update(state_true, current_time=1.0, is_released=True)
        self.assertEqual(evaluator.t_start_hold, 1.0)
        self.assertFalse(evaluator.is_success)

        # At t=2.0: condition interrupted (violating event) -> resets timer
        evaluator.process_state_update(state_false, current_time=2.0, is_released=True)
        self.assertIsNone(evaluator.t_start_hold)
        self.assertFalse(evaluator.is_success)

        # At t=3.0: re-entry -> restarts timer at 3.0
        evaluator.process_state_update(state_true, current_time=3.0, is_released=True)
        self.assertEqual(evaluator.t_start_hold, 3.0)
        self.assertFalse(evaluator.is_success)

        # At t=5.0: duration of 2.0 completed! -> success certified
        evaluator.process_state_update(state_true, current_time=5.0, is_released=True)
        self.assertTrue(evaluator.is_success)
        self.assertEqual(evaluator.completion_time, 5.0)

        # At t=7.0: state relapses to false -> relapsed flag set
        evaluator.process_state_update(state_false, current_time=7.0, is_released=True)
        self.assertTrue(evaluator.relapsed)

        # Finalize at horizon 10.0
        result = evaluator.finalize(state_false, current_time=10.0, expenditure=5.0, distinct_nodes_used=2)
        self.assertTrue(result.success)
        self.assertEqual(result.completion_time, 5.0)
        self.assertEqual(result.restricted_time, 5.0)
        self.assertTrue(result.relapsed)


    def test_evaluator_holding_completion_before_relapse_event(self):
        """Violating event arriving strictly after holding deadline certifies completion and sets relapsed."""
        defn = TargetDefinition(
            target_type=TargetType.REVERSAL,
            holding_duration=5.0,
            horizon=30.0,
            release_deadline=10.0,
            s_star=-1.0,
            is_eligible=True,
        )
        evaluator = TargetEvaluator(target_definition=defn)

        state_true = State.create_empty(N=2, D=1, state_mode=StateMode.BINARY)
        state_true.x.fill(-1.0)
        state_true.y.fill(-1.0)

        state_false = State.create_empty(N=2, D=1, state_mode=StateMode.BINARY)
        state_false.x.fill(1.0)
        state_false.y.fill(1.0)

        # At release t=10.0: hold starts
        evaluator.process_state_update(state_true, current_time=10.0, is_released=True)
        self.assertEqual(evaluator.t_start_hold, 10.0)

        # Violating event occurs at t=15.5 (> 10.0 + 5.0 = 15.0)
        evaluator.process_state_update(state_false, current_time=15.5, is_released=True)
        self.assertTrue(evaluator.is_success)
        self.assertEqual(evaluator.completion_time, 15.0)
        self.assertTrue(evaluator.relapsed)
        self.assertIsNone(evaluator.t_start_hold)


    def test_turnover_target_eligibility_and_conditions(self):
        """Turnover target computes initial top-10% set L_0, checks ambiguity and retention."""
        N = 20
        state = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        # Create strong connected structure
        for i in range(N):
            state.W[i, (i + 1) % N] = 2
            state.W[i, (i + 2) % N] = 1

        # Make node 0 and node 1 high-attention hubs (top 10% for N=20 is 2 nodes)
        for i in range(N):
            if i not in (0, 1):
                state.W[i, 0] += 5
                state.W[i, 1] += 3

        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.TURNOVER,
            initial_state=state,
            holding_duration=10.0,
            horizon=50.0,
        )

        self.assertTrue(evaluator.defn.is_eligible)
        self.assertIsNotNone(evaluator.defn.L_0)
        self.assertEqual(len(evaluator.defn.L_0), 2)
        self.assertIn(0, evaluator.defn.L_0)
        self.assertIn(1, evaluator.defn.L_0)

        # Initially, retention is 1.0 > 0.5 -> condition is False
        self.assertFalse(evaluator.evaluate_target_condition(state))

        # Now replace node 0 and 1 with node 5 and 6 as new hubs (turnover!)
        state_turned = state.copy()
        state_turned.W[:, 0] = 0
        state_turned.W[:, 1] = 0
        for i in range(N):
            if i not in (5, 6):
                state_turned.W[i, 5] += 5
                state_turned.W[i, 6] += 3
        # Ensure connectivity and active fraction hold
        for i in range(N):
            state_turned.W[i, (i + 1) % N] = max(1, state_turned.W[i, (i + 1) % N])

        # New top-10% should be {5, 6}, retained fraction |{0, 1} & {5, 6}| / 2 = 0 <= 0.5
        self.assertTrue(evaluator.evaluate_target_condition(state_turned))

    def test_turnover_ambiguous_rank_cutoff_ineligibility(self):
        """When top-10% set has many ties exceeding 20% of N, target is marked ineligible."""
        N = 20
        state = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        for i in range(N):
            state.W[i, (i + 1) % N] = 1
            state.W[i, 0] += 5  # Node 0 is huge hub
            if i not in (1, 2, 3, 4, 5):
                for j in (1, 2, 3, 4, 5):
                    state.W[i, j] += 2  # Nodes 1..5 have identical received attention

        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.TURNOVER,
            initial_state=state,
        )
        # Nodes 1..5 tie at the cutoff, making |L_0| = 5 > 20% of N=20 (4) -> ineligible
        self.assertFalse(evaluator.defn.is_eligible)
        self.assertIn("Ambiguous rank cutoff", evaluator.defn.ineligibility_reason)



    def test_decentralization_target_eligibility_and_conditions(self):
        """Decentralization target checks initial C_N, active fraction, WCC, and condition C_N <= 0.5 C_N(0)."""
        N = 20
        state = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        # Ring topology plus node 0 as high-attention hub
        for i in range(N):
            state.W[i, (i + 1) % N] = 1
            if i != 0:
                state.W[i, 0] += 5

        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.DECENTRALIZATION,
            initial_state=state,
            holding_duration=10.0,
            horizon=50.0,
        )
        self.assertTrue(evaluator.defn.is_eligible)
        self.assertIsNotNone(evaluator.defn.C_N_0)
        self.assertGreaterEqual(evaluator.defn.C_N_0, 0.002)

        # Initially, C_N(0) > 0.5 * C_N(0) -> condition is False
        self.assertFalse(evaluator.evaluate_target_condition(state))

        # Decentralize: redistribute hub attention uniformly across nodes
        state_decentralized = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        for i in range(N):
            state_decentralized.W[i, (i + 1) % N] = 2
            state_decentralized.W[i, (i + 2) % N] = 2

        self.assertTrue(evaluator.evaluate_target_condition(state_decentralized))

        # Disconnected state: largest WCC < 0.8 -> False
        state_frag = state_decentralized.copy()
        state_frag.W[10:, :] = 0
        state_frag.W[:, 10:] = 0
        self.assertFalse(evaluator.evaluate_target_condition(state_frag))

    def test_decentralization_terminal_target_violation(self):
        """d_G computation for decentralization: surrogate distance to 0.5 C_N(0) and floors."""
        N = 20
        state = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        for i in range(N):
            state.W[i, (i + 1) % N] = 1
            if i != 0:
                state.W[i, 0] += 5

        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.DECENTRALIZATION,
            initial_state=state,
        )

        # Empty graph: undefined attention -> d_G = 10.0 per spec
        state_empty = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        d_G_empty = evaluator.compute_terminal_target_violation(state_empty)
        self.assertEqual(d_G_empty, 10.0)

        # Initial state: C_N / (0.5 * C_N(0)) - 1 = 2 - 1 = 1.0
        d_G_init = evaluator.compute_terminal_target_violation(state)
        self.assertAlmostEqual(d_G_init, 1.0, places=4)

    def test_compute_top_10_percent_set_empty_graph(self):
        """compute_top_10_percent_set must handle empty N=0 array gracefully."""
        from src.control.evaluator import compute_top_10_percent_set
        res = compute_top_10_percent_set(np.zeros((0, 0), dtype=int))
        self.assertEqual(res, set())

if __name__ == "__main__":
    unittest.main()
