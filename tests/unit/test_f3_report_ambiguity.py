"""Regression tests: missing/tied public-report semantics and direction knowledge contract.

Acceptance criteria:
1. An empty non-focal public report must NOT be interpreted as majority 0.0:
   the report is missing/unavailable -> controller waits, no direction lock,
   no actuation budget consumed while waiting.
2. A tied public report (mean exactly 0) is ambiguous under the declared tie
   rule -> no direction lock, controller waits.
3. Negative sample -> target direction +1; positive sample -> target -1
   (the success contract: opposite initial public majority).
4. Missing focal panel already returns None (regression guard).
5. Delayed initial report: the locked direction is attributed to the report
   sample time, exported through effective_settings.
6. Hidden private states must not influence the restricted policy's action
   (no leak of x_private or hidden W).
7. Checkpoint/resume preserves the locked target sign exactly.
8. Evaluator target stays fixed from the TRUE initial public majority; the
   controller's inferred sign may differ (reported as an inference mistake).
"""

import copy
import unittest

import numpy as np

from src.control.contracts import ControlProblem, Observation, TargetType
from src.control.controllers import ObservableSelectorPolicy
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import SampledPublicObservationModel
from src.core.state import State, StateMode

N = 20


def _make_state(y_sign=+1.0, x_sign=+1.0, seed=12345):
    """Synthetic state with public beliefs y and private beliefs x."""
    rng = np.random.default_rng(seed)
    y = np.full((N, 1), y_sign * 0.8, dtype=np.float64)
    x = np.full((N, 1), x_sign * 0.7, dtype=np.float64)
    W = np.zeros((N, N), dtype=np.uint32)
    for i in range(N):
        j = (i + 1) % N
        W[i, j] = 5
        W[j, i] = 5
    return State(
        N=N,
        D=1,
        state_mode=StateMode.CONTINUOUS,
        x=x,
        y=y,
        W=W,
        n=np.zeros(N, dtype=np.uint8),
        Q=np.zeros((N, 1, 2), dtype=np.float64),
        t=0.0,
        opportunity_groups=np.zeros(N, dtype=int),
        K=rng.uniform(0.8, 1.2, size=N),
        tau_x=rng.uniform(0.5, 1.5, size=N),
        nu=rng.uniform(0.1, 0.3, size=N),
        g=rng.uniform(0.05, 0.15, size=N),
        b=np.zeros((N, 1), dtype=np.float64),
    )


def _make_problem(focal=None, N_=N):
    return ControlProblem(
        problem_id="f3_probe",
        target_type=TargetType.REVERSAL,
        target_params={"focal_indices": focal} if focal else {},
        N=N_,
        k=4,
        exposure_cap=10.0,
        amplitude_cap=4.0,
        release_deadline=20.0,
        holding_duration=100.0,
        horizon=320.0,
        accessible_roster=tuple(range(min(6, N_))),
    )


def _make_observation(panel_values, sample_time=0.0, delivery_time=1.0, N_=N):
    """panel_values: list of observed public values for nodes 0..len-1, or None for no panel."""
    panel_mask = np.zeros(N_, dtype=bool)
    public_states = np.full((N_, 1), np.nan)
    if panel_values is not None:
        n = len(panel_values)
        panel_mask[:n] = True
        public_states[:n, 0] = panel_values
    return Observation(
        time=delivery_time,
        public_states=public_states,
        panel_mask=panel_mask,
        observed_edges={},
        accessible_roster=tuple(range(min(6, N_))),
        group_labels=np.zeros(N_, dtype=int),
        sample_time=sample_time,
        delivery_time=delivery_time,
        remaining_exposure=10.0,
        remaining_distinct_nodes=4,
        current_spending_rate=0.0,
    )


def _make_ledger(problem):
    return BudgetLedger(
        N=problem.N,
        D=problem.D,
        k=problem.k,
        exposure_cap=problem.exposure_cap,
        amplitude_cap=problem.amplitude_cap,
        release_deadline=problem.release_deadline,
        accessible_roster=set(problem.accessible_roster),
    )


def _act_once(problem, obs, seed=7):
    """Reset policy, act once, return (action, controller, ledger)."""
    ctrl = ObservableSelectorPolicy(selector="random", seed=seed)
    ledger = _make_ledger(problem)
    ctrl.reset(problem, obs)
    action = ctrl.act(obs, ledger)
    return action, ctrl, ledger


class TestF3ReportAmbiguity(unittest.TestCase):
    """F3: missing/tied reports must be treated as unavailable, not as majority 0."""

    def test_empty_report_helper_returns_unavailable(self):
        """observed_public_mean must return None (not 0.0) when no finite values exist."""
        obs = _make_observation([np.nan, np.nan, np.nan, np.nan])
        self.assertIsNone(obs.observed_public_mean(0))
        obs_no_panel = _make_observation(None)
        self.assertIsNone(obs_no_panel.observed_public_mean(0))

    def test_empty_report_helper_still_averages_finite_values(self):
        """Finite values are still averaged (regression guard for existing behavior)."""
        obs = _make_observation([1.0, np.nan, -1.0])
        self.assertAlmostEqual(obs.observed_public_mean(0), 0.0)

    def test_empty_report_does_not_lock_direction(self):
        """Empty non-focal report: controller must wait, not select -1."""
        problem = _make_problem()
        obs = _make_observation([np.nan] * 4)
        action, ctrl, ledger = _act_once(problem, obs)
        self.assertIsNone(ctrl.s_star, "Empty report must not lock a target direction")
        self.assertEqual(len(action.entries), 0)
        self.assertFalse(action.release)
        # Waiting must not consume actuation budget
        self.assertAlmostEqual(ledger.integrated_exposure, 0.0, places=12)
        self.assertEqual(len(ledger.distinct_nodes_actuated), 0)
        self.assertFalse(ledger.is_released)

    def test_no_panel_does_not_lock_direction(self):
        """Fully empty panel (no panel members) behaves like a missing report."""
        problem = _make_problem()
        obs = _make_observation(None)
        action, ctrl, ledger = _act_once(problem, obs)
        self.assertIsNone(ctrl.s_star)
        self.assertEqual(len(action.entries), 0)
        self.assertAlmostEqual(ledger.integrated_exposure, 0.0, places=12)

    def test_tied_report_is_ambiguous_no_direction_lock(self):
        """Tie rule: mean exactly 0 is ambiguous -> wait, no direction lock."""
        problem = _make_problem()
        obs = _make_observation([1.0, -1.0, 1.0, -1.0])
        action, ctrl, ledger = _act_once(problem, obs)
        self.assertIsNone(ctrl.s_star, "Tied report must not lock a target direction")
        self.assertEqual(len(action.entries), 0)
        self.assertAlmostEqual(ledger.integrated_exposure, 0.0, places=12)
        self.assertEqual(len(ledger.distinct_nodes_actuated), 0)

    def test_negative_sample_locks_positive_target(self):
        """Negative initial public majority -> inferred target sign +1."""
        problem = _make_problem()
        obs = _make_observation([-1.0, -1.0, -1.0, -0.2])
        action, ctrl, _ = _act_once(problem, obs)
        self.assertIsNotNone(ctrl.s_star)
        self.assertGreater(ctrl.s_star, 0.0)
        self.assertEqual(len(action.entries), 2)
        for e in action.entries:
            self.assertGreater(e.field, 0.0)

    def test_positive_sample_locks_negative_target(self):
        """Positive initial public majority -> inferred target sign -1."""
        problem = _make_problem()
        obs = _make_observation([1.0, 1.0, 1.0, 0.2])
        action, ctrl, _ = _act_once(problem, obs)
        self.assertIsNotNone(ctrl.s_star)
        self.assertLess(ctrl.s_star, 0.0)
        for e in action.entries:
            self.assertLess(e.field, 0.0)

    def test_weak_nonzero_sample_locks_by_sign(self):
        """Weak but nonzero majority (|mean| < threshold) still locks by sign.

        The restricted policy only knows the sign of the observed majority;
        threshold gating belongs to the evaluator, not the controller.
        """
        problem = _make_problem()
        obs = _make_observation([0.2, 0.1, 0.05, 0.1])
        action, ctrl, _ = _act_once(problem, obs)
        self.assertIsNotNone(ctrl.s_star)
        self.assertLess(ctrl.s_star, 0.0)

    def test_missing_focal_panel_waits(self):
        """Focal-empty path already returns None (regression guard)."""
        problem = _make_problem(focal=(0, 1))
        obs = _make_observation([np.nan, np.nan, 1.0, 1.0])
        action, ctrl, ledger = _act_once(problem, obs)
        self.assertIsNone(ctrl.s_star)
        self.assertEqual(len(action.entries), 0)
        self.assertAlmostEqual(ledger.integrated_exposure, 0.0, places=12)

    def test_focal_panel_nonempty_locks_from_focal_mean(self):
        """When any focal node is observed, direction is inferred from focal mean."""
        problem = _make_problem(focal=(0, 1))
        obs = _make_observation([1.0, np.nan, -1.0, -1.0])
        action, ctrl, _ = _act_once(problem, obs)
        self.assertIsNotNone(ctrl.s_star)
        self.assertLess(ctrl.s_star, 0.0)

    def test_delayed_initial_report_records_sample_time(self):
        """Locked direction is attributed to the report's sample (snapshot) time."""
        problem = _make_problem()
        obs = _make_observation([1.0, 1.0, -1.0], sample_time=0.0, delivery_time=1.0)
        action, ctrl, _ = _act_once(problem, obs)
        self.assertLess(ctrl.s_star, 0.0)
        self.assertIn("controller_inferred_target_sign", ctrl.effective_settings)
        self.assertAlmostEqual(
            ctrl.effective_settings["controller_inferred_target_sign"], ctrl.s_star, places=12
        )
        self.assertAlmostEqual(ctrl.effective_settings["inference_report_sample_time"], 0.0, places=12)

    def test_action_invariant_under_hidden_private_state_perturbation(self):
        """Restricted policy action must not depend on private states x (no leak)."""
        problem = _make_problem()
        state_a = _make_state(y_sign=+1.0, x_sign=+0.9, seed=1)
        state_b = _make_state(y_sign=+1.0, x_sign=-0.9, seed=1)  # only x differs

        model_a = SampledPublicObservationModel(
            N=N, accessible_roster=problem.accessible_roster,
            rng=np.random.default_rng(42), panel_fraction=0.5, delivery_delay=1.0,
        )
        model_b = SampledPublicObservationModel(
            N=N, accessible_roster=problem.accessible_roster,
            rng=np.random.default_rng(42), panel_fraction=0.5, delivery_delay=1.0,
        )
        obs_a = model_a.sample_observation(state_a, current_time=0.0)
        obs_b = model_b.sample_observation(state_b, current_time=0.0)

        # Both restricted views must exclude private states
        self.assertIsNone(obs_a.private_states)
        self.assertIsNone(obs_b.private_states)

        action_a, _, _ = _act_once(problem, obs_a, seed=11)
        action_b, _, _ = _act_once(problem, obs_b, seed=11)
        self.assertEqual(len(action_a.entries), len(action_b.entries))
        for ea, eb in zip(action_a.entries, action_b.entries):
            self.assertEqual(ea.node, eb.node)
            self.assertAlmostEqual(ea.field, eb.field, places=12)

    def test_no_hidden_memory_leak_in_restricted_view(self):
        """Restricted observation must not expose true memory W to the policy."""
        problem = _make_problem()
        state = _make_state()
        model = SampledPublicObservationModel(
            N=N, accessible_roster=problem.accessible_roster,
            rng=np.random.default_rng(42), panel_fraction=0.5, delivery_delay=1.0,
        )
        obs = model.sample_observation(state, current_time=0.0)
        self.assertIsNone(obs.true_memory)


class TestF3DirectionKnowledgeContract(unittest.TestCase):
    """Direction knowledge contract: evaluator truth vs restricted inference."""

    def test_evaluator_target_fixed_from_true_initial_majority(self):
        """Evaluator s_star is immutable truth from the true initial public majority,
        independent of what any restricted report later shows."""
        state = _make_state(y_sign=+1.0, x_sign=+1.0)
        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.REVERSAL,
            initial_state=state,
            holding_duration=100.0,
            horizon=320.0,
            release_deadline=20.0,
            target_params={},
        )
        self.assertEqual(evaluator.defn.s_star, -1.0)

    def test_controller_inference_may_disagree_and_is_reported(self):
        """Controller inference is based on its restricted report only. If the report
        disagrees with the truth, the mismatch is an inference mistake that is
        exported for counting (never corrected by leaking the true target)."""
        # True initial public majority is negative -> evaluator target +1
        state = _make_state(y_sign=-1.0, x_sign=-1.0)
        evaluator = TargetEvaluator.create_for_state(
            target_type=TargetType.REVERSAL,
            initial_state=state,
            holding_duration=100.0,
            horizon=320.0,
            release_deadline=20.0,
            target_params={},
        )
        self.assertEqual(evaluator.defn.s_star, +1.0)

        # Restricted noisy report suggests the opposite majority
        problem = _make_problem()
        obs = _make_observation([1.0, 1.0, 1.0, 1.0])  # observed mean +1
        _, ctrl, _ = _act_once(problem, obs)
        self.assertLess(ctrl.s_star, 0.0)
        inferred = ctrl.effective_settings["controller_inferred_target_sign"]
        # Direction mismatch flag computable from exports without any private leak
        mismatch = (inferred != evaluator.defn.s_star)
        self.assertTrue(mismatch)


class TestF3ResumeParity(unittest.TestCase):
    """Checkpoint/resume must preserve the locked target sign exactly."""

    def _run_piece(self, problem, initial_state, ctl, **kwargs):
        return run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            observation_model=SampledPublicObservationModel(
                N=N, accessible_roster=problem.accessible_roster,
                rng=np.random.default_rng(888), panel_fraction=0.5, delivery_delay=0.5,
            ),
            target_evaluator=TargetEvaluator.create_for_state(
                target_type=problem.target_type,
                initial_state=initial_state,
                holding_duration=problem.holding_duration,
                horizon=problem.horizon,
                release_deadline=problem.release_deadline,
                target_params=problem.target_params,
            ),
            sim_rng=np.random.default_rng(999),
            controller_rng=np.random.default_rng(777),
            **kwargs,
        )

    def test_resume_preserves_locked_sign(self):
        problem = _make_problem()
        initial_state = _make_state(y_sign=+1.0, x_sign=+0.9)
        ctl = ObservableSelectorPolicy(selector="random", nominal_amplitude=1.0, duration=4.0, seed=7)

        res_full, final_full, _ = self._run_piece(problem, initial_state, ctl)
        inferred_full = res_full.summaries.get("effective_settings", {}).get(
            "controller_inferred_target_sign"
        )
        self.assertIsNotNone(inferred_full)
        self.assertAlmostEqual(inferred_full, -1.0, places=12)

        # Interrupt at t=3.0 (after the sign is locked), then resume
        res_part, _, _ = self._run_piece(
            problem, initial_state, ctl, checkpoint_at=3.0, stop_at_checkpoint=True
        )
        self.assertIsNotNone(res_part.checkpoint)

        res_resumed, final_resumed, _ = run_controlled_simulation(
            initial_state=initial_state.copy(deep=True),
            problem=problem,
            controller=copy.deepcopy(ctl),
            resume_checkpoint=res_part.checkpoint,
        )
        inferred_resumed = res_resumed.summaries.get("effective_settings", {}).get(
            "controller_inferred_target_sign"
        )
        self.assertIsNotNone(inferred_resumed)
        # Locked sign survives checkpoint/resume exactly
        self.assertAlmostEqual(inferred_full, inferred_resumed, places=12)
        # And rollout parity is preserved
        np.testing.assert_array_equal(final_full.x, final_resumed.x)
        np.testing.assert_array_equal(final_full.y, final_resumed.y)


if __name__ == "__main__":
    unittest.main()

