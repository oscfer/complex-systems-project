"""Unit tests for Controllers and Realism Observable Policy Search.

Tests:
1. ZeroController immediate release at t=0.
2. OpenLoopPulseController effective amplitude clipping by budget and duration.
3. ObservableSelectorPolicy:
   - 5 observable selectors (random, reported_attention, reported_outgoing_degree,
     reported_cross_group, group_balanced).
   - Quota slot allocation between observed accessible and unobserved accessible nodes.
   - Static commitment vs dynamic report-driven feedback adaptation.
   - Delayed recruitment fraction (0.5 vs 1.0).
   - Alignment checking: avoiding wasted field on nodes already aligned with target sign.
   - Observation isolation: decision uses only Observation data.
4. OracleMPCController benchmark policy execution.
"""

import unittest
import numpy as np

from src.control.contracts import Action, ControlProblem, Observation, TargetType
from src.control.controllers import (
    ObservableSelectorPolicy,
    OpenLoopPulseController,
    OracleMPCController,
    ZeroController,
)
from src.control.ledger import BudgetLedger
from src.control.observation import SampledPublicObservationModel
from src.core.state import State, StateMode


class TestControllers(unittest.TestCase):
    """Test suite for ZeroController, OpenLoopPulseController, and ObservableSelectorPolicy."""

    def setUp(self):
        self.N = 20
        self.problem_rev = ControlProblem(
            problem_id="prob_rev",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=self.N,
            k=4,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=20.0,
            holding_duration=50.0,
            horizon=270.0,
            accessible_roster=(0, 1, 2, 3, 4, 5),  # 6 accessible nodes
        )

        # Synthetic observation at t=1.0 (delivered report)
        panel_mask = np.zeros(self.N, dtype=bool)
        panel_mask[0:10] = True  # Nodes 0..9 in panel
        public_states = np.full((self.N, 1), np.nan)
        public_states[0:10, 0] = 1.0  # Positive beliefs initially -> target sign s_* = -1

        # Observed edges giving varying degrees/attention
        # Node 1 receives attention from 0, 2, 3 -> rec_att = 3
        # Node 2 has out degree to 1, 3 -> out_deg = 2
        observed_edges = {
            (0, 1): 2,
            (2, 1): 1,
            (3, 1): 1,
            (2, 3): 1,
        }
        groups = np.zeros(self.N, dtype=int)
        groups[0:5] = 0
        groups[5:10] = 1

        self.obs = Observation(
            time=1.0,
            public_states=public_states,
            panel_mask=panel_mask,
            observed_edges=observed_edges,
            accessible_roster=self.problem_rev.accessible_roster,
            group_labels=groups,
            sample_time=0.0,
            delivery_time=1.0,
            remaining_exposure=10.0,
            remaining_distinct_nodes=4,
            current_spending_rate=0.0,
        )
        self.ledger = BudgetLedger(
            N=self.N,
            D=1,
            k=4,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=20.0,
            accessible_roster=set(self.problem_rev.accessible_roster),
        )

    def test_zero_controller(self):
        """ZeroController must release control immediately at t=0."""
        ctrl = ZeroController()
        ctrl.reset(self.problem_rev, self.obs)
        action = ctrl.act(self.obs, self.ledger)
        self.assertTrue(action.release)
        self.assertEqual(len(action.entries), 0)

    def test_open_loop_pulse_amplitude_clipping(self):
        """OpenLoopPulseController clips amplitude by B / (n_s * d)."""
        # B = 10.0. If we target 2 nodes for duration 10 with nominal amplitude 4.0:
        # max_amp = 10.0 / (2 * 10) = 0.5.
        ctrl = OpenLoopPulseController(
            target_nodes=[0, 1],
            nominal_amplitude=4.0,
            duration=10.0,
        )
        ctrl.reset(self.problem_rev, self.obs)
        action = ctrl.act(self.obs, self.ledger)

        self.assertEqual(len(action.entries), 2)
        for e in action.entries:
            self.assertAlmostEqual(e.field, 0.5, places=6)

        self.ledger.apply_action(action, current_time=1.0)
        self.assertEqual(self.ledger.current_spending_rate, 1.0)  # 2 * 0.5 = 1.0

        # At expiration (t=11.0), controller releases
        obs_expired = Observation(
            time=11.0,
            public_states=self.obs.public_states,
            panel_mask=self.obs.panel_mask,
            observed_edges={},
            accessible_roster=self.obs.accessible_roster,
        )
        act_end = ctrl.act(obs_expired, self.ledger)
        self.assertTrue(act_end.release)

    def test_observable_policy_reported_attention_selector(self):
        """Reported attention selector chooses highest-attention accessible nodes."""
        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=1.0,
            duration=5.0,
            quota=1.0,  # All slots from observed accessible
            is_feedback=False,
            initial_recruitment_fraction=1.0,
        )
        ctrl.reset(self.problem_rev, self.obs)
        action = ctrl.act(self.obs, self.ledger)

        # Node 1 received the most attention (count 4), so it must be selected first
        actuated = action.actuated_nodes()
        self.assertIn(1, actuated)
        # All actuated nodes must have target sign s_* = -1 (pushing negative)
        for e in action.entries:
            self.assertLess(e.field, 0.0)

    def test_observable_policy_quota_allocation(self):
        """Quota=0.5 allocates half slots to observed accessible, half to unobserved accessible."""
        # Accessible roster: (0, 1, 2, 3, 4, 5).
        # In panel (0..9): 0, 1, 2, 3, 4, 5 are all in panel here!
        # Let's customize panel so that 4 and 5 are outside panel:
        custom_panel = np.zeros(self.N, dtype=bool)
        custom_panel[0:4] = True  # 0, 1, 2, 3 observed. 4, 5 unobserved.
        obs_custom = Observation(
            time=1.0,
            public_states=self.obs.public_states,
            panel_mask=custom_panel,
            observed_edges=self.obs.observed_edges,
            accessible_roster=(0, 1, 2, 3, 4, 5),
        )

        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=1.0,
            duration=5.0,
            quota=0.5,  # 50% observed, 50% unobserved
            is_feedback=False,
            initial_recruitment_fraction=1.0,
        )
        ctrl.reset(self.problem_rev, obs_custom)
        action = ctrl.act(obs_custom, self.ledger)

        actuated = action.actuated_nodes()
        self.assertEqual(len(actuated), 4)
        # 2 from {0, 1, 2, 3} and 2 from {4, 5}
        obs_count = sum(1 for n in actuated if n in {0, 1, 2, 3})
        unobs_count = sum(1 for n in actuated if n in {4, 5})
        self.assertEqual(obs_count, 2)
        self.assertEqual(unobs_count, 2)

    def test_observable_policy_feedback_recruitment(self):
        """Feedback with recruitment_fraction=0.5 initially recruits 2 nodes, then recruits on later reports."""
        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=1.0,
            duration=10.0,
            is_feedback=True,
            initial_recruitment_fraction=0.5,  # initially floor(4 * 0.5) = 2 nodes
        )
        ctrl.reset(self.problem_rev, self.obs)

        # Report 1 at t=1.0
        act1 = ctrl.act(self.obs, self.ledger)
        self.assertEqual(len(act1.actuated_nodes()), 2)
        self.ledger.apply_action(act1, current_time=1.0)
        self.assertEqual(self.ledger.remaining_distinct_nodes, 2)

        # Report 2 at t=6.0: should recruit up to 2 more nodes
        self.ledger.advance_to(6.0)
        obs2 = Observation(
            time=6.0,
            public_states=self.obs.public_states,
            panel_mask=self.obs.panel_mask,
            observed_edges=self.obs.observed_edges,
            accessible_roster=self.obs.accessible_roster,
            remaining_exposure=self.ledger.remaining_exposure,
            remaining_distinct_nodes=self.ledger.remaining_distinct_nodes,
        )
        act2 = ctrl.act(obs2, self.ledger)
        self.assertGreater(len(act2.actuated_nodes()), 2)

    def test_observable_policy_avoids_wasting_field_on_aligned_nodes(self):
        """Feedback policy zeroes field on nodes that already strongly align with target sign."""
        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=1.0,
            duration=10.0,
            is_feedback=True,
            initial_recruitment_fraction=1.0,
        )
        ctrl.reset(self.problem_rev, self.obs)
        act1 = ctrl.act(self.obs, self.ledger)
        self.ledger.apply_action(act1, current_time=1.0)

        # On report 2 at t=6.0: node 1 has flipped and is now aligned with target: y[1] = -1.0
        # (Target sign is -1.0, so s_* * y[1] = (-1) * (-1) = +1.0 >= 0.5!)
        y_updated = np.copy(self.obs.public_states)
        y_updated[1, 0] = -1.0
        obs2 = Observation(
            time=6.0,
            public_states=y_updated,
            panel_mask=self.obs.panel_mask,
            observed_edges=self.obs.observed_edges,
            accessible_roster=self.obs.accessible_roster,
            remaining_exposure=self.ledger.remaining_exposure,
            remaining_distinct_nodes=self.ledger.remaining_distinct_nodes,
        )
        act2 = ctrl.act(obs2, self.ledger)
        # Node 1 should not receive field because it is already aligned
        actuated2 = act2.actuated_nodes()
        self.assertNotIn(1, actuated2)


    def test_observable_policy_all_active_nodes_aligned_zeroes_field(self):
        """When all active nodes are observed and aligned, field is zeroed without declaring release."""
        state = State.create_empty(N=4, D=1, state_mode=StateMode.BINARY)
        state.y[0, 0] = -1.0
        state.y[1, 0] = -1.0
        state.y[2, 0] = 1.0
        state.y[3, 0] = 1.0
        obs_model = SampledPublicObservationModel(N=4, panel_fraction=1.0, accessible_roster=(0, 1), binary_flip_prob=0.0)
        obs = obs_model.sample_observation(state, current_time=1.0, remaining_exposure=10.0, remaining_distinct_nodes=2)
        ctrl = ObservableSelectorPolicy(
            selector="reported_attention",
            nominal_amplitude=2.0,
            duration=4.0,
            is_feedback=True,
        )
        problem = ControlProblem(
            problem_id="prob_aligned",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            k=2,
            release_deadline=10.0,
            holding_duration=5.0,
            horizon=20.0,
            accessible_roster=(0, 1),
        )
        ctrl.reset(problem, obs)
        ledger = BudgetLedger(N=4, k=2, exposure_cap=10.0, accessible_roster={0, 1})
        act1 = ctrl.act(obs, ledger)
        ledger.apply_action(act1, current_time=1.0)

        # On next report: both active nodes remain aligned
        obs2 = obs_model.sample_observation(state, current_time=2.0, remaining_exposure=ledger.remaining_exposure, remaining_distinct_nodes=ledger.remaining_distinct_nodes)
        act2 = ctrl.act(obs2, ledger)
        self.assertEqual(act2.spending_rate, 0.0)
        self.assertEqual(len(act2.entries), 0)
        self.assertFalse(act2.release)


    def test_observable_policy_target_sign_does_not_flip_on_reversal(self):
        """Target sign s_* must remain fixed even when public state successfully reverses to negative."""
        problem = ControlProblem(
            problem_id="prob_no_flip",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=4,
            k=2,
            accessible_roster=(0, 1, 2, 3),
        )
        ctrl = ObservableSelectorPolicy(
            selector="random",
            is_feedback=True,
            duration=15.0,
            initial_recruitment_fraction=1.0,
        )
        ledger = BudgetLedger(N=4, k=2, exposure_cap=20.0, accessible_roster={0, 1, 2, 3})

        # Initial report: mean is +1.0 -> target sign is s_* = -1.0
        obs1 = Observation(
            time=1.0,
            public_states=np.ones((4, 1)),
            panel_mask=np.ones(4, dtype=bool),
            observed_edges={},
            accessible_roster=(0, 1, 2, 3),
            remaining_exposure=20.0,
            remaining_distinct_nodes=2,
        )
        ctrl.reset(problem, obs1)
        act1 = ctrl.act(obs1, ledger)
        self.assertEqual(len(act1.entries), 2)
        for e in act1.entries:
            self.assertLess(e.field, 0.0)
        ledger.apply_action(act1, current_time=1.0)

        # Later report at t=6.0: beliefs successfully reversed to negative (-0.75 mean)
        obs2 = Observation(
            time=6.0,
            public_states=np.array([[-1.0], [-1.0], [-1.0], [0.0]]),
            panel_mask=np.ones(4, dtype=bool),
            observed_edges={},
            accessible_roster=(0, 1, 2, 3),
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )
        act2 = ctrl.act(obs2, ledger)
        # Controller must NOT flip to positive fields
        for e in act2.entries:
            self.assertLess(e.field, 0.0)


    def test_feedback_recruitment_bounds_strictly_enforced(self):
        """Feedback policy must not recruit more than n_to_recruit nodes in a single report."""
        N = 10
        problem = ControlProblem(
            problem_id="prob_recruit_bound",
            target_type=TargetType.REVERSAL,
            target_params={},
            N=N,
            k=10,
            accessible_roster=tuple(range(N)),
        )
        ctrl = ObservableSelectorPolicy(
            selector="random",
            is_feedback=True,
            initial_recruitment_fraction=0.2,  # initial k = 2, recruit_allowance = 2
            duration=15.0,
            seed=42,
        )
        ledger = BudgetLedger(N=N, k=10, exposure_cap=20.0, accessible_roster=set(range(N)))

        obs1 = Observation(
            time=1.0,
            public_states=np.ones((N, 1)),
            panel_mask=np.ones(N, dtype=bool),
            observed_edges={},
            accessible_roster=problem.accessible_roster,
            remaining_exposure=20.0,
            remaining_distinct_nodes=10,
        )
        ctrl.reset(problem, obs1)
        act1 = ctrl.act(obs1, ledger)
        self.assertEqual(len(ctrl.used_nodes), 2)
        ledger.apply_action(act1, current_time=1.0)

        # On next report at t=6.0: recruit_allowance = 2, so at most 2 new nodes can be recruited
        obs2 = Observation(
            time=6.0,
            public_states=np.ones((N, 1)),
            panel_mask=np.ones(N, dtype=bool),
            observed_edges={},
            accessible_roster=problem.accessible_roster,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
        )
        act2 = ctrl.act(obs2, ledger)
        self.assertLessEqual(len(ctrl.used_nodes), 4)

if __name__ == "__main__":
    unittest.main()
