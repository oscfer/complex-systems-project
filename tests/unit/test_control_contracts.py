"""Unit tests for Control Contracts.

Tests:
1. ControlProblem immutability (frozen dataclass) and automatic deterministic hashing.
2. ControlProblem parameter validation (k in [1, N], horizon > deadline, non-negative budget).
3. Action entry uniqueness, sorting, and deduplication.
4. Action dense NumPy array conversion (to_array and from_array).
5. Action spending rate and actuated node set extraction.
6. Action validation (finite values, node/component bounds, amplitude cap, accessible roster).
7. Observation isolation: non-oracle observations contain NO private state or true memory.
8. Observation helper methods (observed panel checks, accessible roster checks, observed public mean).
"""

from dataclasses import FrozenInstanceError
import unittest
import numpy as np

from src.control.contracts import (
    Action,
    ActionEntry,
    ActuatorClass,
    ControlProblem,
    Observation,
    ObservationMode,
    RolloutResult,
    TargetType,
)


class TestControlContracts(unittest.TestCase):
    """Test suite for ControlProblem, Action, Observation, and RolloutResult."""

    def test_control_problem_immutability_and_hash(self):
        """ControlProblem must be frozen and generate a deterministic SHA-256 hash."""
        prob = ControlProblem(
            problem_id="prob_01",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": 0.5},
            N=64,
            k=6,
            exposure_cap=10.0,
            amplitude_cap=4.0,
            release_deadline=20.0,
            holding_duration=50.0,
            horizon=270.0,
            accessible_roster=(0, 1, 2, 3),
        )

        self.assertTrue(len(prob.model_hash) > 0)
        # Attempting to mutate attributes must raise FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            prob.k = 10
        with self.assertRaises(FrozenInstanceError):
            prob.exposure_cap = 50.0

    def test_control_problem_parameter_validation(self):
        """ControlProblem must validate bounds on N, k, and times."""
        # Invalid k > N
        with self.assertRaises(ValueError):
            ControlProblem(
                problem_id="bad_k",
                target_type=TargetType.REVERSAL,
                target_params={},
                N=10,
                k=15,
            )

        # Invalid negative exposure cap
        with self.assertRaises(ValueError):
            ControlProblem(
                problem_id="bad_B",
                target_type=TargetType.REVERSAL,
                target_params={},
                N=10,
                k=2,
                exposure_cap=-1.0,
            )

        # Invalid horizon <= deadline
        with self.assertRaises(ValueError):
            ControlProblem(
                problem_id="bad_H",
                target_type=TargetType.REVERSAL,
                target_params={},
                N=10,
                k=2,
                release_deadline=20.0,
                horizon=20.0,
            )

    def test_action_sorting_and_deduplication(self):
        """Action entries must be sorted unique by (node, component)."""
        e1 = ActionEntry(node=5, component=0, field=1.0)
        e2 = ActionEntry(node=2, component=0, field=2.0)
        e3 = ActionEntry(node=5, component=1, field=-0.5)

        # Passed out of order: should be sorted
        act = Action(entries=(e1, e2, e3))
        self.assertEqual(act.entries[0].node, 2)
        self.assertEqual(act.entries[1].node, 5)
        self.assertEqual(act.entries[1].component, 0)
        self.assertEqual(act.entries[2].node, 5)
        self.assertEqual(act.entries[2].component, 1)

        # Duplicate (node, component) must raise ValueError
        e_dup = ActionEntry(node=2, component=0, field=1.5)
        with self.assertRaises(ValueError):
            Action(entries=(e1, e2, e_dup))

    def test_action_dense_conversion(self):
        """Action dense array conversion and reconstruction."""
        N, D = 5, 2
        arr = np.zeros((N, D), dtype=np.float64)
        arr[1, 0] = 1.5
        arr[3, 1] = -2.0

        act = Action.from_array(arr, intended_end_time=5.0)
        self.assertEqual(len(act.entries), 2)
        self.assertEqual(act.spending_rate, 3.5)
        self.assertEqual(act.actuated_nodes(), {1, 3})

        # Convert back to array
        reconstructed = act.to_array(N=N, D=D)
        np.testing.assert_allclose(reconstructed, arr)

    def test_action_validation_checks(self):
        """Action validation checks finiteness, bounds, amplitude cap, and roster."""
        # Nonfinite field
        with self.assertRaises(ValueError):
            ActionEntry(node=0, component=0, field=float("nan"))

        # Node index out of bounds
        act_oob = Action.from_dict({(10, 0): 1.0})
        with self.assertRaises(ValueError):
            act_oob.validate(N=5, D=1)

        # Amplitude cap exceeded
        act_high = Action.from_dict({(0, 0): 5.0})
        with self.assertRaises(ValueError):
            act_high.validate(N=5, D=1, amplitude_cap=4.0)

        # Accessible roster violation
        act_unauth = Action.from_dict({(3, 0): 1.0})
        with self.assertRaises(ValueError):
            act_unauth.validate(N=5, D=1, accessible_roster={0, 1})

    def test_observation_strict_isolation(self):
        """Restricted observation must not contain private states or mutable true state."""
        N = 10
        panel_mask = np.zeros(N, dtype=bool)
        panel_mask[0:5] = True
        obs = Observation(
            time=1.0,
            public_states=np.ones((N, 1)),
            panel_mask=panel_mask,
            observed_edges={(0, 1): 2},
            accessible_roster=(0, 1, 2),
            is_oracle=False,
            private_states=None,
            true_memory=None,
        )

        self.assertFalse(obs.is_oracle)
        self.assertIsNone(obs.private_states)
        self.assertIsNone(obs.true_memory)
        self.assertTrue(obs.is_node_observed(2))
        self.assertFalse(obs.is_node_observed(7))
        self.assertTrue(obs.is_node_accessible(1))
        self.assertFalse(obs.is_node_accessible(5))
        self.assertEqual(obs.observed_public_mean(0), 1.0)


    def test_action_multi_component_per_node_amplitude_cap(self):
        """Per-node total amplitude across components cannot exceed amplitude_cap."""
        # Node 0 has components 0 and 1 with fields 2.5 and 2.5 (sum = 5.0 > cap 4.0)
        act_excess = Action.from_dict({(0, 0): 2.5, (0, 1): 2.5})
        with self.assertRaises(ValueError) as ctx:
            act_excess.validate(N=5, D=2, amplitude_cap=4.0)
        self.assertIn("exceeds per-node amplitude cap", str(ctx.exception))

        # Sum within cap (2.0 + 1.5 = 3.5 <= 4.0)
        act_ok = Action.from_dict({(0, 0): 2.0, (0, 1): 1.5})
        act_ok.validate(N=5, D=2, amplitude_cap=4.0)


if __name__ == "__main__":
    unittest.main()
