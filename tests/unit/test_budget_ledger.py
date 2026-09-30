"""Unit tests for BudgetLedger.

Tests:
1. Distinct-node limit k enforcement, slot consumption, and multi-component accounting.
2. Hand-calculated piecewise exposure exhaustion at exact timestamp t=3.5.
3. Automatic field shutoff upon exposure budget exhaustion.
4. Amplitude cap validation and violation rejection.
5. Mandatory switch-off deadline T_c enforcement.
6. Accessible roster membership enforcement.
7. Irreversible release semantics.
8. Relative accounting tolerance bounds.
9. Rejection of zero-length nonzero pulses and backwards time advances.
"""

import unittest
import numpy as np

from src.control.contracts import Action, ActionEntry
from src.control.ledger import BudgetLedger


class TestBudgetLedger(unittest.TestCase):
    """Test suite for BudgetLedger resource accounting and actuator constraints."""

    def test_distinct_node_limit_k_enforcement(self):
        """Actuation cannot exceed configured distinct node cap k."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=20.0, amplitude_cap=4.0)

        # Actuate node 0 and node 1 (total 2 nodes == k)
        action1 = Action.from_dict({(0, 0): 1.0, (1, 0): 1.0}, intended_end_time=5.0)
        end1 = ledger.apply_action(action1, current_time=0.0)
        self.assertEqual(len(ledger.distinct_nodes_actuated), 2)
        self.assertEqual(ledger.remaining_distinct_nodes, 0)
        self.assertEqual(end1, 5.0)

        # Advance to t=2.0 and re-actuate existing node 1 (should succeed: 0 new slots)
        action2 = Action.from_dict({(1, 0): 1.5}, intended_end_time=5.0)
        valid2, msg2 = ledger.validate_action(action2, current_time=2.0)
        self.assertTrue(valid2, msg2)
        ledger.apply_action(action2, current_time=2.0)
        self.assertEqual(len(ledger.distinct_nodes_actuated), 2)

        # Attempt to actuate a third distinct node 2 (should fail)
        action3 = Action.from_dict({(2, 0): 1.0}, intended_end_time=5.0)
        valid3, msg3 = ledger.validate_action(action3, current_time=2.0)
        self.assertFalse(valid3)
        self.assertIn("exceeds distinct-node cap", msg3)
        with self.assertRaises(ValueError):
            ledger.apply_action(action3, current_time=2.0)

    def test_multi_component_distinct_node_accounting(self):
        """Controlling multiple components on the same node consumes only 1 distinct node slot."""
        ledger = BudgetLedger(N=10, D=3, k=1, exposure_cap=20.0, amplitude_cap=4.0)

        # Actuate node 4 on components 0, 1, 2
        action = Action.from_dict({(4, 0): 1.0, (4, 1): 1.0, (4, 2): 1.0}, intended_end_time=2.0)
        valid, msg = ledger.validate_action(action, current_time=0.0)
        self.assertTrue(valid, msg)
        ledger.apply_action(action, current_time=0.0)
        self.assertEqual(len(ledger.distinct_nodes_actuated), 1)
        self.assertIn(4, ledger.distinct_nodes_actuated)
        self.assertEqual(ledger.remaining_distinct_nodes, 0)

    def test_hand_calculated_piecewise_exposure_exhaustion(self):
        """Piecewise exposure: two fields of magnitude 1 for 2 units spend 4;
        a subsequent magnitude-2 field exhausts remaining 3 at exactly time 3.5.
        """
        # B = 7.0, k = 3
        ledger = BudgetLedger(N=10, D=1, k=3, exposure_cap=7.0, amplitude_cap=4.0, release_deadline=20.0)

        # Segment 1: at t=0, apply field of magnitude 1 on nodes 0 and 1 until t=2.0
        # spending rate c_u = 1.0 + 1.0 = 2.0. Duration = 2.0 -> spends 4.0.
        action1 = Action.from_dict({(0, 0): 1.0, (1, 0): 1.0}, intended_end_time=2.0)
        ledger.apply_action(action1, current_time=0.0)
        self.assertEqual(ledger.current_spending_rate, 2.0)

        # Advance to t=2.0
        ledger.advance_to(2.0)
        self.assertAlmostEqual(ledger.integrated_exposure, 4.0, places=10)
        self.assertAlmostEqual(ledger.remaining_exposure, 3.0, places=10)

        # Segment 2: at t=2.0, apply field of magnitude 2 on node 2 with nominal duration until 10.0
        # spending rate c_u = 2.0.
        # remaining budget is 3.0. Max duration = 3.0 / 2.0 = 1.5.
        # exhaustion time = 2.0 + 1.5 = 3.5!
        action2 = Action.from_dict({(2, 0): 2.0}, intended_end_time=10.0)
        eff_end = ledger.apply_action(action2, current_time=2.0)
        self.assertAlmostEqual(eff_end, 3.5, places=10)

        # Advance to t=3.5: budget must be exhausted and field shut off
        ledger.advance_to(3.5)
        self.assertAlmostEqual(ledger.integrated_exposure, 7.0, places=10)
        self.assertAlmostEqual(ledger.remaining_exposure, 0.0, places=10)
        self.assertTrue(ledger.is_exhausted)
        self.assertTrue(ledger.is_released)
        self.assertEqual(ledger.release_time, 3.5)
        self.assertEqual(ledger.current_spending_rate, 0.0)
        self.assertTrue(np.all(ledger.current_field == 0.0))

    def test_amplitude_cap_violation(self):
        """Field amplitudes exceeding u_max must be rejected."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=20.0, amplitude_cap=4.0)

        # Exactly at cap
        action_ok = Action.from_dict({(0, 0): 4.0}, intended_end_time=1.0)
        valid, _ = ledger.validate_action(action_ok, current_time=0.0)
        self.assertTrue(valid)

        # Exceeding cap
        action_bad = Action.from_dict({(0, 0): 4.05}, intended_end_time=1.0)
        valid, msg = ledger.validate_action(action_bad, current_time=0.0)
        self.assertFalse(valid)
        self.assertIn("exceeds cap", msg)
        with self.assertRaises(ValueError):
            ledger.apply_action(action_bad, current_time=0.0)

    def test_release_deadline_enforcement(self):
        """All interventions must permanently shut off at release deadline T_c."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=50.0, amplitude_cap=4.0, release_deadline=20.0)

        # Action scheduled beyond deadline has effective end clipped to 20.0
        action = Action.from_dict({(0, 0): 1.0}, intended_end_time=25.0)
        eff_end = ledger.apply_action(action, current_time=10.0)
        self.assertEqual(eff_end, 20.0)

        # Advance to deadline
        ledger.advance_to(20.0)
        self.assertTrue(ledger.is_released)
        self.assertEqual(ledger.release_time, 20.0)
        self.assertEqual(ledger.current_spending_rate, 0.0)

        # Attempt to actuate at or after deadline
        action_late = Action.from_dict({(0, 0): 1.0}, intended_end_time=22.0)
        valid, msg = ledger.validate_action(action_late, current_time=20.0)
        self.assertFalse(valid)

    def test_accessible_roster_enforcement(self):
        """Nodes outside accessible roster cannot be actuated."""
        accessible = {1, 3, 5, 7}
        ledger = BudgetLedger(
            N=10, D=1, k=4, exposure_cap=20.0, amplitude_cap=4.0, accessible_roster=accessible
        )

        # In-roster node
        action_ok = Action.from_dict({(3, 0): 1.0}, intended_end_time=1.0)
        self.assertTrue(ledger.validate_action(action_ok, current_time=0.0)[0])

        # Out-of-roster node
        action_bad = Action.from_dict({(2, 0): 1.0}, intended_end_time=1.0)
        valid, msg = ledger.validate_action(action_bad, current_time=0.0)
        self.assertFalse(valid)
        self.assertIn("accessible roster", msg)

    def test_irreversible_release_semantics(self):
        """Declared release permanently disables actuation."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=20.0, amplitude_cap=4.0)

        # Declare release at t=5.0
        ledger.release_control(5.0)
        self.assertTrue(ledger.is_released)
        self.assertEqual(ledger.release_time, 5.0)

        # Attempt actuation post-release
        action = Action.from_dict({(0, 0): 1.0}, intended_end_time=7.0)
        valid, msg = ledger.validate_action(action, current_time=6.0)
        self.assertFalse(valid)
        self.assertIn("release is irreversible", msg)

    def test_zero_length_pulse_rejection(self):
        """Zero-length nonzero pulses are rejected."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=20.0, amplitude_cap=4.0)

        action = Action.from_dict({(0, 0): 1.0}, intended_end_time=0.0)
        valid, msg = ledger.validate_action(action, current_time=0.0)
        self.assertFalse(valid)
        self.assertIn("Zero-length", msg)


    def test_zero_budget_immediate_release(self):
        """Exposure cap of 0.0 must immediately trigger release at t=0.0."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=0.0, amplitude_cap=4.0)
        self.assertTrue(ledger.is_exhausted)
        self.assertTrue(ledger.is_released)
        self.assertEqual(ledger.release_time, 0.0)

    def test_validate_action_backwards_in_time(self):
        """Validating an action with current_time in the past must return False."""
        ledger = BudgetLedger(N=10, D=1, k=2, exposure_cap=10.0, amplitude_cap=4.0)
        ledger.advance_to(5.0)
        action = Action.from_dict({(0, 0): 1.0}, intended_end_time=6.0)
        valid, msg = ledger.validate_action(action, current_time=4.0)
        self.assertFalse(valid)
        self.assertIn("backwards in time", msg)



    def test_advance_to_past_action_intended_end_time(self):
        """Advancing past action intended end time integrates only active duration and clears field."""
        ledger = BudgetLedger(N=10, k=2, exposure_cap=10.0, release_deadline=20.0)
        action = Action(entries=(ActionEntry(node=0, component=0, field=1.0),), intended_end_time=2.0)
        eff_end = ledger.apply_action(action, current_time=0.0)
        self.assertEqual(eff_end, 2.0)

        # Advance directly to 5.0 (3.0 units past pulse shutoff)
        ledger.advance_to(5.0)
        self.assertEqual(ledger.current_time, 5.0)
        self.assertAlmostEqual(ledger.integrated_exposure, 2.0, places=10)
        self.assertEqual(ledger.current_spending_rate, 0.0)
        self.assertIsNone(ledger.active_action)
        self.assertTrue(np.all(ledger.current_field == 0.0))

    def test_advance_to_past_budget_exhaustion(self):
        """Advancing past exhaustion time integrates remaining budget, clears field, and releases at exhaustion."""
        ledger = BudgetLedger(N=10, k=2, exposure_cap=3.0, release_deadline=20.0)
        action = Action(entries=(ActionEntry(node=0, component=0, field=2.0),), intended_end_time=10.0)
        eff_end = ledger.apply_action(action, current_time=0.0)
        self.assertAlmostEqual(eff_end, 1.5, places=10)

        # Advance directly to 5.0 (well past 1.5 exhaustion)
        ledger.advance_to(5.0)
        self.assertEqual(ledger.current_time, 5.0)
        self.assertAlmostEqual(ledger.integrated_exposure, 3.0, places=10)
        self.assertTrue(ledger.is_exhausted)
        self.assertTrue(ledger.is_released)
        self.assertAlmostEqual(ledger.release_time, 1.5, places=10)
        self.assertEqual(ledger.current_spending_rate, 0.0)

    def test_advance_to_past_release_deadline(self):
        """Advancing past release deadline shuts off field and declares release at deadline."""
        ledger = BudgetLedger(N=10, k=2, exposure_cap=20.0, release_deadline=10.0)
        action = Action(entries=(ActionEntry(node=0, component=0, field=1.0),), intended_end_time=15.0)
        eff_end = ledger.apply_action(action, current_time=5.0)
        self.assertAlmostEqual(eff_end, 10.0, places=10)

        # Advance directly to 15.0 (past 10.0 deadline)
        ledger.advance_to(15.0)
        self.assertEqual(ledger.current_time, 15.0)
        self.assertAlmostEqual(ledger.integrated_exposure, 5.0, places=10)
        self.assertTrue(ledger.is_released)
        self.assertAlmostEqual(ledger.release_time, 10.0, places=10)
        self.assertEqual(ledger.current_spending_rate, 0.0)

if __name__ == "__main__":
    unittest.main()
