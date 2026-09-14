"""Unit tests for src/core/state.py."""

import os
import tempfile
import unittest
import numpy as np

from src.core.state import State, StateMode


class TestCoreState(unittest.TestCase):
    """Test suite for State container, validation invariants, and serialization."""

    def test_empty_state_initialization(self):
        """Verify empty state initialization with correct shapes, dtypes, and zeros."""
        state = State.create_empty(N=10, D=3, state_mode=StateMode.BINARY)
        self.assertEqual(state.N, 10)
        self.assertEqual(state.D, 3)
        self.assertEqual(state.state_mode, StateMode.BINARY)
        self.assertEqual(state.x.shape, (10, 3))
        self.assertEqual(state.y.shape, (10, 3))
        self.assertEqual(state.W.shape, (10, 10))
        self.assertEqual(state.n.shape, (10,))
        self.assertEqual(state.Q.shape, (10, 3, 2))
        self.assertEqual(state.t, 0.0)
        self.assertEqual(state.W.dtype, np.uint32)
        self.assertEqual(state.n.dtype, np.uint8)
        self.assertEqual(state.Q.dtype, np.float64)
        self.assertTrue(np.all(state.W == 0))
        self.assertTrue(np.all(state.n == 0))
        self.assertTrue(np.all(state.Q == 0.0))

    def test_unbiased_emergence_initialization_binary(self):
        """Verify unbiased emergence initializes binary spins in {-1, +1} with y == x."""
        rng = np.random.default_rng(42)
        state = State.create_unbiased_emergence(N=50, D=3, state_mode=StateMode.BINARY, rng=rng)
        state.validate()

        self.assertTrue(np.array_equal(state.x, state.y))
        self.assertTrue(np.all(np.isin(state.x, [-1.0, 1.0])))
        self.assertTrue(np.all(np.diag(state.W) == 0))
        self.assertEqual(state.total_memory, 0)
        self.assertEqual(state.active_agents_count, 0)
        self.assertEqual(state.discrepancy, 0.0)

    def test_unbiased_emergence_initialization_continuous(self):
        """Verify unbiased emergence initializes continuous states in [-1, 1] with y == x."""
        rng = np.random.default_rng(123)
        state = State.create_unbiased_emergence(N=40, D=3, state_mode=StateMode.CONTINUOUS, rng=rng)
        state.validate()

        self.assertTrue(np.array_equal(state.x, state.y))
        self.assertTrue(np.all(state.x >= -1.0) and np.all(state.x <= 1.0))
        self.assertEqual(state.discrepancy, 0.0)

    def test_state_deep_copy_isolation(self):
        """Verify deep copy provides complete array and buffer independence."""
        rng = np.random.default_rng(99)
        orig = State.create_unbiased_emergence(N=10, D=1, state_mode=StateMode.BINARY, rng=rng)
        orig.W[0, 1] = 5
        orig.holding_active = True
        orig.t_start_hold = 2.5
        orig.t_deadline = 3.5

        clone = orig.copy(deep=True)

        # Mutate clone buffers
        clone.x[0, 0] = -clone.x[0, 0]
        clone.W[0, 1] = 99
        clone.n[0] = 1
        clone.Q[0, 0, 1] = 5.0
        clone.t = 10.0
        clone.holding_active = False

        # Orig should remain untouched
        self.assertNotEqual(orig.x[0, 0], clone.x[0, 0])
        self.assertEqual(orig.W[0, 1], 5)
        self.assertEqual(orig.n[0], 0)
        self.assertEqual(orig.Q[0, 0, 1], 0.0)
        self.assertEqual(orig.t, 0.0)
        self.assertTrue(orig.holding_active)

    def test_state_validation_invariants(self):
        """Verify validate() raises ValueError on invariant violations."""
        state = State.create_empty(N=5, D=1, state_mode=StateMode.BINARY)

        # Non-zero diagonal in W
        state.W[1, 1] = 1
        with self.assertRaises(ValueError):
            state.validate()
        state.W[1, 1] = 0

        # Negative memory in W
        state.W = state.W.astype(np.int32)
        state.W[0, 1] = -1
        with self.assertRaises(ValueError):
            state.validate()
        state.W[0, 1] = 0
        state.W = state.W.astype(np.uint32)

        # Invalid participation
        state.n[0] = 3
        with self.assertRaises(ValueError):
            state.validate()
        state.n[0] = 0

        # Negative simulation time
        state.t = -0.01
        with self.assertRaises(ValueError):
            state.validate()
        state.t = 0.0

        # Out of bounds binary spin
        state.x[0, 0] = 0.0
        with self.assertRaises(ValueError):
            state.validate()
        state.x[0, 0] = 1.0

        # Out of bounds continuous spin
        c_state = State.create_empty(N=5, D=1, state_mode=StateMode.CONTINUOUS)
        c_state.x[0, 0] = 1.05
        with self.assertRaises(ValueError):
            c_state.validate()

    def test_checkpoint_roundtrip(self):
        """Verify save_checkpoint and load_checkpoint preserve all data bit-for-bit."""
        rng = np.random.default_rng(77)
        state = State.create_unbiased_emergence(N=20, D=3, state_mode=StateMode.BINARY, rng=rng)
        state.W[2, 5] = 12
        state.W[5, 2] = 8
        state.n[3] = 1
        state.Q[1, 2, 0] = 0.42
        state.t = 15.678
        state.holding_active = True
        state.t_start_hold = 12.0
        state.t_deadline = 17.0
        state.deadline_generation = 4
        state.transition_occurred = False
        state.pulse_active = True
        state.pulse_val = 1.5
        state.pulse_component = 0
        state.pulse_targets = np.array([True, False] * 10, dtype=bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "checkpoint.npz")
            state.save_checkpoint(chk_path)

            loaded = State.load_checkpoint(chk_path)
            loaded.validate()

            self.assertEqual(loaded.N, state.N)
            self.assertEqual(loaded.D, state.D)
            self.assertEqual(loaded.state_mode, state.state_mode)
            self.assertTrue(np.array_equal(loaded.x, state.x))
            self.assertTrue(np.array_equal(loaded.y, state.y))
            self.assertTrue(np.array_equal(loaded.W, state.W))
            self.assertTrue(np.array_equal(loaded.n, state.n))
            self.assertTrue(np.array_equal(loaded.Q, state.Q))
            self.assertEqual(loaded.t, state.t)
            self.assertEqual(loaded.holding_active, state.holding_active)
            self.assertEqual(loaded.t_start_hold, state.t_start_hold)
            self.assertEqual(loaded.t_deadline, state.t_deadline)
            self.assertEqual(loaded.deadline_generation, state.deadline_generation)
            self.assertEqual(loaded.pulse_active, state.pulse_active)
            self.assertEqual(loaded.pulse_val, state.pulse_val)
            self.assertTrue(np.array_equal(loaded.pulse_targets, state.pulse_targets))

    def test_npz_roundtrip(self):
        """Verify save_npz and load_npz alias methods work identically."""
        state = State.create_empty(N=6, D=1, state_mode=StateMode.BINARY)
        state.x.fill(1.0)
        state.y.fill(-1.0)
        state.W[1, 2] = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            state.save_npz(path)
            loaded = State.load_npz(path)
            loaded.validate()
            self.assertTrue(np.array_equal(loaded.W, state.W))
            self.assertTrue(np.array_equal(loaded.x, state.x))
            self.assertTrue(np.array_equal(loaded.y, state.y))

    def test_state_derived_observables(self):
        """Verify derived properties: row_sums, total_memory, discrepancy, etc."""
        state = State.create_empty(N=4, D=1, state_mode=StateMode.BINARY)
        state.x = np.array([[1.0], [1.0], [-1.0], [-1.0]])
        state.y = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        state.W[0, 1] = 3
        state.W[0, 2] = 2
        state.W[1, 3] = 4
        state.n = np.array([1, 0, 1, 0], dtype=np.uint8)

        self.assertTrue(np.array_equal(state.row_sums, [5, 4, 0, 0]))
        self.assertEqual(state.total_memory, 9)
        self.assertEqual(state.active_agents_count, 2)
        self.assertEqual(state.participation_fraction, 0.5)

        # Discrepancy: x-y = [0, 2, -2, 0]. (x-y)^2 = [0, 4, 4, 0]. Sum = 8.
        # d_2 = 8 / (4 * 4 * 1) = 8 / 16 = 0.5
        self.assertAlmostEqual(state.discrepancy, 0.5, places=10)


if __name__ == "__main__":
    unittest.main()
