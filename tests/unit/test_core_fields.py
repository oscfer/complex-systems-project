"""Unit tests for src/core/fields.py."""

import unittest
import numpy as np

from src.core.fields import (
    FieldPulseSpec,
    compute_attention_matrix,
    compute_external_field,
    compute_private_fields,
    compute_public_fields,
    compute_received_attention,
    compute_single_private_field,
    compute_single_public_field,
    compute_single_row_attention,
)
from src.core.couplings import get_coupling_matrix


class TestCoreFields(unittest.TestCase):
    """Test suite for attention normalization, public and private fields, and external pulses."""

    def test_attention_row_normalization(self):
        """Verify attention matrix normalization and received attention vector properties."""
        W = np.array([
            [0, 3, 1],
            [0, 0, 0],  # empty row
            [2, 2, 0],
        ], dtype=np.uint32)

        A = compute_attention_matrix(W)
        r = compute_received_attention(A)

        # Active row 0: sum is 1.0, entries 3/4 and 1/4
        self.assertAlmostEqual(np.sum(A[0]), 1.0, places=10)
        self.assertAlmostEqual(A[0, 1], 0.75, places=10)
        self.assertAlmostEqual(A[0, 2], 0.25, places=10)

        # Empty row 1: strictly all zeros, no NaN
        self.assertTrue(np.all(A[1] == 0.0))

        # Active row 2: sum is 1.0, entries 0.5 and 0.5
        self.assertAlmostEqual(np.sum(A[2]), 1.0, places=10)
        self.assertAlmostEqual(A[2, 0], 0.5, places=10)
        self.assertAlmostEqual(A[2, 1], 0.5, places=10)

        # Received attention vector: r = sum_i A[i, :]
        expected_r = np.array([0.5, 1.25, 0.25])
        self.assertTrue(np.allclose(r, expected_r))

        # Conservation of received attention: sum_j r_j == N_active == 2
        self.assertAlmostEqual(np.sum(r), 2.0, places=10)

    def test_single_row_attention(self):
        """Verify compute_single_row_attention matches compute_attention_matrix row-by-row."""
        W = np.array([
            [0, 5, 0, 5],
            [0, 0, 0, 0],
            [1, 2, 3, 0],
        ], dtype=np.uint32)

        A_full = compute_attention_matrix(W)
        for i in range(len(W)):
            A_single = compute_single_row_attention(W[i])
            self.assertTrue(np.allclose(A_full[i], A_single))

    def test_public_field_empty_network(self):
        """Verify public field collapses to private coupling and bias when attention is empty."""
        N, D = 4, 2
        A = np.zeros((N, N), dtype=np.float64)
        x = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        y = np.ones((N, D), dtype=np.float64)
        K = 0.5
        h = np.array([[0.1, 0.2], [0.0, 0.0], [0.1, 0.2], [0.0, 0.0]])

        Fy = compute_public_fields(A, y, x, J=1.0, K=K, h=h)
        expected = K * x + h
        self.assertTrue(np.allclose(Fy, expected))

    def test_public_field_vectorized_vs_single(self):
        """Verify vectorized compute_public_fields matches single-site compute_single_public_field."""
        rng = np.random.default_rng(101)
        N, D = 25, 3
        W = rng.integers(0, 5, size=(N, N), dtype=np.uint32)
        np.fill_diagonal(W, 0)
        A = compute_attention_matrix(W)
        x = rng.choice([-1.0, 1.0], size=(N, D))
        y = rng.choice([-1.0, 1.0], size=(N, D))
        h = rng.uniform(-0.5, 0.5, size=(N, D))
        J, K = 1.2, 0.8

        Fy_vec = compute_public_fields(A, y, x, J=J, K=K, h=h)

        for i in range(N):
            for alpha in range(D):
                fy_single = compute_single_public_field(
                    W_row=W[i],
                    y_col=y[:, alpha],
                    x_val=x[i, alpha],
                    J=J,
                    K=K,
                    h_val=h[i, alpha],
                )
                self.assertAlmostEqual(Fy_vec[i, alpha], fy_single, places=10)

    def test_private_field_scalar_d1(self):
        """Verify scalar private field F^x = K * y + b for D=1."""
        N = 5
        x = np.array([[1.0], [-1.0], [1.0], [1.0], [-1.0]])
        y = np.array([[-1.0], [1.0], [1.0], [-1.0], [1.0]])
        b = np.array([[0.2], [0.2], [-0.1], [0.0], [0.5]])
        K = 0.5

        Fx = compute_private_fields(x, y, B=None, K=K, L=1.0, b=b)
        expected = K * y + b
        self.assertTrue(np.allclose(Fx, expected))

    def test_private_field_coupled_d3(self):
        """Verify private field cross-coupling under B_0, B_+, and B_f."""
        # Single agent with spins (+1, +1, +1)
        x = np.array([[1.0, 1.0, 1.0]])
        y = np.array([[1.0, 1.0, 1.0]])
        K, L = 0.5, 1.0

        # B_0: no cross-coupling
        B0 = get_coupling_matrix("B0", D=3)
        Fx_b0 = compute_private_fields(x, y, B=B0, K=K, L=L)
        self.assertTrue(np.allclose(Fx_b0, 0.5 * y))

        # B_+: cross coupling sum is 1.0 for each component
        # B_+ = 0.5 * [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # (x @ B)_alpha = 0.5 * (1 + 1) = 1.0
        B_plus = get_coupling_matrix("B+", D=3)
        Fx_plus = compute_private_fields(x, y, B=B_plus, K=K, L=L)
        # Expected: K * y + L * 1.0 = 0.5 + 1.0 = 1.5
        self.assertTrue(np.allclose(Fx_plus, [[1.5, 1.5, 1.5]]))

        # B_f: cross couplings
        # alpha=0: 0.5*(x1 + x2) = 1.0
        # alpha=1: 0.5*(x0 - x2) = 0.0
        # alpha=2: 0.5*(x0 - x1) = 0.0
        B_f = get_coupling_matrix("Bf", D=3)
        Fx_f = compute_private_fields(x, y, B=B_f, K=K, L=L)
        # Expected: [0.5 + 1.0, 0.5 + 0.0, 0.5 + 0.0] = [1.5, 0.5, 0.5]
        self.assertTrue(np.allclose(Fx_f, [[1.5, 0.5, 0.5]]))

    def test_single_private_field_matches_vectorized(self):
        """Verify compute_single_private_field matches compute_private_fields."""
        rng = np.random.default_rng(202)
        N, D = 30, 3
        x = rng.choice([-1.0, 1.0], size=(N, D))
        y = rng.choice([-1.0, 1.0], size=(N, D))
        b = rng.uniform(-0.3, 0.3, size=(N, D))
        B = get_coupling_matrix("Bf", D=3)
        K, L = 0.7, 1.3

        Fx_vec = compute_private_fields(x, y, B=B, K=K, L=L, b=b)

        for i in range(N):
            for alpha in range(D):
                fx_single = compute_single_private_field(
                    y_val=y[i, alpha],
                    x_row=x[i],
                    alpha=alpha,
                    B=B,
                    K=K,
                    L=L,
                    b_val=b[i, alpha],
                )
                self.assertAlmostEqual(Fx_vec[i, alpha], fx_single, places=10)

    def test_external_field_pulse_timing_and_targets(self):
        """Verify external field pulse timing window and target masking."""
        N, D = 10, 2
        pulse = FieldPulseSpec(
            t_start=2.0,
            t_end=7.0,
            amplitude=1.5,
            component=0,
            target_nodes=np.array([0, 2, 4]),
        )

        # Before pulse
        h_before = compute_external_field(t=1.99, pulse=pulse, N=N, D=D)
        self.assertTrue(np.all(h_before == 0.0))

        # During pulse: target nodes should have amplitude 1.5 on component 0
        h_during = compute_external_field(t=5.0, pulse=pulse, N=N, D=D)
        for i in range(N):
            if i in (0, 2, 4):
                self.assertEqual(h_during[i, 0], 1.5)
            else:
                self.assertEqual(h_during[i, 0], 0.0)
            self.assertEqual(h_during[i, 1], 0.0)

        # After pulse
        h_after = compute_external_field(t=7.0, pulse=pulse, N=N, D=D)
        self.assertTrue(np.all(h_after == 0.0))


if __name__ == "__main__":
    unittest.main()
