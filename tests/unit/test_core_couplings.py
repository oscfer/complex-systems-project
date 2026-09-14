"""Unit tests for src/core/couplings.py."""

import unittest
import numpy as np

from src.core.couplings import (
    calculate_internal_fields,
    calculate_isolated_internal_energy,
    enumerate_triad_energy_spectrum,
    get_coupling_matrix,
    validate_coupling_matrix,
)


class TestCoreCouplings(unittest.TestCase):
    """Test suite for belief couplings, energy spectra, and internal field calculations."""

    def test_triad_energy_spectra_exact(self):
        """Verify exact discrete energy spectra for B_0, B_+, and B_f at L = 1.0."""
        # Uncoupled B_0: all 8 configurations have energy 0.0
        B0 = get_coupling_matrix("B0", D=3)
        spec_b0 = enumerate_triad_energy_spectrum(B0, L=1.0)
        self.assertEqual(spec_b0, {0.0: 8})

        # Reinforcing B_+: 2 ground states at -1.5, 6 excited states at +0.5
        B_plus = get_coupling_matrix("B+", D=3)
        spec_plus = enumerate_triad_energy_spectrum(B_plus, L=1.0)
        self.assertEqual(spec_plus, {-1.5: 2, 0.5: 6})

        # Frustrated B_f: 6 partially-satisfied states at -0.5, 2 frustrated states at +1.5
        B_f = get_coupling_matrix("Bf", D=3)
        spec_f = enumerate_triad_energy_spectrum(B_f, L=1.0)
        self.assertEqual(spec_f, {-0.5: 6, 1.5: 2})

    def test_triad_energy_spectra_scaling(self):
        """Verify linear scaling of energy spectra with coupling parameter L."""
        for L in (0.5, 2.0, 3.5):
            B_plus = get_coupling_matrix("B+", D=3)
            spec_plus = enumerate_triad_energy_spectrum(B_plus, L=L)
            expected_plus = {round(-1.5 * L, 6): 2, round(0.5 * L, 6): 6}
            self.assertEqual(spec_plus, expected_plus)

            B_f = get_coupling_matrix("Bf", D=3)
            spec_f = enumerate_triad_energy_spectrum(B_f, L=L)
            expected_f = {round(-0.5 * L, 6): 6, round(1.5 * L, 6): 2}
            self.assertEqual(spec_f, expected_f)

    def test_gauge_invariant_balanced_triad(self):
        """Verify two-negative-edge balanced triad reproduces the exact B_+ spectrum."""
        # Gauge transform: flipping spin 0 reverses signs on (0, 1) and (0, 2)
        # B_bal has links (-0.5, -0.5, +0.5). Product of signs = (-1)*(-1)*(+1) = +1 (balanced)
        B_bal = 0.5 * np.array([
            [0.0, -1.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
        ], dtype=np.float64)

        validate_coupling_matrix(B_bal)
        spec_bal = enumerate_triad_energy_spectrum(B_bal, L=1.0)
        self.assertEqual(spec_bal, {-1.5: 2, 0.5: 6})

    def test_matrix_structural_invariants(self):
        """Verify validation logic enforces symmetry, zero diagonal, and unit row sums."""
        # Valid matrices should pass
        for name in ("B0", "B+", "Bf"):
            B = get_coupling_matrix(name, D=3)
            validate_coupling_matrix(B)

        # Asymmetric matrix
        B_asym = np.array([
            [0.0, 0.6, 0.4],
            [0.4, 0.0, 0.6],
            [0.4, 0.6, 0.0],
        ])
        with self.assertRaises(ValueError):
            validate_coupling_matrix(B_asym)

        # Non-zero diagonal
        B_diag = np.array([
            [0.1, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        with self.assertRaises(ValueError):
            validate_coupling_matrix(B_diag)

        # Non-unit row sums
        B_nonunit = np.array([
            [0.0, 0.3, 0.3],
            [0.3, 0.0, 0.3],
            [0.3, 0.3, 0.0],
        ])
        with self.assertRaises(ValueError):
            validate_coupling_matrix(B_nonunit)

    def test_internal_fields_vectorization(self):
        """Verify vectorized calculate_internal_fields matches explicit nested loops."""
        rng = np.random.default_rng(505)
        N, D = 60, 3
        x = rng.choice([-1.0, 1.0], size=(N, D))
        B = get_coupling_matrix("Bf", D=3)
        L = 1.8

        fields_vec = calculate_internal_fields(x, B, L=L)

        for i in range(N):
            for alpha in range(D):
                expected = L * sum(B[alpha, beta] * x[i, beta] for beta in range(D) if beta != alpha)
                self.assertAlmostEqual(fields_vec[i, alpha], expected, places=12)

    def test_scalar_baseline_d1(self):
        """Verify scalar D=1 baseline returns 1x1 zero matrix and zero internal field."""
        B_scalar = get_coupling_matrix("scalar", D=1)
        self.assertEqual(B_scalar.shape, (1, 1))
        self.assertEqual(B_scalar[0, 0], 0.0)

        x = np.array([[1.0], [-1.0], [1.0]])
        fields = calculate_internal_fields(x, B_scalar, L=2.0)
        self.assertTrue(np.all(fields == 0.0))


if __name__ == "__main__":
    unittest.main()
