"""Unit tests for src/core/heat_baths.py."""

import math
import unittest
import numpy as np

from src.core.heat_baths import (
    binary_glauber_prob_plus,
    calibrate_continuous_temperature,
    sample_binary_glauber_scalar,
    sample_binary_glauber_vector,
    sample_continuous_inverse_cdf_scalar,
    sample_continuous_inverse_cdf_vector,
)


class TestCoreHeatBaths(unittest.TestCase):
    """Test suite for binary Glauber updates and stable continuous inverse-CDF sampling."""

    def test_binary_glauber_boundary_probabilities(self):
        """Verify Glauber probability at zero and extreme fields without overflow."""
        # Zero field
        self.assertAlmostEqual(binary_glauber_prob_plus(0.0), 0.5, places=12)

        # Extreme positive field: should be 1.0 without overflow
        self.assertEqual(binary_glauber_prob_plus(500.0), 1.0)
        self.assertEqual(binary_glauber_prob_plus(1000.0), 1.0)

        # Extreme negative field: should be 0.0 without overflow/underflow warnings
        self.assertEqual(binary_glauber_prob_plus(-500.0), 0.0)
        self.assertEqual(binary_glauber_prob_plus(-1000.0), 0.0)

    def test_binary_glauber_analytical_gibbs(self):
        """Verify binary Glauber sampling frequency matches exact analytical Gibbs probability."""
        kappa = 0.5
        theoretical_p_plus = 1.0 / (1.0 + math.exp(-2.0 * kappa))  # ~0.7310585786
        n_samples = 100_000

        rng = np.random.default_rng(42)
        samples = np.array([sample_binary_glauber_scalar(kappa, rng) for _ in range(n_samples)])
        empirical_p_plus = np.mean(samples == 1.0)

        std_err = math.sqrt(theoretical_p_plus * (1.0 - theoretical_p_plus) / n_samples)
        self.assertAlmostEqual(empirical_p_plus, theoretical_p_plus, delta=3.5 * std_err)

    def test_continuous_zero_field_uniform_density(self):
        """Verify continuous inverse-CDF at kappa=0 generates exact uniform distribution on [-1, 1]."""
        n_samples = 100_000
        rng = np.random.default_rng(123)

        u_draws = rng.uniform(0.0, 1.0, size=n_samples)
        samples = np.array([sample_continuous_inverse_cdf_scalar(0.0, u) for u in u_draws])

        # Mean should be 0.0, variance should be 1/3
        mean = float(np.mean(samples))
        var = float(np.var(samples))
        self.assertAlmostEqual(mean, 0.0, delta=0.015)
        self.assertAlmostEqual(var, 1.0 / 3.0, delta=0.015)

        # 10-bin Chi-squared test against uniform density
        counts, _ = np.histogram(samples, bins=10, range=(-1.0, 1.0))
        expected_per_bin = n_samples / 10.0
        chi2 = float(np.sum((counts - expected_per_bin) ** 2 / expected_per_bin))
        # 9 degrees of freedom, critical value at alpha=0.01 is 21.67
        self.assertLess(chi2, 21.67)

    def test_continuous_inverse_cdf_regimes_smoothness(self):
        """Verify smoothness across the boundary |kappa| = 1e-5 and extreme fields."""
        u_test = 0.65
        k_boundary_plus = 1.000001e-5
        k_boundary_minus = 0.999999e-5

        z_outer = sample_continuous_inverse_cdf_scalar(k_boundary_plus, u_test)
        z_inner = sample_continuous_inverse_cdf_scalar(k_boundary_minus, u_test)
        self.assertAlmostEqual(z_outer, z_inner, places=8)

        # Extreme fields must clamp strictly to [-1.0, 1.0]
        z_large_pos = sample_continuous_inverse_cdf_scalar(1000.0, 0.5)
        self.assertTrue(-1.0 <= z_large_pos <= 1.0)
        self.assertAlmostEqual(z_large_pos, 1.0, delta=0.01)

        z_large_neg = sample_continuous_inverse_cdf_scalar(-1000.0, 0.5)
        self.assertTrue(-1.0 <= z_large_neg <= 1.0)
        self.assertAlmostEqual(z_large_neg, -1.0, delta=0.01)

        # Boundary preservation
        self.assertEqual(sample_continuous_inverse_cdf_scalar(0.5, 0.0), -1.0)
        self.assertEqual(sample_continuous_inverse_cdf_scalar(0.5, 1.0), 1.0)
        self.assertEqual(sample_continuous_inverse_cdf_scalar(-0.5, 0.0), -1.0)
        self.assertEqual(sample_continuous_inverse_cdf_scalar(-0.5, 1.0), 1.0)

    def test_continuous_analytical_langevin_moments(self):
        """Verify continuous sample means match theoretical Langevin function coth(k) - 1/k."""
        rng = np.random.default_rng(999)
        test_kappas = [0.2, 0.5, 1.0, 2.0]
        n_samples = 80_000

        for kappa in test_kappas:
            # Theoretical Langevin mean: L(kappa) = coth(kappa) - 1/kappa
            expected_mean = (1.0 / math.tanh(kappa)) - (1.0 / kappa)

            u_draws = rng.uniform(0.0, 1.0, size=n_samples)
            samples = np.array([sample_continuous_inverse_cdf_scalar(kappa, u) for u in u_draws])
            sample_mean = float(np.mean(samples))

            # Sample standard error < 1 / sqrt(N)
            std_err = 1.0 / math.sqrt(n_samples)
            self.assertAlmostEqual(sample_mean, expected_mean, delta=3.5 * std_err)

    def test_weak_field_susceptibility_matching(self):
        """Verify that T_cont = T_bin / 3 matches linear susceptibility chi_cont == chi_bin."""
        T_bin = 1.0
        T_cont = calibrate_continuous_temperature(T_bin)
        self.assertAlmostEqual(T_cont, 1.0 / 3.0, places=12)

        F = 0.03
        kappa_bin = F / T_bin      # 0.03
        kappa_cont = F / T_cont    # 0.09

        # Analytical expectations:
        # E[z]_bin = tanh(0.03) = 0.02999100
        # E[z]_cont = coth(0.09) - 1/0.09 = 0.02998380
        e_bin = math.tanh(kappa_bin)
        e_cont = (1.0 / math.tanh(kappa_cont)) - (1.0 / kappa_cont)
        self.assertLess(abs(e_bin - e_cont), 1e-4)

        # Empirical simulation of 200,000 draws
        rng = np.random.default_rng(314)
        bin_samples = sample_binary_glauber_vector(np.full(200_000, kappa_bin), rng)
        cont_samples = sample_continuous_inverse_cdf_vector(np.full(200_000, kappa_cont), rng)

        mean_bin = float(np.mean(bin_samples))
        mean_cont = float(np.mean(cont_samples))

        self.assertAlmostEqual(mean_bin, 0.030, delta=0.008)
        self.assertAlmostEqual(mean_cont, 0.030, delta=0.008)

    def test_two_site_coupled_gibbs_distribution(self):
        """Verify MCMC Glauber heat bath on 2 coupled spins recovers exact Boltzmann distribution."""
        # Coupled Hamiltonian: H(x, y) = -K * x * y - h_y * y - b_x * x
        K = 0.5
        h_y = 0.2
        b_x = -0.1
        T = 1.0

        # Exact Boltzmann probabilities for 4 states (x, y) in {-1, +1}^2
        states = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        energies = np.array([-K * s[0] * s[1] - h_y * s[1] - b_x * s[0] for s in states])
        boltzmann_weights = np.exp(-energies / T)
        theoretical_probs = boltzmann_weights / np.sum(boltzmann_weights)

        # Run 100,000 MCMC steps alternating heat-bath updates
        rng = np.random.default_rng(777)
        x = 1.0
        y = 1.0
        state_counts = {s: 0 for s in states}
        n_steps = 100_000

        for _ in range(n_steps):
            # Update y given x: local field F^y = K * x + h_y
            fy = K * x + h_y
            y = sample_binary_glauber_scalar(fy / T, rng)

            # Update x given y: local field F^x = K * y + b_x
            fx = K * y + b_x
            x = sample_binary_glauber_scalar(fx / T, rng)

            state_counts[(x, y)] += 1

        empirical_counts = np.array([state_counts[s] for s in states])
        expected_counts = theoretical_probs * n_steps

        # Chi-squared test with 3 degrees of freedom (critical value 7.815 at alpha=0.05)
        chi2 = float(np.sum((empirical_counts - expected_counts) ** 2 / expected_counts))
        self.assertLess(chi2, 7.815)


if __name__ == "__main__":
    unittest.main()
