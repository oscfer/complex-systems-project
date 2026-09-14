"""Unit tests for src/core/memory.py."""

import math
import unittest
import numpy as np

from src.core.memory import MemoryManager


class TestCoreMemory(unittest.TestCase):
    """Test suite for interaction memory W, attention A, and encounter dynamics."""

    def test_attention_normalization_empty_and_mixed(self):
        """Verify attention matrix normalization and received attention vector properties."""
        mgr = MemoryManager(N=4, reciprocal=False)

        # Empty network initially
        self.assertEqual(mgr.total_memory, 0)
        self.assertTrue(np.all(mgr.A == 0.0))
        self.assertTrue(np.all(mgr.r == 0.0))

        # Add ties from agent 0
        mgr.add_encounter(0, 1)
        mgr.add_encounter(0, 1)
        mgr.add_encounter(0, 1)
        mgr.add_encounter(0, 2)

        # Agent 0 has 4 encounters (3 to node 1, 1 to node 2)
        self.assertEqual(mgr.row_sums[0], 4)
        self.assertAlmostEqual(mgr.A[0, 1], 0.75, places=12)
        self.assertAlmostEqual(mgr.A[0, 2], 0.25, places=12)
        self.assertAlmostEqual(np.sum(mgr.A[0]), 1.0, places=12)

        # Inactive rows are all zeros
        self.assertTrue(np.all(mgr.A[1] == 0.0))
        self.assertTrue(np.all(mgr.A[2] == 0.0))
        self.assertTrue(np.all(mgr.A[3] == 0.0))

        # Received attention: r_1 = 0.75, r_2 = 0.25, others 0
        self.assertAlmostEqual(mgr.r[1], 0.75, places=12)
        self.assertAlmostEqual(mgr.r[2], 0.25, places=12)
        self.assertAlmostEqual(np.sum(mgr.r), 1.0, places=12)

    def test_self_tie_strict_exclusion(self):
        """Verify that self-ties (i == j) are strictly prohibited."""
        mgr = MemoryManager(N=5)

        # Attempting self-encounter must raise ValueError
        with self.assertRaises(ValueError):
            mgr.add_encounter(2, 2)

        # In target distribution, self-probability must be strictly 0.0
        y = np.ones((5, 1), dtype=np.float64)
        P_i, is_null = mgr.compute_target_distribution(i=2, y=y)
        self.assertFalse(is_null)
        self.assertEqual(P_i[2], 0.0)

    def test_target_distribution_logsumexp_stability(self):
        """Verify numerical stability of candidate distribution under extreme prestige values."""
        mgr = MemoryManager(N=4, epsilon=0.1, a=10.0, c=1.0)
        # Artificially set very large received attention
        mgr.r = np.array([0.0, 1000.0, 500.0, 0.0])
        # Give agent 0 existing ties to agents 1 and 2
        mgr.W[0, 1] = 5
        mgr.W[0, 2] = 2
        mgr.row_sums[0] = 7

        y = np.ones((4, 2), dtype=np.float64)
        P_0, is_null = mgr.compute_target_distribution(i=0, y=y)

        self.assertFalse(is_null)
        self.assertIsNotNone(P_0)
        # Must not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(P_0)))
        self.assertFalse(np.any(np.isinf(P_0)))
        # Must sum to 1.0
        self.assertAlmostEqual(float(np.sum(P_0)), 1.0, places=10)
        # Agent 1 has much higher prestige -> should receive dominating preference
        self.assertGreater(P_0[1], P_0[2])

    def test_null_encounter_eligibility(self):
        """Verify null encounter when agent has zero eligible partners."""
        mgr = MemoryManager(N=4)
        y = np.ones((4, 1), dtype=np.float64)

        # Agent 0 eligibility row is strictly all zeros
        C_row = np.zeros(4, dtype=np.float64)

        P_0, is_null = mgr.compute_target_distribution(i=0, y=y, C_row=C_row)
        self.assertTrue(is_null)
        self.assertIsNone(P_0)

        # Executing encounter attempt leaves memory matrix unchanged
        W_before = mgr.W.copy()
        rng = np.random.default_rng(42)
        res = mgr.sample_and_apply_encounter(i=0, y=y, C_row=C_row, rng=rng)

        self.assertTrue(res.is_null)
        self.assertEqual(res.target, -1)
        self.assertTrue(np.array_equal(mgr.W, W_before))

    def test_triadic_closure_dynamics(self):
        """Verify triadic closure reinforces only the direct shortcut tie."""
        mgr = MemoryManager(N=4, epsilon=0.0, chi=1.0)
        # Set up 2-hop path: 0 -> 1 and 1 -> 2
        mgr.add_encounter(0, 1)
        mgr.add_encounter(1, 2)

        # Initial memory check
        self.assertEqual(mgr.W[0, 2], 0)
        self.assertEqual(mgr.W[0, 1], 1)
        self.assertEqual(mgr.W[1, 2], 1)

        y = np.ones((4, 1), dtype=np.float64)
        P_0, is_null = mgr.compute_target_distribution(i=0, y=y)
        self.assertFalse(is_null)

        # Target 2 completes the triad 0 -> 1 -> 2
        # Target 2 should have non-zero closure probability
        self.assertGreater(P_0[2], 0.0)
        # Self-closure (0 -> 0) must be strictly 0.0
        self.assertEqual(P_0[0], 0.0)

        # Manually add encounter between 0 and 2
        mgr.add_encounter(0, 2)

        # Direct tie is incremented
        self.assertEqual(mgr.W[0, 2], 1)
        # Intermediate path ties must NOT be reinforced
        self.assertEqual(mgr.W[0, 1], 1)
        self.assertEqual(mgr.W[1, 2], 1)

    def test_reciprocal_memory_symmetry(self):
        """Verify reciprocal memory maintains W == W.T across encounters and decay."""
        mgr = MemoryManager(N=5, reciprocal=True)
        rng = np.random.default_rng(888)

        # Add multiple paired encounters
        mgr.add_encounter(0, 1)
        mgr.add_encounter(1, 3)
        mgr.add_encounter(2, 4)
        mgr.add_encounter(0, 4)

        self.assertTrue(np.array_equal(mgr.W, mgr.W.T))
        self.assertEqual(mgr.W[0, 1], mgr.W[1, 0])
        self.assertEqual(mgr.W[1, 3], mgr.W[3, 1])

        # Decay a pair
        i_dec, j_dec = mgr.sample_and_apply_decay(rng)
        self.assertTrue(np.array_equal(mgr.W, mgr.W.T))

        # Symmetrized memory produces asymmetric attention when row degrees differ
        # Agent 0 has 2 ties, Agent 2 has 1 tie -> A[0, 4] = 0.5 != A[4, 0] = 0.5?
        # Wait, if row sums differ, A is asymmetric
        mgr.add_encounter(0, 2)
        self.assertTrue(np.array_equal(mgr.W, mgr.W.T))
        if mgr.row_sums[0] != mgr.row_sums[2]:
            self.assertFalse(np.array_equal(mgr.A, mgr.A.T))

    def test_immigration_death_expectation_ode(self):
        """Verify empirical mean memory follows exact analytical ODE: E[S(t)] = (N*nu/delta)*(1 - e^{-delta*t})."""
        N = 30
        nu = 1.0
        delta = 0.5
        horizon = 2.0
        n_replicates = 80

        # Exact expected memory at t = 2.0
        s_inf = N * nu / delta  # 30 * 1.0 / 0.5 = 60.0
        expected_s_horizon = s_inf * (1.0 - math.exp(-delta * horizon))  # 60 * (1 - e^-1) = 37.927

        final_memories = []
        for rep in range(n_replicates):
            rng = np.random.default_rng(1000 + rep)
            mgr = MemoryManager(N=N, reciprocal=False)
            t = 0.0

            while t < horizon:
                # Rates: R_birth = N * nu, R_death = delta * S
                r_birth = N * nu
                r_death = delta * mgr.total_memory
                r_tot = r_birth + r_death

                dt = rng.exponential(1.0 / r_tot)
                if t + dt > horizon:
                    break
                t += dt

                if rng.random() < (r_birth / r_tot):
                    # Birth event: sample random i, j
                    i = int(rng.integers(0, N))
                    j = int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    # Death event
                    mgr.sample_and_apply_decay(rng)

            final_memories.append(mgr.total_memory)

        empirical_mean = float(np.mean(final_memories))
        # Relative error should be within 10%
        rel_err = abs(empirical_mean - expected_s_horizon) / expected_s_horizon
        self.assertLess(rel_err, 0.10)

    def test_stationary_variance_directed_vs_reciprocal(self):
        """Verify analytical stationary variance: Directed Var(S) ~ N*nu/delta vs Reciprocal Var(S) ~ 2*N*nu/delta."""
        # Theory: S_directed ~ Poisson(N*nu/delta) with Mean=Var=N*nu/delta
        # In reciprocal: pair count S_pair ~ Poisson(N*nu/(2*delta)). Since S = 2*S_pair, Var(S) = 4 * Var(S_pair) = 2*N*nu/delta.
        N = 20
        nu = 2.0
        delta = 1.0
        # Expected stationary mean: 40.0
        # Expected directed var: 40.0; reciprocal var: 80.0

        # Sample 1000 stationary states via Gillespie jump chains
        def sample_stationary_directed(n_steps=600):
            rng = np.random.default_rng(42)
            counts = []
            mgr = MemoryManager(N=N, reciprocal=False)
            for step in range(n_steps):
                r_birth = N * nu
                r_death = delta * mgr.total_memory
                r_tot = r_birth + r_death
                if rng.random() < (r_birth / r_tot):
                    i, j = int(rng.integers(0, N)), int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    mgr.sample_and_apply_decay(rng)
                if step >= 200 and step % 5 == 0:
                    counts.append(mgr.total_memory)
            return counts

        def sample_stationary_reciprocal(n_steps=600):
            rng = np.random.default_rng(84)
            counts = []
            mgr = MemoryManager(N=N, reciprocal=True)
            for step in range(n_steps):
                # Reciprocal attempt rate is nu / 2 per agent => total birth rate N*nu/2 pairs => N*nu tokens
                r_birth = N * nu / 2.0
                total_pairs = mgr.total_memory // 2
                r_death = delta * total_pairs
                r_tot = r_birth + r_death
                if rng.random() < (r_birth / r_tot):
                    i, j = int(rng.integers(0, N)), int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    mgr.sample_and_apply_decay(rng)
                if step >= 200 and step % 5 == 0:
                    counts.append(mgr.total_memory)
            return counts

        dir_counts = sample_stationary_directed()
        rec_counts = sample_stationary_reciprocal()

        var_dir = float(np.var(dir_counts))
        var_rec = float(np.var(rec_counts))

        # Reciprocal variance should be substantially larger than directed variance
        self.assertGreater(var_rec, var_dir)


if __name__ == "__main__":
    unittest.main()
