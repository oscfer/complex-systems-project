"""Adversarial stress harness and empirical challenge suite for the simulator core.

Covers:
1. Extreme floating-point parameter stress on heat baths (kappa in {+-1000, +-1e-7, 0, +-1e5, +-1e-15}).
2. High-precision Kolmogorov-Smirnov goodness-of-fit on continuous heat bath density.
3. Large-scale memory matrix operations (N=256, dense encounters, high decay rates, reciprocal mode).
4. Memory exhaustion boundary conditions (complete decay to total_memory == 0).
5. Exact Poisson scheduler inter-arrival exponential distribution goodness-of-fit (Exp(R_total)).
6. Scheduler branching ratio chi-squared test across all 5 stochastic jump channels.
7. Institutional holding duration edge cases (rapid threshold oscillations, late interruptions, re-entry).
8. Q-learning temperature and field extremes, and strict counterfactual isolation.
"""

from __future__ import annotations

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
from src.core.memory import MemoryManager, EncounterResult
from src.core.fields import (
    compute_attention_matrix,
    compute_received_attention,
    compute_public_fields,
    compute_private_fields,
)
from src.core.couplings import get_coupling_matrix, enumerate_triad_energy_spectrum
from src.core.participation import ParticipationTracker, ParticipationParams
from src.core.q_learning import (
    init_q_table,
    softmax_probabilities,
    select_action,
    update_chosen_q_value,
    step_q_learning,
    QLearningParams,
)
from src.core.scheduler import PoissonScheduler
from src.core.state import State, StateMode


class TestAdversarialCoreStress(unittest.TestCase):
    """Adversarial empirical challenge suite for core dynamics modules."""

    def test_continuous_heat_bath_extreme_kappas(self):
        """Stress-test continuous inverse-CDF heat bath across extreme parameter boundaries."""
        kappas = [
            1000.0, -1000.0,
            1e-7, -1e-7,
            0.0, -0.0,
            1e-5, -1e-5,
            1e5, -1e5,
            1e-15, -1e-15,
        ]
        test_us = [
            0.0, 1e-16, 1e-10, 1e-5, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
            1.0 - 1e-16, 1.0,
        ]

        for k in kappas:
            for u in test_us:
                z = sample_continuous_inverse_cdf_scalar(k, u)
                self.assertFalse(np.isnan(z), f"NaN returned for kappa={k}, u={u}")
                self.assertFalse(np.isinf(z), f"Inf returned for kappa={k}, u={u}")
                self.assertGreaterEqual(z, -1.0, f"z < -1.0 for kappa={k}, u={u}, z={z}")
                self.assertLessEqual(z, 1.0, f"z > 1.0 for kappa={k}, u={u}, z={z}")

        # Physical parity symmetry: z(-kappa, u) == -z(kappa, 1 - u)
        interior_us = [1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-6]
        for k in [1000.0, 50.0, 1.0, 1e-4, 1e-7]:
            for u in interior_us:
                z_pos = sample_continuous_inverse_cdf_scalar(k, u)
                z_neg = sample_continuous_inverse_cdf_scalar(-k, 1.0 - u)
                self.assertAlmostEqual(
                    z_pos, -z_neg, places=10,
                    msg=f"Parity violated for kappa={k}, u={u}: z_pos={z_pos}, -z_neg={-z_neg}"
                )

        # Monotonicity test: u1 < u2 => z(u1) <= z(u2)
        grid_u = np.linspace(0.0, 1.0, 500)
        for k in [-1000.0, -5.0, -1e-5, 0.0, 1e-5, 5.0, 1000.0]:
            z_vals = [sample_continuous_inverse_cdf_scalar(k, u) for u in grid_u]
            diffs = np.diff(z_vals)
            self.assertTrue(
                np.all(diffs >= -1e-14),
                f"Inverse CDF is not monotonic for kappa={k}. Min diff: {np.min(diffs)}"
            )

        # Branch boundary continuity at |kappa| = 1e-5
        u_samples = np.linspace(0.01, 0.99, 100)
        for u in u_samples:
            z_below = sample_continuous_inverse_cdf_scalar(1e-5 * (1.0 - 1e-8), u)
            z_above = sample_continuous_inverse_cdf_scalar(1e-5 * (1.0 + 1e-8), u)
            self.assertAlmostEqual(
                z_below, z_above, places=8,
                msg=f"Discontinuity at boundary kappa=1e-5 for u={u}: {z_below} vs {z_above}"
            )

    def test_continuous_heat_bath_distribution_ks(self):
        """High-precision Kolmogorov-Smirnov goodness-of-fit on continuous heat bath inverse CDF."""
        rng = np.random.default_rng(777)
        n_samples = 100_000
        # Critical value at alpha = 0.01: 1.63 / sqrt(100000) = 0.00515
        ks_critical = 1.63 / math.sqrt(n_samples)

        for kappa in [0.0, 1.5, -2.0]:
            k_vec = np.full(n_samples, kappa)
            samples = sample_continuous_inverse_cdf_vector(k_vec, rng)

            sorted_s = np.sort(samples)
            empirical_cdf = np.arange(1, n_samples + 1) / n_samples

            if abs(kappa) < 1e-8:
                th_cdf = (sorted_s + 1.0) / 2.0
            else:
                th_cdf = (np.exp(kappa * sorted_s) - np.exp(-kappa)) / (np.exp(kappa) - np.exp(-kappa))

            ks_stat = float(np.max(np.abs(empirical_cdf - th_cdf)))
            self.assertLess(
                ks_stat, ks_critical,
                f"KS test failed for kappa={kappa}: D={ks_stat:.5f} >= critical={ks_critical:.5f}"
            )

    def test_binary_heat_bath_extreme_kappas(self):
        """Stress-test binary Glauber conditional probability with extreme couplings."""
        # Exact saturation
        self.assertEqual(binary_glauber_prob_plus(1000.0), 1.0)
        self.assertEqual(binary_glauber_prob_plus(-1000.0), 0.0)
        self.assertEqual(binary_glauber_prob_plus(1e8), 1.0)
        self.assertEqual(binary_glauber_prob_plus(-1e8), 0.0)
        self.assertEqual(binary_glauber_prob_plus(0.0), 0.5)

        # Weak field susceptibility limit: P(+) ~ 0.5 + 0.5 * kappa
        p_weak = binary_glauber_prob_plus(1e-7)
        self.assertAlmostEqual(p_weak, 0.5 + 0.5 * 1e-7, places=13)

        # Vectorized batch evaluation
        rng = np.random.default_rng(123)
        extreme_k = np.array([1000.0, -1000.0, 0.0, 1e-7, -1e-7, 1e6, -1e6])
        spins = sample_binary_glauber_vector(extreme_k, rng)
        self.assertTrue(np.all(np.isin(spins, [-1.0, 1.0])))
        self.assertEqual(spins[0], 1.0)   # kappa = 1000 -> must be +1
        self.assertEqual(spins[1], -1.0)  # kappa = -1000 -> must be -1

    def test_large_memory_matrix_dense_scale(self):
        """Stress-test MemoryManager with N=256, dense encounters, and high decay rates."""
        N = 256
        rng = np.random.default_rng(2026)

        # Initialize dense memory: 256 * 255 non-zero entries
        dense_W = rng.integers(1, 20, size=(N, N), dtype=np.uint32)
        np.fill_diagonal(dense_W, 0)

        mm = MemoryManager(
            N=N,
            reciprocal=False,
            epsilon=0.1,
            a=0.5,
            c=1.0,
            chi=0.15,  # Triadic closure active
            initial_W=dense_W,
        )

        initial_total = int(np.sum(dense_W))
        self.assertEqual(mm.total_memory, initial_total)
        self.assertTrue(np.all(mm.row_sums == np.sum(dense_W, axis=1)))
        self.assertTrue(np.allclose(mm.A, compute_attention_matrix(mm.W)))
        self.assertTrue(np.allclose(mm.r, compute_received_attention(mm.A)))

        # Run dense encounters
        y = rng.choice([-1.0, 1.0], size=(N, 3))
        n_encounters = 1000
        for _ in range(n_encounters):
            i = int(rng.integers(0, N))
            res = mm.sample_and_apply_encounter(i, y, rng=rng)
            self.assertFalse(res.is_null)
            self.assertNotEqual(res.target, i)

        # Invariants after encounters
        self.assertEqual(mm.total_memory, initial_total + n_encounters)
        self.assertTrue(np.all(mm.row_sums == np.sum(mm.W, axis=1)))
        self.assertTrue(np.allclose(mm.A, compute_attention_matrix(mm.W)))
        self.assertTrue(np.allclose(mm.r, compute_received_attention(mm.A), atol=1e-12))

        # Run high-rate decays
        n_decays = 3000
        for _ in range(n_decays):
            i, j = mm.sample_and_apply_decay(rng=rng)
            self.assertGreaterEqual(i, 0)
            self.assertGreaterEqual(j, 0)
            self.assertNotEqual(i, j)

        # Invariants after decays
        expected_total = initial_total + n_encounters - n_decays
        self.assertEqual(mm.total_memory, expected_total)
        self.assertTrue(np.all(mm.row_sums == np.sum(mm.W, axis=1)))
        self.assertTrue(np.allclose(mm.A, compute_attention_matrix(mm.W)))
        self.assertTrue(np.allclose(mm.r, compute_received_attention(mm.A), atol=1e-12))

    def test_large_memory_reciprocal_exhaustion(self):
        """Stress-test reciprocal memory mode symmetry maintenance and complete decay to zero."""
        N = 128
        rng = np.random.default_rng(888)

        raw = rng.integers(1, 10, size=(N, N), dtype=np.uint32)
        sym_W = raw + raw.T
        np.fill_diagonal(sym_W, 0)

        mm = MemoryManager(N=N, reciprocal=True, initial_W=sym_W)
        self.assertTrue(np.array_equal(mm.W, mm.W.T))

        # Perform reciprocal encounters
        y = rng.choice([-1.0, 1.0], size=(N, 1))
        for _ in range(500):
            i = int(rng.integers(0, N))
            res = mm.sample_and_apply_encounter(i, y, rng=rng)
            self.assertFalse(res.is_null)

        self.assertTrue(np.array_equal(mm.W, mm.W.T))
        self.assertEqual(mm.total_memory, int(np.sum(mm.W)))

        # Perform reciprocal decays
        for _ in range(1000):
            i, j = mm.sample_and_apply_decay(rng=rng)
            self.assertGreaterEqual(i, 0)
            self.assertEqual(mm.W[i, j], mm.W[j, i])

        self.assertTrue(np.array_equal(mm.W, mm.W.T))

        # Complete decay exhaustion test
        small_sym = np.array([[0, 3], [3, 0]], dtype=np.uint32)
        small_mm = MemoryManager(N=2, reciprocal=True, initial_W=small_sym)
        self.assertEqual(small_mm.total_memory, 6)

        for _ in range(3):
            i, j = small_mm.sample_and_apply_decay(rng=rng)
            self.assertGreaterEqual(i, 0)

        self.assertEqual(small_mm.total_memory, 0)
        self.assertTrue(np.all(small_mm.W == 0))

        # Further decay on empty memory returns (-1, -1) cleanly without exceptions
        end_i, end_j = small_mm.sample_and_apply_decay(rng=rng)
        self.assertEqual((end_i, end_j), (-1, -1))
        self.assertEqual(small_mm.total_memory, 0)

    def test_scheduler_poisson_interarrival_exponential(self):
        """Verify scheduler Poisson event waiting times are exponentially distributed Exp(R_total)."""
        N = 100
        D = 1
        tau_x = 1.0
        tau_y = 0.5
        nu = 2.0
        delta = 0.05
        tau_n = 2.0

        sched = PoissonScheduler(
            N=N, D=D, tau_x=tau_x, tau_y=tau_y, nu=nu, delta=delta, tau_n=tau_n,
            reciprocal=False, rng=np.random.default_rng(9999)
        )
        S = 400
        sched.set_total_memory(S)

        R_x = N * D / tau_x           # 100.0
        R_y = N * D / tau_y           # 200.0
        R_enc = N * nu                # 200.0
        R_forget = delta * S          # 20.0
        R_n = N / tau_n               # 50.0
        R_tot = R_x + R_y + R_enc + R_forget + R_n # 570.0

        self.assertAlmostEqual(sched.R_total, R_tot, places=10)

        n_events = 100_000
        times = [sched.time]

        for _ in range(n_events):
            sched.step()
            times.append(sched.time)

        times_arr = np.array(times)
        dt = np.diff(times_arr)

        mean_dt = float(np.mean(dt))
        var_dt = float(np.var(dt))
        th_mean = 1.0 / R_tot
        th_var = 1.0 / (R_tot ** 2)

        # Mean within 1.5%, variance within 3%
        self.assertAlmostEqual(mean_dt, th_mean, delta=0.015 * th_mean)
        self.assertAlmostEqual(var_dt, th_var, delta=0.03 * th_var)

        # Kolmogorov-Smirnov test against Exp(R_tot) CDF: F(t) = 1 - exp(-R_tot * t)
        sorted_dt = np.sort(dt)
        empirical_cdf = np.arange(1, n_events + 1) / n_events
        th_cdf = 1.0 - np.exp(-R_tot * sorted_dt)
        ks_stat = float(np.max(np.abs(empirical_cdf - th_cdf)))
        ks_critical = 1.63 / math.sqrt(n_events)
        self.assertLess(
            ks_stat, ks_critical,
            f"Scheduler waiting times failed KS test: D={ks_stat:.5f} >= critical={ks_critical:.5f}"
        )

        # Chi-squared test on channel selection counts
        observed = np.array([
            sched.stats.private_updates,
            sched.stats.public_updates,
            sched.stats.encounters,
            sched.stats.memory_decays,
            sched.stats.participation_updates,
        ], dtype=np.float64)
        expected = np.array([R_x, R_y, R_enc, R_forget, R_n], dtype=np.float64) / R_tot * n_events
        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        # Critical value for df=4, alpha=0.01 is 13.28
        self.assertLess(
            chi2, 13.28,
            f"Scheduler channel branching failed Chi2 test: chi2={chi2:.4f} >= critical=13.28"
        )

    def test_institutional_holding_rapid_oscillations(self):
        """Stress-test institutional holding tracker under 500 rapid threshold oscillations."""
        N = 100
        params = ParticipationParams(theta=0.5, t_hold=1.0)
        tracker = ParticipationTracker(N=N, params=params)
        sched = PoissonScheduler(N=N, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.01)

        def on_deadline(s, event):
            tracker.handle_deadline_event(event.time, event.payload)

        # 500 rapid crossings back and forth across theta
        t = 1.0
        sched.time = t
        for k in range(500):
            t += 0.001
            sched.time = t
            if k % 2 == 0:
                tracker.n[:60] = 1
                tracker.n[60:] = 0
            else:
                tracker.n[:40] = 1
                tracker.n[40:] = 0
            tracker.check_threshold(t, sched)

        # Settle above theta at t = 1.501
        t = 1.501
        sched.time = t
        tracker.n[:70] = 1
        tracker.n[70:] = 0
        tracker.check_threshold(t, sched)
        expected_deadline = t + params.t_hold  # 2.501

        for ev in sched.pq:
            if ev.event_type == "holding_deadline":
                ev.callback = on_deadline

        # Process all events up to t = 3.0
        while sched.pq and sched.time < 3.0:
            sched.step()

        self.assertTrue(tracker.transition_occurred)
        self.assertIsNotNone(tracker.t_transition)
        self.assertAlmostEqual(tracker.t_transition, expected_deadline, places=6)

    def test_institutional_holding_late_interruption(self):
        """Verify interruption 1 microsecond before deadline cancels institutional transition."""
        N = 100
        params = ParticipationParams(theta=0.5, t_hold=1.0)
        tracker = ParticipationTracker(N=N, params=params)
        sched = PoissonScheduler(N=N, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.01)

        def on_deadline(s, event):
            tracker.handle_deadline_event(event.time, event.payload)

        # Start holding at t = 1.0 (deadline 2.0)
        sched.time = 1.0
        tracker.n[:60] = 1
        tracker.n[60:] = 0
        tracker.check_threshold(1.0, sched)

        for ev in sched.pq:
            if ev.event_type == "holding_deadline":
                ev.callback = on_deadline

        # Interrupt 1 microsecond before deadline
        sched.time = 1.999999
        tracker.n[:45] = 1
        tracker.n[45:] = 0
        tracker.check_threshold(1.999999, sched)

        # Step past deadline
        while sched.pq and sched.peek_next_deterministic_time() <= 2.0:
            sched.step()

        self.assertFalse(tracker.transition_occurred)
        self.assertFalse(tracker.holding_active)
        self.assertIsNone(tracker.t_transition)

    def test_q_learning_and_fields_extreme_inputs(self):
        """Stress-test Q-learning updates under extreme temperature and field inputs."""
        # Extreme temperature
        Q_pair = np.array([10.0, -10.0], dtype=np.float64)
        p_minus, p_plus = softmax_probabilities(Q_pair, T_Q=1e-5)
        self.assertAlmostEqual(p_plus, 0.0, places=12)
        self.assertAlmostEqual(p_minus, 1.0, places=12)

        # Large temperature (high noise limit -> 50/50)
        p_minus, p_plus = softmax_probabilities(Q_pair, T_Q=1e6)
        self.assertAlmostEqual(p_plus, 0.5, places=4)

        # TD update counterfactual isolation
        q_entry = np.array([0.0, 0.0], dtype=np.float64)
        # Update chosen action +1 (index 1) with reward 5.0, eta=0.1
        update_chosen_q_value(q_entry, chosen_action=1, reward=5.0, eta=0.1)
        self.assertEqual(q_entry[0], 0.0)  # Counterfactual unchosen action MUST be untouched
        self.assertAlmostEqual(q_entry[1], 0.5, places=12)

        # Zero attention row handling in fields
        zero_W = np.zeros((10, 10), dtype=np.uint32)
        A = compute_attention_matrix(zero_W)
        self.assertTrue(np.all(A == 0.0))

        y = np.ones((10, 1), dtype=np.float64)
        x = np.ones((10, 1), dtype=np.float64)
        Fy = compute_public_fields(A, y, x, J=2.0, K=0.5)
        # With zero attention, social field is 0, so Fy = K * x = 0.5
        self.assertTrue(np.allclose(Fy, 0.5))


if __name__ == "__main__":
    unittest.main()
