"""Empirical verification test suite for the simulator core.

Exhaustively verifies exact analytical invariants and oracles on src/core/:
1. Triad Energy Spectrum Oracle:
   Exhaustively evaluate isolated energy E_B(x) for all 8 spin states under B_+ and B_f at L=1.
   Confirm B_+ values are strictly {-1.5, +0.5} and B_f values are strictly {-0.5, +1.5}.
2. Memory Immigration-Death Expectation Oracle:
   Simulate memory dynamics under pure immigration-death and verify lim_{t->inf} E[S(t)] = N*nu/delta.
3. Q-learning Equivalence Oracle:
   Verify that when Q(+-1) = +-F^y and T_Q = T_y, the softmax probabilities match the Glauber heat bath probabilities to machine precision.
4. Participation Potential Oracle:
   Verify Delta Phi = Delta u_k holds exactly to floating-point precision on symmetric matrices.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.core.couplings import (
    calculate_internal_fields,
    calculate_isolated_internal_energy,
    enumerate_triad_energy_spectrum,
    get_coupling_matrix,
    validate_coupling_matrix,
)
from src.core.heat_baths import binary_glauber_prob_plus
from src.core.memory import MemoryManager
from src.core.participation import (
    ParticipationParams,
    ParticipationTracker,
    compute_participation_potential,
)
from src.core.q_learning import (
    QLearningParams,
    compute_immediate_reward,
    init_q_table,
    select_action,
    softmax_probabilities,
    step_q_learning,
    update_chosen_q_value,
)
from src.core.scheduler import PoissonScheduler


class TestTriadEnergySpectraOracle(unittest.TestCase):
    """Oracle 1: Triad Energy Spectra Exhaustive Verification."""

    def test_exhaustive_8_states_b_plus(self):
        """Exhaustively evaluate all 8 spin states for B_+ at L=1.0.

        Strict invariant: Values must be strictly in {-1.5, +0.5} with degeneracies (2, 6).
        """
        B_plus = get_coupling_matrix("B+", D=3)
        validate_coupling_matrix(B_plus)

        all_states = [
            np.array([s0, s1, s2], dtype=np.float64)
            for s0 in (-1.0, 1.0)
            for s1 in (-1.0, 1.0)
            for s2 in (-1.0, 1.0)
        ]
        self.assertEqual(len(all_states), 8)

        energies = []
        for state in all_states:
            E = calculate_isolated_internal_energy(state, B_plus, L=1.0)
            energies.append(round(E, 10))

            # Ground states: (+1, +1, +1) and (-1, -1, -1)
            if (state == 1.0).all() or (state == -1.0).all():
                self.assertAlmostEqual(E, -1.5, places=12,
                                       msg=f"Ground state {state} expected -1.5, got {E}")
            else:
                self.assertAlmostEqual(E, +0.5, places=12,
                                       msg=f"Excited state {state} expected +0.5, got {E}")

        unique_levels = set(energies)
        self.assertEqual(unique_levels, {-1.5, 0.5},
                         f"B_+ energy levels must be strictly {{-1.5, +0.5}}, got {unique_levels}")

        # Check multiplicity
        counts = {lvl: energies.count(lvl) for lvl in unique_levels}
        self.assertEqual(counts[-1.5], 2, f"Expected 2 ground states, got {counts[-1.5]}")
        self.assertEqual(counts[0.5], 6, f"Expected 6 excited states, got {counts[0.5]}")

        # Enumerate helper check
        spec = enumerate_triad_energy_spectrum(B_plus, L=1.0)
        self.assertEqual(spec, {-1.5: 2, 0.5: 6})

    def test_exhaustive_8_states_b_f(self):
        """Exhaustively evaluate all 8 spin states for B_f at L=1.0.

        Strict invariant: Values must be strictly in {-0.5, +1.5} with degeneracies (6, 2).
        """
        B_f = get_coupling_matrix("Bf", D=3)
        validate_coupling_matrix(B_f)

        all_states = [
            np.array([s0, s1, s2], dtype=np.float64)
            for s0 in (-1.0, 1.0)
            for s1 in (-1.0, 1.0)
            for s2 in (-1.0, 1.0)
        ]

        energies = []
        for state in all_states:
            E = calculate_isolated_internal_energy(state, B_f, L=1.0)
            energies.append(round(E, 10))

            # Frustrated excited states: (+1, -1, -1) and (-1, +1, +1)
            s0, s1, s2 = state
            if (s0 == 1.0 and s1 == -1.0 and s2 == -1.0) or (s0 == -1.0 and s1 == 1.0 and s2 == 1.0):
                self.assertAlmostEqual(E, +1.5, places=12,
                                       msg=f"Frustrated state {state} expected +1.5, got {E}")
            else:
                self.assertAlmostEqual(E, -0.5, places=12,
                                       msg=f"Partially-satisfied state {state} expected -0.5, got {E}")

        unique_levels = set(energies)
        self.assertEqual(unique_levels, {-0.5, 1.5},
                         f"B_f energy levels must be strictly {{-0.5, +1.5}}, got {unique_levels}")

        counts = {lvl: energies.count(lvl) for lvl in unique_levels}
        self.assertEqual(counts[-0.5], 6, f"Expected 6 partially-satisfied states, got {counts[-0.5]}")
        self.assertEqual(counts[1.5], 2, f"Expected 2 frustrated states, got {counts[1.5]}")

        spec = enumerate_triad_energy_spectrum(B_f, L=1.0)
        self.assertEqual(spec, {-0.5: 6, 1.5: 2})

    def test_coupling_linear_scaling_and_signs(self):
        """Verify E_B(x; L) = L * E_B(x; 1.0) to machine precision for varied L."""
        test_L_values = [0.0, 0.25, 0.77, 1.0, 2.5, 3.14159, 10.0, -1.5]
        all_states = [
            np.array([s0, s1, s2], dtype=np.float64)
            for s0 in (-1.0, 1.0)
            for s1 in (-1.0, 1.0)
            for s2 in (-1.0, 1.0)
        ]

        for mat_name in ("B+", "Bf"):
            B = get_coupling_matrix(mat_name, D=3)
            for L in test_L_values:
                for state in all_states:
                    e_1 = calculate_isolated_internal_energy(state, B, L=1.0)
                    e_L = calculate_isolated_internal_energy(state, B, L=L)
                    expected = L * e_1
                    self.assertAlmostEqual(e_L, expected, places=14,
                                           msg=f"Scaling failed for {mat_name}, L={L}, state={state}")

    def test_spin_flip_energy_gradient_consistency(self):
        """Verify exact physical Hamiltonian identity:

        Delta E_alpha = E(x with -x_alpha) - E(x with +x_alpha) == 2 * x_alpha * F^x_{internal, alpha}.
        """
        all_states = [
            np.array([s0, s1, s2], dtype=np.float64)
            for s0 in (-1.0, 1.0)
            for s1 in (-1.0, 1.0)
            for s2 in (-1.0, 1.0)
        ]

        for mat_name in ("B+", "Bf"):
            B = get_coupling_matrix(mat_name, D=3)
            L = 1.35
            for state in all_states:
                fields = calculate_internal_fields(state.reshape(1, 3), B, L=L)[0]
                e_base = calculate_isolated_internal_energy(state, B, L=L)

                for alpha in range(3):
                    flipped_state = state.copy()
                    flipped_state[alpha] = -state[alpha]
                    e_flipped = calculate_isolated_internal_energy(flipped_state, B, L=L)

                    delta_E = e_flipped - e_base
                    expected_delta_E = 2.0 * state[alpha] * fields[alpha]
                    self.assertAlmostEqual(delta_E, expected_delta_E, places=13,
                                           msg=f"Spin-flip Hamiltonian identity failed on {mat_name}, alpha={alpha}")


class TestMemoryImmigrationDeathOracle(unittest.TestCase):
    """Oracle 2: Memory Immigration-Death Expectation Oracle.

    Verifies pure immigration-death Markov jump dynamics:
    dE[S]/dt = N*nu - delta*E[S] => E[S(t)] = (N*nu/delta)*(1 - exp(-delta*t))
    and lim_{t->inf} E[S(t)] = N*nu/delta.
    """

    def test_transient_expectation_ode_curve(self):
        """Verify empirical mean memory follows exact transient analytical ODE curve.

        E[S(t)] = (N * nu / delta) * (1 - exp(-delta * t)) across intermediate checkpoints.
        """
        N = 25
        nu = 1.5
        delta = 0.8
        S_inf = N * nu / delta  # 25 * 1.5 / 0.8 = 46.875
        t_checkpoints = [0.25, 0.5, 1.0, 1.5, 2.0]
        n_replicates = 300

        # Storage for sampled trajectories at checkpoints
        checkpoints_data = {t: [] for t in t_checkpoints}

        for rep in range(n_replicates):
            rng = np.random.default_rng(20000 + rep)
            mgr = MemoryManager(N=N, reciprocal=False)
            t = 0.0
            cp_idx = 0
            n_cp = len(t_checkpoints)

            while cp_idx < n_cp:
                next_cp = t_checkpoints[cp_idx]
                r_birth = float(N) * nu
                r_death = delta * float(mgr.total_memory)
                r_tot = r_birth + r_death

                dt = rng.exponential(1.0 / r_tot)
                if t + dt >= next_cp:
                    # Record memory state at checkpoint
                    checkpoints_data[next_cp].append(mgr.total_memory)
                    cp_idx += 1
                    continue

                t += dt
                if rng.random() < (r_birth / r_tot):
                    i = int(rng.integers(0, N))
                    j = int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    mgr.sample_and_apply_decay(rng)

        # Check each checkpoint against analytical expectation
        for t_cp in t_checkpoints:
            expected_S = S_inf * (1.0 - math.exp(-delta * t_cp))
            samples = np.array(checkpoints_data[t_cp])
            emp_mean = float(np.mean(samples))
            std_err = float(np.std(samples) / math.sqrt(len(samples)))

            # Empirical mean should be within 3 standard errors of analytical expectation
            z_score = abs(emp_mean - expected_S) / std_err
            self.assertLess(z_score, 3.2,
                            msg=f"At t={t_cp}: expected {expected_S:.3f}, got {emp_mean:.3f}, z={z_score:.2f}")

    def test_asymptotic_stationary_expectation_directed(self):
        """Verify lim_{t->inf} E[S(t)] = N*nu/delta for directed memory.

        Also verifies Poisson variance: Var(S) ~ N*nu/delta with Fano factor ~ 1.0.
        Uses independent continuous-time trajectory endpoints at t = 10 / delta.
        """
        N = 30
        nu = 2.0
        delta = 1.0
        S_inf = N * nu / delta  # 60.0
        horizon = 10.0  # 10 relaxation times (exp(-10) < 5e-5)
        n_replicates = 600

        endpoints = []
        for rep in range(n_replicates):
            rng = np.random.default_rng(30000 + rep)
            mgr = MemoryManager(N=N, reciprocal=False)
            t = 0.0
            while t < horizon:
                r_birth = float(N) * nu
                r_death = delta * float(mgr.total_memory)
                r_tot = r_birth + r_death
                dt = rng.exponential(1.0 / r_tot)
                if t + dt > horizon:
                    break
                t += dt
                if rng.random() < (r_birth / r_tot):
                    i = int(rng.integers(0, N))
                    j = int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    mgr.sample_and_apply_decay(rng)
            endpoints.append(mgr.total_memory)

        samples = np.array(endpoints)
        emp_mean = float(np.mean(samples))
        emp_var = float(np.var(samples))

        # Relative error of asymptotic mean < 3%
        rel_error = abs(emp_mean - S_inf) / S_inf
        self.assertLess(rel_error, 0.03,
                        f"Asymptotic mean discrepancy: expected {S_inf}, got {emp_mean}, rel_err={rel_error:.4f}")

        # Poisson property check: Var(S) / Mean(S) ~ 1.0 (Fano factor in [0.85, 1.15])
        fano_factor = emp_var / emp_mean
        self.assertTrue(0.85 <= fano_factor <= 1.15,
                        f"Fano factor for directed immigration-death expected ~1.0, got {fano_factor:.3f}")

    def test_asymptotic_stationary_expectation_reciprocal(self):
        """Verify lim_{t->inf} E[S(t)] = N*nu/delta for reciprocal paired memory.

        Also verifies paired variance doubling: Var(S) ~ 2 * N*nu/delta with Fano factor ~ 2.0.
        Uses independent continuous-time trajectory endpoints at t = 10 / delta.
        """
        N = 30
        nu = 2.0
        delta = 1.0
        S_inf = N * nu / delta  # 60.0
        horizon = 10.0  # 10 relaxation times
        n_replicates = 600

        endpoints = []
        for rep in range(n_replicates):
            rng = np.random.default_rng(40000 + rep)
            mgr = MemoryManager(N=N, reciprocal=True)
            t = 0.0
            while t < horizon:
                r_birth = float(N) * nu / 2.0
                total_pairs = mgr.total_memory // 2
                r_death = delta * float(total_pairs)
                r_tot = r_birth + r_death
                dt = rng.exponential(1.0 / r_tot)
                if t + dt > horizon:
                    break
                t += dt
                if rng.random() < (r_birth / r_tot):
                    i = int(rng.integers(0, N))
                    j = int(rng.integers(0, N - 1))
                    if j >= i:
                        j += 1
                    mgr.add_encounter(i, j)
                else:
                    mgr.sample_and_apply_decay(rng)
            endpoints.append(mgr.total_memory)

        samples = np.array(endpoints)
        emp_mean = float(np.mean(samples))
        emp_var = float(np.var(samples))

        # Expected mean is still N*nu/delta = 60.0
        rel_error = abs(emp_mean - S_inf) / S_inf
        self.assertLess(rel_error, 0.03,
                        f"Reciprocal asymptotic mean: expected {S_inf}, got {emp_mean}, rel_err={rel_error:.4f}")

        # Reciprocal Fano factor Var / Mean ~ 2.0 (in [1.70, 2.30])
        fano_factor = emp_var / emp_mean
        self.assertTrue(1.70 <= fano_factor <= 2.30,
                        f"Reciprocal Fano factor expected ~2.0, got {fano_factor:.3f}")

    def test_poisson_scheduler_immigration_death_integration(self):
        """Verify PoissonScheduler properly updates memory propensity and converges to N*nu/delta."""
        N = 20
        nu = 1.0
        delta = 0.5
        S_inf = N * nu / delta  # 40.0

        rng = np.random.default_rng(50505)
        sched = PoissonScheduler(
            N=N, D=1, tau_x=1e9, tau_y=1e9, nu=nu, delta=delta,
            tau_n=None, reciprocal=False, rng=rng
        )

        mgr = MemoryManager(N=N, reciprocal=False)

        def cb_enc():
            i = int(rng.integers(0, N))
            j = int(rng.integers(0, N - 1))
            if j >= i:
                j += 1
            mgr.add_encounter(i, j)
            sched.set_total_memory(mgr.total_memory)

        def cb_dec():
            mgr.sample_and_apply_decay(rng)
            sched.set_total_memory(mgr.total_memory)

        sched.register_callbacks(cb_encounter=cb_enc, cb_decay=cb_dec)

        # Run until horizon H = 20.0 (well past relaxation: 20 * 0.5 = 10 relaxation times)
        sched.run_until(20.0)

        # Check total memory at end of run
        self.assertGreater(sched.stats.encounters, 0)
        self.assertGreater(sched.stats.memory_decays, 0)
        # Final memory should be around 40
        self.assertTrue(25 <= mgr.total_memory <= 55,
                        f"Scheduler final memory {mgr.total_memory} expected around {S_inf}")


class TestQLearningGlauberEquivalenceOracle(unittest.TestCase):
    """Oracle 3: Q-learning Softmax vs Glauber Heat Bath Equivalence.

    When Q(+-1) = +-F^y and T_Q = T_y:
    Softmax: P(+1) = 1 / (1 + exp(-(Q(+1) - Q(-1))/T_Q)) = 1 / (1 + exp(-2*F^y/T_y))
    Glauber: P(+1) = exp(kappa) / (2 cosh kappa) = 1 / (1 + exp(-2*kappa)) where kappa = F^y/T_y.
    These are mathematically identical.
    """

    def test_machine_precision_equivalence_grid(self):
        """Exhaustively verify softmax == Glauber to machine precision across dense (F^y, T) grid."""
        field_values = np.linspace(-25.0, 25.0, 201)
        temperature_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

        max_discrepancy_plus = 0.0
        max_discrepancy_minus = 0.0

        for Fy in field_values:
            for T in temperature_values:
                # Q-learning setup
                Q_pair = np.array([-Fy, Fy], dtype=np.float64)
                p_minus_q, p_plus_q = softmax_probabilities(Q_pair, T_Q=T)

                # Glauber setup
                kappa = Fy / T
                p_plus_glauber = binary_glauber_prob_plus(kappa)
                p_minus_glauber = 1.0 - p_plus_glauber

                diff_plus = abs(p_plus_q - p_plus_glauber)
                diff_minus = abs(p_minus_q - p_minus_glauber)

                if diff_plus > max_discrepancy_plus:
                    max_discrepancy_plus = diff_plus
                if diff_minus > max_discrepancy_minus:
                    max_discrepancy_minus = diff_minus

        # Must hold to machine precision (< 2e-15)
        self.assertLess(max_discrepancy_plus, 2e-15,
                        f"Max p_plus discrepancy {max_discrepancy_plus} exceeded machine precision limit")
        self.assertLess(max_discrepancy_minus, 2e-15,
                        f"Max p_minus discrepancy {max_discrepancy_minus} exceeded machine precision limit")

    def test_extreme_field_saturation_equivalence(self):
        """Verify overflow-proof saturation to {0.0, 1.0} for extreme fields."""
        T = 1.0
        for extreme_Fy in (400.0, 1000.0, 1e6):
            # Very positive field: P(+1) = 1.0, P(-1) = 0.0
            Q_pos = np.array([-extreme_Fy, extreme_Fy], dtype=np.float64)
            p_minus_q, p_plus_q = softmax_probabilities(Q_pos, T_Q=T)
            p_plus_g = binary_glauber_prob_plus(extreme_Fy / T)
            self.assertEqual(p_plus_q, 1.0)
            self.assertEqual(p_minus_q, 0.0)
            self.assertEqual(p_plus_g, 1.0)

            # Very negative field: P(+1) = 0.0, P(-1) = 1.0
            Q_neg = np.array([extreme_Fy, -extreme_Fy], dtype=np.float64)
            p_minus_q, p_plus_q = softmax_probabilities(Q_neg, T_Q=T)
            p_plus_g = binary_glauber_prob_plus(-extreme_Fy / T)
            self.assertEqual(p_plus_q, 0.0)
            self.assertEqual(p_minus_q, 1.0)
            self.assertEqual(p_plus_g, 0.0)

    def test_q_offset_invariance(self):
        """Verify softmax probabilities are strictly invariant to additive baseline shifts in Q.

        P(s | Q + C) == P(s | Q) == P_Glauber(s).
        """
        Fy = 1.75
        T = 0.8
        p_glauber_plus = binary_glauber_prob_plus(Fy / T)

        for C in (-1e5, -100.0, -1.0, 0.0, 1.0, 100.0, 1e5):
            Q_shifted = np.array([C - Fy, C + Fy], dtype=np.float64)
            _, p_plus = softmax_probabilities(Q_shifted, T_Q=T)
            self.assertAlmostEqual(p_plus, p_glauber_plus, places=14,
                                   msg=f"Offset invariance failed for C={C}")

    def test_dynamical_td_convergence_to_glauber_policy(self):
        """Verify TD updates Q(s) <- Q(s) + eta * [s*Fy - Q(s)] converge to the Glauber distribution."""
        Fy = 1.2
        T = 1.0
        eta = 0.02
        rng = np.random.default_rng(60606)

        Q_pair = np.array([0.0, 0.0], dtype=np.float64)

        # Alternate or sample actions under stationary field
        # Repeated exposure to both actions
        for _ in range(500):
            # Expose +1
            r_plus = compute_immediate_reward(1, Fy)
            update_chosen_q_value(Q_pair, 1, r_plus, eta)

            # Expose -1
            r_minus = compute_immediate_reward(-1, Fy)
            update_chosen_q_value(Q_pair, -1, r_minus, eta)

        # Learned values must be within 1e-4 of +-Fy
        self.assertAlmostEqual(Q_pair[1], +Fy, places=4)
        self.assertAlmostEqual(Q_pair[0], -Fy, places=4)

        # Softmax probability on learned values must match Glauber
        _, p_plus_learned = softmax_probabilities(Q_pair, T_Q=T)
        p_plus_glauber = binary_glauber_prob_plus(Fy / T)
        self.assertAlmostEqual(p_plus_learned, p_plus_glauber, places=4)


class TestParticipationPotentialOracle(unittest.TestCase):
    """Oracle 4: Participation Potential Difference Oracle (Delta Phi == Delta u_k).

    Phi(n) = sum_i (b*q_i - c)*n_i + (g/2) * sum_{i,j} A_{ij}*n_i*n_j
    On symmetric attention matrix A (A == A.T):
    Phi(n with n_k=1) - Phi(n with n_k=0) == Delta u_k(n with n_k=0)
    """

    def test_exhaustive_all_states_complete_graph(self):
        """Verify Delta Phi == Delta u_k for all 2^N states on symmetric 3-node graph."""
        A = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=np.float64)

        b, g, c = 1.4, 0.9, 0.35
        I = 1
        x1 = np.array([-0.8, 0.2, 1.0], dtype=np.float64)
        params = ParticipationParams(I=I, b=b, g=g, c=c)
        tracker = ParticipationTracker(N=3, params=params)
        q = tracker.compute_opposition(x1)

        max_discrepancy = 0.0

        for n0 in (0, 1):
            for n1 in (0, 1):
                for n2 in (0, 1):
                    base_n = np.array([n0, n1, n2], dtype=np.uint8)

                    for k in range(3):
                        n0_state = base_n.copy()
                        n0_state[k] = 0
                        n1_state = base_n.copy()
                        n1_state[k] = 1

                        phi_1 = compute_participation_potential(n1_state, q, A, b=b, g=g, c=c)
                        phi_0 = compute_participation_potential(n0_state, q, A, b=b, g=g, c=c)
                        delta_phi = phi_1 - phi_0

                        tracker.n = n0_state.copy()
                        rho_k = float(np.dot(A[k], n0_state))
                        delta_u_k = tracker.compute_payoff_diff(q[k], rho_k)

                        diff = abs(delta_phi - delta_u_k)
                        if diff > max_discrepancy:
                            max_discrepancy = diff

        self.assertLess(max_discrepancy, 1e-14,
                        f"Max potential difference discrepancy {max_discrepancy} exceeded 1e-14")

    def test_random_symmetric_matrices_various_sizes(self):
        """Verify Delta Phi == Delta u_k holds across diverse population sizes N in {5, 10, 20, 50}."""
        rng = np.random.default_rng(70707)

        for N in (5, 10, 20, 50):
            # Generate random symmetric matrix with zero diagonal
            raw_mat = rng.uniform(0.0, 1.0, size=(N, N))
            sym_mat = 0.5 * (raw_mat + raw_mat.T)
            np.fill_diagonal(sym_mat, 0.0)

            # Normalize so row sums are roughly 1.0 (or keep symmetric directly)
            # Note: to preserve symmetry, we can normalize by overall max or use symmetric scaling
            # For pure potential verification, ANY symmetric matrix with zero diagonal satisfies Delta Phi = Delta u_k
            A = sym_mat / np.max(np.sum(sym_mat, axis=1))

            b = float(rng.uniform(0.5, 2.0))
            g = float(rng.uniform(0.5, 2.0))
            c = float(rng.uniform(0.1, 1.0))
            I = int(rng.choice([-1, 1]))
            x1 = rng.uniform(-1.0, 1.0, size=N)

            params = ParticipationParams(I=I, b=b, g=g, c=c)
            tracker = ParticipationTracker(N=N, params=params)
            q = tracker.compute_opposition(x1)

            # Test 30 random state configurations and random flip agents
            for _ in range(30):
                n_base = rng.choice([0, 1], size=N).astype(np.uint8)
                k = int(rng.integers(0, N))

                n0_state = n_base.copy()
                n0_state[k] = 0
                n1_state = n_base.copy()
                n1_state[k] = 1

                phi_1 = compute_participation_potential(n1_state, q, A, b=b, g=g, c=c)
                phi_0 = compute_participation_potential(n0_state, q, A, b=b, g=g, c=c)
                delta_phi = phi_1 - phi_0

                rho_k = float(np.dot(A[k], n0_state))
                delta_u_k = tracker.compute_payoff_diff(q[k], rho_k)

                diff = abs(delta_phi - delta_u_k)
                self.assertLess(diff, 1e-14,
                                f"Failed on N={N}, k={k}: diff={diff}")

    def test_asymmetric_attention_breaks_potential_identity(self):
        """Verify that when attention matrix A is asymmetric, Delta Phi != Delta u_k.

        Confirms the mathematical theorem that symmetry (A == A.T) is required.
        """
        # Strictly asymmetric matrix
        A_asym = np.array([
            [0.0, 0.8, 0.2],
            [0.1, 0.0, 0.9],
            [0.4, 0.6, 0.0],
        ], dtype=np.float64)
        self.assertFalse(np.allclose(A_asym, A_asym.T))

        b, g, c = 1.0, 2.0, 0.5
        params = ParticipationParams(I=1, b=b, g=g, c=c)
        tracker = ParticipationTracker(N=3, params=params)
        q = np.array([0.5, 0.5, 0.5])

        # Test state n = [0, 1, 0] with agent 0 switching to 1
        n0 = np.array([0, 1, 0], dtype=np.uint8)
        n1 = np.array([1, 1, 0], dtype=np.uint8)

        phi_1 = compute_participation_potential(n1, q, A_asym, b=b, g=g, c=c)
        phi_0 = compute_participation_potential(n0, q, A_asym, b=b, g=g, c=c)
        delta_phi = phi_1 - phi_0

        rho_0 = float(np.dot(A_asym[0], n0))  # A[0, 1] * 1 = 0.8
        delta_u_0 = tracker.compute_payoff_diff(q[0], rho_0)

        # Expected discrepancy: (g/2) * (A_asym[1, 0] - A_asym[0, 1]) = (2/2) * (0.1 - 0.8) = -0.7
        discrepancy = delta_phi - delta_u_0
        self.assertNotAlmostEqual(discrepancy, 0.0, places=3,
                                  msg="Asymmetric matrix must not satisfy exact potential difference")
        self.assertAlmostEqual(discrepancy, 0.5 * g * (A_asym[1, 0] - A_asym[0, 1]), places=14)

    def test_nonzero_diagonal_discrepancy(self):
        """Verify that non-zero diagonal A_kk > 0 introduces exactly (1/2) * g * A_kk discrepancy."""
        A_diag = np.array([
            [0.4, 0.5],
            [0.5, 0.0],
        ], dtype=np.float64)  # Symmetric but A_00 != 0

        b, g, c = 1.0, 1.5, 0.2
        params = ParticipationParams(I=1, b=b, g=g, c=c)
        tracker = ParticipationTracker(N=2, params=params)
        q = np.array([0.5, 0.5])

        n0 = np.array([0, 0], dtype=np.uint8)
        n1 = np.array([1, 0], dtype=np.uint8)

        phi_1 = compute_participation_potential(n1, q, A_diag, b=b, g=g, c=c)
        phi_0 = compute_participation_potential(n0, q, A_diag, b=b, g=g, c=c)
        delta_phi = phi_1 - phi_0

        rho_0 = float(np.dot(A_diag[0], n0))
        delta_u_0 = tracker.compute_payoff_diff(q[0], rho_0)

        discrepancy = delta_phi - delta_u_0
        expected_discrepancy = 0.5 * g * A_diag[0, 0]
        self.assertAlmostEqual(discrepancy, expected_discrepancy, places=14)


if __name__ == "__main__":
    unittest.main()
