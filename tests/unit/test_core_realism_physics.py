"""Unit tests for Realism Physics Revision.

Tests:
1. Clipped lognormal multipliers, moments, bounds, and sigma=0 baseline equivalence.
2. Truncated normal biases, truncation bounds, and sigma=0 baseline equivalence.
3. Susceptibility generation and bounds.
4. QuenchedHeterogeneity container and generator.
5. Opportunity groups partition, interface matrix E, eligibility matrix C, and cross-group probability.
6. MemoryManager bounded row capacity caps, atomic replacement, and self-tie invariance.
7. Local field computations with heterogeneous K_i, b_{i, alpha}, and g_i * u_{i, alpha}.
8. PoissonScheduler with heterogeneous tau_x, nu, and weighted node sampling.
9. State serialization and checkpoint roundtrip with realism fields.
"""

import os
import tempfile
import unittest
import numpy as np

from src.core.heterogeneity import (
    DEFAULT_BIAS_BOUNDS_OVER_J,
    DEFAULT_BIAS_SD_OVER_J,
    DEFAULT_CROSS_GROUP_ELIGIBILITY,
    DEFAULT_MULTIPLIER_BOUNDS,
    DEFAULT_NUM_OPPORTUNITY_GROUPS,
    DEFAULT_ROW_CAP,
    DEFAULT_SUSCEPTIBILITY_BOUNDS,
    DEFAULT_WITHIN_GROUP_ELIGIBILITY,
    QuenchedHeterogeneity,
    calculate_cross_group_exploration_probability,
    clipped_lognormal_multiplier,
    generate_opportunity_groups,
    generate_quenched_heterogeneity,
    generate_susceptibilities,
    generate_truncated_normal_biases,
)
from src.core.state import State, StateMode
from src.core.memory import MemoryManager
from src.core.fields import compute_public_fields, compute_private_fields
from src.core.scheduler import PoissonScheduler


class TestRealismHeterogeneity(unittest.TestCase):
    """Test suite for quenched heterogeneity sampling functions and dataclass."""

    def test_clipped_lognormal_multiplier_zero_sigma(self):
        """Zero sigma must identically return 1.0, preserving homogeneous baseline."""
        res = clipped_lognormal_multiplier(0.0, size=100)
        self.assertEqual(res.shape, (100,))
        self.assertTrue(np.all(res == 1.0))

    def test_clipped_lognormal_multiplier_bounds(self):
        """Samples must stay strictly within clipping bounds [0.5, 2.0]."""
        rng = np.random.default_rng(12345)
        samples = clipped_lognormal_multiplier(0.5, size=20000, bounds=(0.5, 2.0), rng=rng)
        self.assertGreaterEqual(np.min(samples), 0.5)
        self.assertLessEqual(np.max(samples), 2.0)
        # Check that clipping actually bounded extreme values
        raw = np.exp(0.5 * rng.standard_normal(20000) - 0.125)
        self.assertTrue(np.any(raw < 0.5) or np.any(raw > 2.0))

    def test_clipped_lognormal_multiplier_mean(self):
        """Sample mean should match expected clipped lognormal moments."""
        rng = np.random.default_rng(15092026)
        samples = clipped_lognormal_multiplier(0.5, size=50000, rng=rng)
        mean_val = float(np.mean(samples))
        # With bounds [0.5, 2.0] and sigma=0.5, mean is approximately 0.978 +/- 0.02
        self.assertAlmostEqual(mean_val, 0.978, delta=0.03)

    def test_clipped_lognormal_multiplier_invalid_args(self):
        """Negative sigma or invalid bounds must raise ValueError."""
        with self.assertRaises(ValueError):
            clipped_lognormal_multiplier(-0.1)
        with self.assertRaises(ValueError):
            clipped_lognormal_multiplier(0.5, bounds=(2.0, 0.5))

    def test_truncated_normal_biases_zero_sigma(self):
        """Zero sigma_over_J must identically return zeros."""
        b = generate_truncated_normal_biases(N=50, D=3, J=1.0, sigma_over_J=0.0)
        self.assertEqual(b.shape, (50, 3))
        self.assertTrue(np.all(b == 0.0))

    def test_truncated_normal_biases_bounds_and_scaling(self):
        """Biases must stay strictly in [-0.45 * J, 0.45 * J]."""
        rng = np.random.default_rng(42)
        J = 2.5
        b = generate_truncated_normal_biases(
            N=5000,
            D=3,
            J=J,
            sigma_over_J=0.15,
            bounds_over_J=(-0.45, 0.45),
            rng=rng,
        )
        self.assertEqual(b.shape, (5000, 3))
        self.assertGreaterEqual(np.min(b), -0.45 * J)
        self.assertLessEqual(np.max(b), 0.45 * J)
        # Mean should be close to 0 due to symmetry
        self.assertAlmostEqual(float(np.mean(b)), 0.0, delta=0.02)

    def test_generate_susceptibilities(self):
        """Susceptibilities must fall in [0.5, 1.5], or degenerate to 1.0."""
        rng = np.random.default_rng(999)
        g = generate_susceptibilities(N=1000, bounds=(0.5, 1.5), rng=rng)
        self.assertEqual(g.shape, (1000,))
        self.assertGreaterEqual(np.min(g), 0.5)
        self.assertLessEqual(np.max(g), 1.5)

        g_deg = generate_susceptibilities(N=100, bounds=(1.0, 1.0))
        self.assertTrue(np.all(g_deg == 1.0))

    def test_quenched_heterogeneity_generator(self):
        """Unified generator should generate valid traits matching specifications."""
        rng = np.random.default_rng(777)
        qh = generate_quenched_heterogeneity(
            N=100,
            D=3,
            J=1.2,
            K_nominal=0.6,
            tau_x_nominal=4.0,
            nu_nominal=2.5,
            rng=rng,
        )
        self.assertEqual(qh.K.shape, (100,))
        self.assertEqual(qh.tau_x.shape, (100,))
        self.assertEqual(qh.nu.shape, (100,))
        self.assertEqual(qh.g.shape, (100,))
        self.assertEqual(qh.b.shape, (100, 3))
        self.assertEqual(qh.q_K.shape, (100,))
        self.assertEqual(qh.q_tau_x.shape, (100,))
        self.assertEqual(qh.q_nu.shape, (100,))

        # Dictionary serialization
        d = qh.to_dict()
        self.assertEqual(set(d.keys()), {"K", "tau_x", "nu", "g", "b", "q_K", "q_tau_x", "q_nu"})

    def test_quenched_heterogeneity_zero_variance_baseline(self):
        """Zero sigma parameters must recreate exact nominal baseline."""
        qh = generate_quenched_heterogeneity(
            N=20,
            D=1,
            J=1.0,
            K_nominal=0.5,
            tau_x_nominal=5.0,
            nu_nominal=2.0,
            K_sigma=0.0,
            tau_x_sigma=0.0,
            nu_sigma=0.0,
            bias_sigma_over_J=0.0,
            g_bounds=(1.0, 1.0),
        )
        self.assertTrue(np.all(qh.K == 0.5))
        self.assertTrue(np.all(qh.tau_x == 5.0))
        self.assertTrue(np.all(qh.nu == 2.0))
        self.assertTrue(np.all(qh.g == 1.0))
        self.assertTrue(np.all(qh.b == 0.0))


class TestOpportunityStructures(unittest.TestCase):
    """Test suite for constrained encounter opportunity groups and eligibility."""

    def test_opportunity_groups_partition(self):
        """Opportunity groups should partition N agents into equal-sized blocks."""
        N = 1000
        num_groups = 4
        groups, E, C = generate_opportunity_groups(
            N=N,
            num_groups=num_groups,
            between_eligibility=0.05,
            within_eligibility=1.0,
            randomize=False,
        )
        self.assertEqual(groups.shape, (N,))
        counts = np.bincount(groups, minlength=num_groups)
        self.assertTrue(np.all(counts == 250))

        # Interface matrix E
        self.assertEqual(E.shape, (4, 4))
        self.assertTrue(np.all(np.diag(E) == 1.0))
        self.assertTrue(np.all(E[~np.eye(4, dtype=bool)] == 0.05))

        # Full eligibility matrix C
        self.assertEqual(C.shape, (N, N))
        self.assertTrue(np.all(np.diag(C) == 0.0))

        # Check within-group vs cross-group entries
        for i in range(10):
            for j in range(10):
                if i == j:
                    self.assertEqual(C[i, j], 0.0)
                elif groups[i] == groups[j]:
                    self.assertEqual(C[i, j], 1.0)
                else:
                    self.assertEqual(C[i, j], 0.05)

    def test_cross_group_exploration_probability(self):
        """Cross-group probability formula matches theoretical value 0.130890."""
        # For N=1000, 4 groups: within=249 * 1.0, outside=750 * 0.05 = 37.5
        # cross_p = 37.5 / (249 + 37.5) = 37.5 / 286.5 = 0.13089005...
        p = calculate_cross_group_exploration_probability(
            N=1000,
            num_groups=4,
            between_eligibility=0.05,
            within_eligibility=1.0,
        )
        self.assertAlmostEqual(p, 37.5 / 286.5, places=6)
        self.assertAlmostEqual(p, 0.130890, places=5)


class TestBoundedMemoryCaps(unittest.TestCase):
    """Test suite for row-bounded memory capacity caps and atomic replacement."""

    def test_memory_row_caps_initialization(self):
        """MemoryManager must initialize and update row caps."""
        mm = MemoryManager(N=10, capacity=40)
        self.assertEqual(mm.capacity, 40)
        self.assertEqual(mm.row_caps.shape, (10,))
        self.assertTrue(np.all(mm.row_caps == 40))

        # Update row caps
        mm.set_row_caps(20)
        self.assertEqual(mm.capacity, 20)
        self.assertTrue(np.all(mm.row_caps == 20))

    def test_atomic_replacement_at_capacity(self):
        """When row is full (s_i == C_i), adding an encounter preserves row sum exactly."""
        rng = np.random.default_rng(42)
        mm = MemoryManager(N=5, capacity=10, rng=rng)

        # Fill row 0 to capacity 10 with encounters to node 1
        for _ in range(10):
            mm.add_encounter(0, 1)

        self.assertEqual(mm.row_sums[0], 10)
        self.assertEqual(mm.W[0, 1], 10)

        # Now add encounter from 0 to 2
        mm.add_encounter(0, 2)

        # Total memory and row sum must remain exactly 10
        self.assertEqual(mm.row_sums[0], 10)
        self.assertEqual(mm.W[0, 2], 1)
        self.assertEqual(mm.W[0, 1], 9)
        self.assertEqual(int(np.sum(mm.W[0])), 10)
        self.assertTrue(np.all(mm.W >= 0))

    def test_self_replacement_invariance(self):
        """When the chosen unit to remove targets the same agent, state remains unchanged."""
        rng = np.random.default_rng(42)
        mm = MemoryManager(N=5, capacity=5, rng=rng)

        # Fill row 0 with 5 encounters to node 1
        for _ in range(5):
            mm.add_encounter(0, 1)

        self.assertEqual(mm.W[0, 1], 5)

        # Adding another encounter to node 1: node 1 MUST be the removed unit, net change = 0
        mm.add_encounter(0, 1)

        self.assertEqual(mm.row_sums[0], 5)
        self.assertEqual(mm.W[0, 1], 5)
        self.assertEqual(int(np.sum(mm.W[0])), 5)
        self.assertTrue(np.all(mm.W >= 0))

    def test_multistep_atomic_replacement_stress(self):
        """Multiple consecutive encounters at capacity preserve non-negativity and row sums."""
        rng = np.random.default_rng(12345)
        mm = MemoryManager(N=8, capacity=15, rng=rng)

        # Populate some random encounters
        for _ in range(200):
            i = rng.integers(0, 8)
            j = rng.integers(0, 8)
            if i != j:
                mm.add_encounter(i, j)

        # Check all rows never exceed capacity
        for i in range(8):
            self.assertLessEqual(mm.row_sums[i], 15)
            self.assertEqual(mm.row_sums[i], int(np.sum(mm.W[i])))
            self.assertEqual(mm.W[i, i], 0)
        self.assertTrue(np.all(mm.W >= 0))


class TestHeterogeneousFieldsAndScheduler(unittest.TestCase):
    """Test suite for heterogeneous fields and scheduler rate integration."""

    def test_public_fields_heterogeneous(self):
        """Public fields must support array K_i and g_i * u_{i, alpha}."""
        N, D = 4, 2
        A = np.zeros((N, N), dtype=np.float64)
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[2, 3] = 1.0
        A[3, 2] = 1.0

        y = np.ones((N, D), dtype=np.float64)
        x = np.ones((N, D), dtype=np.float64) * 0.5
        K = np.array([0.2, 0.4, 0.6, 0.8])
        g = np.array([0.5, 1.0, 1.5, 2.0])
        u = np.full((N, D), 0.1)

        Fy = compute_public_fields(A=A, y=y, x=x, J=1.0, K=K, g=g, u=u)
        self.assertEqual(Fy.shape, (N, D))

        # For agent i: Fy_i = 1.0 * (1.0) + K_i * 0.5 + g_i * 0.1
        for i in range(N):
            expected = 1.0 + K[i] * 0.5 + g[i] * 0.1
            np.testing.assert_allclose(Fy[i], expected)

    def test_private_fields_heterogeneous(self):
        """Private fields must support array K_i and intrinsic biases b_{i, alpha}."""
        N, D = 4, 2
        x = np.ones((N, D), dtype=np.float64)
        y = np.ones((N, D), dtype=np.float64) * 0.5
        K = np.array([0.3, 0.5, 0.7, 0.9])
        b = np.array([
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.4, -0.4],
        ])

        Fx = compute_private_fields(x=x, y=y, K=K, b=b)
        self.assertEqual(Fx.shape, (N, D))

        # For agent i: Fx_i = K_i * y_i + b_i
        for i in range(N):
            expected = K[i] * 0.5 + b[i]
            np.testing.assert_allclose(Fx[i], expected)

    def test_scheduler_heterogeneous_rates(self):
        """Scheduler propensities must correctly sum over heterogeneous rates."""
        N, D = 4, 1
        tau_x = np.array([2.0, 4.0, 4.0, 8.0])
        nu = np.array([1.0, 2.0, 3.0, 4.0])

        sched = PoissonScheduler(
            N=N,
            D=D,
            tau_x=tau_x,
            tau_y=10.0,
            nu=nu,
            delta=0.1,
            reciprocal=False,
        )

        # R_x = D * sum(1 / tau_{x,i}) = 1 * (0.5 + 0.25 + 0.25 + 0.125) = 1.125
        self.assertAlmostEqual(sched.R_x, 1.125)
        # R_enc = sum(nu_i) = 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        self.assertAlmostEqual(sched.R_enc, 10.0)

    def test_scheduler_weighted_node_sampling(self):
        """Weighted node sampling methods must follow respective rate distributions."""
        N, D = 3, 1
        # tau_x: [1.0, 2.0, 4.0] => weights 1/tau_x: [1.0, 0.5, 0.25] => sum 1.75 => probs [4/7, 2/7, 1/7]
        tau_x = np.array([1.0, 2.0, 4.0])
        # nu: [1.0, 3.0, 6.0] => probs [0.1, 0.3, 0.6]
        nu = np.array([1.0, 3.0, 6.0])

        rng = np.random.default_rng(42)
        sched = PoissonScheduler(
            N=N,
            D=D,
            tau_x=tau_x,
            tau_y=5.0,
            nu=nu,
            delta=0.1,
            rng=rng,
        )

        # Sample private nodes
        private_samples = [sched.sample_private_node() for _ in range(14000)]
        private_counts = np.bincount(private_samples, minlength=3)
        private_probs = private_counts / 14000.0
        np.testing.assert_allclose(private_probs, [4/7, 2/7, 1/7], atol=0.02)

        # Sample encounter initiators
        enc_samples = [sched.sample_encounter_initiator() for _ in range(10000)]
        enc_counts = np.bincount(enc_samples, minlength=3)
        enc_probs = enc_counts / 10000.0
        np.testing.assert_allclose(enc_probs, [0.1, 0.3, 0.6], atol=0.02)


class TestStateRealismIntegration(unittest.TestCase):
    """Test suite for State factory, deep copy, and checkpoint serialization with realism fields."""

    def test_state_create_realism_state(self):
        """Factory State.create_realism_state must initialize all realism attributes."""
        rng = np.random.default_rng(101)
        state = State.create_realism_state(
            N=20,
            D=3,
            state_mode=StateMode.BINARY,
            row_cap=40,
            enable_opportunity_groups=True,
            num_groups=4,
            rng=rng,
        )
        state.validate()

        self.assertIsNotNone(state.K)
        self.assertIsNotNone(state.tau_x)
        self.assertIsNotNone(state.nu)
        self.assertIsNotNone(state.g)
        self.assertIsNotNone(state.b)
        self.assertIsNotNone(state.row_caps)
        self.assertIsNotNone(state.opportunity_groups)

        self.assertEqual(state.K.shape, (20,))
        self.assertEqual(state.tau_x.shape, (20,))
        self.assertEqual(state.nu.shape, (20,))
        self.assertEqual(state.g.shape, (20,))
        self.assertEqual(state.b.shape, (20, 3))
        self.assertEqual(state.row_caps.shape, (20,))
        self.assertEqual(state.opportunity_groups.shape, (20,))
        self.assertTrue(np.all(state.row_caps == 40))

    def test_state_deep_copy_realism_fields(self):
        """Deep copy must isolate all realism buffers."""
        rng = np.random.default_rng(102)
        s1 = State.create_realism_state(N=10, D=1, rng=rng)
        s2 = s1.copy(deep=True)

        # Mutate s2 buffers
        s2.K[0] += 10.0
        s2.b[0, 0] += 5.0
        s2.row_caps[0] += 10

        self.assertNotEqual(s1.K[0], s2.K[0])
        self.assertNotEqual(s1.b[0, 0], s2.b[0, 0])
        self.assertNotEqual(s1.row_caps[0], s2.row_caps[0])

    def test_state_checkpoint_roundtrip_realism(self):
        """Saving and loading checkpoint preserves all realism fields exactly."""
        rng = np.random.default_rng(103)
        state = State.create_realism_state(
            N=16,
            D=3,
            state_mode=StateMode.CONTINUOUS,
            row_cap=30,
            enable_opportunity_groups=True,
            rng=rng,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "realism_state.npz")
            state.save_checkpoint(ckpt_path)
            loaded = State.load_checkpoint(ckpt_path)
            loaded.validate()

            self.assertEqual(loaded.N, state.N)
            self.assertEqual(loaded.D, state.D)
            self.assertEqual(loaded.state_mode, state.state_mode)
            np.testing.assert_array_equal(loaded.x, state.x)
            np.testing.assert_array_equal(loaded.y, state.y)
            np.testing.assert_array_equal(loaded.W, state.W)
            np.testing.assert_array_equal(loaded.K, state.K)
            np.testing.assert_array_equal(loaded.tau_x, state.tau_x)
            np.testing.assert_array_equal(loaded.nu, state.nu)
            np.testing.assert_array_equal(loaded.g, state.g)
            np.testing.assert_array_equal(loaded.b, state.b)
            np.testing.assert_array_equal(loaded.row_caps, state.row_caps)
            np.testing.assert_array_equal(loaded.opportunity_groups, state.opportunity_groups)


class TestRealismAdversarialFixes(unittest.TestCase):
    """Adversarial test suite validating reviewer fixes and edge-case invariants."""

    def test_float_scalar_capacity_support(self):
        """Float scalar capacity like 40.0 or np.float64(40) must be accepted cleanly."""
        mm = MemoryManager(N=4, capacity=40.0)
        self.assertEqual(mm.capacity, 40)
        self.assertTrue(np.all(mm.row_caps == 40))

        # Float in set_row_caps
        mm.set_row_caps(20.0)
        self.assertEqual(mm.capacity, 20)

        # Non-integer float must raise ValueError
        with self.assertRaises(ValueError):
            mm.set_row_caps(20.5)

        # Non-positive float must raise ValueError
        with self.assertRaises(ValueError):
            mm.set_row_caps(0.0)

    def test_reciprocal_memory_rejection_with_capacity(self):
        """Reciprocal memory with row caps must be rejected by specification."""
        with self.assertRaises(ValueError):
            MemoryManager(N=4, reciprocal=True, capacity=40)

        mm_recip = MemoryManager(N=4, reciprocal=True)
        with self.assertRaises(ValueError):
            mm_recip.set_row_caps(40)

    def test_fields_1d_broadcasting_guard(self):
        """When D=1, 1D array parameters of length N must never broadcast to (N, N)."""
        N, D = 5, 1
        x = np.ones((N, D), dtype=np.float64)
        y = np.ones((N, D), dtype=np.float64)
        A = np.zeros((N, N), dtype=np.float64)

        b_1d = np.linspace(-0.2, 0.2, N)
        Fx = compute_private_fields(x=x, y=y, b=b_1d)
        self.assertEqual(Fx.shape, (N, 1), f"Expected (N, 1), got {Fx.shape}")

        g_1d = np.ones(N) * 1.2
        u_1d = np.ones(N) * 0.3
        h_1d = np.ones(N) * 0.1
        Fy = compute_public_fields(A=A, y=y, x=x, h=h_1d, g=g_1d, u=u_1d)
        self.assertEqual(Fy.shape, (N, 1), f"Expected (N, 1), got {Fy.shape}")

    def test_capacity_validation_against_existing_memory(self):
        """set_row_caps and State.validate must reject capacities smaller than existing memory."""
        mm = MemoryManager(N=4)
        for _ in range(5):
            mm.add_encounter(0, 1)

        self.assertEqual(mm.row_sums[0], 5)

        # Trying to cap at 3 when row has 5 must fail
        with self.assertRaises(ValueError):
            mm.set_row_caps(3)

        # Trying to set per-row caps where row 0 exceeds cap must fail
        with self.assertRaises(ValueError):
            mm.set_row_caps(np.array([4, 10, 10, 10]))

        # Setting cap >= 5 succeeds
        mm.set_row_caps(5)
        self.assertEqual(mm.capacity, 5)

        # State.validate check
        W = np.zeros((4, 4), dtype=np.uint32)
        W[0, 1] = 10
        state = State(
            N=4,
            D=1,
            state_mode=StateMode.BINARY,
            x=np.ones((4, 1)),
            y=np.ones((4, 1)),
            W=W,
            n=np.zeros(4, dtype=np.uint8),
            Q=np.zeros((4, 1, 2)),
            row_caps=np.full(4, 5, dtype=np.uint32),
        )
        with self.assertRaises(ValueError):
            state.validate()

    def test_truncated_biases_invalid_bounds_rejected(self):
        """Invalid bounds_over_J where low > high must raise ValueError."""
        with self.assertRaises(ValueError):
            generate_truncated_normal_biases(N=10, D=1, bounds_over_J=(0.5, -0.5))

    def test_cross_group_exploration_probability_edge_cases(self):
        """N < num_groups must raise ValueError; N mod 4 != 0 evaluates exact population average."""
        with self.assertRaises(ValueError):
            calculate_cross_group_exploration_probability(N=3, num_groups=4)

        # N=10, num_groups=4: groups have sizes [3, 3, 2, 2]
        p_exact = calculate_cross_group_exploration_probability(
            N=10,
            num_groups=4,
            between_eligibility=0.05,
            within_eligibility=1.0,
        )
        self.assertGreater(p_exact, 0.0)
        self.assertLess(p_exact, 1.0)

    def test_opportunity_groups_attribute_preserved(self):
        """MemoryManager must preserve and expose opportunity_groups."""
        groups = np.array([0, 1, 2, 3])
        mm = MemoryManager(N=4, opportunity_groups=groups)
        self.assertIsNotNone(mm.opportunity_groups)
        np.testing.assert_array_equal(mm.opportunity_groups, groups)

    def test_scheduler_precomputed_probabilities_and_invokers(self):
        """Fast invokers and precomputed sampling probabilities behave identically to original."""
        N, D = 4, 1
        tau_x = np.array([1.0, 2.0, 2.0, 4.0])
        nu = np.array([2.0, 2.0, 4.0, 8.0])
        sched = PoissonScheduler(N=N, D=D, tau_x=tau_x, nu=nu)

        self.assertIsNotNone(sched._p_tau_x)
        self.assertIsNotNone(sched._p_nu)
        self.assertAlmostEqual(float(np.sum(sched._p_tau_x)), 1.0)
        self.assertAlmostEqual(float(np.sum(sched._p_nu)), 1.0)

        received_nodes = []
        def enc_cb(node):
            received_nodes.append(node)

        sched.register_callbacks(cb_encounter=enc_cb)
        self.assertIsNotNone(sched._invoker_encounter)

        # Force trigger step
        for _ in range(50):
            sched.step()

        self.assertGreater(len(received_nodes), 0)
        for node in received_nodes:
            self.assertIn(node, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
