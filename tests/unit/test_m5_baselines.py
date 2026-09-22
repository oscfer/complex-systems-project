"""Focused acceptance tests for reference experiments."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.control.contracts import RolloutResult
from src.experiment.baselines import (
    BaselineCondition,
    BaselineDynamics,
    BaselineExperimentConfig,
    ConstructedOriginSpec,
    FiniteReferencePolicy,
    build_binary_flip_generator,
    build_constructed_origin,
    enumerate_reference_catalog,
    exact_binomial_lower_bound,
    fixed_release_survival_probability,
    monte_carlo_fixed_release_probability,
    register_generated_checkpoint,
    run_reference_baselines,
    summarize_reliability,
    wilson_interval,
)


def _result(
    success: bool,
    *,
    eligible: bool = True,
    valid: bool = True,
    restricted_time: float = 10.0,
    origin_id: str = "o",
) -> RolloutResult:
    return RolloutResult(
        problem_id="p",
        policy_id="uniform_field",
        origin_id=origin_id,
        future_id=0,
        is_valid=valid,
        is_eligible=eligible,
        success=success,
        completion_time=restricted_time if success else None,
        restricted_time=restricted_time,
        expenditure=0.0,
        distinct_nodes_used=0,
    )


class TestReferenceEnsembles(unittest.TestCase):
    def test_constructed_origin_is_reproducible_and_exact(self):
        spec = ConstructedOriginSpec(N=16, memory_draws_per_row=6, memory_capacity_per_row=8)
        first = build_constructed_origin(2, master_seed=17, spec=spec)
        second = build_constructed_origin(2, master_seed=17, spec=spec)
        self.assertEqual(first.origin_id, second.origin_id)
        np.testing.assert_array_equal(first.state.x, second.state.x)
        np.testing.assert_array_equal(first.state.y, second.state.y)
        np.testing.assert_array_equal(first.state.W, second.state.W)
        np.testing.assert_array_equal(first.state.K, second.state.K)
        np.testing.assert_array_equal(first.state.opportunity_groups, second.state.opportunity_groups)
        np.testing.assert_array_equal(first.state.W.sum(axis=1), np.full(16, 6))
        np.testing.assert_array_equal(first.state.row_caps, np.full(16, 8))
        np.testing.assert_array_equal(first.state.x, first.state.y)

    def test_distinct_origin_indices_produce_distinct_states(self):
        spec = ConstructedOriginSpec(N=16, memory_draws_per_row=4)
        first = build_constructed_origin(0, master_seed=9, spec=spec)
        second = build_constructed_origin(1, master_seed=9, spec=spec)
        self.assertFalse(np.array_equal(first.state.W, second.state.W))
        other_seed = build_constructed_origin(0, master_seed=10, spec=spec)
        self.assertNotEqual(first.origin_id, other_seed.origin_id)

    def test_homogeneous_reference_traits_are_constant(self):
        spec = ConstructedOriginSpec(
            N=8,
            memory_draws_per_row=3,
            heterogeneous=False,
            constrained_opportunities=False,
        )
        origin = build_constructed_origin(0, master_seed=3, spec=spec)
        np.testing.assert_allclose(origin.state.K, 0.5)
        np.testing.assert_allclose(origin.state.tau_x, 5.0)
        np.testing.assert_allclose(origin.state.nu, 2.0)
        np.testing.assert_allclose(origin.state.g, 1.0)
        np.testing.assert_allclose(origin.state.b, 0.0)
        self.assertIsNone(origin.state.opportunity_groups)

    def test_generated_checkpoint_registration_is_deep(self):
        state = build_constructed_origin(
            0, spec=ConstructedOriginSpec(N=8, memory_draws_per_row=2)
        ).state
        state.t = 12.5
        registered = register_generated_checkpoint(
            state, "generated_001", establishment_time=12.5, source_manifest="pilot.json"
        )
        before = registered.state.x.copy()
        state.x *= -1
        np.testing.assert_array_equal(registered.state.x, before)
        self.assertEqual(registered.metadata["physical_age"], 12.5)
        self.assertEqual(registered.metadata["source_manifest"], "pilot.json")

    def test_constructed_spec_rejects_capacity_below_row_mass(self):
        with self.assertRaises(ValueError):
            ConstructedOriginSpec(N=8, memory_draws_per_row=5, memory_capacity_per_row=4)


class TestReliabilityStatistics(unittest.TestCase):
    def test_exact_lower_bound_edges(self):
        self.assertEqual(exact_binomial_lower_bound(0, 100), 0.0)
        self.assertAlmostEqual(exact_binomial_lower_bound(100, 100), 0.05 ** 0.01, places=12)
        with self.assertRaises(ValueError):
            exact_binomial_lower_bound(2, 1)

    def test_exact_bound_reproduces_predeclared_p0_threshold(self):
        minimum = next(
            successes
            for successes in range(101)
            if exact_binomial_lower_bound(successes, 100) >= 0.8
        )
        self.assertEqual(minimum, 87)

    def test_wilson_interval_contains_point_estimate(self):
        low, high = wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_summary_excludes_invalid_and_ineligible_runs(self):
        runs = [
            _result(True),
            _result(False),
            _result(True, eligible=False),
            _result(False, valid=False),
        ]
        summary = summarize_reliability(
            BaselineCondition.UNIFORM_FIELD, BaselineDynamics.ADAPTIVE, runs
        )
        self.assertEqual(summary.total_rollouts, 4)
        self.assertEqual(summary.valid_rollouts, 3)
        self.assertEqual(summary.eligible_rollouts, 2)
        self.assertEqual(summary.ineligible_rollouts, 1)
        self.assertEqual(summary.invalid_rollouts, 1)
        self.assertEqual(summary.success_probability, 0.5)
        self.assertFalse(summary.certification_available)
        self.assertEqual(summary.certification_reason, "invalid_or_incomplete_runs")
        self.assertTrue(math.isnan(summary.wilson_low))
        self.assertTrue(math.isnan(summary.wilson_high))

    def test_point_target_is_not_a_reliability_certificate(self):
        summary = summarize_reliability(
            BaselineCondition.UNIFORM_FIELD,
            BaselineDynamics.ADAPTIVE,
            [_result(True) for _ in range(4)] + [_result(False)],
        )
        self.assertTrue(summary.meets_point_target)
        self.assertFalse(summary.reliability_certified)
        self.assertFalse(summary.certification_available)
        self.assertEqual(
            summary.certification_reason,
            "nested_futures_require_origin_clustered_inference",
        )
        self.assertTrue(math.isnan(summary.wilson_low))

    def test_empty_eligible_population_is_explicit(self):
        summary = summarize_reliability(
            BaselineCondition.UNCONTROLLED,
            BaselineDynamics.FROZEN,
            [_result(False, eligible=False)],
        )
        self.assertEqual(summary.eligible_rollouts, 0)
        self.assertFalse(summary.reliability_certified)
        self.assertTrue(math.isnan(summary.mean_restricted_time))


class TestMatchedReferenceRunner(unittest.TestCase):
    def setUp(self):
        spec = ConstructedOriginSpec(
            N=4,
            memory_draws_per_row=2,
            memory_capacity_per_row=4,
            public_positive_probability=1.0,
            random_global_sign=False,
            heterogeneous=True,
            between_group_eligibility=0.2,
            within_group_eligibility=0.8,
        )
        self.origin = build_constructed_origin(0, master_seed=41, spec=spec)
        self.config = BaselineExperimentConfig(
            uniform_amplitude=1.0,
            pulse_duration=1.0,
            release_deadline=1.0,
            holding_duration=1.0,
            horizon=3.0,
            master_seed=91,
            sample_interval=1.0,
        )

    def test_runner_creates_matched_zero_and_uniform_branches(self):
        comparison = run_reference_baselines([self.origin], 2, self.config)
        self.assertEqual(comparison.matched_pairs, 2)
        self.assertEqual(len(comparison.results), 8)
        for future_id in range(2):
            matched = [r for r in comparison.results if r.future_id == future_id]
            self.assertEqual(len(matched), 4)
            self.assertEqual(len({r.summaries["matched_physical_seed"] for r in matched}), 1)
        policies = {result.policy_id for result in comparison.results}
        self.assertEqual(
            policies,
            {
                "uncontrolled:adaptive",
                "uniform_field:adaptive",
                "uncontrolled:frozen",
                "uniform_field:frozen",
            },
        )
        for result in comparison.results:
            self.assertEqual(result.summaries["between_eligibility"], 0.2)
            self.assertEqual(result.summaries["within_eligibility"], 0.8)
            if result.summaries["dynamics"] == "frozen":
                self.assertEqual(result.event_counts["encounters"], 0)
                self.assertEqual(result.event_counts["memory_decays"], 0)

    def test_uniform_reference_uses_all_nodes_and_exact_exposure(self):
        comparison = run_reference_baselines([self.origin], 1, self.config)
        controlled = next(r for r in comparison.results if r.policy_id == "uniform_field:adaptive")
        uncontrolled = next(r for r in comparison.results if r.policy_id == "uncontrolled:adaptive")
        self.assertEqual(controlled.distinct_nodes_used, 4)
        self.assertAlmostEqual(controlled.expenditure, 4.0, places=10)
        self.assertEqual(uncontrolled.distinct_nodes_used, 0)
        self.assertEqual(uncontrolled.expenditure, 0.0)

    def test_runner_does_not_mutate_registered_origin(self):
        before = self.origin.state.to_dict()
        run_reference_baselines([self.origin], 1, self.config)
        np.testing.assert_array_equal(self.origin.state.x, before["x"])
        np.testing.assert_array_equal(self.origin.state.y, before["y"])
        np.testing.assert_array_equal(self.origin.state.W, before["W"])
        self.assertEqual(self.origin.state.t, before["t"])

    def test_runner_is_reproducible(self):
        first = run_reference_baselines([self.origin], 1, self.config)
        second = run_reference_baselines([self.origin], 1, self.config)
        left = [(r.success, r.completion_time, r.restricted_time, r.expenditure) for r in first.results]
        right = [(r.success, r.completion_time, r.restricted_time, r.expenditure) for r in second.results]
        self.assertEqual(left, right)


class TestFiniteReference(unittest.TestCase):
    def setUp(self):
        self.attention = np.zeros((4, 4), dtype=np.float64)
        for node in range(4):
            self.attention[node, (node - 1) % 4] = 0.5
            self.attention[node, (node + 1) % 4] = 0.5

    def test_generator_is_valid(self):
        states, generator = build_binary_flip_generator(self.attention)
        self.assertEqual(states.shape, (16, 4))
        np.testing.assert_allclose(generator.sum(axis=1), 0.0, atol=1e-14)
        off_diagonal = generator.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        self.assertTrue(np.all(off_diagonal >= 0.0))
        self.assertTrue(np.all(np.diag(generator) <= 0.0))

    def test_fixed_release_probability_is_bounded_and_hold_monotone(self):
        policy = FiniteReferencePolicy((0, 1), 1.0, 1.0)
        short = fixed_release_survival_probability(self.attention, policy, hold_time=0.25)
        long = fixed_release_survival_probability(self.attention, policy, hold_time=1.0)
        self.assertGreaterEqual(short, 0.0)
        self.assertLessEqual(short, 1.0)
        self.assertGreaterEqual(short + 1e-14, long)

    def test_catalog_enforces_caps_and_sorts_by_probability(self):
        catalog = enumerate_reference_catalog(self.attention)
        self.assertGreater(len(catalog), 1)
        self.assertEqual(sum(result.policy.nodes == () for result in catalog), 1)
        self.assertTrue(all(len(result.policy.nodes) <= 2 for result in catalog))
        self.assertTrue(all(result.exposure <= 2.0 + 1e-12 for result in catalog))
        probabilities = [result.survival_probability for result in catalog]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertTrue(
            any(result.policy.nodes == (0,) and result.policy.amplitude == 2.0 for result in catalog)
        )

    def test_cycle_permutation_symmetry(self):
        values = [
            fixed_release_survival_probability(
                self.attention, FiniteReferencePolicy((node,), 1.0, 1.0)
            )
            for node in range(4)
        ]
        np.testing.assert_allclose(values, values[0], atol=1e-13)

    def test_catalog_matches_independent_planning_reference(self):
        catalog = enumerate_reference_catalog(self.attention)
        self.assertEqual(len(catalog), 25)
        self.assertAlmostEqual(catalog[0].survival_probability, 0.09236217822321065, places=13)
        self.assertEqual(len(catalog[0].policy.nodes), 2)
        self.assertAlmostEqual(catalog[0].policy.amplitude, 1.0)
        trials = 20000
        observed = monte_carlo_fixed_release_probability(
            self.attention, catalog[0].policy, trials=trials, seed=37015
        )
        expected = catalog[0].survival_probability
        standard_error = math.sqrt(expected * (1.0 - expected) / trials)
        self.assertLess(abs(observed - expected), 5.0 * standard_error + 1.0 / trials)


if __name__ == "__main__":
    unittest.main()
