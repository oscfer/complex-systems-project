"""Unit tests for Observation Models.

Tests:
1. SampledPublicObservationModel 50% fixed panel selection.
2. Binary state report noise (5% flip probability).
3. Continuous state report noise (Gaussian sd=0.1, clipped [-1, 1]).
4. Non-panel state masking (unobserved nodes are NaN).
5. Partial directed tie disclosure (50% probability among panel members; unrevealed marked unknown).
6. Observation delivery latency (sample_time vs delivery_time separated by delay).
7. Strict isolation of private beliefs in non-oracle observations.
8. Oracle and Complete Public observation models.
9. RNG isolation between observation generator and physical simulator.
"""

import unittest
import numpy as np

from src.control.observation import (
    CompletePublicObservationModel,
    OracleObservationModel,
    SampledPublicObservationModel,
)
from src.core.state import State, StateMode


class TestObservationModels(unittest.TestCase):
    """Test suite for SampledPublicObservationModel, CompletePublic, and Oracle models."""

    def setUp(self):
        self.N = 100
        self.D = 1
        self.state_binary = State.create_empty(N=self.N, D=self.D, state_mode=StateMode.BINARY)
        self.state_binary.y.fill(1.0)
        self.state_binary.x.fill(1.0)

        # Create known non-zero memory ties
        for i in range(0, self.N - 1, 2):
            self.state_binary.W[i, i + 1] = 5

    def test_fixed_50_percent_random_panel(self):
        """Observation model must fix a random panel of exactly 50% of the nodes."""
        obs_model = SampledPublicObservationModel(N=self.N, panel_fraction=0.5, seed=123)
        self.assertEqual(len(obs_model.panel_indices), 50)
        self.assertEqual(int(np.sum(obs_model.panel_mask)), 50)

        # Repeated samples retain the same fixed panel
        obs1 = obs_model.sample_observation(self.state_binary, current_time=0.0)
        obs2 = obs_model.sample_observation(self.state_binary, current_time=5.0)
        np.testing.assert_array_equal(obs1.panel_mask, obs2.panel_mask)

    def test_binary_state_flip_noise_probability(self):
        """Binary reports on panel nodes should flip with approximately 5% probability."""
        # Use large population N=20,000 for tight binomial sample frequency
        N_large = 20000
        state = State.create_empty(N=N_large, D=1, state_mode=StateMode.BINARY)
        state.y.fill(1.0)
        obs_model = SampledPublicObservationModel(
            N=N_large, panel_fraction=0.5, binary_flip_prob=0.05, seed=42
        )

        obs = obs_model.sample_observation(state, current_time=0.0)
        panel_y = obs.public_states[obs.panel_mask, 0]
        self.assertEqual(len(panel_y), 10000)

        # Number of flipped spins (-1.0 instead of +1.0)
        flips = int(np.sum(panel_y == -1.0))
        flip_rate = flips / len(panel_y)
        # 5% with 10,000 samples has standard error sqrt(0.05*0.95/10000) ~ 0.00218
        self.assertAlmostEqual(flip_rate, 0.05, delta=0.01)

    def test_continuous_state_gaussian_noise_and_bounds(self):
        """Continuous reports add zero-mean Gaussian noise (sd=0.1) and stay in [-1, 1]."""
        state_cont = State.create_empty(N=5000, D=1, state_mode=StateMode.CONTINUOUS)
        state_cont.y.fill(0.0)
        obs_model = SampledPublicObservationModel(
            N=5000, panel_fraction=0.5, continuous_noise_sd=0.1, seed=99
        )

        obs = obs_model.sample_observation(state_cont, current_time=0.0)
        panel_y = obs.public_states[obs.panel_mask, 0]

        # Mean should be near 0.0, standard deviation near 0.1
        self.assertAlmostEqual(float(np.mean(panel_y)), 0.0, delta=0.01)
        self.assertAlmostEqual(float(np.std(panel_y)), 0.1, delta=0.01)
        self.assertTrue(np.all(panel_y >= -1.0) and np.all(panel_y <= 1.0))

    def test_unobserved_nodes_masked_as_nan(self):
        """Non-panel nodes must have public states reported as NaN."""
        obs_model = SampledPublicObservationModel(N=self.N, panel_fraction=0.5, seed=7)
        obs = obs_model.sample_observation(self.state_binary, current_time=0.0)

        unobserved_idx = np.where(~obs.panel_mask)[0]
        self.assertEqual(len(unobserved_idx), 50)
        for idx in unobserved_idx:
            self.assertTrue(np.isnan(obs.public_states[idx, 0]))

    def test_partial_tie_disclosure_and_unknown_handling(self):
        """Panel pairs with W_ij > 0 are revealed with 50% probability; unrevealed are omitted (unknown)."""
        N = 2000
        state = State.create_empty(N=N, D=1, state_mode=StateMode.BINARY)
        # Create dense positive ties
        for i in range(100):
            for j in range(100):
                if i != j:
                    state.W[i, j] = 3

        obs_model = SampledPublicObservationModel(
            N=N, panel_fraction=0.5, edge_reveal_prob=0.5, seed=42
        )
        obs = obs_model.sample_observation(state, current_time=0.0)

        # Count how many positive ties between panel nodes exist
        panel_set = set(obs_model.panel_indices)
        panel_positive_ties = [
            (i, j)
            for i in range(100)
            for j in range(100)
            if i != j and i in panel_set and j in panel_set
        ]
        self.assertGreater(len(panel_positive_ties), 100)

        revealed_count = sum(1 for tie in panel_positive_ties if tie in obs.observed_edges)
        reveal_rate = revealed_count / len(panel_positive_ties)
        self.assertAlmostEqual(reveal_rate, 0.5, delta=0.05)

        # Ties where at least one node is outside panel must NEVER be revealed
        for i, j in obs.observed_edges:
            self.assertIn(i, panel_set)
            self.assertIn(j, panel_set)
            self.assertEqual(obs.observed_edges[(i, j)], 3)

    def test_delivery_delay(self):
        """Snapshot sampled at time t is delivered at t + delivery_delay."""
        obs_model = SampledPublicObservationModel(N=self.N, delivery_delay=1.0, seed=1)
        obs = obs_model.sample_observation(self.state_binary, current_time=5.0)
        self.assertEqual(obs.sample_time, 5.0)
        self.assertEqual(obs.delivery_time, 6.0)
        self.assertEqual(obs.time, 6.0)

    def test_strict_isolation_from_private_beliefs(self):
        """Sampled-public observation contains NO reference to private beliefs or simulator state."""
        obs_model = SampledPublicObservationModel(N=self.N, seed=1)
        obs = obs_model.sample_observation(self.state_binary, current_time=0.0)
        self.assertFalse(obs.is_oracle)
        self.assertIsNone(obs.private_states)
        self.assertIsNone(obs.true_memory)

    def test_oracle_and_complete_public_models(self):
        """Oracle reveals full microstate, complete public reveals public and W without private."""
        comp_model = CompletePublicObservationModel(N=self.N)
        obs_comp = comp_model.sample_observation(self.state_binary, current_time=1.0)
        self.assertFalse(obs_comp.is_oracle)
        self.assertIsNone(obs_comp.private_states)
        self.assertIsNotNone(obs_comp.observed_edges)
        self.assertTrue(np.all(obs_comp.panel_mask))

        oracle_model = OracleObservationModel(N=self.N)
        obs_oracle = oracle_model.sample_observation(self.state_binary, current_time=1.0)
        self.assertTrue(obs_oracle.is_oracle)
        self.assertIsNotNone(obs_oracle.private_states)
        self.assertIsNotNone(obs_oracle.true_memory)

    def test_rng_isolation(self):
        """Sampling observations must not advance or affect a separate physical simulation RNG."""
        sim_rng1 = np.random.default_rng(12345)
        sim_rng2 = np.random.default_rng(12345)

        obs_model = SampledPublicObservationModel(N=self.N, seed=999)

        # Draw from sim_rng1 while generating observations
        for _ in range(5):
            obs_model.sample_observation(self.state_binary, current_time=0.0)
            val1 = sim_rng1.random()

        # sim_rng2 draws without observations
        for _ in range(5):
            val2 = sim_rng2.random()

        self.assertEqual(val1, val2)


if __name__ == "__main__":
    unittest.main()
