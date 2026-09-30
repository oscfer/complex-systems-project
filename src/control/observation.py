"""Observation generation and latency-managed disclosure models.

Implements:
1. SampledPublicObservationModel:
   - 50% fixed random panel.
   - Binary flip noise (5%) and continuous Gaussian noise (sd=0.1, clipped [-1, 1]).
   - Partial directed tie revelation (50% probability for positive ties between panel members).
   - Unrevealed pairs marked unknown, not zero.
   - Delivery delay (default 1.0 * tau_y) separating snapshot time from decision time.
   - Strict isolation from true private beliefs and simulator RNG streams.
2. CompletePublicObservationModel:
   - Idealized public benchmark (full public state y and full memory W, no private x).
3. OracleObservationModel:
   - Full microstate observation tagged explicitly as is_oracle=True.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.control.contracts import Observation, ObservationMode
from src.core.state import State, StateMode


class ObservationModel(ABC):
    """Abstract base class for state observation generators."""

    @abstractmethod
    def sample_observation(
        self,
        state: State,
        current_time: float,
        remaining_exposure: float = 0.0,
        remaining_distinct_nodes: int = 0,
        current_spending_rate: float = 0.0,
    ) -> Observation:
        """Capture an observation snapshot from physical state."""
        pass


class SampledPublicObservationModel(ObservationModel):
    """Principal realism observation model: delayed, noisy, incomplete public panel."""

    def __init__(
        self,
        N: int,
        panel_fraction: float = 0.5,
        binary_flip_prob: float = 0.05,
        continuous_noise_sd: float = 0.1,
        edge_reveal_prob: float = 0.5,
        delivery_delay: float = 1.0,
        accessible_roster: Optional[Tuple[int, ...]] = None,
        accessible_fraction: float = 0.25,
        group_labels: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        seed: int = 42,
    ) -> None:
        self.N = N
        self.panel_fraction = float(panel_fraction)
        self.binary_flip_prob = float(binary_flip_prob)
        self.continuous_noise_sd = float(continuous_noise_sd)
        self.edge_reveal_prob = float(edge_reveal_prob)
        self.delivery_delay = float(delivery_delay)

        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # 1. Sample fixed random panel of 50% of nodes
        n_panel = max(1, int(round(N * self.panel_fraction)))
        n_panel = min(n_panel, N)
        panel_perm = self.rng.permutation(N)
        self.panel_indices = np.sort(panel_perm[:n_panel])
        self.panel_mask = np.zeros(N, dtype=bool)
        self.panel_mask[self.panel_indices] = True

        # 2. Sample accessible roster (25% of nodes) independently if not provided
        if accessible_roster is not None:
            self.accessible_roster = tuple(sorted(accessible_roster))
        else:
            n_acc = max(1, int(round(N * float(accessible_fraction))))
            n_acc = min(n_acc, N)
            acc_perm = self.rng.permutation(N)
            self.accessible_roster = tuple(sorted(acc_perm[:n_acc].tolist()))

        self.group_labels = np.copy(group_labels) if group_labels is not None else None

    def sample_observation(
        self,
        state: State,
        current_time: float,
        remaining_exposure: float = 0.0,
        remaining_distinct_nodes: int = 0,
        current_spending_rate: float = 0.0,
    ) -> Observation:
        """Capture a noisy, masked observation snapshot from the state."""
        N, D = state.N, state.D
        sample_time = float(current_time)
        delivery_time = sample_time + self.delivery_delay

        # Public states: masked to NaN for non-panel nodes
        public_obs = np.full((N, D), np.nan, dtype=np.float64)

        if state.state_mode == StateMode.BINARY:
            panel_vals = np.copy(state.y[self.panel_indices, :])
            flips = self.rng.random(panel_vals.shape) < self.binary_flip_prob
            panel_vals[flips] = -panel_vals[flips]
            public_obs[self.panel_indices, :] = panel_vals
        else:
            panel_vals = np.copy(state.y[self.panel_indices, :])
            noise = self.rng.normal(0.0, self.continuous_noise_sd, size=panel_vals.shape)
            public_obs[self.panel_indices, :] = np.clip(panel_vals + noise, -1.0, 1.0)

        # Directed tie disclosure for panel members (vectorized lookup on submatrix)
        observed_edges: Dict[Tuple[int, int], int] = {}
        W_panel = state.W[np.ix_(self.panel_indices, self.panel_indices)]
        nonzero_r, nonzero_c = np.nonzero(W_panel)
        # Exclude self-loops
        valid = nonzero_r != nonzero_c
        nonzero_r = nonzero_r[valid]
        nonzero_c = nonzero_c[valid]

        if len(nonzero_r) > 0:
            revealed = self.rng.random(len(nonzero_r)) < self.edge_reveal_prob
            rev_r = nonzero_r[revealed]
            rev_c = nonzero_c[revealed]
            for r_idx, c_idx in zip(rev_r, rev_c):
                orig_i = int(self.panel_indices[r_idx])
                orig_j = int(self.panel_indices[c_idx])
                observed_edges[(orig_i, orig_j)] = int(W_panel[r_idx, c_idx])

        # Groups
        groups = np.copy(state.opportunity_groups) if state.opportunity_groups is not None else self.group_labels

        return Observation(
            time=delivery_time,  # Delivered at delivery_time
            public_states=public_obs,
            panel_mask=np.copy(self.panel_mask),
            observed_edges=observed_edges,
            accessible_roster=self.accessible_roster,
            group_labels=groups,
            sample_time=sample_time,
            delivery_time=delivery_time,
            remaining_exposure=float(remaining_exposure),
            remaining_distinct_nodes=int(remaining_distinct_nodes),
            current_spending_rate=float(current_spending_rate),
            noise_model_version="sampled_public_v1",
            is_oracle=False,
            private_states=None,
            true_memory=None,
        )


class CompletePublicObservationModel(ObservationModel):
    """Complete public state benchmark without noise or missingness."""

    def __init__(
        self,
        N: int,
        accessible_roster: Optional[Tuple[int, ...]] = None,
        delivery_delay: float = 1.0,
    ) -> None:
        self.N = N
        self.delivery_delay = float(delivery_delay)
        self.accessible_roster = tuple(accessible_roster) if accessible_roster is not None else tuple(range(N))

    def sample_observation(
        self,
        state: State,
        current_time: float,
        remaining_exposure: float = 0.0,
        remaining_distinct_nodes: int = 0,
        current_spending_rate: float = 0.0,
    ) -> Observation:
        sample_time = float(current_time)
        delivery_time = sample_time + self.delivery_delay

        observed_edges: Dict[Tuple[int, int], int] = {}
        nonzero_idx = np.argwhere(state.W > 0)
        for i, j in nonzero_idx:
            if i != j:
                observed_edges[(int(i), int(j))] = int(state.W[i, j])

        return Observation(
            time=delivery_time,
            public_states=np.copy(state.y),
            panel_mask=np.ones(state.N, dtype=bool),
            observed_edges=observed_edges,
            accessible_roster=self.accessible_roster,
            group_labels=np.copy(state.opportunity_groups) if state.opportunity_groups is not None else None,
            sample_time=sample_time,
            delivery_time=delivery_time,
            remaining_exposure=float(remaining_exposure),
            remaining_distinct_nodes=int(remaining_distinct_nodes),
            current_spending_rate=float(current_spending_rate),
            noise_model_version="complete_public",
            is_oracle=False,
            private_states=None,
            true_memory=None,
        )


class OracleObservationModel(ObservationModel):
    """Full microstate observation (private beliefs and complete memory included)."""

    def __init__(
        self,
        N: int,
        accessible_roster: Optional[Tuple[int, ...]] = None,
        delivery_delay: float = 0.0,
    ) -> None:
        self.N = N
        self.delivery_delay = float(delivery_delay)
        self.accessible_roster = tuple(accessible_roster) if accessible_roster is not None else tuple(range(N))

    def sample_observation(
        self,
        state: State,
        current_time: float,
        remaining_exposure: float = 0.0,
        remaining_distinct_nodes: int = 0,
        current_spending_rate: float = 0.0,
    ) -> Observation:
        sample_time = float(current_time)
        delivery_time = sample_time + self.delivery_delay

        observed_edges: Dict[Tuple[int, int], int] = {}
        nonzero_idx = np.argwhere(state.W > 0)
        for i, j in nonzero_idx:
            if i != j:
                observed_edges[(int(i), int(j))] = int(state.W[i, j])

        return Observation(
            time=delivery_time,
            public_states=np.copy(state.y),
            panel_mask=np.ones(state.N, dtype=bool),
            observed_edges=observed_edges,
            accessible_roster=self.accessible_roster,
            group_labels=np.copy(state.opportunity_groups) if state.opportunity_groups is not None else None,
            sample_time=sample_time,
            delivery_time=delivery_time,
            remaining_exposure=float(remaining_exposure),
            remaining_distinct_nodes=int(remaining_distinct_nodes),
            current_spending_rate=float(current_spending_rate),
            noise_model_version="oracle",
            is_oracle=True,
            private_states=np.copy(state.x),
            true_memory=np.copy(state.W),
        )
