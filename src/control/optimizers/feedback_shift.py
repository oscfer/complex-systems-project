"""Feedback-dependent shift algorithms for optimal transformation control.

Implements:
1. FeedbackTargetShiftController adhering strictly to the Observation layer contract.
2. Noise-resilient public state tracking with exponential belief smoothing.
3. Dynamic target shifting across the intervention horizon:
   - Detecting target alignment on active nodes
   - Zeroing field on aligned nodes to preserve exposure budget
   - Reallocating remaining exposure budget to unaligned or newly recruited accessible nodes
   - Respecting distinct-node cardinality limit k and exposure cap B
4. Graceful fallback allocation for missing observation panels, ties, or budget exhaustion.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.contracts import Action, ActionEntry, ControlProblem, Observation, TargetType
from src.control.controllers import Controller, TIE_TOLERANCE
from src.control.ledger import BudgetLedger
from src.control.optimizers.heuristics import (
    allocate_greedy_budget,
    compute_degree_scores,
    compute_influence_scores,
    select_heuristic_candidates,
)


class ShiftStrategy(str, Enum):
    """Policy strategies for feedback-driven target shifting."""
    THRESHOLD = "threshold"
    GREEDY_MARGINAL = "greedy_marginal"
    PERIODIC = "periodic"


class FeedbackTargetShiftController(Controller):
    """Dynamic controller shifting active targets based on delayed, noisy public observations.

    Architectural Compliance:
    - Operates STRICTLY on Layer 2 via the Observation and BudgetLedger contracts.
    - NEVER touches or reads private_states (x_i) or true_memory (W).
    - Robust against binary flip noise, continuous Gaussian noise, and missing panel masks.
    """

    def __init__(
        self,
        nominal_amplitude: float = 1.0,
        duration: float = 5.0,
        initial_recruitment_fraction: float = 0.5,
        alignment_threshold: float = 0.5,
        smoothing_alpha: float = 0.7,
        quota: float = 0.5,
        shift_strategy: ShiftStrategy = ShiftStrategy.THRESHOLD,
        shift_period: Optional[float] = None,
        component: int = 0,
        seed: int = 42,
    ) -> None:
        if nominal_amplitude < 0.0:
            raise ValueError(f"nominal_amplitude must be >= 0, got {nominal_amplitude}")
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {duration}")
        if not (0.0 <= initial_recruitment_fraction <= 1.0):
            raise ValueError(f"initial_recruitment_fraction must be in [0, 1], got {initial_recruitment_fraction}")
        if not (0.0 <= alignment_threshold <= 1.0):
            raise ValueError(f"alignment_threshold must be in [0, 1], got {alignment_threshold}")
        if not (0.0 <= smoothing_alpha <= 1.0):
            raise ValueError(f"smoothing_alpha must be in [0, 1], got {smoothing_alpha}")
        if not (0.0 <= quota <= 1.0):
            raise ValueError(f"quota must be in [0, 1], got {quota}")
        if component < 0:
            raise ValueError(f"component must be >= 0, got {component}")
        if shift_period is not None and shift_period <= 0.0:
            raise ValueError(f"shift_period must be > 0, got {shift_period}")

        self.nominal_amplitude = float(nominal_amplitude)
        self.duration = float(duration)
        self.initial_recruitment_fraction = float(initial_recruitment_fraction)
        self.alignment_threshold = float(alignment_threshold)
        self.smoothing_alpha = float(smoothing_alpha)
        self.quota = float(quota)
        self.shift_strategy = ShiftStrategy(shift_strategy)
        self.shift_period = float(shift_period) if shift_period is not None else None
        self.component = int(component)
        self.seed = int(seed)

        self.problem: Optional[ControlProblem] = None
        self.rng: Optional[np.random.Generator] = None

        # Internal episode tracking
        self.s_star: Optional[float] = None
        self.committed_end_time: Optional[float] = None
        self.initial_actuation_done: bool = False
        self.actuated_distinct_nodes: Set[int] = set()
        self.active_nodes: Set[int] = set()
        self.smoothed_public_states: Dict[int, float] = {}
        self.last_shift_time: float = 0.0

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Reset controller state for a new rollout episode."""
        if self.component < 0 or self.component >= problem.D:
            raise ValueError(f"component must be in [0, {problem.D}), got {self.component}")
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)
        self.s_star = None
        self.committed_end_time = None
        self.initial_actuation_done = False
        self.actuated_distinct_nodes = set()
        self.active_nodes = set()
        self.smoothed_public_states = {}
        self.last_shift_time = 0.0

    def _update_smoothed_beliefs(self, observation: Observation) -> None:
        """Update noise-smoothed public state beliefs using ONLY observed panel nodes."""
        N = observation.public_states.shape[0]
        alpha = self.smoothing_alpha

        if self.component < 0 or self.component >= observation.public_states.shape[1]:
            return

        for node in range(N):
            if observation.is_node_observed(node):
                val = float(observation.public_states[node, self.component])
                if np.isfinite(val):
                    if node in self.smoothed_public_states:
                        self.smoothed_public_states[node] = alpha * val + (1.0 - alpha) * self.smoothed_public_states[node]
                    else:
                        self.smoothed_public_states[node] = val

    def _determine_target_direction(self, observation: Observation) -> Optional[float]:
        """Determine target direction s_* from observed public panel mean.

        Follows the declared direction-knowledge contract (see
        ObservableSelectorPolicy): a missing report (observed mean is None) or a
        tied report (|mean| <= TIE_TOLERANCE) is ambiguous — no direction is
        locked, None is returned, and the caller must wait without consuming
        actuation budget instead of fabricating a majority.
        """
        if self.s_star is not None:
            return self.s_star

        if self.problem is not None and "s_star" in self.problem.target_params:
            s_star = float(self.problem.target_params["s_star"])
        elif self.problem is not None and "target_sign" in self.problem.target_params:
            s_star = float(self.problem.target_params["target_sign"])
        elif self.problem is not None and self.problem.target_type == TargetType.REVERSAL:
            obs_mean = observation.observed_public_mean(self.component)
            if obs_mean is None or abs(obs_mean) <= TIE_TOLERANCE:
                # Missing or tied report: direction is ambiguous; do not lock.
                return None
            s_star = -1.0 if obs_mean >= 0.0 else 1.0
        else:
            s_star = 1.0

        self.s_star = s_star
        return s_star

    def _is_node_aligned(self, node: int, s_star: float) -> bool:
        """Check if node has achieved desired target alignment under smoothed belief."""
        if node not in self.smoothed_public_states:
            return False
        val = self.smoothed_public_states[node]
        if self.problem is not None and self.problem.target_type == TargetType.REVERSAL:
            return (s_star * val) >= self.alignment_threshold
        return False

    def _rank_accessible_candidates(
        self,
        candidate_pool: Sequence[int],
        observation: Observation,
        s_star: float,
    ) -> List[int]:
        """Rank candidate nodes to prioritize for intervention using observable information only."""
        if not candidate_pool:
            return []
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        N = observation.public_states.shape[0]
        valid_pool = [int(n) for n in candidate_pool if 0 <= n < N]
        if not valid_pool:
            return []

        rec_att = compute_influence_scores(N, observed_edges=observation.observed_edges, mode="received_attention")

        is_structural = (
            self.problem is not None
            and self.problem.target_type in (TargetType.DECENTRALIZATION, TargetType.TURNOVER)
        )

        scored: List[Tuple[float, float, float, int]] = []
        for n in valid_pool:
            influence = float(rec_att[n])
            if n in self.smoothed_public_states:
                alignment = s_star * self.smoothed_public_states[n]
                opposing_priority = -alignment
            else:
                opposing_priority = 0.0

            tie_breaker = float(self.rng.random())
            if is_structural:
                scored.append((influence, opposing_priority, tie_breaker, n))
            else:
                scored.append((opposing_priority, influence, tie_breaker, n))

        scored.sort(reverse=True)
        return [n for _, _, _, n in scored]

    def _select_roster_by_quota(
        self,
        candidate_pool: Sequence[int],
        observation: Observation,
        s_star: float,
        k_target: int,
    ) -> List[int]:
        """Select up to k_target candidates respecting observation panel quota."""
        if k_target <= 0 or not candidate_pool:
            return []

        if observation.panel_mask is None or self.quota >= 1.0:
            ranked = self._rank_accessible_candidates(candidate_pool, observation, s_star)
            return ranked[:k_target]

        obs_cands = [n for n in candidate_pool if observation.is_node_observed(n)]
        unobs_cands = [n for n in candidate_pool if not observation.is_node_observed(n)]

        n_obs = int(round(k_target * self.quota))
        n_unobs = k_target - n_obs

        # Redistribute if either pool is too small
        if len(obs_cands) < n_obs:
            deficit = n_obs - len(obs_cands)
            n_obs = len(obs_cands)
            n_unobs += deficit
        if len(unobs_cands) < n_unobs:
            deficit = n_unobs - len(unobs_cands)
            n_unobs = len(unobs_cands)
            n_obs = min(len(obs_cands), n_obs + deficit)

        ranked_obs = self._rank_accessible_candidates(obs_cands, observation, s_star)
        ranked_unobs = self._rank_accessible_candidates(unobs_cands, observation, s_star)

        selected = list(dict.fromkeys(ranked_obs[:n_obs] + ranked_unobs[:n_unobs]))
        return selected[:k_target]

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        """Compute dynamic actuation action based on latest delivered observation."""
        if self.problem is None:
            raise RuntimeError("Controller must be reset before act()")

        t = observation.time
        self._update_smoothed_beliefs(observation)
        s_star = self._determine_target_direction(observation)

        # 0. Ambiguous direction (missing/tied report): wait without spending.
        if s_star is None:
            return Action(entries=(), intended_end_time=t)

        # 1. Release check
        if ledger.is_released or ledger.remaining_exposure <= ledger.tol:
            return Action(entries=(), intended_end_time=t, release=True)

        # 2. Deadline and committed pulse end check
        if self.committed_end_time is not None and t >= self.committed_end_time - 1e-12:
            return Action(entries=(), intended_end_time=t, release=True)

        # 3. Initial actuation
        if not self.initial_actuation_done:
            self.committed_end_time = min(t + self.duration, self.problem.release_deadline)
            self.initial_actuation_done = True
            self.last_shift_time = t

            roster = list(observation.accessible_roster)
            if not roster:
                return Action(entries=(), release=True)

            # Filter for unaligned accessible nodes
            unaligned_roster = [n for n in roster if not self._is_node_aligned(n, s_star)]
            if not unaligned_roster:
                # All accessible nodes are already aligned! Release cleanly with 0 budget spent.
                return Action(entries=(), intended_end_time=t, release=True)

            k_cap = min(self.problem.k, ledger.remaining_distinct_nodes, len(unaligned_roster))
            if k_cap <= 0:
                return Action(entries=(), intended_end_time=t, release=True)

            # Initial recruitment fraction
            initial_k = max(1, int(np.floor(k_cap * self.initial_recruitment_fraction)))

            selected = self._select_roster_by_quota(unaligned_roster, observation, s_star, initial_k)

            if not selected:
                return Action(entries=(), intended_end_time=t, release=True)

            pulse_dur = self.committed_end_time - t
            allocations = allocate_greedy_budget(
                selected,
                exposure_cap=ledger.remaining_exposure,
                amplitude_cap=min(self.nominal_amplitude, self.problem.amplitude_cap),
                duration=pulse_dur,
                k_limit=self.problem.k,
                allocation_strategy="uniform",
            )

            if not allocations:
                return Action(entries=(), intended_end_time=t, release=True)

            for n in selected:
                if allocations.get(n, 0.0) > 1e-15:
                    self.actuated_distinct_nodes.add(n)
                    self.active_nodes.add(n)

            if not self.active_nodes:
                return Action(entries=(), intended_end_time=t, release=True)

            entries = [
                ActionEntry(node=n, component=self.component, field=float(s_star * alloc_amp))
                for n, alloc_amp in allocations.items()
                if alloc_amp > 1e-15
            ]
            return Action(entries=tuple(entries), intended_end_time=self.committed_end_time, release=False)

        # 4. Subsequent observations: Dynamic Target Shifting
        rem_dur = self.committed_end_time - t
        if rem_dur <= 1e-12 or ledger.remaining_exposure <= ledger.tol:
            return Action(entries=(), intended_end_time=t, release=True)

        # Evaluate alignment on currently active nodes
        aligned_nodes: Set[int] = set()
        unaligned_nodes: Set[int] = set()

        for n in self.active_nodes:
            if self._is_node_aligned(n, s_star):
                aligned_nodes.add(n)
            else:
                unaligned_nodes.add(n)

        # Execute Target Shift:
        # Aligned nodes shut off to preserve exposure budget
        if aligned_nodes:
            for n in aligned_nodes:
                self.active_nodes.remove(n)
            self.last_shift_time = t

        # Recruit or reactivate unaligned accessible nodes if active slots available
        unaligned_accessible = [
            n for n in observation.accessible_roster
            if n not in self.active_nodes and not self._is_node_aligned(n, s_star)
        ]
        target_active_k = max(1, int(np.floor(self.problem.k * self.initial_recruitment_fraction)))
        slots_needed = max(0, target_active_k - len(self.active_nodes))

        if slots_needed > 0 and unaligned_accessible:
            candidate_selection = self._select_roster_by_quota(
                unaligned_accessible, observation, s_star, len(unaligned_accessible)
            )
            recruited: List[int] = []
            rem_k = ledger.remaining_distinct_nodes
            for n in candidate_selection:
                if len(recruited) >= slots_needed:
                    break
                if n in self.actuated_distinct_nodes:
                    # Free reactivation: already paid distinct node slot
                    recruited.append(n)
                elif rem_k > 0:
                    recruited.append(n)
                    rem_k -= 1

            for n in recruited:
                self.actuated_distinct_nodes.add(n)
                self.active_nodes.add(n)
            if recruited:
                self.last_shift_time = t

        # If greedy marginal strategy is active, swap out most aligned active node for top candidate
        if self.shift_strategy == ShiftStrategy.GREEDY_MARGINAL and len(self.active_nodes) > 1:
            rem_k_slots = ledger.remaining_distinct_nodes
            unaligned_candidates = [
                n for n in observation.accessible_roster
                if n not in self.active_nodes and not self._is_node_aligned(n, s_star)
            ]
            feasible_candidates = [
                n for n in unaligned_candidates
                if n in self.actuated_distinct_nodes or rem_k_slots > 0
            ]
            if feasible_candidates:
                is_structural = (
                    self.problem is not None
                    and self.problem.target_type in (TargetType.DECENTRALIZATION, TargetType.TURNOVER)
                )
                if is_structural:
                    rec_att = compute_influence_scores(
                        observation.public_states.shape[0],
                        observed_edges=observation.observed_edges,
                        mode="received_attention",
                    )
                    least_impactful_active = min(self.active_nodes, key=lambda n: float(rec_att[n]))
                    ranked_candidates = self._rank_accessible_candidates(feasible_candidates, observation, s_star)
                    best_candidate = ranked_candidates[0]
                    if float(rec_att[best_candidate]) > float(rec_att[least_impactful_active]):
                        self.active_nodes.remove(least_impactful_active)
                        self.active_nodes.add(best_candidate)
                        self.actuated_distinct_nodes.add(best_candidate)
                        self.last_shift_time = t
                else:
                    most_aligned_active = max(
                        self.active_nodes,
                        key=lambda n: s_star * self.smoothed_public_states.get(n, 0.0),
                    )
                    ranked_candidates = self._rank_accessible_candidates(feasible_candidates, observation, s_star)
                    best_candidate = ranked_candidates[0]
                    active_align = s_star * self.smoothed_public_states.get(most_aligned_active, 0.0)
                    cand_align = s_star * self.smoothed_public_states.get(best_candidate, 0.0)
                    if active_align > 0.2 and active_align > cand_align:
                        self.active_nodes.remove(most_aligned_active)
                        self.active_nodes.add(best_candidate)
                        self.actuated_distinct_nodes.add(best_candidate)
                        self.last_shift_time = t

        elif self.shift_strategy == ShiftStrategy.PERIODIC and len(self.active_nodes) > 0:
            period = self.shift_period if self.shift_period is not None else max(1.0, self.duration / max(1, self.problem.k))
            if (t - self.last_shift_time) >= period - 1e-12:
                rem_k_slots = ledger.remaining_distinct_nodes
                unaligned_candidates = [
                    n for n in observation.accessible_roster
                    if n not in self.active_nodes and not self._is_node_aligned(n, s_star)
                ]
                feasible_candidates = [
                    n for n in unaligned_candidates
                    if n in self.actuated_distinct_nodes or rem_k_slots > 0
                ]
                if feasible_candidates:
                    most_aligned_active = max(
                        self.active_nodes,
                        key=lambda n: s_star * self.smoothed_public_states.get(n, 0.0),
                    )
                    ranked_candidates = self._rank_accessible_candidates(feasible_candidates, observation, s_star)
                    best_candidate = ranked_candidates[0]
                    self.active_nodes.remove(most_aligned_active)
                    self.active_nodes.add(best_candidate)
                    self.actuated_distinct_nodes.add(best_candidate)
                    self.last_shift_time = t

        # Check remaining active nodes
        if len(self.active_nodes) == 0:
            # All active nodes are aligned, but pulse end has not elapsed
            # Section 9 compliance: retain slots, set field to zero without unspending or early releasing
            return Action(entries=(), intended_end_time=self.committed_end_time, release=False)

        # Allocate remaining budget across active nodes
        allocations = allocate_greedy_budget(
            list(self.active_nodes),
            exposure_cap=ledger.remaining_exposure,
            amplitude_cap=min(self.nominal_amplitude, self.problem.amplitude_cap),
            duration=rem_dur,
            k_limit=len(self.active_nodes),
            allocation_strategy="uniform",
        )

        if not allocations:
            return Action(entries=(), release=True)

        entries = [
            ActionEntry(node=n, component=self.component, field=float(s_star * alloc_amp))
            for n, alloc_amp in allocations.items()
            if alloc_amp > 1e-15
        ]
        return Action(entries=tuple(entries), intended_end_time=self.committed_end_time, release=False)
