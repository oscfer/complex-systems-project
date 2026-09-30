"""Heuristic subset search optimizers for optimal transformation control.

Implements:
1. PolicyCandidate dataclass representing concrete intervention schedules.
2. Candidate proposal generator seeded with degree, influence, and random heuristics.
3. Neighborhood mutation operator (add, remove, swap, pulse mutation, sign inversion, uniform resets).
4. Lexicographic multi-objective ranking (feasibility, restricted time, success rate, plateau tie-breaking, exposure).
5. HeuristicSubsetSearchOptimizer executing batched deterministic policy discovery across reference ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.contracts import ControlProblem, RolloutResult, SearchState, TargetType
from src.control.controllers import Controller, OpenLoopPulseController, ZeroController, ObservableSelectorPolicy
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import CompletePublicObservationModel, ObservationModel
from src.control.optimizers.heuristics import (
    allocate_greedy_budget,
    compute_degree_scores,
    compute_influence_scores,
    select_heuristic_candidates,
)
from src.core.state import State
from src.experiment.baselines.ensembles import ReferenceOrigin


@dataclass(frozen=True)
class PolicyCandidate:
    """Concrete parameterization of an intervention candidate policy."""
    candidate_id: str
    target_nodes: Tuple[int, ...]
    signs: Tuple[Tuple[int, float], ...]  # Sorted tuples of (node, sign)
    nominal_amplitude: float
    duration: float
    component: int = 0
    heuristic_name: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=lambda: {"information_level": "oracle"}, hash=False, compare=False)

    @property
    def signs_dict(self) -> Dict[int, float]:
        return dict(self.signs)

    @property
    def signature(self) -> Tuple[Any, ...]:
        """Deterministic hashable signature for exact deduplication."""
        return (
            tuple(sorted(self.target_nodes)),
            round(float(self.nominal_amplitude), 6),
            round(float(self.duration), 6),
            tuple(sorted((int(n), round(float(s), 6)) for n, s in self.signs)),
            self.component,
        )

    def to_controller(self) -> Controller:
        """Instantiate a concrete Controller executing this candidate's schedule."""
        if len(self.target_nodes) == 0 or self.nominal_amplitude <= 1e-15 or self.duration <= 1e-15:
            return ZeroController()
        return OpenLoopPulseController(
            target_nodes=self.target_nodes,
            nominal_amplitude=self.nominal_amplitude,
            duration=self.duration,
            signs=self.signs_dict,
            component=self.component,
        )


@dataclass
class OptimizationResult:
    """Full record of policy search optimizer execution."""
    incumbent: PolicyCandidate
    incumbent_score: Tuple[Any, ...]
    search_state: SearchState
    evaluated_candidates: List[PolicyCandidate]
    candidate_rollouts: Dict[str, List[RolloutResult]]
    candidate_scores: Dict[str, Tuple[Any, ...]]
    candidate_summaries: Dict[str, Dict[str, Any]]
    search_exhausted: bool
    total_rollouts: int
    static_incumbent: Optional[Any] = None
    static_incumbent_score: Optional[Tuple[Any, ...]] = None
    feedback_incumbent: Optional[Any] = None
    feedback_incumbent_score: Optional[Tuple[Any, ...]] = None
    search_coverage: Optional[Dict[str, Any]] = None


def compute_candidate_score(
    candidate: PolicyCandidate,
    rollouts: Sequence[RolloutResult],
    p0: float = 0.8,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Compute lexicographic ranking score tuple for a candidate.

    Ranking specification (§7, §8):
    1. Feasible estimated probability (p_hat >= p0) first.
    2. If feasible: smaller mean restricted time, then larger p_hat.
    3. If infeasible: larger p_hat, then smaller mean restricted time.
    4. Plateau tie-breaking: smaller average terminal target violation d_G.
    5. Resource tie-breaking: lower exposure, fewer nodes, then candidate ID.

    Returns:
        (score_tuple, summary_dict)
    """
    valid = [r for r in rollouts if r.is_valid]
    eligible = [r for r in valid if r.is_eligible]
    n_trials = len(eligible)
    # A reversal that later relapses is not a lasting transformation.
    successes = sum(bool(r.success and not r.relapsed) for r in eligible)
    certified_once = sum(bool(r.success) for r in eligible)
    p_hat = successes / n_trials if n_trials > 0 else 0.0

    mean_time = float(np.mean([
        r.restricted_time if r.success and not r.relapsed
        else float(r.summaries.get("horizon", r.restricted_time))
        for r in eligible
    ])) if n_trials > 0 else float('inf')
    mean_d_G = float(np.mean([r.terminal_target_violation for r in eligible])) if n_trials > 0 else 10.0
    mean_exposure = float(np.mean([r.expenditure for r in eligible])) if n_trials > 0 else float('inf')
    if hasattr(candidate, "selector"):
        n_nodes = round(float(np.mean([r.distinct_nodes_used for r in eligible])), 8) if n_trials > 0 else 0.0
    else:
        n_nodes = len(candidate.target_nodes)

    if np.isnan(mean_time):
        mean_time = float('inf')
    if np.isnan(mean_d_G):
        mean_d_G = 10.0
    if np.isnan(mean_exposure):
        mean_exposure = float('inf')
    if np.isnan(p_hat):
        p_hat = 0.0

    is_feasible = p_hat >= p0 - 1e-12 and n_trials > 0

    if is_feasible:
        # Tier 0 (Feasible)
        score_tuple = (
            0,
            round(mean_time, 8),
            round(-p_hat, 8),
            round(mean_d_G, 8),
            round(mean_exposure, 8),
            round(float(n_nodes), 8) if isinstance(n_nodes, float) else n_nodes,
            candidate.candidate_id,
        )
    else:
        # Tier 1 (Infeasible)
        score_tuple = (
            1,
            round(-p_hat, 8),
            round(mean_time, 8),
            round(mean_d_G, 8),
            round(mean_exposure, 8),
            round(float(n_nodes), 8) if isinstance(n_nodes, float) else n_nodes,
            candidate.candidate_id,
        )

    summary = {
        "candidate_id": candidate.candidate_id,
        "selector": getattr(candidate, "selector", None),
        "heuristic_name": candidate.heuristic_name,
        "target_nodes": list(candidate.target_nodes),
        "nominal_amplitude": candidate.nominal_amplitude,
        "duration": candidate.duration,
        "trials": n_trials,
        "successes": successes,
        "certified_once": certified_once,
        "ranking_outcome": "certified_and_no_relapse_through_horizon",
        "p_hat": p_hat,
        "mean_restricted_time": mean_time,
        "mean_terminal_target_violation": mean_d_G,
        "mean_exposure": mean_exposure,
        "is_feasible": is_feasible,
    }
    return score_tuple, summary


FAMILY_LABELS = ("static", "feedback")


def candidate_family(candidate: Any) -> str:
    """Honest family label for a candidate ('static' or 'feedback')."""
    return "feedback" if bool(getattr(candidate, "feedback", False)) else "static"


def rank_candidates_by_family(
    candidate_scores: Dict[str, Tuple[Any, ...]],
    candidate_registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Tuple[str, Tuple[Any, ...]]]]:
    """Separate per-family rankings (audit F1b).

    The static and feedback families are each ranked independently by the
    lexicographic score from ``compute_candidate_score``. The returned
    mapping is explicitly per-family; it is NOT a global ranking and must
    not be presented as one. Use :func:`rank_candidates_global` for a true
    cross-family ordering.
    """
    families: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {
        label: [] for label in FAMILY_LABELS
    }
    for cid, score in candidate_scores.items():
        cand = candidate_registry.get(cid) if candidate_registry else None
        label = candidate_family(cand)
        families.setdefault(label, []).append((cid, tuple(score)))
    for label in families:
        families[label].sort(key=lambda kv: (tuple(kv[1]), kv[0]))
    return families


def rank_candidates_global(
    candidate_scores: Dict[str, Tuple[Any, ...]],
) -> List[Tuple[str, Tuple[Any, ...]]]:
    """TRUE global ranking across all families (audit F1b).

    Ranks every evaluated candidate jointly by its lexicographic score,
    regardless of family. This is what the legacy key
    ``ranked_search_candidates`` was named after but historically was not:
    that key held a concatenation of separately ranked families.
    """
    return sorted(
        ((cid, tuple(score)) for cid, score in candidate_scores.items()),
        key=lambda kv: (tuple(kv[1]), kv[0]),
    )


def summarize_node_counts(result: Any) -> Dict[str, Any]:
    """Initial / peak-active / cumulative-distinct node accounting (audit F1c).

    ``effective_node_count`` recorded by the controller is the INITIAL
    deployment only; the rollout may recruit more nodes afterwards. This
    helper reports the three quantities separately:

    - ``initial_node_count``: nodes deployed at the first actuation
      (``effective_settings['effective_node_count']``; falls back to the
      first recorded actuating action).
    - ``peak_active_nodes``: maximum number of nodes carrying a nonzero
      field at any one instant (each new action REPLACES the active field,
      so the maximum over recorded actuating entries is the true peak).
      ``None`` when no action history was recorded (e.g. search rollouts).
    - ``cumulative_distinct_nodes``: distinct nodes ever actuated
      (``RolloutResult.distinct_nodes_used``, from the budget ledger).
    """
    effective = (getattr(result, "summaries", {}) or {}).get("effective_settings") or {}
    recorder_doc = (getattr(result, "summaries", {}) or {}).get("recorder") or {}
    history = recorder_doc.get("action_history") or []
    actuating = [
        a for a in history
        if float(a.get("spending_rate") or 0.0) > 1e-15 and a.get("field_entries")
    ]

    initial = effective.get("effective_node_count")
    initial_source = "effective_settings.effective_node_count"
    if initial is None and actuating:
        initial = len(actuating[0]["actuated_nodes"])
        initial_source = "first recorded actuating action"

    peak = None
    peak_source = None
    if actuating:
        peak = max(len(a["field_entries"]) for a in actuating)
        peak_source = "max concurrent nonzero-field entries in recorded action history"

    cumulative = int(getattr(result, "distinct_nodes_used", 0) or 0)
    return {
        "initial_node_count": initial,
        "peak_active_nodes": peak,
        "cumulative_distinct_nodes": cumulative,
        "legacy_effective_node_count": effective.get("effective_node_count"),
        "method": {
            "initial": initial_source if initial is not None else "unavailable",
            "peak": peak_source or "unavailable (no recorded action history)",
            "cumulative": "len(budget ledger distinct_nodes_actuated)",
        },
    }


class HeuristicSubsetSearchOptimizer:
    """Algorithmic heuristic search optimizer for static snapshots."""

    def __init__(
        self,
        problem: ControlProblem,
        candidate_cap: int = 24,
        futures_per_candidate: int = 2,
        p0: float = 0.8,
        batch_size: int = 4,
        search_seed: int = 42,
        duplicate_attempt_limit: int = 100,
        component: Optional[int] = None,
    ) -> None:
        if candidate_cap < 1:
            raise ValueError(f"candidate_cap must be >= 1, got {candidate_cap}")
        if futures_per_candidate < 1:
            raise ValueError(f"futures_per_candidate must be >= 1, got {futures_per_candidate}")
        if not 0.0 < p0 < 1.0:
            raise ValueError(f"p0 must be in (0, 1), got {p0}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.problem = problem
        self.candidate_cap = int(candidate_cap)
        self.futures_per_candidate = int(futures_per_candidate)
        self.p0 = float(p0)
        self.batch_size = int(batch_size)
        self.search_seed = int(search_seed)
        self.duplicate_attempt_limit = int(duplicate_attempt_limit)
        self.component = (
            int(component)
            if component is not None
            else int(self.problem.target_params.get("component", 0))
        )
        if self.component < 0 or self.component >= self.problem.D:
            raise ValueError(f"component must be in [0, {self.problem.D}), got {self.component}")
        self.batch_size = min(int(batch_size), self.candidate_cap)

        self.rng = np.random.default_rng(self.search_seed)
        self.seen_signatures: Set[Tuple[Any, ...]] = set()

    def _determine_target_sign(self, state: State, component: int = 0) -> float:
        """Compute target direction from observed or initial public state."""
        if "s_star" in self.problem.target_params:
            return float(self.problem.target_params["s_star"])
        if "target_sign" in self.problem.target_params:
            return float(self.problem.target_params["target_sign"])
        if self.problem.target_type == TargetType.REVERSAL:
            mean_y = float(np.mean(state.y[:, component]))
            return -1.0 if mean_y >= 0.0 else 1.0
        return 1.0

    def generate_seed_candidates(self, state: State) -> List[PolicyCandidate]:
        """Generate initial diverse candidate pool from degree and influence heuristics."""
        N = state.N
        k = self.problem.k
        roster = list(self.problem.accessible_roster) if self.problem.accessible_roster else list(range(N))
        target_sign = self._determine_target_sign(state, component=self.component)

        default_amp = min(1.0, self.problem.amplitude_cap)
        default_dur = min(5.0, self.problem.release_deadline)

        seeds: List[PolicyCandidate] = []

        # 1. Zero-control baseline
        seeds.append(
            PolicyCandidate(
                candidate_id="seed_zero",
                target_nodes=(),
                signs=(),
                nominal_amplitude=0.0,
                duration=0.0,
                component=self.component,
                heuristic_name="zero_control",
            )
        )

        # Compute heuristic scores on static snapshot memory W
        in_deg = compute_degree_scores(N, W=state.W, mode="in")
        out_deg = compute_degree_scores(N, W=state.W, mode="out")
        rec_att = compute_influence_scores(N, W=state.W, mode="received_attention")
        cross_grp = (
            compute_influence_scores(N, W=state.W, mode="cross_group", group_labels=state.opportunity_groups)
            if state.opportunity_groups is not None
            else np.zeros(N)
        )

        heuristic_specs = [
            ("in_degree", in_deg),
            ("out_degree", out_deg),
            ("received_attention", rec_att),
            ("cross_group", cross_grp),
        ]

        for name, scores in heuristic_specs:
            nodes = select_heuristic_candidates(scores, k=k, accessible_roster=roster, rng=self.rng)
            if nodes:
                cand_signs = tuple((n, target_sign) for n in sorted(nodes))
                # Greedily allocate budget
                alloc = allocate_greedy_budget(
                    nodes,
                    exposure_cap=self.problem.exposure_cap,
                    amplitude_cap=self.problem.amplitude_cap,
                    duration=default_dur,
                    k_limit=k,
                    allocation_strategy="uniform",
                )
                amp = max(alloc.values()) if alloc else default_amp
                seeds.append(
                    PolicyCandidate(
                        candidate_id=f"seed_{name}",
                        target_nodes=tuple(sorted(nodes)),
                        signs=cand_signs,
                        nominal_amplitude=amp,
                        duration=default_dur,
                        component=self.component,
                        heuristic_name=name,
                    )
                )

        # Random accessible candidate
        rand_nodes = select_heuristic_candidates(np.zeros(N), k=k, accessible_roster=roster, rng=self.rng)
        if rand_nodes:
            cand_signs = tuple((n, target_sign) for n in sorted(rand_nodes))
            seeds.append(
                PolicyCandidate(
                    candidate_id="seed_random",
                    target_nodes=tuple(sorted(rand_nodes)),
                    signs=cand_signs,
                    nominal_amplitude=default_amp,
                    duration=default_dur,
                    component=self.component,
                    heuristic_name="random",
                )
            )

        # Single top node candidates for leading member of each heuristic
        for name, scores in [("top_in_degree", in_deg), ("top_attention", rec_att)]:
            top_node = select_heuristic_candidates(scores, k=1, accessible_roster=roster, rng=self.rng)
            if top_node:
                n = top_node[0]
                # Single node can receive higher amplitude up to amplitude_cap and budget
                alloc = allocate_greedy_budget(
                    [n],
                    exposure_cap=self.problem.exposure_cap,
                    amplitude_cap=self.problem.amplitude_cap,
                    duration=default_dur,
                    k_limit=1,
                )
                amp = alloc.get(n, default_amp)
                seeds.append(
                    PolicyCandidate(
                        candidate_id=f"seed_{name}",
                        target_nodes=(n,),
                        signs=((n, target_sign),),
                        nominal_amplitude=amp,
                        duration=default_dur,
                        component=self.component,
                        heuristic_name=name,
                    )
                )

        # Greedy priority budget allocation variant on received_attention
        att_nodes = select_heuristic_candidates(rec_att, k=k, accessible_roster=roster, rng=self.rng)
        if att_nodes:
            # Shorter pulse with higher intensity
            short_dur = min(2.5, default_dur)
            alloc = allocate_greedy_budget(
                att_nodes,
                exposure_cap=self.problem.exposure_cap,
                amplitude_cap=self.problem.amplitude_cap,
                duration=short_dur,
                k_limit=k,
                allocation_strategy="greedy_priority",
            )
            amp = max(alloc.values()) if alloc else default_amp
            seeds.append(
                PolicyCandidate(
                    candidate_id="seed_attention_priority",
                    target_nodes=tuple(sorted(att_nodes)),
                    signs=tuple((n, target_sign) for n in att_nodes),
                    nominal_amplitude=amp,
                    duration=short_dur,
                    component=self.component,
                    heuristic_name="attention_priority",
                )
            )

        # Deduplicate seeds
        unique_seeds: List[PolicyCandidate] = []
        for cand in seeds:
            sig = cand.signature
            if sig not in self.seen_signatures:
                self.seen_signatures.add(sig)
                unique_seeds.append(cand)
        return unique_seeds

    def mutate_candidate(
        self,
        incumbent: PolicyCandidate,
        proposal_idx: int,
        state: State,
    ) -> Optional[PolicyCandidate]:
        """Generate a mutated candidate from incumbent respecting budget ledger limits."""
        roster = list(self.problem.accessible_roster) if self.problem.accessible_roster else list(range(state.N))
        target_sign = self._determine_target_sign(state, component=self.component)
        k = self.problem.k

        for _ in range(self.duplicate_attempt_limit):
            # Alternate every 4th proposal with uniformly sampled random subset
            if proposal_idx % 4 == 0:
                subset_size = int(self.rng.integers(1, k + 1))
                perm = self.rng.permutation(len(roster))
                nodes = tuple(sorted([roster[i] for i in perm[:subset_size]]))
                amp = float(self.rng.choice([0.5, 1.0, 2.0, self.problem.amplitude_cap]))
                amp = min(amp, self.problem.amplitude_cap)
                dur = float(self.rng.choice([1.0, 2.5, 5.0, 10.0]))
                dur = min(dur, self.problem.release_deadline)
                alloc = allocate_greedy_budget(
                    nodes,
                    exposure_cap=self.problem.exposure_cap,
                    amplitude_cap=self.problem.amplitude_cap,
                    duration=dur,
                    k_limit=k,
                )
                clipped_amp = min(amp, max(alloc.values())) if alloc else (0.0 if self.problem.exposure_cap <= 1e-12 else amp)
                cand = PolicyCandidate(
                    candidate_id=f"prop_{proposal_idx}",
                    target_nodes=nodes,
                    signs=tuple((n, target_sign) for n in nodes),
                    nominal_amplitude=clipped_amp,
                    duration=dur,
                    component=self.component,
                    heuristic_name="mutation_uniform_reset",
                )
                if cand.signature not in self.seen_signatures:
                    self.seen_signatures.add(cand.signature)
                    return cand

            # Neighborhood mutation around incumbent
            cur_nodes = set(incumbent.target_nodes)
            available = [n for n in roster if n not in cur_nodes]
            mut_type = int(self.rng.integers(0, 5))

            new_nodes = set(cur_nodes)
            new_amp = (
                incumbent.nominal_amplitude
                if incumbent.nominal_amplitude > 1e-15
                else min(1.0, self.problem.amplitude_cap)
            )
            new_dur = (
                incumbent.duration
                if incumbent.duration > 1e-15
                else min(5.0, self.problem.release_deadline)
            )

            if mut_type == 0 and len(new_nodes) < k and available:
                # Add node
                add_n = int(self.rng.choice(available))
                new_nodes.add(add_n)
                mut_name = "add_node"

            elif mut_type == 1 and len(new_nodes) > 1:
                # Remove node
                rem_n = int(self.rng.choice(list(new_nodes)))
                new_nodes.remove(rem_n)
                mut_name = "remove_node"

            elif mut_type == 2 and new_nodes and available:
                # Swap node
                rem_n = int(self.rng.choice(list(new_nodes)))
                add_n = int(self.rng.choice(available))
                new_nodes.remove(rem_n)
                new_nodes.add(add_n)
                mut_name = "swap_node"

            elif mut_type == 3:
                # Mutate amplitude / duration
                catalog_amps = [0.5, 1.0, 2.0, self.problem.amplitude_cap]
                catalog_durs = [1.0, 2.5, 5.0, 10.0]
                valid_amps = [a for a in catalog_amps if a <= self.problem.amplitude_cap]
                new_amp = float(self.rng.choice(valid_amps)) if valid_amps else float(self.problem.amplitude_cap)
                valid_durs = [d for d in catalog_durs if d <= self.problem.release_deadline]
                new_dur = float(self.rng.choice(valid_durs)) if valid_durs else float(self.problem.release_deadline)
                mut_name = "mutate_pulse"

            else:
                # Invert single sign (for structural or exploratory check)
                mut_name = "invert_sign"

            sorted_nodes = tuple(sorted(new_nodes))
            if not sorted_nodes:
                continue

            new_signs_dict = {n: incumbent.signs_dict.get(n, target_sign) for n in sorted_nodes}
            if mut_name == "invert_sign" and sorted_nodes:
                inv_n = int(self.rng.choice(sorted_nodes))
                new_signs_dict[inv_n] = -float(new_signs_dict[inv_n])

            # Greedy check against budget
            alloc = allocate_greedy_budget(
                sorted_nodes,
                exposure_cap=self.problem.exposure_cap,
                amplitude_cap=self.problem.amplitude_cap,
                duration=new_dur,
                k_limit=k,
            )
            clipped_amp = min(new_amp, max(alloc.values())) if alloc else (0.0 if self.problem.exposure_cap <= 1e-12 else new_amp)

            cand = PolicyCandidate(
                candidate_id=f"prop_{proposal_idx}",
                target_nodes=sorted_nodes,
                signs=tuple((n, new_signs_dict[n]) for n in sorted_nodes),
                nominal_amplitude=clipped_amp,
                duration=new_dur,
                component=self.component,
                heuristic_name=f"mutation_{mut_name}",
            )
            if cand.signature not in self.seen_signatures:
                self.seen_signatures.add(cand.signature)
                return cand

        return None

    def optimize(
        self,
        origin: ReferenceOrigin | State,
        observation_model: Optional[ObservationModel] = None,
        J: float = 1.0,
        T: float = 0.3,
        delta: float = 0.1,
        epsilon: float = 0.1,
        closure: float = 0.3,
        sample_interval: float = 5.0,
        adaptive_memory: bool = True,
        a: float = 0.0,
        c: float = 1.0,
        tau_y: float = 1.0,
    ) -> OptimizationResult:
        """Run heuristic policy search optimization across deterministic batches.

        Args:
            origin: Reference origin or initial physical state.
            observation_model: Observation generator model (defaults to CompletePublic).
            J, T, delta, epsilon, closure: Core physical simulation traits.
            sample_interval: Interval for scheduled observation snapshots.
            adaptive_memory: Whether encounter/decay dynamics run adaptively.

        Returns:
            OptimizationResult containing discovered incumbent policy and search history.
        """
        initial_state = origin.state if hasattr(origin, "state") else origin
        if hasattr(origin, "origin_id"):
            origin_id = origin.origin_id
        else:
            state_bytes = np.ascontiguousarray(initial_state.W).tobytes() + np.ascontiguousarray(initial_state.y).tobytes()
            origin_id = "state_" + hashlib.sha256(state_bytes).hexdigest()[:8]
        between_eligibility = origin.between_eligibility if isinstance(origin, ReferenceOrigin) else 0.05
        within_eligibility = origin.within_eligibility if isinstance(origin, ReferenceOrigin) else 1.0

        if observation_model is None:
            observation_model = CompletePublicObservationModel(
                N=initial_state.N,
                accessible_roster=self.problem.accessible_roster,
                delivery_delay=0.0,
            )

        self.rng = np.random.default_rng(self.search_seed)
        self.seen_signatures = set()

        # 1. Initialize seed candidates
        candidate_queue: List[PolicyCandidate] = self.generate_seed_candidates(initial_state)

        search_state = SearchState(evaluation_budget=self.candidate_cap)
        candidate_rollouts: Dict[str, List[RolloutResult]] = {}
        candidate_scores: Dict[str, Tuple[Any, ...]] = {}
        candidate_summaries: Dict[str, Dict[str, Any]] = {}
        evaluated_candidates: List[PolicyCandidate] = []

        incumbent: Optional[PolicyCandidate] = None
        incumbent_score: Optional[Tuple[Any, ...]] = None
        search_exhausted = False
        proposal_idx = 0
        total_rollouts = 0

        # Optimization loop across batches
        while search_state.evaluations_used < self.candidate_cap:
            batch: List[PolicyCandidate] = []
            while len(batch) < self.batch_size and search_state.evaluations_used + len(batch) < self.candidate_cap:
                if candidate_queue:
                    batch.append(candidate_queue.pop(0))
                else:
                    proposal_idx += 1
                    base_cand = incumbent or (batch[0] if batch else PolicyCandidate(
                        candidate_id="base_zero",
                        target_nodes=(),
                        signs=(),
                        nominal_amplitude=0.0,
                        duration=0.0,
                        component=self.component,
                        heuristic_name="zero_control",
                    ))
                    mutated = self.mutate_candidate(base_cand, proposal_idx, initial_state)
                    if mutated is not None:
                        batch.append(mutated)
                    else:
                        search_exhausted = True
                        break

            if not batch:
                break

            # Evaluate each candidate in batch across deterministic futures
            for cand in batch:
                rollouts: List[RolloutResult] = []
                controller = cand.to_controller()
                for future_id in range(self.futures_per_candidate):
                    # Deterministic simulator and controller seeds derived from master seeds
                    sim_seed = int(
                        hashlib.sha256(f"{self.problem.search_seed}|{origin_id}|{future_id}|sim".encode()).hexdigest()[:8],
                        16,
                    )
                    ctl_seed = int(
                        hashlib.sha256(f"{self.problem.proposal_seed}|{origin_id}|{future_id}|ctl".encode()).hexdigest()[:8],
                        16,
                    )

                    evaluator = TargetEvaluator.create_for_state(
                        target_type=self.problem.target_type,
                        initial_state=initial_state,
                        holding_duration=self.problem.holding_duration,
                        horizon=self.problem.horizon,
                        release_deadline=self.problem.release_deadline,
                        problem_id=self.problem.problem_id,
                        policy_id=cand.candidate_id,
                        origin_id=origin_id,
                        future_id=future_id,
                        target_params=self.problem.target_params,
                    )

                    res, _, _ = run_controlled_simulation(
                        initial_state=initial_state.copy(deep=True),
                        problem=self.problem,
                        controller=controller,
                        observation_model=observation_model,
                        target_evaluator=evaluator,
                        J=J,
                        T=T,
                        delta=delta,
                        epsilon=epsilon,
                        closure=closure,
                        sample_interval=sample_interval,
                        adaptive_memory=adaptive_memory,
                        between_eligibility=between_eligibility,
                        within_eligibility=within_eligibility,
                        a=a,
                        c=c,
                        tau_y=tau_y,
                        sim_rng=np.random.default_rng(sim_seed),
                        controller_rng=np.random.default_rng(ctl_seed),
                    )
                    rollouts.append(res)
                    total_rollouts += 1

                score, summary = compute_candidate_score(cand, rollouts, p0=self.p0)
                candidate_rollouts[cand.candidate_id] = rollouts
                candidate_scores[cand.candidate_id] = score
                candidate_summaries[cand.candidate_id] = summary
                evaluated_candidates.append(cand)
                search_state.evaluations_used += 1
                search_state.candidate_registry[cand.candidate_id] = cand
                search_state.completed_evaluations.extend(rollouts)

                # Check if this candidate improves upon incumbent
                if incumbent_score is None or score < incumbent_score:
                    incumbent = cand
                    incumbent_score = score
                    search_state.incumbent_policy_id = cand.candidate_id
                    search_state.incumbent_score = score

            search_state.proposal_step += 1
            if search_exhausted:
                break

        if incumbent is None:
            # Fallback to zero controller
            incumbent = PolicyCandidate(
                candidate_id="fallback_zero",
                target_nodes=(),
                signs=(),
                nominal_amplitude=0.0,
                duration=0.0,
                component=self.component,
                heuristic_name="zero_control",
            )
            incumbent_score = (1, 0.0, float('inf'), 10.0, 0.0, 0, incumbent.candidate_id)
            search_state.incumbent_policy_id = incumbent.candidate_id
            search_state.incumbent_score = incumbent_score

        return OptimizationResult(
            incumbent=incumbent,
            incumbent_score=incumbent_score,
            search_state=search_state,
            evaluated_candidates=evaluated_candidates,
            candidate_rollouts=candidate_rollouts,
            candidate_scores=candidate_scores,
            candidate_summaries=candidate_summaries,
            search_exhausted=search_exhausted,
            total_rollouts=total_rollouts,
        )


@dataclass(frozen=True)
class ObservablePolicyCandidate:
    """Observable policy candidate (restricted observations)."""
    candidate_id: str
    selector: str
    observed_slot_quota: float
    feedback: bool
    initial_recruitment_fraction: float
    nominal_amplitude: float
    duration: float
    component: int = 0
    heuristic_name: str = "observable_search"
    target_nodes: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=lambda: {"information_level": "observable"}, hash=False, compare=False)
    node_fraction: float = 0.1
    effective_feedback_mode: str = "genuinely_adaptive"

    @property
    def signature(self) -> Tuple[Any, ...]:
        return (
            self.selector,
            round(float(self.node_fraction), 6),
            round(float(self.observed_slot_quota), 6),
            self.feedback,
            round(float(self.initial_recruitment_fraction), 6),
            round(float(self.nominal_amplitude), 6),
            round(float(self.duration), 6),
            self.component,
        )

    def to_controller(self) -> Controller:
        if self.nominal_amplitude <= 1e-15 or self.duration <= 1e-15:
            return ZeroController()
        return ObservableSelectorPolicy(
            selector=self.selector,
            node_fraction=self.node_fraction,
            nominal_amplitude=self.nominal_amplitude,
            duration=self.duration,
            quota=self.observed_slot_quota,
            is_feedback=self.feedback,
            initial_recruitment_fraction=self.initial_recruitment_fraction,
            component=self.component,
        )


class ObservablePolicySearchOptimizer:
    """Algorithmic heuristic search optimizer for restricted observations."""

    def __init__(
        self,
        problem: ControlProblem,
        candidate_cap: int = 24,
        futures_per_candidate: int = 2,
        p0: float = 0.8,
        search_seed: int = 42,
        component: Optional[int] = None,
        selectors: Sequence[str] = ("random", "reported_attention", "reported_outgoing_degree", "reported_cross_group", "group_balanced"),
        observed_slot_quotas: Sequence[float] = (0.0, 0.5, 1.0),
        feedback_choices: Sequence[bool] = (False, True),
        nominal_amplitudes: Sequence[float] = (0.25, 1.0, 4.0),
        durations: Sequence[float] = (1.0, 5.0, 20.0),
        candidate_node_fractions: Sequence[float] = (0.05, 0.1),
    ) -> None:
        self.problem = problem
        self.candidate_cap = int(candidate_cap)
        self.futures_per_candidate = int(futures_per_candidate)
        self.p0 = float(p0)
        self.search_seed = int(search_seed)
        self.component = int(component) if component is not None else int(self.problem.target_params.get("component", 0))

        self.selectors = list(selectors)
        self.observed_slot_quotas = list(observed_slot_quotas)
        self.feedback_choices = list(feedback_choices)
        self.nominal_amplitudes = list(nominal_amplitudes)
        self.durations = list(durations)
        self.candidate_node_fractions = list(candidate_node_fractions)

        self.rng = np.random.default_rng(self.search_seed)

    def generate_candidates(self) -> List[ObservablePolicyCandidate]:
        from src.control.controllers import audit_policy_timing

        # Zero controller as baseline
        default_n_frac = self.candidate_node_fractions[0] if self.candidate_node_fractions else 0.1
        zero = ObservablePolicyCandidate(
            candidate_id="observable_zero",
            selector="random",
            observed_slot_quota=0.0,
            feedback=False,
            initial_recruitment_fraction=1.0,
            nominal_amplitude=0.0,
            duration=0.0,
            component=self.component,
            node_fraction=default_n_frac,
            effective_feedback_mode="effectively_static",
        )

        all_static: List[ObservablePolicyCandidate] = []
        all_feedback: List[ObservablePolicyCandidate] = []
        seen_static = set()
        seen_feedback = set()

        idx = 1
        raw_proposed = 0  # every constructed candidate, before deduplication
        for sel in self.selectors:
            for n_frac in self.candidate_node_fractions:
                for quota in self.observed_slot_quotas:
                    for fb in self.feedback_choices:
                        for amp in self.nominal_amplitudes:
                            for dur in self.durations:
                                if amp <= 1e-15 or dur <= 1e-15:
                                    continue
                                if amp > self.problem.amplitude_cap or dur > self.problem.release_deadline:
                                    continue

                                # Audit report/actuation timing explicitly
                                timing = audit_policy_timing(
                                    delivery_delay=1.0,
                                    duration=dur,
                                    sample_interval=5.0,
                                    release_deadline=self.problem.release_deadline,
                                )
                                eff_mode = timing["classification"] if fb else "effectively_static"

                                init_fracs = [0.5, 1.0] if fb else [1.0]
                                for frac in init_fracs:
                                    cand = ObservablePolicyCandidate(
                                        candidate_id=f"obs_cand_{idx}",
                                        selector=sel,
                                        observed_slot_quota=float(quota),
                                        feedback=bool(fb),
                                        initial_recruitment_fraction=float(frac),
                                        nominal_amplitude=float(amp),
                                        duration=float(dur),
                                        component=self.component,
                                        node_fraction=float(n_frac),
                                        effective_feedback_mode=eff_mode,
                                    )
                                    idx += 1
                                    raw_proposed += 1
                                    if fb:
                                        if cand.signature not in seen_feedback:
                                            seen_feedback.add(cand.signature)
                                            all_feedback.append(cand)
                                    else:
                                        if cand.signature not in seen_static:
                                            seen_static.add(cand.signature)
                                            all_static.append(cand)

        # Explicit proposal coverage (audit F1b): proposed vs unique vs
        # evaluated, per family, with the actual caps that constrained them.
        self.last_proposal_stats: Dict[str, Any] = {
            "candidates_proposed": int(raw_proposed),
            "candidates_unique": int(len(all_static) + len(all_feedback)),
            "candidates_unique_by_family": {
                "static": int(len(all_static)),
                "feedback": int(len(all_feedback)),
            },
            "candidate_cap": int(self.candidate_cap),
            "actual_caps": {
                "candidate_cap": int(self.candidate_cap),
                "amplitude_cap": float(self.problem.amplitude_cap),
                "exposure_cap": float(self.problem.exposure_cap),
                "distinct_node_cap_k": int(self.problem.k),
                "accessible_roster_size": len(self.problem.accessible_roster or ()),
                "release_deadline": float(self.problem.release_deadline),
            },
        }

        # Reproducible shuffle within families
        self.rng.shuffle(all_static)
        self.rng.shuffle(all_feedback)

        # For feedback family, prioritize candidates that are genuinely adaptive
        adaptive_fb = [c for c in all_feedback if c.effective_feedback_mode == "genuinely_adaptive"]
        other_fb = [c for c in all_feedback if c.effective_feedback_mode != "genuinely_adaptive"]
        all_feedback_sorted = adaptive_fb + other_fb

        # Determine budget allocation between static and feedback families
        has_static = False in self.feedback_choices
        has_feedback = True in self.feedback_choices

        if has_static and has_feedback:
            n_static_budget = max(1, self.candidate_cap // 2)
            n_feedback_budget = max(1, self.candidate_cap - n_static_budget)
        elif has_static:
            n_static_budget = self.candidate_cap
            n_feedback_budget = 0
        else:
            n_static_budget = 0
            n_feedback_budget = self.candidate_cap

        final_candidates: List[ObservablePolicyCandidate] = []
        if n_static_budget > 0:
            final_candidates.append(zero)

        # Step 1: Cover distinct selectors in remaining slots, alternating between feedback and static
        turn_feedback = bool(n_feedback_budget > 0)
        for sel in self.selectors:
            if len(final_candidates) >= self.candidate_cap:
                break
            n_cur_fb = sum(1 for c in final_candidates if c.feedback)
            n_cur_st = sum(1 for c in final_candidates if not c.feedback)

            pools_to_try = []
            if turn_feedback:
                if n_cur_fb < n_feedback_budget:
                    pools_to_try.append(all_feedback_sorted)
                if n_cur_st < n_static_budget:
                    pools_to_try.append(all_static)
            else:
                if n_cur_st < n_static_budget:
                    pools_to_try.append(all_static)
                if n_cur_fb < n_feedback_budget:
                    pools_to_try.append(all_feedback_sorted)

            cand_found = None
            for pool in pools_to_try:
                for c in pool:
                    if c.selector == sel and c not in final_candidates:
                        cand_found = c
                        break
                if cand_found is not None:
                    break

            if cand_found is not None:
                final_candidates.append(cand_found)
                turn_feedback = not turn_feedback

        # Step 2: Fill remaining static and feedback slots up to budgets
        n_cur_fb = sum(1 for c in final_candidates if c.feedback)
        n_cur_st = sum(1 for c in final_candidates if not c.feedback)

        for c in all_static:
            if len(final_candidates) >= self.candidate_cap:
                break
            if n_cur_st >= n_static_budget:
                break
            if c not in final_candidates:
                final_candidates.append(c)
                n_cur_st += 1

        for c in all_feedback_sorted:
            if len(final_candidates) >= self.candidate_cap:
                break
            if n_cur_fb >= n_feedback_budget:
                break
            if c not in final_candidates:
                final_candidates.append(c)
                n_cur_fb += 1

        # Step 3: If any slots remain, fill from remaining unique candidates
        for c in (all_static + all_feedback_sorted):
            if len(final_candidates) >= self.candidate_cap:
                break
            if c not in final_candidates:
                final_candidates.append(c)

        return final_candidates

    def optimize(
        self,
        training_origins: Sequence[Any],
        J: float = 1.0,
        T: float = 0.3,
        delta: float = 0.1,
        epsilon: float = 0.1,
        closure: float = 0.3,
        sample_interval: float = 5.0,
        adaptive_memory: bool = True,
        a: float = 0.0,
        c: float = 1.0,
        tau_y: float = 1.0,
        observation_model_kwargs: Optional[Dict[str, Any]] = None,
        registry: Optional[Any] = None,
    ) -> OptimizationResult:
        candidates = self.generate_candidates()

        search_state = SearchState(evaluation_budget=self.candidate_cap)
        candidate_rollouts: Dict[str, List[RolloutResult]] = {}
        candidate_scores: Dict[str, Tuple[Any, ...]] = {}
        candidate_summaries: Dict[str, Dict[str, Any]] = {}
        evaluated_candidates: List[PolicyCandidate] = []

        incumbent = None
        incumbent_score = None
        static_incumbent = None
        static_incumbent_score = None
        feedback_incumbent = None
        feedback_incumbent_score = None
        total_rollouts = 0

        from src.control.observation import SampledPublicObservationModel

        obs_kwargs = observation_model_kwargs or {}
        obs_kwargs.setdefault("panel_fraction", 0.5)
        obs_kwargs.setdefault("edge_reveal_prob", 0.5)

        for cand in candidates:
            rollouts: List[RolloutResult] = []
            controller = cand.to_controller()

            for origin_idx, origin in enumerate(training_origins):
                initial_state = origin.state if hasattr(origin, "state") else origin
                if hasattr(origin, "origin_id"):
                    origin_id = origin.origin_id
                else:
                    state_bytes = np.ascontiguousarray(initial_state.W).tobytes() + np.ascontiguousarray(initial_state.y).tobytes()
                    origin_id = "state_" + hashlib.sha256(state_bytes).hexdigest()[:8]
                between_eligibility = origin.between_eligibility if hasattr(origin, "between_eligibility") else 0.05
                within_eligibility = origin.within_eligibility if hasattr(origin, "within_eligibility") else 1.0

                for future_id in range(self.futures_per_candidate):
                    # Check registry cache for previously completed rollout
                    if registry is not None and registry.is_completed("search", cand.candidate_id, origin_id, future_id, "adaptive"):
                        cached_res = registry.get_rollout("search", cand.candidate_id, origin_id, future_id, "adaptive")
                        if cached_res is not None:
                            rollouts.append(cached_res)
                            total_rollouts += 1
                            continue

                    t_unit_start = time.perf_counter()
                    sim_seed = int(
                        hashlib.sha256(f"{self.problem.search_seed}|{origin_id}|{future_id}|sim".encode()).hexdigest()[:8],
                        16,
                    )
                    ctl_seed = int(
                        hashlib.sha256(f"{self.problem.proposal_seed}|{origin_id}|{future_id}|ctl".encode()).hexdigest()[:8],
                        16,
                    )

                    observation_model = SampledPublicObservationModel(
                        N=initial_state.N,
                        accessible_roster=self.problem.accessible_roster,
                        rng=np.random.default_rng(ctl_seed),
                        **obs_kwargs,
                    )

                    evaluator = TargetEvaluator.create_for_state(
                        target_type=self.problem.target_type,
                        initial_state=initial_state,
                        holding_duration=self.problem.holding_duration,
                        horizon=self.problem.horizon,
                        release_deadline=self.problem.release_deadline,
                        problem_id=self.problem.problem_id,
                        policy_id=cand.candidate_id,
                        origin_id=origin_id,
                        future_id=future_id,
                        target_params=self.problem.target_params,
                    )

                    res, _, _ = run_controlled_simulation(
                        initial_state=initial_state.copy(deep=True),
                        problem=self.problem,
                        controller=controller,
                        observation_model=observation_model,
                        target_evaluator=evaluator,
                        J=J,
                        T=T,
                        delta=delta,
                        epsilon=epsilon,
                        closure=closure,
                        sample_interval=sample_interval,
                        adaptive_memory=adaptive_memory,
                        between_eligibility=between_eligibility,
                        within_eligibility=within_eligibility,
                        a=a,
                        c=c,
                        tau_y=tau_y,
                        sim_rng=np.random.default_rng(sim_seed),
                        controller_rng=np.random.default_rng(ctl_seed),
                    )
                    elapsed_unit = time.perf_counter() - t_unit_start
                    if registry is not None:
                        registry.record_rollout("search", cand.candidate_id, origin_id, future_id, "adaptive", res, elapsed_unit)
                    # F1c: initial / peak-active / cumulative-distinct node
                    # counts. Search rollouts carry no recorder, so peak-active
                    # may be unavailable and is reported as None (never
                    # silently equated to the initial deployment).
                    res.summaries["node_count_decomposition"] = summarize_node_counts(res)
                    rollouts.append(res)
                    total_rollouts += 1

            score, summary = compute_candidate_score(cand, rollouts, p0=self.p0)  # type: ignore
            candidate_rollouts[cand.candidate_id] = rollouts
            candidate_scores[cand.candidate_id] = score
            candidate_summaries[cand.candidate_id] = summary
            evaluated_candidates.append(cand)  # type: ignore
            search_state.evaluations_used += 1
            search_state.candidate_registry[cand.candidate_id] = cand  # type: ignore
            search_state.completed_evaluations.extend(rollouts)
            # Track family incumbents
            if cand.feedback:
                if feedback_incumbent_score is None or score < feedback_incumbent_score:
                    feedback_incumbent = cand
                    feedback_incumbent_score = score
            else:
                if static_incumbent_score is None or score < static_incumbent_score:
                    static_incumbent = cand
                    static_incumbent_score = score

            # Track overall incumbent
            if incumbent_score is None or score < incumbent_score:
                incumbent = cand
                incumbent_score = score
                search_state.incumbent_policy_id = cand.candidate_id
                search_state.incumbent_score = score

            if search_state.evaluations_used >= self.candidate_cap:
                break

        if incumbent is None:
            incumbent = candidates[0]
            incumbent_score = (1, 0.0, float('inf'), 10.0, 0.0, 0, incumbent.candidate_id)
            search_state.incumbent_policy_id = incumbent.candidate_id
            search_state.incumbent_score = incumbent_score

        # Search coverage report (audit F1b): proposed/unique/evaluated counts,
        # per-family coverage, training origins/futures, actual caps, and the
        # failure-distance tie-break definition with per-candidate values.
        proposal = dict(getattr(self, "last_proposal_stats", {}) or {})
        evaluated_by_family: Dict[str, int] = {"static": 0, "feedback": 0}
        for cand in evaluated_candidates:  # type: ignore
            evaluated_by_family[candidate_family(cand)] = (
                evaluated_by_family.get(candidate_family(cand), 0) + 1
            )
        training_origin_ids: List[str] = []
        for origin in training_origins:
            if hasattr(origin, "origin_id"):
                training_origin_ids.append(origin.origin_id)
            else:
                initial_state = origin.state if hasattr(origin, "state") else origin
                state_bytes = (
                    np.ascontiguousarray(initial_state.W).tobytes()
                    + np.ascontiguousarray(initial_state.y).tobytes()
                )
                training_origin_ids.append(
                    "state_" + hashlib.sha256(state_bytes).hexdigest()[:8]
                )
        search_coverage: Dict[str, Any] = {
            **proposal,
            "candidates_evaluated": len(evaluated_candidates),  # type: ignore
            "family_coverage": {
                "static": {
                    "unique_proposed": int(proposal.get("candidates_unique_by_family", {}).get("static", 0)),
                    "evaluated": evaluated_by_family.get("static", 0),
                },
                "feedback": {
                    "unique_proposed": int(proposal.get("candidates_unique_by_family", {}).get("feedback", 0)),
                    "evaluated": evaluated_by_family.get("feedback", 0),
                },
            },
            "training_origin_ids": training_origin_ids,
            "training_futures_per_candidate": int(self.futures_per_candidate),
            "training_future_ids": list(range(int(self.futures_per_candidate))),
            "rollouts_total": int(total_rollouts),
            "ranking_semantics": (
                "candidate_scores rank each family separately; use "
                "rank_candidates_by_family for per-family orderings and "
                "rank_candidates_global for a true cross-family ordering. "
                "A concatenation of family rankings is NOT a global ranking."
            ),
            "failure_distance_tie_break": {
                "definition": (
                    "Among infeasible candidates (is_feasible=False), the "
                    "lexicographic score orders by higher success fraction, "
                    "then smaller mean restricted time, then smaller mean "
                    "terminal target violation d_G (failure distance to the "
                    "target at horizon), then lower exposure, fewer nodes, "
                    "candidate id."
                ),
                "per_candidate_failure_distance": {
                    cid: float(summary.get("mean_terminal_target_violation", float("nan")))
                    for cid, summary in candidate_summaries.items()
                },
            },
        }

        return OptimizationResult(
            incumbent=incumbent,  # type: ignore
            incumbent_score=incumbent_score,
            search_state=search_state,
            evaluated_candidates=evaluated_candidates,
            candidate_rollouts=candidate_rollouts,
            candidate_scores=candidate_scores,
            candidate_summaries=candidate_summaries,
            search_exhausted=True,
            total_rollouts=total_rollouts,
            static_incumbent=static_incumbent,
            static_incumbent_score=static_incumbent_score,
            feedback_incumbent=feedback_incumbent,
            feedback_incumbent_score=feedback_incumbent_score,
            search_coverage=search_coverage,
        )
