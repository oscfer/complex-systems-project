"""Bounded recipient-group search from restricted observations.

This is a search mechanism, not a forecast model or a scientific result. The
caller supplies a forecast evaluator built from permitted information. Never
bind that evaluator to the actual hidden state or an evaluation future.
Ranking reuses the project's lasting-success/restricted-time objective. Reported
attention and audience coverage only initialize proposals; they do not pick the
winner. Validation forecasts use separate seeds and cannot retune the winner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from src.control.contracts import Observation, RolloutResult
from src.control.optimizers.search import PolicyCandidate, compute_candidate_score

ForecastEvaluator = Callable[[PolicyCandidate, tuple[int, ...]], Sequence[RolloutResult]]


@dataclass(frozen=True)
class SubsetSearchConfig:
    sizes: tuple[int, ...] = (4, 10)
    amplitude: float = 2.0
    duration: float = 19.0
    component: int = 0
    candidate_limit: int = 24
    beam_width: int = 3
    search_futures: int = 3
    validation_futures: int = 3
    reliability_target: float = 0.8
    seed: int = 0
    representative_limit: int = 2

    def __post_init__(self):
        counts = (*self.sizes, self.candidate_limit, self.beam_width,
                  self.search_futures, self.validation_futures, self.representative_limit)
        if not self.sizes or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in counts):
            raise ValueError('Sizes and search budgets must be positive integers')
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError('Recipient sizes must be distinct')
        if self.candidate_limit < 4 * len(self.sizes):
            raise ValueError('Reserve four initial proposals per recipient size')
        if not 0 < self.reliability_target <= 1:
            raise ValueError('Invalid reliability target')
        if not np.isfinite(self.amplitude) or self.amplitude <= 0 or not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('Amplitude and duration must be finite and positive')


@dataclass
class SubsetSearchResult:
    selected: PolicyCandidate
    candidates: list[PolicyCandidate]
    summaries: list[dict]
    validation_summaries: list[dict]
    frontier: tuple[PolicyCandidate, ...]
    representatives: tuple[PolicyCandidate, ...]
    search_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    forecast_calls: int
    forecast_rollouts: int
    budget_exhausted: bool
    influence_ranks: dict[int, int]


def _proposal_orders(observation: Observation, rng: np.random.Generator):
    """Four diverse orders; W[i,j] means listener i attends to recipient j."""
    if observation.is_oracle or observation.private_states is not None or observation.true_memory is not None:
        raise ValueError('Recipient search requires restricted observations without hidden payloads')
    N = observation.public_states.shape[0]
    roster = sorted(set(observation.accessible_roster))
    if not roster or any(i < 0 or i >= N for i in roster):
        raise ValueError('Invalid accessible roster')
    W = np.zeros((N, N), dtype=float)
    for (i, j), count in observation.observed_edges.items():
        if not (0 <= i < N and 0 <= j < N) or not np.isfinite(count) or count < 0:
            raise ValueError('Invalid revealed edge')
        if i != j:
            W[i, j] = count
    incoming = W.sum(axis=0)
    tie_order = list(rng.permutation(roster))
    ties = {int(n): i for i, n in enumerate(tie_order)}
    high = sorted(roster, key=lambda n: (-incoming[n], ties[n]))
    # Missing incoming data is not evidence of low prominence.
    low = sorted(roster, key=lambda n: (incoming[n] <= 0, incoming[n], ties[n]))
    # Maximize marginal distinct audience coverage, reducing redundant listeners.
    # This is only a proposal heuristic; no diffusion guarantee is assumed.
    rows = W.sum(axis=1, keepdims=True)
    attention = np.divide(W, rows, out=np.zeros_like(W), where=rows > 0)
    covered = np.zeros(N)
    available = set(roster)
    coverage = []
    while available:
        node = max(available, key=lambda n: (float(np.maximum(attention[:, n]-covered, 0).sum()), -ties[n]))
        covered = np.maximum(covered, attention[:, node])
        coverage.append(node)
        available.remove(node)
    ranks = _influence_ranks(roster, incoming, ties)
    return roster, incoming, ranks, [('attention', high), ('low_observed_attention', low),
                              ('audience_coverage', coverage), ('random', tie_order)]


def _influence_ranks(roster: list[int], incoming: np.ndarray, tie_order: dict[int, int]) -> dict[int, int]:
    """Observed-power cost ranks: 1 is least positive reported attention.

    Equal positive counts get equal dense ranks. Unknown incoming attention is
    assigned a conservative cost above every measured rank, not rewarded as
    powerless. These are observable proxies, not true influence estimates.
    """
    levels = sorted({float(incoming[n]) for n in roster if incoming[n] > 0})
    rank = {value: i + 1 for i, value in enumerate(levels)}
    return {n: rank.get(float(incoming[n]), len(levels) + 1) for n in roster}


def _dominates(a: tuple, b: tuple, tol: float = 1e-8) -> bool:
    """Pareto domination on declared objectives (p_hat, time, count, mean/max
    observed-influence rank). p_hat maximized, all others minimized."""
    p_a, t_a, c_a, mr_a, mx_a = a
    p_b, t_b, c_b, mr_b, mx_b = b
    no_worse = (p_a >= p_b - tol and t_a <= t_b + tol and c_a <= c_b + tol
                and mr_a <= mr_b + tol and mx_a <= mx_b + tol)
    better = (p_a > p_b + tol or t_a < t_b - tol or c_a < c_b - tol
              or mr_a < mr_b - tol or mx_a < mx_b - tol)
    return no_worse and better


def _estimated_frontier(evaluated: dict) -> list:
    """Non-dominated candidates under the frozen objective vector."""
    entries = []
    for nodes, value in evaluated.items():
        cand, summary = value[0], value[2]
        entries.append((nodes, (
            summary['p_hat'], summary['mean_restricted_time'], float(len(nodes)),
            summary['mean_influence_rank'], summary['max_influence_rank'])))
    frontier = []
    for nodes, objectives in entries:
        if not any(_dominates(other, objectives) for other_n, other in entries if other_n != nodes):
            frontier.append(nodes)
    return sorted(frontier)


def search_observed_subsets(
    observation: Observation, evaluator: ForecastEvaluator, *, target_sign: float,
    config: SubsetSearchConfig = SubsetSearchConfig(),
) -> SubsetSearchResult:
    """Compare whole recipient sets, then search membership exchanges.

    target_sign must be the controller's permitted inferred/declared direction,
    never silently copied from a hidden evaluation target. Fixed sizes are
    separate resource choices; speed is ranked subject to estimated reliability,
    then existing resource tie breakers. This does not certify the fastest or
    smallest possible set. A failed forecast counts as failure, not a dropped trial.

    The evaluator returns one RolloutResult per supplied future seed. It must use
    identical forecast assumptions/timing for all candidates and the shared
    event-level hold evaluator. The returned winner's validation results are
    evidence about prediction stability, not real-population confirmation.
    """
    if target_sign not in (-1, 1):
        raise ValueError('An unambiguous permitted target direction is required')
    if not 0 <= config.component < observation.public_states.shape[1]:
        raise ValueError('Invalid target component')
    rng = np.random.default_rng(config.seed)
    roster, incoming, influence_rank, orders = _proposal_orders(observation, rng)
    if max(config.sizes) > len(roster):
        raise ValueError('Requested group exceeds accessible roster')
    seed_count = config.search_futures + config.validation_futures
    seeds = tuple(int(s) for s in rng.choice(2**31, size=seed_count, replace=False))
    search_seeds = seeds[:config.search_futures]
    validation_seeds = seeds[config.search_futures:]
    evaluated: dict[tuple[int, ...], tuple[PolicyCandidate, tuple, dict]] = {}

    def forecast(candidate, future_seeds):
        rollouts = list(evaluator(candidate, future_seeds))
        if len(rollouts) != len(future_seeds):
            raise ValueError('Forecast evaluator returned an incomplete future batch')
        if any(not r.is_valid or not r.is_eligible for r in rollouts):
            raise ValueError('Invalid/ineligible forecasts cannot be silently removed from ranking')
        score, summary = compute_candidate_score(candidate, rollouts, p0=config.reliability_target)
        summary['evidence_kind'] = 'forecast_only'
        summary['reported_incoming_attention_sum'] = float(incoming[list(candidate.target_nodes)].sum())
        return score, summary

    def evaluate(nodes, method):
        nodes = tuple(sorted(int(n) for n in nodes))
        if nodes in evaluated or len(evaluated) >= config.candidate_limit:
            return
        if len(nodes) not in config.sizes or len(set(nodes)) != len(nodes) or not set(nodes).issubset(roster):
            raise ValueError('Invalid recipient proposal')
        candidate = PolicyCandidate(candidate_id='subset_'+'_'.join(map(str, nodes)),
            target_nodes=nodes, signs=tuple((n, float(target_sign)) for n in nodes),
            nominal_amplitude=config.amplitude, duration=config.duration,
            component=config.component, heuristic_name=method,
            metadata={'information_level':'restricted_forecast','sample_time':observation.sample_time})
        score, summary = forecast(candidate, search_seeds)
        ranks = [influence_rank[n] for n in nodes]
        summary.update({'mean_influence_rank': float(np.mean(ranks)), 'max_influence_rank': float(max(ranks))})
        evaluated[nodes] = (candidate, score, summary)

    for size in config.sizes:
        for method, order in orders:
            evaluate(order[:size], method)

    # Search one-node exchanges around multiple promising groups. Shuffle a
    # complete neighborhood before applying the cap to avoid label-order bias.
    expanded = set()
    while len(evaluated) < config.candidate_limit:
        ranked = sorted(evaluated.items(), key=lambda pair: pair[1][1])
        parents = [(nodes, value) for nodes, value in ranked if nodes not in expanded][:config.beam_width]
        if not parents:
            break
        proposals = set()
        for nodes, _ in parents:
            expanded.add(nodes)
            for removed in nodes:
                for added in roster:
                    if added not in nodes:
                        changed = tuple(sorted((set(nodes)-{removed})|{added}))
                        if changed not in evaluated:
                            proposals.add(changed)
        ordered = sorted(proposals)
        for idx in rng.permutation(len(ordered)):
            evaluate(ordered[int(idx)], 'membership_exchange')
            if len(evaluated) >= config.candidate_limit:
                break
    best = min(evaluated.values(), key=lambda item: item[1])[0]
    # Estimated Pareto frontier over the frozen objectives, then predeclared
    # representatives: lexicographic speed winner; the lowest-count frontier
    # candidate whose estimate meets the reliability target (else the best
    # estimated candidate); the lowest mean observed-influence-rank frontier
    # candidate under the same fallback. At most representative_limit distinct
    # representatives receive extra validation futures; validation is
    # audit-only, never selection.
    frontier_nodes = _estimated_frontier(evaluated)
    pool = frontier_nodes or sorted(evaluated)

    def _pick(key):
        feasible = [n for n in pool if evaluated[n][2]['is_feasible']]
        if feasible:
            return min(feasible, key=key)
        return min(pool, key=lambda n: (-evaluated[n][2]['p_hat'], key(n)))

    reps = {best.target_nodes}
    if len(reps) < config.representative_limit:
        reps.add(_pick(lambda n: (evaluated[n][2]['max_influence_rank'], evaluated[n][2]['mean_influence_rank'], len(n))))
    if len(reps) < config.representative_limit:
        reps.add(_pick(lambda n: (len(n), evaluated[n][2]['mean_restricted_time'])))
    representatives = tuple(sorted(reps)[:config.representative_limit])
    validation_summaries = []
    for nodes in representatives:
        _, validation = forecast(evaluated[nodes][0], validation_seeds)
        validation['used_for_selection'] = False
        validation['representative_role'] = (
            'lexicographic_winner' if nodes == best.target_nodes else 'frontier_representative')
        validation_summaries.append(validation)
    return SubsetSearchResult(best, [v[0] for v in evaluated.values()],
        [v[2] for v in evaluated.values()], validation_summaries,
        tuple(evaluated[n][0] for n in frontier_nodes), tuple(evaluated[n][0] for n in representatives),
        search_seeds, validation_seeds,
        len(evaluated)+len(representatives),
        len(evaluated)*config.search_futures+len(representatives)*config.validation_futures,
        len(evaluated) == config.candidate_limit, influence_rank)
