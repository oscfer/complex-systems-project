"""Heuristic scoring and greedy budget allocation for optimal transformation control.

Implements:
1. Degree-based heuristics (in-degree, out-degree, total degree).
2. Influence-based heuristics (received attention, attention share, cross-group ties).
3. Candidate subset selection respecting accessible roster, observation masks, and quotas.
4. Greedy budget allocation enforcing distinct-node limits (k), exposure caps (B), and amplitude caps (u_max).
5. Robust fallback allocation handling unobserved, uninformative, or missing graph states.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


def compute_degree_scores(
    N: int,
    W: Optional[np.ndarray] = None,
    observed_edges: Optional[Dict[Tuple[int, int], int]] = None,
    mode: str = "total",
) -> np.ndarray:
    """Compute degree scores from adjacency/memory matrix or revealed edges.

    Args:
        N: Total number of nodes in population.
        W: Optional (N, N) full memory or adjacency matrix.
        observed_edges: Optional mapping of directed ties (i, j) -> count.
        mode: Degree mode: 'in' (received), 'out' (outgoing), or 'total' (in + out).

    Returns:
        (N,) float64 array of degree scores.
    """
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    valid_modes = {"in", "out", "total"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")

    in_deg = np.zeros(N, dtype=np.float64)
    out_deg = np.zeros(N, dtype=np.float64)

    if W is not None:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.shape != (N, N):
            raise ValueError(f"W shape {W_arr.shape} does not match (N, N) = ({N}, {N})")
        # Exclude self-loops
        W_noself = W_arr.copy()
        np.fill_diagonal(W_noself, 0.0)
        out_deg += np.sum(W_noself > 0, axis=1)
        in_deg += np.sum(W_noself > 0, axis=0)

    elif observed_edges is not None:
        for (i, j), count in observed_edges.items():
            if count > 0 and 0 <= i < N and 0 <= j < N and i != j:
                out_deg[i] += 1.0
                in_deg[j] += 1.0

    if mode == "in":
        return in_deg
    elif mode == "out":
        return out_deg
    else:
        return in_deg + out_deg


def compute_influence_scores(
    N: int,
    W: Optional[np.ndarray] = None,
    observed_edges: Optional[Dict[Tuple[int, int], int]] = None,
    mode: str = "received_attention",
    group_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute influence and attention concentration scores from memory or revealed ties.

    Args:
        N: Total number of nodes in population.
        W: Optional (N, N) interaction memory matrix.
        observed_edges: Optional mapping of revealed ties (i, j) -> count.
        mode: Influence mode: 'received_attention', 'attention_share', or 'cross_group'.
        group_labels: Optional (N,) opportunity group labels for cross-group scoring.

    Returns:
        (N,) float64 array of non-negative influence scores.
    """
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    valid_modes = {"received_attention", "attention_share", "cross_group"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")

    if W is not None:
        W_arr = np.asarray(W, dtype=np.float64)
        if W_arr.shape != (N, N):
            raise ValueError(f"W shape {W_arr.shape} does not match (N, N) = ({N}, {N})")
        W_noself = W_arr.copy()
        np.fill_diagonal(W_noself, 0.0)
    else:
        W_noself = None

    if mode == "cross_group":
        cross_scores = np.zeros(N, dtype=np.float64)
        if group_labels is not None:
            groups = np.asarray(group_labels)
            if len(groups) != N:
                raise ValueError(f"group_labels length {len(groups)} does not match N = {N}")
        else:
            groups = np.zeros(N, dtype=int)
        if W_noself is not None:
            for i in range(N):
                for j in range(N):
                    if i != j and groups[i] != groups[j] and W_noself[i, j] > 0:
                        cross_scores[j] += float(W_noself[i, j])
        elif observed_edges is not None:
            for (i, j), count in observed_edges.items():
                if count > 0 and 0 <= i < N and 0 <= j < N and i != j:
                    if groups[i] != groups[j]:
                        cross_scores[j] += float(count)
        return cross_scores

    # Received attention / attention share
    rec_att = np.zeros(N, dtype=np.float64)
    if W_noself is not None:
        row_sums = W_noself.sum(axis=1)
        nonzero_rows = np.where(row_sums > 0)[0]
        A = np.zeros((N, N), dtype=np.float64)
        for i in nonzero_rows:
            A[i, :] = W_noself[i, :] / row_sums[i]
        rec_att = A.sum(axis=0)
    elif observed_edges is not None:
        # Compute row-normalized attention proxy from revealed edges
        row_sums = np.zeros(N, dtype=np.float64)
        for (i, j), count in observed_edges.items():
            if count > 0 and 0 <= i < N and 0 <= j < N and i != j:
                row_sums[i] += float(count)
        for (i, j), count in observed_edges.items():
            if count > 0 and 0 <= i < N and 0 <= j < N and i != j:
                if row_sums[i] > 0:
                    rec_att[j] += float(count) / row_sums[i]

    if mode == "received_attention":
        return rec_att

    # attention_share
    tot = float(np.sum(rec_att))
    if tot > 1e-12:
        return rec_att / tot
    return np.zeros(N, dtype=np.float64)


def select_heuristic_candidates(
    scores: np.ndarray,
    k: int,
    accessible_roster: Optional[Sequence[int]] = None,
    panel_mask: Optional[np.ndarray] = None,
    quota: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    fallback_mode: str = "random",
) -> List[int]:
    """Select candidate nodes respecting accessible roster, observation quota, and scores.

    Args:
        scores: (N,) heuristic scores.
        k: Maximum distinct nodes to select.
        accessible_roster: Eligible node indices. If None, all nodes are eligible.
        panel_mask: Optional (N,) boolean panel mask.
        quota: Fraction of slots assigned to observed accessible nodes (in [0, 1]).
        rng: Optional NumPy random generator for tie-breaking and unobserved sampling.
        fallback_mode: Fallback strategy ('random' or 'sequential').

    Returns:
        Ordered list of selected unique node indices of length <= min(k, len(accessible_roster)).
    """
    if k <= 0:
        return []
    if not (0.0 <= quota <= 1.0):
        raise ValueError(f"quota must be in [0, 1], got {quota}")
    valid_fallback_modes = {"random", "sequential"}
    if fallback_mode not in valid_fallback_modes:
        raise ValueError(f"Invalid fallback_mode '{fallback_mode}'. Expected one of {valid_fallback_modes}")
    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)

    if accessible_roster is not None:
        candidates = list(dict.fromkeys(int(n) for n in accessible_roster if 0 <= n < N))
    else:
        candidates = list(range(N))

    if len(candidates) == 0:
        return []

    k_eff = min(k, len(candidates))
    rng = rng if rng is not None else np.random.default_rng(42)

    # If no panel mask or quota == 1.0, rank directly on all candidates
    if panel_mask is not None:
        panel_mask = np.asarray(panel_mask, dtype=bool)
        if len(panel_mask) != N:
            raise ValueError(f"panel_mask length {len(panel_mask)} does not match N = {N}")

    if panel_mask is None or quota >= 1.0:
        scored = [(float(scores[n]), float(rng.random()), n) for n in candidates]
        scored.sort(reverse=True)
        return [n for _, _, n in scored[:k_eff]]

    # Split candidates by panel mask
    obs_cands = [n for n in candidates if bool(panel_mask[n])]
    unobs_cands = [n for n in candidates if not bool(panel_mask[n])]

    n_obs = int(round(k_eff * quota))
    n_unobs = k_eff - n_obs

    # Redistribute if either pool is too small
    if len(obs_cands) < n_obs:
        deficit = n_obs - len(obs_cands)
        n_obs = len(obs_cands)
        n_unobs += deficit
    if len(unobs_cands) < n_unobs:
        deficit = n_unobs - len(unobs_cands)
        n_unobs = len(unobs_cands)
        n_obs = min(len(obs_cands), n_obs + deficit)

    # Rank observed candidates by score
    scored_obs = [(float(scores[n]), float(rng.random()), n) for n in obs_cands]
    scored_obs.sort(reverse=True)
    selected_obs = [n for _, _, n in scored_obs[:n_obs]]

    # For unobserved candidates, use scores if available, or fallback
    if fallback_mode == "random":
        perm = rng.permutation(len(unobs_cands))
        selected_unobs = [unobs_cands[i] for i in perm[:n_unobs]]
    else:
        scored_unobs = [(float(scores[n]), float(rng.random()), n) for n in unobs_cands]
        scored_unobs.sort(reverse=True)
        selected_unobs = [n for _, _, n in scored_unobs[:n_unobs]]

    selected = list(dict.fromkeys(selected_obs + selected_unobs))
    return selected[:k_eff]


def allocate_greedy_budget(
    candidate_nodes: Sequence[int],
    exposure_cap: float,
    amplitude_cap: float,
    duration: float,
    k_limit: int,
    allocation_strategy: str = "uniform",
    weights: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Greedily allocate intervention field amplitudes respecting budget contracts.

    Guarantees:
    - At most k_limit distinct nodes are allocated nonzero field.
    - For each node i, 0 <= |u_i| <= amplitude_cap.
    - Integrated exposure sum_i |u_i| * duration <= exposure_cap + 1e-12.
    - Gracefully handles budget exhaustion, zero duration, and empty candidate lists.

    Args:
        candidate_nodes: Priority-ordered candidate node indices.
        exposure_cap: Total exposure budget B >= 0.
        amplitude_cap: Maximum instantaneous field amplitude u_max > 0.
        duration: Actuation pulse duration d > 0.
        k_limit: Distinct-node limit k >= 1.
        allocation_strategy: Strategy: 'uniform', 'greedy_priority', or 'proportional'.
        weights: Optional priority weights for 'proportional' strategy.

    Returns:
        Mapping from node index to allocated absolute field amplitude.
    """
    if exposure_cap <= 1e-12 or duration <= 1e-12 or k_limit <= 0 or amplitude_cap <= 1e-12:
        return {}

    unique_nodes: List[int] = []
    seen: Set[int] = set()
    for n in candidate_nodes:
        if n not in seen:
            seen.add(n)
            unique_nodes.append(int(n))
        if len(unique_nodes) >= k_limit:
            break

    if len(unique_nodes) == 0:
        return {}

    valid_strategies = {"uniform", "greedy_priority", "proportional"}
    if allocation_strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{allocation_strategy}'. Expected one of {valid_strategies}")

    if allocation_strategy == "uniform":
        n_act = len(unique_nodes)
        max_amp_by_budget = exposure_cap / (n_act * duration)
        a_eff = min(amplitude_cap, max_amp_by_budget)
        if a_eff <= 1e-15:
            return {}
        return {n: float(a_eff) for n in unique_nodes}

    elif allocation_strategy == "greedy_priority":
        allocations: Dict[int, float] = {}
        rem_exposure = float(exposure_cap)
        for n in unique_nodes:
            if rem_exposure <= 1e-15:
                break
            max_amp = rem_exposure / duration
            a_n = min(amplitude_cap, max_amp)
            if a_n > 1e-15:
                allocations[n] = float(a_n)
                rem_exposure = max(0.0, rem_exposure - a_n * duration)
        return allocations

    else:  # proportional
        weights_dict = weights or {}
        raw_weights = [max(0.0, float(weights_dict.get(n, 1.0))) for n in unique_nodes]
        sum_w = sum(raw_weights)
        if sum_w <= 1e-12:
            raw_weights = [1.0] * len(unique_nodes)
            sum_w = float(len(unique_nodes))

        # Iteratively allocate budget respecting amplitude_cap via water-filling
        allocations = {n: 0.0 for n in unique_nodes}
        active_indices = list(range(len(unique_nodes)))
        rem_exposure = float(exposure_cap)

        while active_indices and rem_exposure > 1e-12:
            sub_w = sum(raw_weights[i] for i in active_indices)
            if sub_w <= 1e-12:
                if sum_w <= 1e-12:
                    equal_share = (rem_exposure / len(active_indices)) / duration
                    for i in active_indices:
                        allocations[unique_nodes[i]] = min(amplitude_cap, equal_share)
                break

            clipped_indices: List[int] = []
            unclipped_indices: List[int] = []
            for i in active_indices:
                desired_amp = (rem_exposure * (raw_weights[i] / sub_w)) / duration
                if desired_amp >= amplitude_cap - 1e-12:
                    clipped_indices.append(i)
                else:
                    unclipped_indices.append(i)

            if not clipped_indices:
                for i in active_indices:
                    allocations[unique_nodes[i]] = float((rem_exposure * (raw_weights[i] / sub_w)) / duration)
                rem_exposure = 0.0
                break

            for i in clipped_indices:
                allocations[unique_nodes[i]] = float(amplitude_cap)
                rem_exposure = max(0.0, rem_exposure - amplitude_cap * duration)

            active_indices = unclipped_indices

        return {n: a for n, a in allocations.items() if a > 1e-15}
