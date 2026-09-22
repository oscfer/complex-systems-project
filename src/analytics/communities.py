"""Graph Community Detection, Leiden Modularity, and Component Tracking.

This module implements Stage 4 network topology diagnostics:
- Symmetrized projection G = W + W^T for undirected community structure.
- Weighted Leiden modularity optimization across resolution scales gamma in {0.5, 1.0, 2.0}.
- Partition stability evaluation via the Adjusted Rand Index (ARI).
- Weakly Connected Components (WCC) and Strongly Connected Components (SCC) tracking.
- Isolated node fraction f_iso and largest component fraction S_max / N.

Simulation Stage Mapping:
    Stage 4: Measurement, Analytics & Neural Diagnostics
    (Observes output from Stage 0-3 simulation trajectories).
"""

from __future__ import annotations

from collections import deque
from math import comb
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def symmetrize_adjacency(W: np.ndarray) -> np.ndarray:
    """Compute the undirected symmetrized adjacency matrix G = W + W^T.

    [Analytics]
    Discards edge directionality to measure undirected interaction intensity
    for community detection. Does NOT measure directed control. Zeroes self-loops.

    Args:
        W: (N, N) non-negative encounter memory matrix.

    Returns:
        (N, N) symmetric matrix G with zero diagonal.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("Matrix W must contain finite values")
    if np.any(W < 0):
        raise ValueError("Matrix W must contain non-negative values")
    G = W + W.T
    np.fill_diagonal(G, 0.0)
    return G


def calculate_partition_stability(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the Adjusted Rand Index (ARI) between two community partitions.

    [Analytics]
    Evaluates partition consistency across random optimizer restarts or temporal snapshots.
    Invariant to community label permutations. Returns 1.0 for identical clustering.

    Args:
        p1: Community label vector of shape (N,).
        p2: Community label vector of shape (N,).

    Returns:
        ARI score in [-1.0, 1.0].
    """
    p1 = np.asarray(p1).ravel()
    p2 = np.asarray(p2).ravel()
    n = len(p1)
    if n != len(p2):
        raise ValueError(f"Partition lengths do not match: {n} vs {len(p2)}")
    if n <= 1:
        return 1.0

    u1, inv1 = np.unique(p1, return_inverse=True)
    u2, inv2 = np.unique(p2, return_inverse=True)
    k1 = len(u1)
    k2 = len(u2)

    table = np.zeros((k1, k2), dtype=np.int64)
    for i in range(n):
        table[inv1[i], inv2[i]] += 1

    sum_comb_table = sum(comb(int(count), 2) for count in table.flat if count > 1)
    sum_comb_a = sum(comb(int(a), 2) for a in table.sum(axis=1) if a > 1)
    sum_comb_b = sum(comb(int(b), 2) for b in table.sum(axis=0) if b > 1)
    total_comb = comb(n, 2)

    if total_comb == 0:
        return 1.0

    expected_index = (sum_comb_a * sum_comb_b) / total_comb
    max_index = (sum_comb_a + sum_comb_b) / 2.0

    if max_index == expected_index:
        return 1.0 if sum_comb_table == sum_comb_a == sum_comb_b else 0.0

    ari = (sum_comb_table - expected_index) / (max_index - expected_index)
    return float(np.clip(ari, -1.0, 1.0))


def connected_components_analysis(W: np.ndarray) -> Dict[str, Any]:
    """Track Weakly Connected Components (WCC) and Strongly Connected Components (SCC).

    [Analytics]
    Evaluates:
        - WCC count and sizes on the undirected support of positive ties.
        - SCC count and sizes via Tarjan's algorithm on directed positive ties.
        - Largest component fraction S_max / N.
        - Isolated node fraction f_iso (nodes with 0 in-degree and 0 out-degree).

    Args:
        W: (N, N) encounter memory or adjacency matrix.

    Returns:
        Dictionary containing:
            "num_wcc": Count of weakly connected components.
            "num_scc": Count of strongly connected components.
            "largest_wcc_fraction": S_max / N in [0.0, 1.0].
            "isolated_fraction": Fraction of completely isolated nodes in [0.0, 1.0].
            "wcc_labels": 1D array of WCC component IDs per node.
            "scc_labels": 1D array of SCC component IDs per node.
            "wcc_sizes": List of component sizes for WCCs.
            "scc_sizes": List of component sizes for SCCs.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("Matrix W must contain finite values")
    if np.any(W < 0):
        raise ValueError("Matrix W must contain non-negative values")
    N = W.shape[0]
    if N == 0:
        return {
            "num_wcc": 0,
            "num_scc": 0,
            "largest_wcc_fraction": 0.0,
            "isolated_fraction": 0.0,
            "wcc_labels": np.array([], dtype=np.int32),
            "scc_labels": np.array([], dtype=np.int32),
            "wcc_sizes": [],
            "scc_sizes": [],
        }

    # 1. WCC: Undirected reachability on support of W
    supp_undir = (W > 0) | (W.T > 0)
    np.fill_diagonal(supp_undir, False)
    rows_u, cols_u = np.nonzero(supp_undir)
    adj_undir: List[List[int]] = [[] for _ in range(N)]
    for r, c in zip(rows_u, cols_u):
        adj_undir[r].append(int(c))

    wcc_labels = np.full(N, -1, dtype=np.int32)
    current_wcc = 0
    wcc_sizes: List[int] = []

    for i in range(N):
        if wcc_labels[i] == -1:
            size = 0
            q = deque([i])
            wcc_labels[i] = current_wcc
            while q:
                u = q.popleft()
                size += 1
                for v in adj_undir[u]:
                    if wcc_labels[v] == -1:
                        wcc_labels[v] = current_wcc
                        q.append(v)
            wcc_sizes.append(size)
            current_wcc += 1

    # 2. SCC: Iterative Kosaraju algorithm on directed edges W[i, j] > 0
    supp_dir = (W > 0)
    np.fill_diagonal(supp_dir, False)
    rows_d, cols_d = np.nonzero(supp_dir)
    adj_dir: List[List[int]] = [[] for _ in range(N)]
    for r, c in zip(rows_d, cols_d):
        adj_dir[r].append(int(c))

    # Pass 1: compute DFS finish order using explicit stack (no recursion depth limit)
    visited = [False] * N
    order: List[int] = []
    for i in range(N):
        if not visited[i]:
            stack: List[Tuple[int, int]] = [(i, 0)]
            visited[i] = True
            while stack:
                u, edge_idx = stack[-1]
                if edge_idx < len(adj_dir[u]):
                    stack[-1] = (u, edge_idx + 1)
                    v = adj_dir[u][edge_idx]
                    if not visited[v]:
                        visited[v] = True
                        stack.append((v, 0))
                else:
                    stack.pop()
                    order.append(u)

    # Build transpose adjacency list
    adj_rev: List[List[int]] = [[] for _ in range(N)]
    for u in range(N):
        for v in adj_dir[u]:
            adj_rev[v].append(u)

    # Pass 2: assign SCC labels in reverse finish order
    scc_labels = np.full(N, -1, dtype=np.int32)
    current_scc = 0
    scc_sizes: List[int] = []

    for root in reversed(order):
        if scc_labels[root] == -1:
            stack_scc = [root]
            scc_labels[root] = current_scc
            size = 0
            while stack_scc:
                u = stack_scc.pop()
                size += 1
                for v in adj_rev[u]:
                    if scc_labels[v] == -1:
                        scc_labels[v] = current_scc
                        stack_scc.append(v)
            scc_sizes.append(size)
            current_scc += 1

    # 3. Component summary fractions
    iso_count = sum(1 for i in range(N) if len(adj_undir[i]) == 0)
    iso_fraction = float(iso_count / N)
    largest_wcc_fraction = float(max(wcc_sizes) / N) if wcc_sizes else 0.0

    return {
        "num_wcc": current_wcc,
        "num_scc": current_scc,
        "largest_wcc_fraction": largest_wcc_fraction,
        "isolated_fraction": iso_fraction,
        "wcc_labels": wcc_labels,
        "scc_labels": scc_labels,
        "wcc_sizes": wcc_sizes,
        "scc_sizes": scc_sizes,
    }


def calculate_modularity(G: np.ndarray, partition: np.ndarray, gamma: float = 1.0) -> float:
    """Calculate weighted modularity Q(gamma) for an undirected graph G.

    [Analytics]
    Q(gamma) = (1 / 2m) sum_{i, j} [ G_{ij} - gamma * (k_i * k_j) / (2m) ] * delta(sigma_i, sigma_j)

    Args:
        G: (N, N) symmetric non-negative adjacency matrix.
        partition: 1D array of community assignments sigma_i.
        gamma: Modularity resolution parameter (default 1.0).

    Returns:
        Modularity score Q in [-1.0, 1.0], or np.nan if 2m == 0.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {G.shape}")
    if not np.isfinite(G).all():
        raise ValueError("Matrix G must contain finite values")
    if np.any(G < 0):
        raise ValueError("Matrix G must contain non-negative values")
    partition = np.asarray(partition).ravel()
    N = G.shape[0]
    if len(partition) != N:
        raise ValueError(f"Partition length mismatch: len(partition)={len(partition)} vs N={N}")
    k = G.sum(axis=1)
    two_m = float(k.sum())
    if two_m == 0.0:
        return np.nan

    unique_c = np.unique(partition)
    Q = 0.0
    for c in unique_c:
        nodes = np.where(partition == c)[0]
        subgraph_weight = G[np.ix_(nodes, nodes)].sum()
        community_strength = k[nodes].sum()
        Q += subgraph_weight - gamma * (community_strength ** 2) / two_m

    return float(Q / two_m)


def _local_move(
    G: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
    initial_partition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run the fast local-moving phase used by Leiden."""
    N = G.shape[0]
    k = G.sum(axis=1)
    two_m = float(k.sum())
    if two_m == 0.0:
        return np.arange(N)

    partition = (
        np.arange(N, dtype=np.int32)
        if initial_partition is None
        else np.asarray(initial_partition, dtype=np.int32).copy()
    )
    _, partition = np.unique(partition, return_inverse=True)
    comm_strength: Dict[int, float] = {
        int(c): float(np.sum(k[partition == c])) for c in np.unique(partition)
    }

    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        nonzero_cols = np.flatnonzero(G[i])
        for j in nonzero_cols:
            neighbors[i].append((int(j), float(G[i, j])))

    order = list(range(N))
    rng.shuffle(order)

    improved = True
    passes = 0
    max_passes = 50

    while improved and passes < max_passes:
        improved = False
        passes += 1
        rng.shuffle(order)

        for i in order:
            if k[i] == 0.0:
                continue
            cur_comm = partition[i]

            # Weights to neighboring communities via sparse adjacency
            weights_to_comm: Dict[int, float] = {}
            for j, w in neighbors[i]:
                c = partition[j]
                weights_to_comm[c] = weights_to_comm.get(c, 0.0) + w

            d_cur = weights_to_comm.get(cur_comm, 0.0)
            K_cur_minus_i = comm_strength[cur_comm] - k[i]
            loss_remove = 2.0 * d_cur - 2.0 * gamma * k[i] * K_cur_minus_i / two_m

            best_gain = 0.0
            best_comm: Optional[int] = cur_comm

            # Consider moving to empty singleton
            gain_singleton = -loss_remove
            if gain_singleton > best_gain + 1e-12:
                best_gain = gain_singleton
                best_comm = None

            for c, d_c in weights_to_comm.items():
                if c == cur_comm:
                    continue
                K_c = comm_strength[c]
                gain_add = 2.0 * d_c - 2.0 * gamma * k[i] * K_c / two_m
                net_gain = gain_add - loss_remove
                if net_gain > best_gain + 1e-12:
                    best_gain = net_gain
                    best_comm = c

            if best_comm != cur_comm and best_gain > 1e-12:
                improved = True
                comm_strength[cur_comm] -= k[i]
                if best_comm is None:
                    new_c = max(comm_strength.keys()) + 1
                    partition[i] = new_c
                    comm_strength[new_c] = float(k[i])
                else:
                    partition[i] = best_comm
                    comm_strength[best_comm] += k[i]

    _, remapped = np.unique(partition, return_inverse=True)
    return remapped


def _refine_partition(
    G: np.ndarray,
    parent: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
    theta: float = 0.01,
) -> np.ndarray:
    """Leiden refinement with subset well-connectedness and randomized merges.

    This follows the refinement phase in Traag, Waltman & van Eck (2019):
    singleton nodes are considered only when well connected to their parent
    subset; eligible target subsets must also satisfy the gamma-connectedness
    condition, and a positive-gain target is sampled with probability
    proportional to ``exp(delta_Q / theta)``. Merges are restricted to
    adjacent subsets, so refined communities remain connected.
    """
    N = G.shape[0]
    strengths = G.sum(axis=1)
    two_m = float(strengths.sum())
    refined = np.arange(N, dtype=np.int32)
    community_strength = {i: float(strengths[i]) for i in range(N)}
    community_size = {i: 1 for i in range(N)}
    parent_strength = {
        int(c): float(np.sum(strengths[parent == c])) for c in np.unique(parent)
    }
    order = np.arange(N)
    rng.shuffle(order)

    for i in order:
        if strengths[i] == 0.0 or community_size[int(refined[i])] != 1:
            continue
        parent_id = int(parent[i])
        weight_to_parent = float(np.sum(G[i, parent == parent_id]))
        required_parent = gamma * strengths[i] * (parent_strength[parent_id] - strengths[i]) / two_m
        if weight_to_parent + 1e-12 < required_parent:
            continue
        weights: Dict[int, float] = {}
        for j in np.flatnonzero(G[i] > 0):
            if parent[j] != parent[i]:
                continue
            target = int(refined[j])
            if target == int(refined[i]):
                continue
            weights[target] = weights.get(target, 0.0) + float(G[i, j])
        candidates: List[int] = []
        gains: List[float] = []
        for target, edge_weight in weights.items():
            gain = 2.0 * edge_weight - 2.0 * gamma * strengths[i] * community_strength[target] / two_m
            required_target = gamma * strengths[i] * community_strength[target] / two_m
            if edge_weight + 1e-12 >= required_target and gain > 1e-12:
                candidates.append(target)
                gains.append(gain)
        if candidates:
            scaled = (np.asarray(gains) - max(gains)) / max(theta, 1e-12)
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum()
            best = int(rng.choice(np.asarray(candidates), p=probabilities))
            old = int(refined[i])
            refined[i] = best
            community_strength[best] += float(strengths[i])
            community_strength[old] = 0.0
            community_size[best] += 1
            community_size[old] = 0

    _, remapped = np.unique(refined, return_inverse=True)
    return remapped.astype(np.int32)


def _aggregate_graph(G: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """Aggregate a weighted graph by a partition, retaining self-loop mass."""
    count = int(np.max(partition)) + 1 if len(partition) else 0
    aggregated = np.zeros((count, count), dtype=np.float64)
    rows, cols = np.nonzero(G)
    for i, j in zip(rows, cols):
        aggregated[partition[i], partition[j]] += G[i, j]
    return aggregated


def _leiden_local_move(
    G: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Optimize modularity with Leiden local-move, refinement, and aggregation."""
    original_n = G.shape[0]
    if original_n == 0 or float(G.sum()) == 0.0:
        return np.arange(original_n, dtype=np.int32)

    current_graph = G.copy()
    original_to_current = np.arange(original_n, dtype=np.int32)
    initial: Optional[np.ndarray] = None

    for _ in range(max(1, original_n)):
        parent = _local_move(current_graph, gamma, rng, initial)
        refined = _refine_partition(current_graph, parent, gamma, rng)
        num_refined = len(np.unique(refined))

        if num_refined == current_graph.shape[0]:
            final = parent[original_to_current]
            _, final = np.unique(final, return_inverse=True)
            return final.astype(np.int32)

        parent_by_refined = np.zeros(num_refined, dtype=np.int32)
        for refined_id in range(num_refined):
            parent_by_refined[refined_id] = int(parent[np.flatnonzero(refined == refined_id)[0]])
        _, parent_by_refined = np.unique(parent_by_refined, return_inverse=True)

        original_to_current = refined[original_to_current]
        current_graph = _aggregate_graph(current_graph, refined)
        initial = parent_by_refined.astype(np.int32)

    final = initial[original_to_current] if initial is not None else original_to_current
    _, final = np.unique(final, return_inverse=True)
    return final.astype(np.int32)


def detect_communities(
    W: np.ndarray,
    gamma: float = 1.0,
    restarts: int = 10,
    rng: Optional[Any] = None,
    n_restarts: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform weighted Leiden modularity optimization and component tracking.

    [Analytics]
    Optimizes modularity on the symmetrized projection G = W + W^T with zero diagonal.
    Executes multiple independent random starts, retaining the partition achieving the highest Q,
    and evaluating partition stability across restarts via pairwise Adjusted Rand Index (ARI).

    Args:
        W: Interaction memory matrix (directed or undirected).
        gamma: Resolution parameter (default: 1.0; checks at 0.5 and 2.0).
        restarts: Number of optimization restarts (default: 10).
        rng: Optional np.random.Generator or integer seed.
        n_restarts: Alias for restarts.
        seed: Optional random seed integer.

    Returns:
        Dictionary containing:
            "partition": 1D integer array of community assignments (0 .. K-1).
            "modularity_Q": Highest achieved modularity score Q.
            "num_communities": Count of detected communities.
            "stability_ari": Average pairwise ARI across restarts.
            "wcc": Component tracking results from connected_components_analysis.
            "scc": SCC diagnostic metrics.
            "gamma": Resolution parameter used.
            "restarts": Number of restarts evaluated.
    """
    if n_restarts is not None:
        restarts = n_restarts

    if rng is None:
        generator = np.random.default_rng(seed) if isinstance(seed, (int, np.integer)) else np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        generator = np.random.default_rng(rng)
    else:
        generator = rng

    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("Matrix W must contain finite values")
    if np.any(W < 0):
        raise ValueError("Matrix W must contain non-negative values")
    N = W.shape[0]

    # Symmetrize if asymmetric, zero diagonal
    if not np.allclose(W, W.T):
        G = symmetrize_adjacency(W)
    else:
        G = W.copy()
        np.fill_diagonal(G, 0.0)

    cc_info = connected_components_analysis(W)

    two_m = float(G.sum())
    if two_m == 0.0 or N == 0:
        return {
            "partition": np.arange(N, dtype=np.int32),
            "modularity_Q": np.nan,
            "num_communities": N,
            "stability_ari": 1.0,
            "wcc": cc_info,
            "scc": cc_info,
            "gamma": gamma,
            "restarts": restarts,
        }

    partitions: List[np.ndarray] = []
    mod_scores: List[float] = []

    best_Q = -float("inf")
    best_p: Optional[np.ndarray] = None

    for _ in range(max(1, restarts)):
        p = _leiden_local_move(G, gamma, generator)
        q = calculate_modularity(G, p, gamma)
        partitions.append(p)
        mod_scores.append(q)
        if q > best_Q or best_p is None:
            best_Q = q
            best_p = p

    assert best_p is not None

    # Evaluate partition stability across restarts via ARI
    if len(partitions) > 1:
        ari_values = [
            calculate_partition_stability(partitions[i], partitions[j])
            for i in range(len(partitions))
            for j in range(i + 1, len(partitions))
        ]
        stability_ari = float(np.mean(ari_values)) if ari_values else 1.0
    else:
        stability_ari = 1.0

    return {
        "partition": best_p,
        "modularity_Q": best_Q,
        "num_communities": len(np.unique(best_p)),
        "stability_ari": stability_ari,
        "wcc": cc_info,
        "scc": cc_info,
        "gamma": gamma,
        "restarts": restarts,
    }
