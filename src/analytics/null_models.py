"""Degree-Preserving Edge-Swap Null Model and Multiset Permutation.

This module implements Stage 4 randomization benchmarks:
- Extraction of simple unweighted support graphs S = (V, E_supp).
- Degree-preserving double-edge swap Markov chains strictly conserving the support degree sequence.
- Uniform multiset permutation of positive edge weights across the randomized support.

Simulation Stage Mapping:
    Stage 4: Measurement, Analytics & Neural Diagnostics
    (Baseline comparison model for community structure significance).
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Union
import numpy as np


def extract_support_graph(G: np.ndarray) -> np.ndarray:
    """Extract the unweighted undirected support graph S = (V, E_supp).

    [Analytics]
    S_{ij} = 1 if G_{ij} > 0 and i != j else 0.
    Diagonal elements are strictly zero.

    Args:
        G: (N, N) weighted adjacency matrix.

    Returns:
        (N, N) binary adjacency matrix (int32).
    """
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {G.shape}")
    if not np.isfinite(G).all():
        raise ValueError("Matrix G must contain finite values")
    if np.any(G < 0):
        raise ValueError("Matrix G must contain non-negative values")
    S = (G > 0).astype(np.int32)
    np.fill_diagonal(S, 0)
    return S


def generate_degree_preserving_null(
    W: np.ndarray,
    num_swaps: int = 10,
    rng: Optional[Any] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate randomized degree-preserving null graph via double-edge swaps and weight permutation.

    [Analytics]
    Algorithm:
        1. Extract the simple support graph S.
        2. Perform Markov chain of degree-preserving edge swaps without introducing self-loops
           or parallel edges, strictly conserving the unweighted degree sequence.
        3. Extract the exact multiset of observed positive edge weights and uniformly permute
           them across the randomized edges.

    Invariants Strictly Preserved:
        - Unweighted support degree sequence d_i = sum_j I(W_{ij} > 0).
        - Total support edge count |E_supp|.
        - Exact multiset of positive edge weights.

    Args:
        W: (N, N) encounter memory or adjacency matrix.
        num_swaps: Number of valid double-edge swaps to attempt.
        rng: Optional np.random.Generator or integer seed.
        seed: Optional integer seed for reproducibility.

    Returns:
        (N, N) null model matrix W_null with identical degree sequence and weight multiset.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Input must be a 2D square matrix, got shape {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("Matrix W must contain finite values")
    if np.any(W < 0):
        raise ValueError("Matrix W must contain non-negative values")
    if num_swaps <= 0 or W.size <= 1:
        return W.copy()

    if rng is None:
        generator = np.random.default_rng(seed) if isinstance(seed, (int, np.integer)) else np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        generator = np.random.default_rng(rng)
    else:
        generator = rng

    N = W.shape[0]
    is_symmetric = np.allclose(W, W.T)

    if is_symmetric:
        # Undirected edge swaps on upper triangle
        edges: List[Tuple[int, int]] = []
        weights: List[Any] = []
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > 0:
                    edges.append((i, j))
                    weights.append(W[i, j])

        M = len(edges)
        if M < 2:
            return W.copy()

        edge_set: Set[Tuple[int, int]] = set(edges)
        edge_list: List[Tuple[int, int]] = list(edges)

        swaps_done = 0
        max_attempts = max(num_swaps * 100, 2000)
        attempts = 0
        consecutive_failures = 0

        while swaps_done < num_swaps and attempts < max_attempts and consecutive_failures < 2000:
            attempts += 1
            idx1, idx2 = generator.choice(M, size=2, replace=False)
            u, v = edge_list[idx1]
            x, y = edge_list[idx2]

            if len({u, v, x, y}) < 4:
                consecutive_failures += 1
                continue

            if generator.random() < 0.5:
                e1 = (min(u, x), max(u, x))
                e2 = (min(v, y), max(v, y))
            else:
                e1 = (min(u, y), max(u, y))
                e2 = (min(v, x), max(v, x))

            if e1 in edge_set or e2 in edge_set or e1 == e2:
                consecutive_failures += 1
                continue

            # Apply swap
            edge_set.remove(edge_list[idx1])
            edge_set.remove(edge_list[idx2])
            edge_set.add(e1)
            edge_set.add(e2)
            edge_list[idx1] = e1
            edge_list[idx2] = e2
            swaps_done += 1
            consecutive_failures = 0

        # Permute positive weight multiset across the (potentially randomized) edge support
        perm_weights = generator.permutation(weights)

        W_null = np.zeros_like(W, dtype=W.dtype)
        for (u, v), w in zip(edge_list, perm_weights):
            W_null[u, v] = w
            W_null[v, u] = w

        return W_null
    else:
        # Directed edge swaps preserving in-degrees and out-degrees
        edges_dir: List[Tuple[int, int]] = []
        weights_dir: List[Any] = []
        for i in range(N):
            for j in range(N):
                if i != j and W[i, j] > 0:
                    edges_dir.append((i, j))
                    weights_dir.append(W[i, j])

        M = len(edges_dir)
        if M < 2:
            return W.copy()

        edge_set_dir: Set[Tuple[int, int]] = set(edges_dir)
        edge_list_dir: List[Tuple[int, int]] = list(edges_dir)

        swaps_done = 0
        max_attempts = max(num_swaps * 100, 2000)
        attempts = 0
        consecutive_failures = 0

        while swaps_done < num_swaps and attempts < max_attempts and consecutive_failures < 2000:
            attempts += 1
            idx1, idx2 = generator.choice(M, size=2, replace=False)
            u, v = edge_list_dir[idx1]
            x, y = edge_list_dir[idx2]

            if u == x or v == y or u == y or x == v:
                consecutive_failures += 1
                continue

            e1 = (u, y)
            e2 = (x, v)

            if e1 in edge_set_dir or e2 in edge_set_dir or e1 == e2:
                consecutive_failures += 1
                continue

            edge_set_dir.remove(edge_list_dir[idx1])
            edge_set_dir.remove(edge_list_dir[idx2])
            edge_set_dir.add(e1)
            edge_set_dir.add(e2)
            edge_list_dir[idx1] = e1
            edge_list_dir[idx2] = e2
            swaps_done += 1
            consecutive_failures = 0

        # Permute positive weight multiset across the (potentially randomized) edge support
        perm_weights = generator.permutation(weights_dir)
        W_null = np.zeros_like(W, dtype=W.dtype)
        for (u, v), w in zip(edge_list_dir, perm_weights):
            W_null[u, v] = w

        return W_null
