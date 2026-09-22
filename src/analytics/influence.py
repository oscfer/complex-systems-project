"""Influence and Attention Concentration Analytics.

This module implements Stage 4 diagnostics for measuring social influence,
attention inequality, and ranking persistence from the interaction memory matrix W.

Simulation Stage Mapping:
    Stage 4: Measurement, Analytics & Neural Diagnostics
    (Observes output from Stage 0-3 simulation trajectories).

Key Capabilities:
    - Row-normalized attention matrix A and active-row fraction f_active.
    - Received attention vector r and normalized attention shares p.
    - Herfindahl concentration index C in [1/N, 1].
    - Effective number of influential leaders N_eff = 1 / C in [1, N].
    - Normalized concentration C_N in [0, 1] with safe handling of empty graphs.
    - Spearman rank correlation persistence across temporal lags with midpoint tie handling.
    - Top-fraction leadership membership overlap (Jaccard index) with conservative tie policy.
"""

from __future__ import annotations

from typing import Any, Dict, Set
import numpy as np


def calculate_attention_matrix(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute row-normalized attention matrix A and active-row fraction f_active.

    [Analytics]
    Maps interaction memory counts W into row-stochastic attention weights:
        A_{ij} = W_{ij} / sum_k W_{ik}  if sum_k W_{ik} > 0 else 0.0

    Args:
        W: (N, N) non-negative encounter memory matrix with zero diagonal.

    Returns:
        Tuple of:
            A: (N, N) row-normalized attention matrix (float64).
            f_active: Fraction of agents with at least one recorded interaction (float).
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
        return np.zeros((0, 0), dtype=np.float64), 0.0

    row_sums = W.sum(axis=1, dtype=np.float64)
    active_mask = row_sums > 0.0
    f_active = float(np.mean(active_mask))

    A = np.zeros((N, N), dtype=np.float64)
    nonzero_idx = np.where(active_mask)[0]
    for i in nonzero_idx:
        A[i, :] = W[i, :] / row_sums[i]

    return A, f_active


def calculate_influence_metrics(W: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive influence inequality and attention concentration metrics.

    [Analytics]
    Evaluates:
        - Received attention: r_j = sum_i A_{ij}
        - Normalized attention shares: p_j = r_j / sum_k r_k
        - Herfindahl concentration: C = sum_j p_j^2
        - Normalized concentration: C_N = (N*C - 1) / (N - 1) for N > 1
        - Effective number of leaders: 1 / C
        - Nonempty-row fraction: f_active

    Guarantees:
        Safely handles empty graphs (W = 0 or total attention = 0) and single-agent
        populations (N <= 1) by returning np.nan without raising ZeroDivisionError.

    Args:
        W: (N, N) encounter memory matrix.

    Returns:
        Dictionary containing:
            "A": Row-normalized attention matrix (N, N).
            "r": Received attention vector (N,).
            "p": Attention share vector (N,), or all-NaN if undefined.
            "C": Herfindahl concentration in [1/N, 1], or NaN.
            "C_N": Normalized concentration in [0, 1], or NaN.
            "effective_leaders": Effective number of influential agents, or NaN.
            "f_active": Active-row fraction in [0, 1].
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
            "A": np.zeros((0, 0), dtype=np.float64),
            "r": np.array([], dtype=np.float64),
            "p": np.array([], dtype=np.float64),
            "C": np.nan,
            "C_N": np.nan,
            "effective_leaders": np.nan,
            "f_active": 0.0,
        }

    A, f_active = calculate_attention_matrix(W)
    r = A.sum(axis=0)
    r_sum = float(r.sum())

    if r_sum == 0.0 or N <= 1:
        return {
            "A": A,
            "r": r,
            "p": np.full(N, np.nan, dtype=np.float64),
            "C": np.nan,
            "C_N": np.nan,
            "effective_leaders": np.nan,
            "f_active": f_active,
        }

    p = r / r_sum
    C = float(np.clip(np.sum(p ** 2), 1.0 / N, 1.0))
    # Normalized concentration C_N in [0, 1]
    C_N = float(np.clip((N * C - 1.0) / (N - 1.0), 0.0, 1.0)) if N > 1 else np.nan
    effective_leaders = float(np.clip(1.0 / C, 1.0, float(N))) if C > 0.0 else np.nan

    return {
        "A": A,
        "r": r,
        "p": p,
        "C": C,
        "C_N": C_N,
        "effective_leaders": effective_leaders,
        "f_active": f_active,
    }


def spearman_rank_persistence(r1: np.ndarray, r2: np.ndarray) -> float:
    """Calculate Spearman rank correlation between attention vectors across temporal lags.

    [Analytics]
    Assigns average (midpoint fractional) ranks to tied values.
    Returns np.nan if either rank vector has zero variance (e.g. constant attention)
    or if vectors have fewer than 2 elements.

    Args:
        r1: Attention vector at initial snapshot t.
        r2: Attention vector at delayed snapshot t + Delta t.

    Returns:
        Spearman rank correlation coefficient in [-1.0, 1.0], or np.nan if undefined.
    """
    r1 = np.asarray(r1, dtype=np.float64).ravel()
    r2 = np.asarray(r2, dtype=np.float64).ravel()

    if len(r1) != len(r2) or len(r1) < 2:
        return np.nan

    if not np.isfinite(r1).all() or not np.isfinite(r2).all():
        return np.nan

    def _fractional_ranks(x: np.ndarray) -> np.ndarray:
        n = len(x)
        sort_indices = np.argsort(x)
        sorted_x = x[sort_indices]
        ranks = np.zeros(n, dtype=np.float64)

        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            avg_rank = 1.0 + (i + j) / 2.0
            ranks[sort_indices[i : j + 1]] = avg_rank
            i = j + 1
        return ranks

    R1 = _fractional_ranks(r1)
    R2 = _fractional_ranks(r2)

    std1 = np.std(R1)
    std2 = np.std(R2)
    if std1 == 0.0 or std2 == 0.0:
        return np.nan

    corr = np.corrcoef(R1, R2)[0, 1]
    return float(np.clip(corr, -1.0, 1.0))


def _get_top_fraction_nodes(r: np.ndarray, fraction: float) -> Set[int]:
    """Identify nodes in top fraction of received attention with tie inclusion policy."""
    n = len(r)
    if fraction <= 0.0 or n == 0:
        return set()
    if fraction >= 1.0:
        return set(range(n))

    k = max(1, int(np.ceil(n * fraction)))
    sorted_vals = np.sort(r)[::-1]
    cutoff_val = sorted_vals[k - 1]
    return set(int(idx) for idx in np.where(r >= cutoff_val)[0])


def top_fraction_overlap(
    r1: np.ndarray,
    r2: np.ndarray,
    fraction: float = 0.10,
) -> Dict[str, Any]:
    """Calculate Jaccard membership overlap for the top fraction of influential nodes.

    [Analytics]
    Applies an explicit, unbiased tie policy: any agent tied with the threshold score
    at the boundary rank is included in the top set. Never breaks ties arbitrarily by ID.

    Args:
        r1: Attention vector at initial time t.
        r2: Attention vector at delayed time t + Delta t.
        fraction: Nominal fraction of top agents to compare (0.0 <= fraction <= 1.0).

    Returns:
        Dictionary containing:
            "jaccard_overlap": Intersection size divided by union size in [0.0, 1.0].
            "size_t1": Realized count of agents in top fraction at t (including ties).
            "size_t2": Realized count of agents in top fraction at t + Delta t (including ties).
    """
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")

    r1 = np.asarray(r1, dtype=np.float64).ravel()
    r2 = np.asarray(r2, dtype=np.float64).ravel()
    if len(r1) != len(r2):
        raise ValueError(f"Vector lengths do not match: {len(r1)} vs {len(r2)}")

    s1 = _get_top_fraction_nodes(r1, fraction)
    s2 = _get_top_fraction_nodes(r2, fraction)

    union = s1 | s2
    intersection = s1 & s2

    if len(union) == 0:
        overlap = 1.0 if fraction == 0.0 else 0.0
    else:
        overlap = float(len(intersection) / len(union))

    return {
        "jaccard_overlap": overlap,
        "size_t1": len(s1),
        "size_t2": len(s2),
    }
