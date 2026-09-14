"""Local fields and attention matrix computations for Emergent & Collective Order Simulator (Stages 0, 1, 2).

Calculates row-normalized attention matrices A, received attention r,
public driving fields F^y, private driving fields F^x, and scheduled external pulses h(t).
Supports both full vectorized evaluations (batch) and single-site evaluations (Poisson events).

Simulation Stage Mapping:
    - Stage 0: Kernel timing benchmarks (single-site and batch field evaluations).
    - Stage 1: Core belief dynamics (social fields J * A @ y, private coupling K * x).
    - Stage 2: Controlled perturbations (external field pulse injection h(t)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


def compute_attention_matrix(W: np.ndarray) -> np.ndarray:
    """Compute row-normalized attention matrix A from memory matrix W (Stages 1 & 4).

    A_{ij} = W_{ij} / sum_k W_{ik} for active rows; 0.0 for empty rows.
    Guarantees no division-by-zero warnings or NaNs.

    Args:
        W: Integer memory count matrix of shape (N, N).

    Returns:
        Attention matrix A of shape (N, N), dtype float64, with row sums in {0.0, 1.0}.
    """
    row_sums = np.sum(W, axis=1, keepdims=True, dtype=np.float64)
    nonzero = row_sums > 0.0
    A = np.zeros(W.shape, dtype=np.float64)
    np.divide(W, row_sums, out=A, where=nonzero)
    return A


def compute_received_attention(A: np.ndarray) -> np.ndarray:
    """Compute received attention vector r = sum_i A_{ij}.

    Args:
        A: Row-normalized attention matrix of shape (N, N).

    Returns:
        Received attention vector r of shape (N,), dtype float64.
        Sum of r equals the number of active agents (with row sum > 0).
    """
    return np.sum(A, axis=0, dtype=np.float64)


def compute_single_row_attention(W_row: np.ndarray) -> np.ndarray:
    """Compute normalized attention vector for a single agent row: A[i, :].

    Args:
        W_row: Outgoing memory counts for agent i, shape (N,).

    Returns:
        Attention vector of shape (N,), dtype float64. All zeros if row sum is 0.
    """
    s_i = float(np.sum(W_row))
    if s_i > 0.0:
        return W_row.astype(np.float64) / s_i
    return np.zeros_like(W_row, dtype=np.float64)


def compute_public_fields(
    A: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    J: float = 1.0,
    K: Union[float, np.ndarray] = 0.5,
    h: Optional[Union[float, np.ndarray]] = None,
    g: Optional[Union[float, np.ndarray]] = None,
    u: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Vectorized calculation of public local fields F^y across all agents and components.

    F^y_{i, alpha} = J * sum_j A_{ij} y_{j, alpha} + K_i * x_{i, alpha} + h_{i, alpha}(t) + g_i * u_{i, alpha}(t)

    Args:
        A: Row-normalized attention matrix of shape (N, N).
        y: Expressed belief states of shape (N, D).
        x: Private belief states of shape (N, D).
        J: Social alignment coupling strength >= 0 (default 1.0).
        K: Private-public consistency coupling strength (scalar or array of shape (N,)).
        h: Optional external applied field (scalar, shape (D,), or shape (N, D)).
        g: Optional intervention susceptibility (scalar or array of shape (N,)).
        u: Optional commanded intervention field (scalar, shape (D,), or shape (N, D)).

    Returns:
        Array of public local fields of shape (N, D), dtype float64.
    """
    N = x.shape[0]
    D = x.shape[1] if x.ndim > 1 else 1

    if isinstance(K, np.ndarray):
        if K.ndim == 1:
            K_term = K[:, np.newaxis] * x
        else:
            K_term = K * x
    else:
        K_term = K * x

    Fy = J * np.dot(A, y) + K_term
    if h is not None:
        h_arr = np.asarray(h) if isinstance(h, (list, tuple, np.ndarray)) else h
        if isinstance(h_arr, np.ndarray) and h_arr.ndim == 1 and h_arr.shape[0] == N and D == 1:
            h_arr = h_arr[:, np.newaxis]
        Fy = Fy + h_arr
    if u is not None:
        u_arr = np.asarray(u) if isinstance(u, (list, tuple, np.ndarray)) else u
        if isinstance(u_arr, np.ndarray) and u_arr.ndim == 1 and u_arr.shape[0] == N and D == 1:
            u_arr = u_arr[:, np.newaxis]
        if g is not None:
            g_arr = np.asarray(g) if isinstance(g, (list, tuple, np.ndarray)) else g
            if isinstance(g_arr, np.ndarray) and g_arr.ndim == 1:
                g_term = g_arr[:, np.newaxis] * u_arr
            else:
                g_term = g_arr * u_arr
        else:
            g_term = u_arr
        Fy = Fy + g_term
    return Fy


calculate_public_fields = compute_public_fields


def compute_single_public_field(
    W_row: np.ndarray,
    y_col: np.ndarray,
    x_val: float,
    J: float = 1.0,
    K: float = 0.5,
    h_val: float = 0.0,
    g_val: float = 1.0,
    u_val: float = 0.0,
) -> float:
    """Compute single-component public local field F^y_{i, alpha} in O(k_i) time.

    Args:
        W_row: Outgoing memory vector W[i, :] of shape (N,).
        y_col: Expressed state vector for component alpha: y[:, alpha] of shape (N,).
        x_val: Agent i's private belief value x[i, alpha].
        J: Social alignment strength.
        K: Private-public consistency strength.
        h_val: External field value h[i, alpha].
        g_val: Intervention susceptibility g_i.
        u_val: Commanded intervention field u_{i, alpha}.

    Returns:
        Scalar field value F^y_{i, alpha}.
    """
    s_i = float(np.sum(W_row))
    if s_i > 0.0:
        social = J * (float(np.dot(W_row, y_col)) / s_i)
    else:
        social = 0.0
    return float(social + K * x_val + h_val + g_val * u_val)


def compute_private_fields(
    x: np.ndarray,
    y: np.ndarray,
    B: Optional[np.ndarray] = None,
    K: Union[float, np.ndarray] = 0.5,
    L: float = 1.0,
    b: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Vectorized calculation of private local fields F^x across all agents and components.

    F^x_{i, alpha} = K_i * y_{i, alpha} + L * sum_{beta != alpha} B_{alpha, beta} * x_{i, beta} + b_{i, alpha}

    Args:
        x: Private belief states of shape (N, D).
        y: Expressed belief states of shape (N, D).
        B: Belief coupling matrix of shape (D, D). If None or D=1, coupling is 0.
        K: Private-public consistency coupling strength (scalar or array of shape (N,)).
        L: Within-agent cross-belief coupling strength >= 0 (default 1.0).
        b: Optional intrinsic private bias (scalar, shape (D,), or shape (N, D)).

    Returns:
        Array of private local fields of shape (N, D), dtype float64.
    """
    N = x.shape[0]
    D = x.shape[1] if x.ndim > 1 else 1

    if isinstance(K, np.ndarray):
        if K.ndim == 1:
            K_term = K[:, np.newaxis] * y
        else:
            K_term = K * y
    else:
        K_term = K * y

    Fx = K_term
    if B is not None and x.shape[1] > 1 and not np.allclose(B, 0.0) and L != 0.0:
        # Since B is symmetric and diag(B) = 0: (x @ B)_{i, alpha} = sum_{beta != alpha} B_{alpha, beta} x_{i, beta}
        Fx = Fx + L * np.dot(x, B)
    if b is not None:
        b_arr = np.asarray(b) if isinstance(b, (list, tuple, np.ndarray)) else b
        if isinstance(b_arr, np.ndarray) and b_arr.ndim == 1 and b_arr.shape[0] == N and D == 1:
            b_arr = b_arr[:, np.newaxis]
        Fx = Fx + b_arr
    return Fx


calculate_private_fields = compute_private_fields


def compute_single_private_field(
    y_val: float,
    x_row: np.ndarray,
    alpha: int,
    B: Optional[np.ndarray] = None,
    K: float = 0.5,
    L: float = 1.0,
    b_val: float = 0.0,
) -> float:
    """Compute single-component private local field F^x_{i, alpha} in O(D) time.

    Args:
        y_val: Agent i's expressed stance y[i, alpha].
        x_row: Agent i's private belief vector x[i, :] of shape (D,).
        alpha: Component index (0-indexed).
        B: Belief coupling matrix of shape (D, D).
        K: Private-public consistency strength K_i.
        L: Within-agent cross-belief coupling strength.
        b_val: Intrinsic private bias value b_{i, alpha}.

    Returns:
        Scalar field value F^x_{i, alpha}.
    """
    field_val = K * y_val + b_val
    if B is not None and len(x_row) > 1 and L != 0.0:
        field_val += L * float(np.dot(B[alpha], x_row))
    return float(field_val)


@dataclass
class FieldPulseSpec:
    """Specification of scheduled external field pulse."""

    t_start: float
    t_end: float
    amplitude: float
    component: int = 0
    target_nodes: Optional[np.ndarray] = None


def compute_external_field(
    t: float,
    pulse: Optional[FieldPulseSpec],
    N: int,
    D: int,
) -> np.ndarray:
    """Compute applied external field array h(t) of shape (N, D) at time t.

    Returns zero array if pulse is None or current time is outside [t_start, t_end).
    """
    h = np.zeros((N, D), dtype=np.float64)
    if pulse is not None and pulse.t_start <= t < pulse.t_end:
        if pulse.target_nodes is None:
            h[:, pulse.component] = pulse.amplitude
        else:
            h[pulse.target_nodes, pulse.component] = pulse.amplitude
    return h
