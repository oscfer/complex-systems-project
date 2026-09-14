"""Belief coupling matrices and energy calculations for D=1 and D=3 (Stages 1 & 2).

Provides:
1. Standard benchmark coupling matrices: uncoupled B_0, reinforcing triad B_+,
   and frustrated triad B_f.
2. Structural validation of coupling matrices (symmetry, zero diagonal, unit absolute row sums).
3. Vectorized internal fields calculation: L * (x @ B).
4. Exact isolated binary internal energy: E_B(x) = -L * sum_{alpha < beta} B_{alpha, beta} x_alpha x_beta.
5. Exact enumeration and verification of triad energy spectra.

Simulation Stage Mapping:
    - Stage 1: Core multi-dimensional belief couplings (uncoupled, reinforcing, frustrated).
    - Stage 2: Extension campaigns evaluating coupling transitions and cross-component alignment.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def get_coupling_matrix(name: str = "B0", D: int = 3) -> np.ndarray:
    """Return standard belief coupling matrix B of shape (D, D).

    Supported names:
    - For D=1: 'scalar', 'none', 'zero' -> returns (1, 1) zero matrix.
    - For D=3:
      - 'B0', 'uncoupled', 'zero': 3x3 zeros.
      - 'B+', 'b_plus', 'reinforcing': reinforcing triangle with mutual positive coupling.
      - 'Bf', 'b_f', 'frustrated': frustrated triangle with one opposing link.

    Args:
        name: Identifier of coupling matrix.
        D: Belief dimensionality (1 or 3).

    Returns:
        Symmetric coupling matrix B of shape (D, D), dtype float64.
    """
    if D == 1:
        return np.zeros((1, 1), dtype=np.float64)

    if D != 3:
        raise ValueError(f"Belief dimensionality must be D=1 or D=3, got D={D}")

    key = name.strip().lower()
    if key in ("b0", "uncoupled", "none", "zero"):
        return np.zeros((3, 3), dtype=np.float64)
    elif key in ("b+", "b_plus", "reinforcing"):
        return 0.5 * np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
    elif key in ("bf", "b_f", "frustrated"):
        return 0.5 * np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, -1.0, 0.0],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown coupling matrix name: '{name}'. Supported: 'B0', 'B+', 'Bf'.")


def validate_coupling_matrix(B: np.ndarray) -> None:
    """Validate structural invariants: square, symmetric, zero diagonal, unit absolute row sums.

    Raises:
        ValueError: If invariants are violated.
    """
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"Coupling matrix must be square 2D array, got shape {B.shape}")
    if not np.allclose(B, B.T):
        raise ValueError("Coupling matrix must be symmetric: B == B.T")
    if not np.allclose(np.diag(B), 0.0):
        raise ValueError("Coupling matrix must have strictly zero diagonal: diag(B) == 0")
    if not np.allclose(B, 0.0):
        row_abs_sums = np.sum(np.abs(B), axis=1)
        if not np.allclose(row_abs_sums, 1.0):
            raise ValueError(f"Nonzero coupling matrix rows must have absolute sum 1.0, got {row_abs_sums}")


def calculate_internal_fields(x: np.ndarray, B: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Vectorized calculation of within-agent cross-belief coupling fields:

    F^x_{internal, i, alpha} = L * sum_{beta != alpha} B_{alpha, beta} * x_{i, beta}

    Since B is symmetric and diag(B) = 0:
    (x @ B)_{i, alpha} = sum_beta x_{i, beta} * B_{beta, alpha} = sum_{beta != alpha} B_{alpha, beta} * x_{i, beta}

    Args:
        x: Private belief array of shape (N, D).
        B: Belief coupling matrix of shape (D, D).
        L: Coupling strength scalar >= 0.

    Returns:
        Internal fields array of shape (N, D), dtype float64.
    """
    if B.shape[0] <= 1 or np.allclose(B, 0.0) or L == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    return float(L) * np.dot(x, B)


def calculate_isolated_internal_energy(x: np.ndarray, B: np.ndarray, L: float = 1.0) -> float:
    """Compute isolated binary internal energy:

    E_B(x) = -L * sum_{alpha < beta} B_{alpha, beta} * x_alpha * x_beta

    Args:
        x: 1D spin vector of shape (D,), values in {-1.0, +1.0}.
        B: Belief coupling matrix of shape (D, D).
        L: Coupling strength scalar.

    Returns:
        Isolated internal energy float.
    """
    D = len(x)
    energy = 0.0
    for alpha in range(D):
        for beta in range(alpha + 1, D):
            energy -= float(L) * B[alpha, beta] * x[alpha] * x[beta]
    return float(energy)


def enumerate_triad_energy_spectrum(B: np.ndarray, L: float = 1.0) -> Dict[float, int]:
    """Exhaustively enumerate all 2^3 = 8 binary configurations for D=3.

    Calculates the exact multiplicity spectrum of internal energy levels.

    Args:
        B: 3x3 coupling matrix.
        L: Coupling strength.

    Returns:
        Dictionary mapping rounded energy level to integer state count.
    """
    validate_coupling_matrix(B)
    counts: Dict[float, int] = {}
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                config = np.array([s0, s1, s2], dtype=np.float64)
                e = round(calculate_isolated_internal_energy(config, B, L=L), 6)
                counts[e] = counts.get(e, 0) + 1
    return counts
