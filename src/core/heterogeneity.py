"""Quenched heterogeneity and constrained encounter opportunity structures (Realism Physics Revision).

Implements:
1. Clipped lognormal multipliers q_i(sigma) = clip(exp(sigma * Z_i - sigma^2 / 2), 0.5, 2.0).
2. Truncated normal intrinsic biases b_{i, alpha} / J in [-0.45, 0.45] with sd = 0.15.
3. Intervention susceptibilities g_i in [0.5, 1.5].
4. QuenchedHeterogeneity dataclass and unified generator for node parameters (K_i, tau_{x,i}, nu_i, g_i, b_{i, alpha}).
5. Constrained encounter opportunity structures: 4 equal-sized groups with cross-group eligibility 0.05 and within-group eligibility 1.0.

Simulation Stage Mapping:
    - Stage 1: Core emergence and realism controls (quenched disorder, opportunity structures).
    - Stage 2: Controlled perturbations with heterogeneous response and susceptibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np


DEFAULT_MULTIPLIER_BOUNDS = (0.5, 2.0)
DEFAULT_BIAS_SD_OVER_J = 0.15
DEFAULT_BIAS_BOUNDS_OVER_J = (-0.45, 0.45)
DEFAULT_SUSCEPTIBILITY_BOUNDS = (0.5, 1.5)
DEFAULT_NUM_OPPORTUNITY_GROUPS = 4
DEFAULT_CROSS_GROUP_ELIGIBILITY = 0.05
DEFAULT_WITHIN_GROUP_ELIGIBILITY = 1.0
DEFAULT_ROW_CAP = 40


def clipped_lognormal_multiplier(
    sigma: float,
    size: Union[int, Tuple[int, ...]] = 1,
    bounds: Tuple[float, float] = DEFAULT_MULTIPLIER_BOUNDS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample independent clipped lognormal multipliers q_i(sigma).

    q_i(sigma) = clip(exp(sigma * Z_i - sigma^2 / 2), bounds[0], bounds[1]),
    where Z_i ~ Normal(0, 1).

    If sigma == 0.0, identically returns 1.0 across all elements, preserving
    homogeneous baseline physics with zero variance.

    Args:
        sigma: Standard deviation parameter sigma >= 0.0.
        size: Shape tuple or integer element count.
        bounds: Clipping interval (min_val, max_val), default (0.5, 2.0).
        rng: Optional NumPy random Generator.

    Returns:
        Array of clipped lognormal multipliers of shape `size`.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if bounds[0] <= 0.0:
        raise ValueError(f"bounds[0] must be strictly positive, got {bounds[0]}")
    if bounds[0] > bounds[1]:
        raise ValueError(f"Invalid bounds: {bounds}")

    if sigma == 0.0:
        return np.ones(size, dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal(size=size)
    raw = np.exp(sigma * z - 0.5 * (sigma * sigma))
    return np.clip(raw, bounds[0], bounds[1])


def generate_truncated_normal_biases(
    N: int,
    D: int = 1,
    J: float = 1.0,
    sigma_over_J: float = DEFAULT_BIAS_SD_OVER_J,
    bounds_over_J: Tuple[float, float] = DEFAULT_BIAS_BOUNDS_OVER_J,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample intrinsic private belief predispositions b_{i, alpha}.

    b_{i, alpha} / J is drawn from Normal(0, sigma_over_J^2) truncated by
    rejection to [bounds_over_J[0], bounds_over_J[1]].

    If sigma_over_J == 0.0, identically returns zeros, preserving the unbiased
    homogeneous baseline.

    Args:
        N: Population size (int >= 1).
        D: Belief dimensionality (int >= 1).
        J: Social coupling scale J >= 0.0.
        sigma_over_J: Standard deviation of b / J (default 0.15).
        bounds_over_J: Truncation interval for b / J (default (-0.45, 0.45)).
        rng: Optional NumPy random Generator.

    Returns:
        Array of intrinsic biases of shape (N, D), dtype float64.
    """
    if N < 1 or D < 1:
        raise ValueError(f"N and D must be >= 1, got N={N}, D={D}")
    if sigma_over_J < 0.0:
        raise ValueError(f"sigma_over_J must be non-negative, got {sigma_over_J}")
    if bounds_over_J[0] > bounds_over_J[1]:
        raise ValueError(f"Invalid bounds_over_J: {bounds_over_J}")

    if sigma_over_J == 0.0:
        return np.zeros((N, D), dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    low, high = bounds_over_J
    total_needed = N * D
    collected: list[float] = []

    # Rejection sampling
    while len(collected) < total_needed:
        batch_size = max(total_needed - len(collected), 256)
        candidates = rng.normal(0.0, sigma_over_J, size=batch_size * 2)
        valid = candidates[(candidates >= low) & (candidates <= high)]
        collected.extend(valid[: total_needed - len(collected)])

    b_over_J = np.array(collected[:total_needed], dtype=np.float64).reshape((N, D))
    return b_over_J * J


def generate_susceptibilities(
    N: int,
    bounds: Tuple[float, float] = DEFAULT_SUSCEPTIBILITY_BOUNDS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample intervention susceptibility multipliers g_i.

    Drawn uniformly from [bounds[0], bounds[1]].
    If bounds[0] == bounds[1] == 1.0, identically returns 1.0.

    Args:
        N: Population size (int >= 1).
        bounds: Interval (min_g, max_g), default (0.5, 1.5).
        rng: Optional NumPy random Generator.

    Returns:
        Array of susceptibilities of shape (N,), dtype float64.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if bounds[0] < 0.0 or bounds[0] > bounds[1]:
        raise ValueError(f"Invalid susceptibility bounds: {bounds}")

    if bounds[0] == bounds[1]:
        return np.full(N, bounds[0], dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    return rng.uniform(bounds[0], bounds[1], size=N)


@dataclass
class QuenchedHeterogeneity:
    """Container for agent-specific quenched disorder parameters."""

    K: np.ndarray             # Private-public consistency coupling K_i, shape (N,)
    tau_x: np.ndarray         # Private relaxation timescale tau_{x,i}, shape (N,)
    nu: np.ndarray            # Encounter attempt rate nu_i, shape (N,)
    g: np.ndarray             # Intervention susceptibility g_i, shape (N,)
    b: np.ndarray             # Intrinsic private bias b_{i, alpha}, shape (N, D)
    q_K: np.ndarray           # Raw multiplier for K, shape (N,)
    q_tau_x: np.ndarray       # Raw multiplier for tau_x, shape (N,)
    q_nu: np.ndarray          # Raw multiplier for nu, shape (N,)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert trait arrays to dictionary."""
        return {
            "K": self.K.copy(),
            "tau_x": self.tau_x.copy(),
            "nu": self.nu.copy(),
            "g": self.g.copy(),
            "b": self.b.copy(),
            "q_K": self.q_K.copy(),
            "q_tau_x": self.q_tau_x.copy(),
            "q_nu": self.q_nu.copy(),
        }


def generate_quenched_heterogeneity(
    N: int,
    D: int = 1,
    J: float = 1.0,
    K_nominal: float = 0.5,
    tau_x_nominal: float = 5.0,
    nu_nominal: float = 2.0,
    K_sigma: float = 0.35,
    tau_x_sigma: float = 0.5,
    nu_sigma: float = 0.5,
    bias_sigma_over_J: float = DEFAULT_BIAS_SD_OVER_J,
    bias_bounds_over_J: Tuple[float, float] = DEFAULT_BIAS_BOUNDS_OVER_J,
    g_bounds: Tuple[float, float] = DEFAULT_SUSCEPTIBILITY_BOUNDS,
    multiplier_bounds: Tuple[float, float] = DEFAULT_MULTIPLIER_BOUNDS,
    rng: Optional[np.random.Generator] = None,
) -> QuenchedHeterogeneity:
    """Generate independent quenched heterogeneity traits for N agents.

    Independent standard normal draws are used for different traits:
    - K_i = K_nominal * q_i(K_sigma)
    - tau_{x,i} = tau_x_nominal * q_i(tau_x_sigma)
    - nu_i = nu_nominal * q_i(nu_sigma)
    - g_i ~ Uniform(g_bounds[0], g_bounds[1])
    - b_{i, alpha} ~ TruncatedNormal(0, bias_sigma_over_J * J)

    Args:
        N: Population size.
        D: Belief dimensionality.
        J: Social coupling scale.
        K_nominal: Nominal private-public coupling K.
        tau_x_nominal: Nominal private relaxation timescale tau_x.
        nu_nominal: Nominal encounter rate nu.
        K_sigma: Standard deviation for K lognormal multiplier (default 0.35).
        tau_x_sigma: Standard deviation for tau_x lognormal multiplier (default 0.5).
        nu_sigma: Standard deviation for nu lognormal multiplier (default 0.5).
        bias_sigma_over_J: Standard deviation for b / J (default 0.15).
        bias_bounds_over_J: Truncation bounds for b / J (default (-0.45, 0.45)).
        g_bounds: Susceptibility bounds (default (0.5, 1.5)).
        multiplier_bounds: Multiplier clipping bounds (default (0.5, 2.0)).
        rng: Optional NumPy random Generator.

    Returns:
        QuenchedHeterogeneity instance containing all generated trait arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    q_K = clipped_lognormal_multiplier(K_sigma, size=N, bounds=multiplier_bounds, rng=rng)
    q_tau_x = clipped_lognormal_multiplier(tau_x_sigma, size=N, bounds=multiplier_bounds, rng=rng)
    q_nu = clipped_lognormal_multiplier(nu_sigma, size=N, bounds=multiplier_bounds, rng=rng)

    K = K_nominal * q_K
    tau_x = tau_x_nominal * q_tau_x
    nu = nu_nominal * q_nu

    g = generate_susceptibilities(N, bounds=g_bounds, rng=rng)
    b = generate_truncated_normal_biases(
        N=N,
        D=D,
        J=J,
        sigma_over_J=bias_sigma_over_J,
        bounds_over_J=bias_bounds_over_J,
        rng=rng,
    )

    return QuenchedHeterogeneity(
        K=K,
        tau_x=tau_x,
        nu=nu,
        g=g,
        b=b,
        q_K=q_K,
        q_tau_x=q_tau_x,
        q_nu=q_nu,
    )


def generate_opportunity_groups(
    N: int,
    num_groups: int = DEFAULT_NUM_OPPORTUNITY_GROUPS,
    between_eligibility: float = DEFAULT_CROSS_GROUP_ELIGIBILITY,
    within_eligibility: float = DEFAULT_WITHIN_GROUP_ELIGIBILITY,
    randomize: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign nodes to equal-sized opportunity groups and construct eligibility matrices.

    Creates an imposed opportunity structure:
    - 4 equal-sized opportunity groups (or specified `num_groups`).
    - Within-group eligibility = 1.0 (default).
    - Cross-group / between-group eligibility = 0.05 (default).

    Args:
        N: Population size (int >= num_groups).
        num_groups: Number of opportunity groups (default 4).
        between_eligibility: Cross-group encounter eligibility (default 0.05).
        within_eligibility: Within-group encounter eligibility (default 1.0).
        randomize: Whether to randomize group label assignments among nodes.
        rng: Optional NumPy random Generator.

    Returns:
        Tuple of (groups, E, C):
            groups: Array of shape (N,) with group labels in {0, ..., num_groups - 1}.
            E: Group interface matrix of shape (num_groups, num_groups).
            C: Full agent eligibility matrix of shape (N, N) with zero diagonal.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if N < num_groups:
        raise ValueError(f"Population size N={N} cannot be smaller than num_groups={num_groups}")
    if between_eligibility < 0.0 or within_eligibility < 0.0:
        raise ValueError("Eligibility factors must be non-negative")

    if rng is None:
        rng = np.random.default_rng()

    # Equal-sized partition
    base_size = N // num_groups
    rem = N % num_groups
    groups = np.empty(N, dtype=np.int64)

    idx = 0
    for g in range(num_groups):
        sz = base_size + (1 if g < rem else 0)
        groups[idx : idx + sz] = g
        idx += sz

    if randomize:
        rng.shuffle(groups)

    # Population-level interface matrix E
    E = np.full((num_groups, num_groups), float(between_eligibility), dtype=np.float64)
    np.fill_diagonal(E, float(within_eligibility))

    # Full agent eligibility matrix C_ij = E_{g_i, g_j} with C_ii = 0
    C = E[np.ix_(groups, groups)].copy()
    np.fill_diagonal(C, 0.0)

    return groups, E, C


def calculate_cross_group_exploration_probability(
    N: int,
    num_groups: int = DEFAULT_NUM_OPPORTUNITY_GROUPS,
    between_eligibility: float = DEFAULT_CROSS_GROUP_ELIGIBILITY,
    within_eligibility: float = DEFAULT_WITHIN_GROUP_ELIGIBILITY,
) -> float:
    """Calculate theoretical cross-group exploration probability under block eligibility.

    For an agent i in an opportunity group g of size n_g:
    - Within-group nonself agents: n_g - 1 with weight within_eligibility.
    - Outside-group agents: N - n_g with weight between_eligibility.
    - Cross probability for agent i = cross_weight / (within_weight + cross_weight).

    Returns the exact population-averaged cross-group exploration probability across
    all N agents for any balanced partition.
    For N=1000, num_groups=4, between=0.05, within=1.0: 37.5 / 286.5 ~= 0.130890.

    Args:
        N: Total population size (int >= num_groups).
        num_groups: Number of opportunity groups (int >= 1).
        between_eligibility: Cross-group eligibility factor.
        within_eligibility: Within-group eligibility factor.

    Returns:
        Theoretical probability that an exploration draw targets an agent outside i's group.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if N < num_groups:
        raise ValueError(f"Population size N={N} cannot be smaller than num_groups={num_groups}")
    if between_eligibility < 0.0 or within_eligibility < 0.0:
        raise ValueError("Eligibility factors must be non-negative")

    base_size = N // num_groups
    rem = N % num_groups
    group_sizes = [base_size + (1 if g < rem else 0) for g in range(num_groups)]

    total_prob = 0.0
    for n_g in group_sizes:
        within_count = n_g - 1
        outside_count = N - n_g
        within_weight = float(within_count) * float(within_eligibility)
        cross_weight = float(outside_count) * float(between_eligibility)
        total_weight = within_weight + cross_weight
        p_g = (cross_weight / total_weight) if total_weight > 0.0 else 0.0
        total_prob += n_g * p_g

    return total_prob / float(N)
