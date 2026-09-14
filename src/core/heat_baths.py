"""Heat bath update algorithms for discrete and continuous belief states (Stages 0 & 1).

Provides:
1. Binary Glauber heat bath with exact overflow-proof conditional probabilities.
2. Bounded continuous [-1, 1] inverse-CDF heat bath using a 3-regime numerically stable
   formulation (Taylor expansion for weak fields, factored log-sum-exp for positive fields,
   and exact physical parity symmetry for negative fields).
3. Weak-field susceptibility temperature calibration: T_cont = T_bin / 3.

Simulation Stage Mapping:
    - Stage 0: Stochastic acceptance sampling throughput benchmarking.
    - Stage 1: Core state updates for discrete and continuous belief dimensions.
"""

from __future__ import annotations

import math
from typing import Optional, Union
import numpy as np


def binary_glauber_prob_plus(kappa: float) -> float:
    """Exact, overflow-proof conditional probability P(s = +1 | kappa = F / T) (Stage 1).

    P(s = +1 | kappa) = e^kappa / (2 cosh kappa) = 1 / (1 + e^{-2 kappa}).
    Evaluated without overflow for all kappa in (-inf, +inf).

    Args:
        kappa: Effective coupling ratio F / T.

    Returns:
        Probability in [0.0, 1.0].
    """
    if kappa >= 0.0:
        return 1.0 / (1.0 + math.exp(-2.0 * kappa))
    else:
        e2k = math.exp(2.0 * kappa)
        return e2k / (1.0 + e2k)


def sample_binary_glauber_scalar(kappa: float, rng: np.random.Generator) -> float:
    """Sample a binary spin in {-1.0, +1.0} given effective field kappa = F / T.

    Args:
        kappa: Field-to-temperature ratio F / T.
        rng: NumPy random Generator.

    Returns:
        +1.0 or -1.0.
    """
    p_plus = binary_glauber_prob_plus(kappa)
    return 1.0 if rng.random() < p_plus else -1.0


def sample_binary_glauber_vector(kappa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized binary Glauber sampling across an array of kappa values.

    Args:
        kappa: Array of shape (N, D) or arbitrary shape.
        rng: NumPy random Generator.

    Returns:
        Array of spins in {-1.0, +1.0} matching kappa.shape.
    """
    # Numerically stable vectorized logistic probability
    pos_mask = kappa >= 0.0
    p_plus = np.empty_like(kappa, dtype=np.float64)

    # For kappa >= 0: 1 / (1 + exp(-2 kappa))
    p_plus[pos_mask] = 1.0 / (1.0 + np.exp(-2.0 * kappa[pos_mask]))

    # For kappa < 0: exp(2 kappa) / (1 + exp(2 kappa))
    e2k = np.exp(2.0 * kappa[~pos_mask])
    p_plus[~pos_mask] = e2k / (1.0 + e2k)

    draws = rng.random(size=kappa.shape)
    return np.where(draws < p_plus, 1.0, -1.0)


def sample_continuous_inverse_cdf_scalar(kappa: float, u: float) -> float:
    """Sample bounded continuous state z' in [-1, 1] via 3-regime stable inverse-CDF.

    The continuous conditional density is f(s | F) = e^{kappa s} / Z(kappa) on s in [-1, 1]
    with flat reference measure ds.

    Inverse-CDF formula:
        z' = (1 / kappa) * ln[ (1 - U) e^{-kappa} + U e^kappa ]

    Regimes:
        1. |kappa| <= 1e-5: 2nd-order Taylor expansion around kappa = 0:
           z' = (2U - 1) + 2U(1 - U) kappa - (4/3) U(1 - U)(2U - 1) kappa^2
        2. kappa > 1e-5: Factored form:
           z' = 1 + (1 / kappa) * ln[ U + (1 - U) e^{-2 kappa} ]
        3. kappa < -1e-5: Physical parity symmetry:
           z'(-kappa, U) = -z'(kappa, 1 - U)
           z' = - [ 1 + (1 / |kappa|) * ln[ (1 - U) + U e^{-2 |kappa|} ] ]

    Args:
        kappa: Effective coupling F / T.
        u: Uniform random draw in [0, 1].

    Returns:
        Sampled state value strictly clamped to [-1.0, 1.0].
    """
    if u <= 0.0:
        return -1.0
    if u >= 1.0:
        return 1.0

    if abs(kappa) <= 1e-5:
        z0 = 2.0 * u - 1.0
        z1 = 2.0 * u * (1.0 - u)
        z2 = -(4.0 / 3.0) * u * (1.0 - u) * (2.0 * u - 1.0)
        z = z0 + z1 * kappa + z2 * (kappa * kappa)
    elif kappa > 0.0:
        e2k = math.exp(-2.0 * kappa) if kappa < 350.0 else 0.0
        arg = u + (1.0 - u) * e2k
        z = 1.0 + math.log(max(arg, 1e-300)) / kappa
    else:
        pos_k = -kappa
        u_sym = 1.0 - u
        e2k = math.exp(-2.0 * pos_k) if pos_k < 350.0 else 0.0
        arg = u_sym + (1.0 - u_sym) * e2k
        z = -(1.0 + math.log(max(arg, 1e-300)) / pos_k)

    if z < -1.0:
        return -1.0
    elif z > 1.0:
        return 1.0
    return z


def sample_continuous_inverse_cdf_vector(kappa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized sampling of bounded continuous states in [-1, 1].

    Args:
        kappa: Array of effective couplings F / T.
        rng: NumPy random Generator.

    Returns:
        Array of sampled continuous values in [-1.0, 1.0] matching kappa.shape.
    """
    flat_kappa = kappa.ravel()
    u_draws = rng.uniform(0.0, 1.0, size=flat_kappa.shape)
    out = np.empty_like(flat_kappa, dtype=np.float64)

    for idx in range(len(flat_kappa)):
        out[idx] = sample_continuous_inverse_cdf_scalar(flat_kappa[idx], u_draws[idx])

    return out.reshape(kappa.shape)


def calibrate_continuous_temperature(T_bin: float) -> float:
    """Calibrate continuous temperature to match binary weak-field susceptibility:

    chi_cont = 1 / (3 T_cont) vs chi_bin = 1 / T_bin => T_cont = T_bin / 3.

    Args:
        T_bin: Binary temperature > 0.

    Returns:
        Continuous temperature T_cont = T_bin / 3.
    """
    if T_bin <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {T_bin}")
    return T_bin / 3.0
