"""Q-learning expressed-state updates and softmax action selection.

Replaces instantaneous heat baths for expressed states y_{i, alpha} in {-1, +1}
with experience-accumulating Q-tables Q_{i, alpha}(s).
Implements:
1. Two-action Q-table mapping: idx 0 <-> s = -1; idx 1 <-> s = +1.
2. Softmax action selection at temperature T_Q via overflow-proof logistic reduction.
3. Pre-action immediate reward calculation: R = s * F^y_{i, alpha}.
4. Chosen-action TD update: Q(s*) <- Q(s*) + eta * [R - Q(s*)].
5. Strict counterfactual isolation: unchosen action value Q(-s*) is completely untouched.

Simulation Stage Mapping:
    - Stage 3: Adaptive learning campaigns comparing experience accumulation (eta in {0.01, 0.1, 0.5})
      against baseline memory-only emergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class QLearningParams:
    """Parameters for Q-learning expressed-state decision maker."""

    eta: float = 0.1     # Learning rate in (0, 1]
    T_Q: float = 1.0     # Softmax temperature > 0


def action_to_index(s: int) -> int:
    """Map action s in {-1, +1} to array index {0, 1}."""
    return (s + 1) // 2


def index_to_action(idx: int) -> int:
    """Map array index in {0, 1} to action {-1, +1}."""
    return 2 * idx - 1


def init_q_table(N: int, D: int) -> np.ndarray:
    """Initialize Q-value table of shape (N, D, 2) with zeros."""
    return np.zeros((N, D, 2), dtype=np.float64)


def softmax_probabilities(Q_pair: np.ndarray, T_Q: float) -> Tuple[float, float]:
    """Compute softmax probabilities (P(s = -1), P(s = +1)) via stable logistic difference.

    P(+1) = 1 / (1 + exp(-(Q[+1] - Q[-1]) / T_Q)).

    Args:
        Q_pair: Array of length 2 containing [Q(s=-1), Q(s=+1)].
        T_Q: Softmax temperature > 0.

    Returns:
        Tuple of (P(-1), P(+1)).
    """
    if T_Q <= 0.0:
        raise ValueError(f"Softmax temperature T_Q must be strictly positive, got {T_Q}")

    delta = (float(Q_pair[1]) - float(Q_pair[0])) / float(T_Q)
    if delta >= 350.0:
        p_plus = 1.0
    elif delta <= -350.0:
        p_plus = 0.0
    else:
        p_plus = 1.0 / (1.0 + np.exp(-delta))

    return (1.0 - p_plus, p_plus)


def select_action(Q_pair: np.ndarray, T_Q: float, rng: np.random.Generator) -> int:
    """Sample action s in {-1, +1} from softmax distribution on Q_pair.

    Args:
        Q_pair: Array of shape (2,) containing [Q(-1), Q(+1)].
        T_Q: Temperature > 0.
        rng: NumPy random Generator.

    Returns:
        +1 or -1.
    """
    p_minus, p_plus = softmax_probabilities(Q_pair, T_Q)
    return 1 if rng.random() < p_plus else -1


def compute_immediate_reward(action: int, field: float) -> float:
    """Compute immediate reward: R = s * F^y.

    Evaluated using pre-action private state and neighbors' current expressed states.
    """
    return float(action * field)


def update_chosen_q_value(
    Q_pair: np.ndarray,
    chosen_action: int,
    reward: float,
    eta: float,
) -> None:
    """Update chosen action value in-place via TD rule.

    Q(s*) <- Q(s*) + eta * [reward - Q(s*)].

    Crucial invariant: unchosen action Q(-s*) is strictly untouched.
    """
    idx = action_to_index(chosen_action)
    Q_pair[idx] += float(eta) * (float(reward) - float(Q_pair[idx]))


def step_q_learning(
    i: int,
    alpha: int,
    Q_table: np.ndarray,
    y: np.ndarray,
    field: float,
    params: QLearningParams,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """Execute single Q-learning update step for agent i and component alpha:

    1. Selects action s* from softmax[Q_{i, alpha} / T_Q].
    2. Computes reward R = s* * field.
    3. Updates chosen action value Q_{i, alpha}(s*) using learning rate eta.
    4. Sets y[i, alpha] = s*.

    Args:
        i: Agent index.
        alpha: Component index.
        Q_table: Q-value array of shape (N, D, 2).
        y: Expressed belief state array of shape (N, D).
        field: Current pre-action public local field F^y_{i, alpha}.
        params: QLearningParams.
        rng: NumPy random Generator.

    Returns:
        Tuple of (chosen_action: int, reward: float).
    """
    Q_pair = Q_table[i, alpha]
    s_star = select_action(Q_pair, params.T_Q, rng)
    reward = compute_immediate_reward(s_star, field)
    update_chosen_q_value(Q_pair, s_star, reward, params.eta)
    y[i, alpha] = float(s_star)
    return s_star, reward
