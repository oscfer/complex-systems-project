"""Tiny frozen binary generator benchmark specified for validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FiniteReferencePolicy:
    nodes: Tuple[int, ...]
    amplitude: float
    duration: float = 1.0


@dataclass(frozen=True)
class FiniteReferenceResult:
    policy: FiniteReferencePolicy
    survival_probability: float
    exposure: float


def enumerate_binary_states(N: int) -> np.ndarray:
    if N < 1:
        raise ValueError("N must be positive")
    values = np.empty((1 << N, N), dtype=np.float64)
    for index in range(1 << N):
        values[index] = [1.0 if (index >> i) & 1 else -1.0 for i in range(N)]
    return values


def build_binary_flip_generator(
    attention: np.ndarray,
    J: float = 1.0,
    temperature: float = 0.7,
    tau_public: float = 1.0,
    field: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ordered binary states and the heat-bath flip generator."""

    matrix = np.asarray(attention, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention must be square")
    population = matrix.shape[0]
    if temperature <= 0.0 or tau_public <= 0.0:
        raise ValueError("temperature and tau_public must be positive")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("attention diagonal must be zero")
    control = (
        np.zeros(population, dtype=np.float64)
        if field is None
        else np.asarray(field, dtype=np.float64)
    )
    if control.shape != (population,) or np.any(~np.isfinite(control)):
        raise ValueError("field must be a finite vector of length N")

    states = enumerate_binary_states(population)
    generator = np.zeros((len(states), len(states)), dtype=np.float64)
    for state_index, state in enumerate(states):
        for node in range(population):
            local_field = J * float(matrix[node] @ state) + float(control[node])
            rate = (
                1.0 - state[node] * math.tanh(local_field / temperature)
            ) / (2.0 * tau_public)
            generator[state_index, state_index ^ (1 << node)] = rate
        generator[state_index, state_index] = -float(np.sum(generator[state_index]))
    return states, generator


def _uniformized_row_action(
    initial: np.ndarray,
    generator: np.ndarray,
    duration: float,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Compute p exp(Qt) by Poisson uniformization for small dense generators."""

    distribution = np.asarray(initial, dtype=np.float64)
    matrix = np.asarray(generator, dtype=np.float64)
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    if duration == 0.0:
        return distribution.copy()
    rate = float(np.max(-np.diag(matrix)))
    if rate <= 0.0:
        return distribution.copy()
    transition = np.eye(matrix.shape[0]) + matrix / rate
    mean = rate * duration
    weight = math.exp(-mean)
    term = distribution.copy()
    result = weight * term
    cumulative = weight
    order = 0
    while 1.0 - cumulative > tolerance:
        order += 1
        term = term @ transition
        weight *= mean / order
        result += weight * term
        cumulative += weight
        if order > 100000:
            raise RuntimeError("uniformization failed to converge")
    return result


def fixed_release_survival_probability(
    attention: np.ndarray,
    policy: FiniteReferencePolicy,
    initial_state: Sequence[float] | None = None,
    target_mean_min: float = 0.5,
    hold_time: float = 0.5,
    J: float = 1.0,
    temperature: float = 0.7,
    tau_public: float = 1.0,
) -> float:
    """Probability of target occupancy at release and no target exit during hold."""

    matrix = np.asarray(attention, dtype=np.float64)
    population = matrix.shape[0]
    starting_state = (
        np.full(population, -1.0)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64)
    )
    if starting_state.shape != (population,) or np.any(~np.isin(starting_state, [-1.0, 1.0])):
        raise ValueError("initial_state must contain N binary spins")
    if any(node < 0 or node >= population for node in policy.nodes):
        raise ValueError("policy node out of bounds")

    field = np.zeros(population, dtype=np.float64)
    field[list(policy.nodes)] = float(policy.amplitude)
    states, controlled_generator = build_binary_flip_generator(
        matrix, J, temperature, tau_public, field
    )
    _, zero_generator = build_binary_flip_generator(
        matrix, J, temperature, tau_public, np.zeros(population)
    )
    initial_index = sum(
        (1 << node) for node, value in enumerate(starting_state) if value > 0
    )
    initial_distribution = np.zeros(len(states), dtype=np.float64)
    initial_distribution[initial_index] = 1.0
    at_release = _uniformized_row_action(
        initial_distribution, controlled_generator, policy.duration
    )

    target_indices = np.flatnonzero(
        np.mean(states, axis=1) >= target_mean_min - 1e-12
    )
    if len(target_indices) == 0:
        return 0.0
    killed_generator = zero_generator[np.ix_(target_indices, target_indices)]
    survived = _uniformized_row_action(
        at_release[target_indices], killed_generator, hold_time
    )
    return float(np.clip(np.sum(survived), 0.0, 1.0))


def monte_carlo_fixed_release_probability(
    attention: np.ndarray,
    policy: FiniteReferencePolicy,
    trials: int,
    seed: int = 37015,
    initial_state: Sequence[float] | None = None,
    target_mean_min: float = 0.5,
    hold_time: float = 0.5,
    J: float = 1.0,
    temperature: float = 0.7,
    tau_public: float = 1.0,
) -> float:
    """Independent attempt-clock Monte Carlo check of the finite reference."""

    if trials < 1:
        raise ValueError("trials must be positive")
    matrix = np.asarray(attention, dtype=np.float64)
    population = matrix.shape[0]
    if matrix.shape != (population, population):
        raise ValueError("attention must be square")
    starting_state = (
        np.full(population, -1.0)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64)
    )
    if starting_state.shape != (population,) or np.any(~np.isin(starting_state, [-1.0, 1.0])):
        raise ValueError("initial_state must contain N binary spins")
    field = np.zeros(population, dtype=np.float64)
    field[list(policy.nodes)] = policy.amplitude
    rng = np.random.default_rng(seed)
    successful = 0

    for _ in range(trials):
        state = starting_state.copy()
        survived = True
        for duration, applied, must_stay in (
            (policy.duration, field, False),
            (hold_time, np.zeros(population), True),
        ):
            if must_stay and float(np.mean(state)) < target_mean_min:
                survived = False
                break
            time = 0.0
            while True:
                time += float(rng.exponential(tau_public / population))
                if time >= duration:
                    break
                node = int(rng.integers(population))
                local_field = J * float(matrix[node] @ state) + float(applied[node])
                probability_plus = (1.0 + math.tanh(local_field / temperature)) / 2.0
                state[node] = 1.0 if rng.random() < probability_plus else -1.0
                if must_stay and float(np.mean(state)) < target_mean_min:
                    survived = False
                    break
            if not survived:
                break
        successful += int(survived)
    return successful / trials


def enumerate_reference_catalog(
    attention: np.ndarray,
    amplitudes: Iterable[float] = (0.25, 1.0, 4.0),
    max_controlled_nodes: int = 2,
    exposure_cap: float = 2.0,
    duration: float = 1.0,
    target_mean_min: float = 0.5,
    hold_time: float = 0.5,
) -> Tuple[FiniteReferenceResult, ...]:
    """Exhaustively score the finite catalog, deduplicating the empty policy."""

    population = int(np.asarray(attention).shape[0])
    if max_controlled_nodes < 0 or max_controlled_nodes > population:
        raise ValueError("invalid max_controlled_nodes")
    policies: List[FiniteReferencePolicy] = [FiniteReferencePolicy((), 0.0, duration)]
    seen = {((), 0.0, float(duration))}
    for size in range(1, max_controlled_nodes + 1):
        for nodes in combinations(range(population), size):
            for nominal_amplitude in amplitudes:
                nominal = float(nominal_amplitude)
                if nominal <= 0.0 or duration <= 0.0:
                    continue
                effective = min(nominal, exposure_cap / (size * duration))
                key = (tuple(nodes), effective, float(duration))
                if effective > 0.0 and key not in seen:
                    seen.add(key)
                    policies.append(FiniteReferencePolicy(tuple(nodes), effective, duration))
    results = [
        FiniteReferenceResult(
            policy=policy,
            survival_probability=fixed_release_survival_probability(
                attention,
                policy,
                target_mean_min=target_mean_min,
                hold_time=hold_time,
            ),
            exposure=len(policy.nodes) * abs(policy.amplitude) * policy.duration,
        )
        for policy in policies
    ]
    results.sort(
        key=lambda result: (
            -result.survival_probability,
            result.exposure,
            result.policy.nodes,
        )
    )
    return tuple(results)
