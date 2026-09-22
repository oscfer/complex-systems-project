"""Matched zero-control and all-node uniform-field reference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.control.contracts import ControlProblem, ObservationMode, RolloutResult, TargetType
from src.control.controllers import OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import CompletePublicObservationModel
from src.experiment.baselines.ensembles import ReferenceOrigin


class BaselineCondition(str, Enum):
    UNCONTROLLED = "uncontrolled"
    UNIFORM_FIELD = "uniform_field"


class BaselineDynamics(str, Enum):
    ADAPTIVE = "adaptive"
    FROZEN = "frozen"


class ReliabilityEstimand(str, Enum):
    POPULATION = "population"
    CONDITIONAL_ORIGIN = "conditional_origin"


@dataclass(frozen=True)
class BaselineExperimentConfig:
    """Fixed reference settings; optimization/search is intentionally absent."""

    target_type: TargetType = TargetType.REVERSAL
    uniform_amplitude: float = 1.0
    pulse_duration: float = 5.0
    release_deadline: float = 20.0
    holding_duration: float = 50.0
    horizon: float = 270.0
    reliability_target: float = 0.8
    reliability_alpha: float = 0.05
    master_seed: int = 1592026
    J: float = 1.0
    temperature: float = 0.3
    delta: float = 0.1
    exploration: float = 0.1
    closure: float = 0.3
    sample_interval: float = 5.0
    dynamics: Tuple[BaselineDynamics, ...] = (
        BaselineDynamics.ADAPTIVE,
        BaselineDynamics.FROZEN,
    )
    reliability_estimand: ReliabilityEstimand = ReliabilityEstimand.POPULATION

    def __post_init__(self) -> None:
        if self.uniform_amplitude <= 0.0:
            raise ValueError("uniform_amplitude must be positive")
        if self.pulse_duration <= 0.0:
            raise ValueError("pulse_duration must be positive")
        if self.release_deadline < self.pulse_duration:
            raise ValueError("release_deadline must cover the complete reference pulse")
        if self.holding_duration <= 0.0 or self.horizon <= self.release_deadline:
            raise ValueError("invalid holding duration or horizon")
        if not 0.0 < self.reliability_target < 1.0:
            raise ValueError("reliability_target must be in (0, 1)")
        if not 0.0 < self.reliability_alpha < 1.0:
            raise ValueError("reliability_alpha must be in (0, 1)")
        if self.target_type != TargetType.REVERSAL:
            raise ValueError("The uniform-field reference is defined for reversal targets")
        if not self.dynamics or len(set(self.dynamics)) != len(self.dynamics):
            raise ValueError("dynamics must contain unique adaptive/frozen branches")
        if any(not isinstance(value, BaselineDynamics) for value in self.dynamics):
            raise ValueError("dynamics entries must be BaselineDynamics values")
        if not isinstance(self.reliability_estimand, ReliabilityEstimand):
            raise ValueError("reliability_estimand must be a ReliabilityEstimand")


@dataclass(frozen=True)
class ReliabilitySummary:
    condition: BaselineCondition
    dynamics: BaselineDynamics
    estimand: ReliabilityEstimand
    total_rollouts: int
    valid_rollouts: int
    eligible_rollouts: int
    ineligible_rollouts: int
    invalid_rollouts: int
    successes: int
    success_probability: float
    wilson_low: float
    wilson_high: float
    exact_one_sided_lower: float
    reliability_target: float
    meets_point_target: bool
    reliability_certified: bool
    certification_available: bool
    certification_reason: str
    independent_units: int
    mean_restricted_time: float


@dataclass(frozen=True)
class BaselineComparison:
    results: Tuple[RolloutResult, ...]
    summaries: Dict[Tuple[BaselineCondition, BaselineDynamics], ReliabilitySummary]
    matched_pairs: int


def _binomial_upper_tail(success_threshold: int, trials: int, p: float) -> float:
    if success_threshold <= 0:
        return 1.0
    if success_threshold > trials:
        return 0.0
    q = 1.0 - p
    return float(sum(
        math.comb(trials, j) * (p ** j) * (q ** (trials - j))
        for j in range(success_threshold, trials + 1)
    ))


def exact_binomial_lower_bound(successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound without an optional SciPy dependency."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if trials == 0 or successes == 0:
        return 0.0
    low, high = 0.0, successes / trials
    for _ in range(80):
        mid = (low + high) / 2.0
        if _binomial_upper_tail(successes, trials, mid) < alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_reliability(
    condition: BaselineCondition,
    dynamics: BaselineDynamics,
    results: Sequence[RolloutResult],
    reliability_target: float = 0.8,
    alpha: float = 0.05,
    estimand: ReliabilityEstimand = ReliabilityEstimand.POPULATION,
) -> ReliabilitySummary:
    valid = [result for result in results if result.is_valid]
    eligible = [result for result in valid if result.is_eligible]
    successes = sum(bool(result.success) for result in eligible)
    trials = len(eligible)
    probability = successes / trials if trials else 0.0
    unique_origins = {result.origin_id for result in eligible}
    origin_counts = {
        origin_id: sum(result.origin_id == origin_id for result in eligible)
        for origin_id in unique_origins
    }
    invalid_count = len(results) - len(valid)
    if invalid_count:
        certification_available = False
        certification_reason = "invalid_or_incomplete_runs"
    elif estimand == ReliabilityEstimand.POPULATION and any(
        count != 1 for count in origin_counts.values()
    ):
        certification_available = False
        certification_reason = "nested_futures_require_origin_clustered_inference"
    elif estimand == ReliabilityEstimand.CONDITIONAL_ORIGIN and len(unique_origins) != 1:
        certification_available = False
        certification_reason = "conditional_estimand_requires_exactly_one_origin"
    else:
        certification_available = trials > 0
        certification_reason = "available" if trials > 0 else "no_eligible_runs"

    low, high = (
        wilson_interval(successes, trials)
        if certification_available
        else (math.nan, math.nan)
    )
    exact_low = (
        exact_binomial_lower_bound(successes, trials, alpha)
        if certification_available
        else math.nan
    )
    mean_time = float(np.mean([result.restricted_time for result in eligible])) if eligible else math.nan
    return ReliabilitySummary(
        condition=condition,
        dynamics=dynamics,
        estimand=estimand,
        total_rollouts=len(results),
        valid_rollouts=len(valid),
        eligible_rollouts=trials,
        ineligible_rollouts=len(valid) - trials,
        invalid_rollouts=invalid_count,
        successes=successes,
        success_probability=probability,
        wilson_low=low,
        wilson_high=high,
        exact_one_sided_lower=exact_low,
        reliability_target=reliability_target,
        meets_point_target=probability >= reliability_target,
        reliability_certified=(certification_available and exact_low >= reliability_target),
        certification_available=certification_available,
        certification_reason=certification_reason,
        independent_units=(len(unique_origins) if estimand == ReliabilityEstimand.POPULATION else trials),
        mean_restricted_time=mean_time,
    )


def _stable_seed(master_seed: int, origin_id: str, future_id: int, role: str) -> int:
    payload = f"{master_seed}|{origin_id}|{future_id}|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _make_problem(origin: ReferenceOrigin, config: BaselineExperimentConfig) -> ControlProblem:
    population = origin.state.N
    exposure = population * config.uniform_amplitude * config.pulse_duration
    return ControlProblem(
        problem_id=f"m5_{origin.origin_id}_{config.target_type.value}",
        target_type=config.target_type,
        target_params={},
        N=population,
        D=origin.state.D,
        k=population,
        exposure_cap=exposure,
        amplitude_cap=config.uniform_amplitude,
        release_deadline=config.release_deadline,
        holding_duration=config.holding_duration,
        horizon=config.horizon,
        accessible_roster=tuple(range(population)),
        observation_mode=ObservationMode.COMPLETE_PUBLIC,
        proposal_seed=config.master_seed,
        search_seed=config.master_seed,
        config_version="m5-reference-baselines-1.0",
    )


def run_reference_baselines(
    origins: Iterable[ReferenceOrigin],
    futures_per_origin: int,
    config: BaselineExperimentConfig | None = None,
) -> BaselineComparison:
    """Run matched uncontrolled and budget-nonbinding uniform-field branches.

    Both branches reuse the same simulator seed for each origin/future pair.
    The uniform reference still passes through the ledger with ``k=N`` and
    exactly sufficient exposure, so no accounting contract is bypassed.
    """

    if futures_per_origin < 1:
        raise ValueError("futures_per_origin must be >= 1")
    config = config or BaselineExperimentConfig()
    all_results: List[RolloutResult] = []
    pair_count = 0

    for origin in origins:
        problem = _make_problem(origin, config)
        target_sign = -1.0 if float(np.mean(origin.state.y[:, 0])) > 0.0 else 1.0
        signs = {node: target_sign for node in range(origin.state.N)}
        for future_id in range(futures_per_origin):
            simulator_seed = _stable_seed(config.master_seed, origin.origin_id, future_id, "physical")
            for dynamics in config.dynamics:
                for condition in (BaselineCondition.UNCONTROLLED, BaselineCondition.UNIFORM_FIELD):
                    controller = (
                        ZeroController()
                        if condition == BaselineCondition.UNCONTROLLED
                        else OpenLoopPulseController(
                            target_nodes=range(origin.state.N),
                            nominal_amplitude=config.uniform_amplitude,
                            duration=config.pulse_duration,
                            signs=signs,
                        )
                    )
                    policy_id = f"{condition.value}:{dynamics.value}"
                    evaluator = TargetEvaluator.create_for_state(
                        target_type=config.target_type,
                        initial_state=origin.state,
                        holding_duration=config.holding_duration,
                        horizon=config.horizon,
                        release_deadline=config.release_deadline,
                        problem_id=problem.problem_id,
                        policy_id=policy_id,
                        origin_id=origin.origin_id,
                        future_id=future_id,
                    )
                    result, _, _ = run_controlled_simulation(
                        initial_state=origin.branch(),
                        problem=problem,
                        controller=controller,
                        observation_model=CompletePublicObservationModel(
                            N=origin.state.N,
                            accessible_roster=tuple(range(origin.state.N)),
                            delivery_delay=0.0,
                        ),
                        target_evaluator=evaluator,
                        J=config.J,
                        T=config.temperature,
                        delta=config.delta,
                        epsilon=config.exploration,
                        closure=config.closure,
                        sample_interval=config.sample_interval,
                        adaptive_memory=(dynamics == BaselineDynamics.ADAPTIVE),
                        between_eligibility=origin.between_eligibility,
                        within_eligibility=origin.within_eligibility,
                        sim_rng=np.random.default_rng(simulator_seed),
                        controller_rng=np.random.default_rng(
                            _stable_seed(
                                config.master_seed,
                                origin.origin_id,
                                future_id,
                                policy_id,
                            )
                        ),
                    )
                    result.summaries.update({
                        "condition": condition.value,
                        "dynamics": dynamics.value,
                        "ensemble_id": origin.ensemble_id,
                        "matched_physical_seed": simulator_seed,
                        "fixed_horizon": config.horizon,
                        "between_eligibility": origin.between_eligibility,
                        "within_eligibility": origin.within_eligibility,
                    })
                    all_results.append(result)
            pair_count += 1

    summaries: Dict[Tuple[BaselineCondition, BaselineDynamics], ReliabilitySummary] = {}
    for dynamics in config.dynamics:
        for condition in (BaselineCondition.UNCONTROLLED, BaselineCondition.UNIFORM_FIELD):
            policy_id = f"{condition.value}:{dynamics.value}"
            selected = [result for result in all_results if result.policy_id == policy_id]
            summaries[(condition, dynamics)] = summarize_reliability(
                condition,
                dynamics,
                selected,
                reliability_target=config.reliability_target,
                alpha=config.reliability_alpha,
                estimand=config.reliability_estimand,
            )
    return BaselineComparison(tuple(all_results), summaries, pair_count)
