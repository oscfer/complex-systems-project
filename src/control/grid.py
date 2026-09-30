"""Bounded exploratory diagnostic response grid and failure trajectory analytics.

Provides:
- ExploratoryGridConfig & GridAxis: Explicitly declared axes, values, seeds, policies,
  and compute budget caps before execution.
- Dedicated exploratory origins: Disjoint from confirmation evaluation origins to prevent
  data leakage or tuning bias.
- Bounded grid execution: Strict enforcement of compute caps, clearly marking any unevaluated
  cells without fabricating interpolated data.
- Descriptive trajectory patterns from recorded aggregate states and actions.
  These patterns do not establish local response or network propagation causes.
- Raw per-cell export: Saves comprehensive data for reproducible reloading and 2D response mapping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.control.contracts import Action, ControlProblem, RolloutResult, TargetType
from src.control.controllers import ObservableSelectorPolicy
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetDefinition, TargetEvaluator
from src.control.recording import RecorderConfig, RolloutRecorder
from src.core.state import State, StateMode
from src.experiment.baselines.reference import wilson_interval
from src.execution.runner import NumpyEncoder, _atomic_write_json, _canonical_json_bytes, _json_safe


@dataclass(frozen=True)
class GridAxis:
    """Declared parameter axis for the 2D exploratory response grid."""
    name: str              # e.g., "exposure_cap" or "node_fraction"
    label: str             # e.g., "Exposure Cap B" or "Candidate Node Fraction k/N"
    values: Tuple[float, ...]
    parameter_target: str  # "problem" (alters ControlProblem) or "policy" (alters controller)
    description: str = ""


@dataclass
class GridCellResult:
    """Measured result for a single cell in the exploratory response grid."""
    coord_x: float
    coord_y: float
    coord_idx: Tuple[int, int]
    is_evaluated: bool
    status: str            # "evaluated" or "unevaluated_budget_cap"
    n_rollouts: int = 0
    n_success: int = 0
    n_eligible: int = 0
    success_probability: Optional[float] = None
    success_ci: Optional[Tuple[float, float]] = None
    mean_restricted_time: Optional[float] = None
    peak_target_alignment: Optional[float] = None  # Legacy public-only metric.
    peak_joint_alignment: Optional[float] = None
    transformed_state_retained_to_horizon: Optional[float] = None
    post_release_target_occupancy: Optional[float] = None
    post_release_retention: Optional[float] = None
    failure_breakdown: Dict[str, int] = field(default_factory=lambda: {
        "weak_joint_shift": 0,
        "partial_joint_shift": 0,
        "target_not_sustained": 0,
        "budget_limited": 0,
        "unclassified": 0,
        "success": 0,
    })
    rollout_summaries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coord_x": float(self.coord_x),
            "coord_y": float(self.coord_y),
            "coord_idx": list(self.coord_idx),
            "is_evaluated": bool(self.is_evaluated),
            "status": self.status,
            "n_rollouts": int(self.n_rollouts),
            "n_success": int(self.n_success),
            "n_eligible": int(self.n_eligible),
            "success_probability": self.success_probability,
            "success_ci": list(self.success_ci) if self.success_ci is not None else None,
            "mean_restricted_time": self.mean_restricted_time,
            "peak_target_alignment": self.peak_target_alignment,
            "peak_joint_alignment": self.peak_joint_alignment,
            "transformed_state_retained_to_horizon": self.transformed_state_retained_to_horizon,
            "post_release_target_occupancy": self.post_release_target_occupancy,
            "post_release_retention": self.post_release_retention,
            "failure_breakdown": dict(self.failure_breakdown),
            "rollout_summaries": self.rollout_summaries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GridCellResult:
        ci = tuple(d["success_ci"]) if d.get("success_ci") is not None else None
        return cls(
            coord_x=float(d["coord_x"]),
            coord_y=float(d["coord_y"]),
            coord_idx=(int(d["coord_idx"][0]), int(d["coord_idx"][1])),
            is_evaluated=bool(d["is_evaluated"]),
            status=str(d["status"]),
            n_rollouts=int(d.get("n_rollouts", 0)),
            n_success=int(d.get("n_success", 0)),
            n_eligible=int(d.get("n_eligible", d.get("n_rollouts", 0))),
            success_probability=float(d["success_probability"]) if d.get("success_probability") is not None else None,
            success_ci=ci,
            mean_restricted_time=float(d["mean_restricted_time"]) if d.get("mean_restricted_time") is not None else None,
            peak_target_alignment=float(d["peak_target_alignment"]) if d.get("peak_target_alignment") is not None else None,
            peak_joint_alignment=float(d["peak_joint_alignment"]) if d.get("peak_joint_alignment") is not None else None,
            transformed_state_retained_to_horizon=float(d["transformed_state_retained_to_horizon"]) if d.get("transformed_state_retained_to_horizon") is not None else None,
            post_release_target_occupancy=float(d["post_release_target_occupancy"]) if d.get("post_release_target_occupancy") is not None else None,
            post_release_retention=float(d["post_release_retention"]) if d.get("post_release_retention") is not None else None,
            failure_breakdown=dict(d.get("failure_breakdown", {})),
            rollout_summaries=list(d.get("rollout_summaries", [])),
        )


@dataclass
class ExploratoryGridConfig:
    """Pre-declared configuration and compute budget for the exploratory grid."""
    grid_id: str = "exploratory_response_grid"
    axis_x: GridAxis = field(default_factory=lambda: GridAxis(
        name="node_fraction",
        label="Candidate Node Fraction (k/N)",
        values=(0.1, 0.25, 0.5),
        parameter_target="policy",
        description="Fraction of population accessible to policy actuation",
    ))
    axis_y: GridAxis = field(default_factory=lambda: GridAxis(
        name="exposure_cap",
        label="Exposure Budget Cap (B)",
        values=(5.0, 10.0, 20.0),
        parameter_target="problem",
        description="Integrated exposure budget cap",
    ))
    selector: str = "reported_attention"
    nominal_amplitude: float = 2.0
    duration: float = 5.0
    quota: float = 1.0
    is_feedback: bool = False
    component: int = 0
    target_type: str = "reversal"
    target_threshold: float = 0.5
    holding_duration: float = 5.0
    release_deadline: float = 10.0
    horizon: float = 25.0
    n_exploratory_origins: int = 2
    futures_per_origin: int = 1
    master_seed: int = 55443
    compute_budget_cap: int = 30  # Maximum total rollouts allowed across entire grid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "axis_x": {
                "name": self.axis_x.name,
                "label": self.axis_x.label,
                "values": list(self.axis_x.values),
                "parameter_target": self.axis_x.parameter_target,
                "description": self.axis_x.description,
            },
            "axis_y": {
                "name": self.axis_y.name,
                "label": self.axis_y.label,
                "values": list(self.axis_y.values),
                "parameter_target": self.axis_y.parameter_target,
                "description": self.axis_y.description,
            },
            "selector": self.selector,
            "nominal_amplitude": self.nominal_amplitude,
            "duration": self.duration,
            "quota": self.quota,
            "is_feedback": self.is_feedback,
            "component": self.component,
            "target_type": self.target_type,
            "target_threshold": self.target_threshold,
            "holding_duration": self.holding_duration,
            "release_deadline": self.release_deadline,
            "horizon": self.horizon,
            "n_exploratory_origins": self.n_exploratory_origins,
            "futures_per_origin": self.futures_per_origin,
            "master_seed": self.master_seed,
            "compute_budget_cap": self.compute_budget_cap,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExploratoryGridConfig:
        ax_x_d = d["axis_x"]
        ax_y_d = d["axis_y"]
        return cls(
            grid_id=str(d.get("grid_id", "exploratory_response_grid")),
            axis_x=GridAxis(
                name=ax_x_d["name"],
                label=ax_x_d["label"],
                values=tuple(float(v) for v in ax_x_d["values"]),
                parameter_target=ax_x_d["parameter_target"],
                description=ax_x_d.get("description", ""),
            ),
            axis_y=GridAxis(
                name=ax_y_d["name"],
                label=ax_y_d["label"],
                values=tuple(float(v) for v in ax_y_d["values"]),
                parameter_target=ax_y_d["parameter_target"],
                description=ax_y_d.get("description", ""),
            ),
            selector=str(d.get("selector", "reported_attention")),
            nominal_amplitude=float(d.get("nominal_amplitude", 2.0)),
            duration=float(d.get("duration", 5.0)),
            quota=float(d.get("quota", 1.0)),
            is_feedback=bool(d.get("is_feedback", False)),
            component=int(d.get("component", 0)),
            target_type=str(d.get("target_type", "reversal")),
            target_threshold=float(d.get("target_threshold", 0.5)),
            holding_duration=float(d.get("holding_duration", 5.0)),
            release_deadline=float(d.get("release_deadline", 10.0)),
            horizon=float(d.get("horizon", 25.0)),
            n_exploratory_origins=int(d.get("n_exploratory_origins", 2)),
            futures_per_origin=int(d.get("futures_per_origin", 1)),
            master_seed=int(d.get("master_seed", 55443)),
            compute_budget_cap=int(d.get("compute_budget_cap", 30)),
        )


@dataclass
class ExploratoryGridResult:
    """Execution outcome of the bounded exploratory diagnostic response grid."""
    config: ExploratoryGridConfig
    cells: List[GridCellResult]
    total_rollouts_executed: int
    compute_budget_exhausted: bool
    elapsed_seconds: float
    overall_failure_breakdown: Dict[str, int]
    raw_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "total_rollouts_executed": int(self.total_rollouts_executed),
            "compute_budget_exhausted": bool(self.compute_budget_exhausted),
            "elapsed_seconds": float(self.elapsed_seconds),
            "overall_failure_breakdown": dict(self.overall_failure_breakdown),
            "raw_data": self.raw_data,
        }

    def save_json(self, filepath: Union[str, Path]) -> str:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = _json_safe(self.to_dict())
        checksum = hashlib.sha256(_canonical_json_bytes(serializable)).hexdigest()
        envelope = {
            "schema_version": 1,
            "sha256": checksum,
            "payload": serializable,
        }
        _atomic_write_json(path, envelope)
        return str(path)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> ExploratoryGridResult:
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        payload = envelope["payload"]
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExploratoryGridResult:
        cfg = ExploratoryGridConfig.from_dict(d["config"])
        cells = [GridCellResult.from_dict(cd) for cd in d["cells"]]
        return cls(
            config=cfg,
            cells=cells,
            total_rollouts_executed=int(d["total_rollouts_executed"]),
            compute_budget_exhausted=bool(d["compute_budget_exhausted"]),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            overall_failure_breakdown=dict(d.get("overall_failure_breakdown", {})),
            raw_data=dict(d.get("raw_data", {})),
        )


# -----------------------------------------------------------------------------
# Descriptive Trajectory Classifier
# -----------------------------------------------------------------------------

def classify_rollout_failure_mechanism(
    result: RolloutResult,
    target_threshold: float = 0.5,
    s_star: float = 1.0,
) -> str:
    """Classify *observed* rollout patterns; do not infer causal mechanisms.

    Scalar trajectories provide population means, so they cannot distinguish
    a lack of response at actuated nodes from failed network propagation. The
    0.10 joint-alignment split is an exploratory descriptive threshold only.
    """
    if result.success:
        return "success"

    rec = getattr(result, "recorder", None)
    if rec is None or not rec.scalar_points:
        return "unclassified"

    points = rec.scalar_points
    if any(
        action.budget_clamped or action.shutoff_or_release == "budget_exhaustion"
        for action in rec.action_history
    ):
        return "budget_limited"

    if result.holding_start_time is not None or any(
        point.target_predicate
        or point.hold_progress > 0.0
        or min(s_star * point.m_x, s_star * point.m_y) >= target_threshold
        for point in points
    ):
        return "target_not_sustained"

    def joint_alignment(point: Any) -> float:
        return min(s_star * point.m_x, s_star * point.m_y)

    shift = max(joint_alignment(point) for point in points) - joint_alignment(points[0])
    return "weak_joint_shift" if shift <= 0.10 + 1e-12 else "partial_joint_shift"


# -----------------------------------------------------------------------------
# Exploratory Grid Execution Engine
# -----------------------------------------------------------------------------

def run_diagnostic_grid(
    config: Any,
    profile: Optional[Any] = None,
    out_dir: Optional[Union[str, Path]] = None,
    progress: bool = False,
    origin_provider: Optional[Callable[[Any, int], State]] = None,
) -> ExploratoryGridResult:
    """Execute the bounded exploratory diagnostic response grid.

    Uses dedicated exploratory origins disjoint from confirmation evaluation origins.
    Respects compute budget cap strictly; unevaluated cells are clearly marked.
    """
    if not isinstance(config, ExploratoryGridConfig) and isinstance(profile, ExploratoryGridConfig):
        config, profile = profile, config

    if profile is None:
        raise ValueError("A profile must be provided to run_diagnostic_grid.")
    is_scenario = hasattr(profile, "scenario_spec_path")
    if is_scenario and origin_provider is None:
        raise ValueError("Tiered-population diagnostic grid requires a scenario origin_provider")

    from src.config.profiles import build_realism_runner
    from src.experiment.ensembles import derive_origin_seed
    from src.control.campaign import _state_origin_hash

    t_start_grid = time.perf_counter()

    # 1. Prepare dedicated exploratory origins (disjoint seed stream)
    # Using seed_base + 50000 ensures complete isolation from evaluation origins
    exploratory_seed_base = int(profile.seed_base) + 50000
    n_exp_origins = int(config.n_exploratory_origins)
    exploratory_states: List[State] = []
    exploratory_origin_ids: List[str] = []

    preparation_attempts: List[Dict[str, Any]] = []
    max_attempts = max(50, 50 * n_exp_origins)
    for attempt_idx in range(max_attempts):
        if len(exploratory_states) >= n_exp_origins:
            break
        seed = derive_origin_seed(exploratory_seed_base, attempt_idx)
        if origin_provider is not None:
            st = origin_provider(profile, seed)
        else:
            runner = build_realism_runner(profile, origin_seed=seed)
            traj = runner.run(horizon=profile.establishment_time)
            st = traj.final_state
        eligibility = TargetEvaluator.create_for_state(
            target_type=TargetType(config.target_type), initial_state=st,
            holding_duration=config.holding_duration, horizon=config.horizon,
            release_deadline=config.release_deadline,
            target_params={"threshold": config.target_threshold, "component": config.component,
                           "eligibility_mode": getattr(profile, "eligibility_mode", "legacy_strong")},
        )
        preparation_attempts.append({
            "index": attempt_idx, "seed": int(seed),
            "accepted": bool(eligibility.defn.is_eligible),
            "reason": eligibility.defn.ineligibility_reason,
        })
        if eligibility.defn.is_eligible or is_scenario:
            accepted_idx = len(exploratory_states)
            exploratory_states.append(st)
            exploratory_origin_ids.append(f"origin_exp_{accepted_idx:03d}_{_state_origin_hash(st)}")
    if len(exploratory_states) < n_exp_origins:
        raise ValueError(
            f"Diagnostic grid found {len(exploratory_states)}/{n_exp_origins} eligible origins "
            f"in {max_attempts} preparation attempts"
        )

    # 2. Physics & Base parameters
    tau_y = float(profile.tau_public)
    physics = {
        "J": float(profile.J),
        "T": float(profile.temperature),
        "delta": float(profile.delta),
        "epsilon": float(profile.exploration),
        "closure": float(profile.closure),
        "sample_interval": float(profile.sample_interval_tau_public) * tau_y,
        "a": float(profile.prestige_global),
        "c": float(profile.matching),
        "tau_y": tau_y,
    }
    obs_kwargs = {
        "panel_fraction": float(profile.panel_fraction),
        "edge_reveal_prob": float(profile.edge_reveal_prob),
        "binary_flip_prob": float(profile.binary_flip_prob),
        "delivery_delay": float(profile.delivery_delay_tau_public) * tau_y,
    }

    # 3. Grid Coordinates Iteration
    x_vals = list(config.axis_x.values)
    y_vals = list(config.axis_y.values)
    cells: List[GridCellResult] = []
    total_rollouts_done = 0
    budget_exhausted = False
    overall_failure_counts = {
        "weak_joint_shift": 0,
        "partial_joint_shift": 0,
        "target_not_sustained": 0,
        "budget_limited": 0,
        "unclassified": 0,
        "success": 0,
    }

    raw_cell_data: Dict[str, Any] = {}

    for i_x, val_x in enumerate(x_vals):
        for i_y, val_y in enumerate(y_vals):
            coord_key = f"cell_x{i_x}_y{i_y}"

            # Check compute budget cap
            planned_cell_rollouts = n_exp_origins * int(config.futures_per_origin)
            if total_rollouts_done + planned_cell_rollouts > config.compute_budget_cap:
                budget_exhausted = True
                cell_res = GridCellResult(
                    coord_x=float(val_x),
                    coord_y=float(val_y),
                    coord_idx=(i_x, i_y),
                    is_evaluated=False,
                    status="unevaluated_budget_cap",
                )
                cells.append(cell_res)
                raw_cell_data[coord_key] = cell_res.to_dict()
                continue

            # Determine parameter assignments for this cell
            # Exposure cap, node fraction, amplitude, etc.
            cur_exposure_cap = float(val_y) if config.axis_y.name == "exposure_cap" else (
                float(val_x) if config.axis_x.name == "exposure_cap" else float(profile.exposure_cap_normalized) * profile.N * profile.J
            )
            cur_node_fraction = float(val_x) if config.axis_x.name == "node_fraction" else (
                float(val_y) if config.axis_y.name == "node_fraction" else 0.25
            )

            N = int(profile.N)
            D = int(profile.D)
            k_val = max(1, int(round(cur_node_fraction * N)))

            # Keep the accessible pool fixed across cells so only the declared
            # node cap and exposure budget vary.
            rng_acc = np.random.default_rng(config.master_seed + 100)
            accessible_count = max(1, int(round(float(profile.accessible_fraction) * N)))
            acc_nodes = tuple(sorted(rng_acc.choice(N, size=accessible_count, replace=False).tolist()))

            problem = ControlProblem(
                problem_id=f"grid_{config.grid_id}_{coord_key}",
                target_type=TargetType(config.target_type),
                target_params={"threshold": config.target_threshold, "component": config.component,
                           "eligibility_mode": getattr(profile, "eligibility_mode", "legacy_strong")},
                N=N,
                D=D,
                k=k_val,
                exposure_cap=cur_exposure_cap,
                amplitude_cap=float(profile.amplitude_cap_over_J) * profile.J,
                release_deadline=float(config.release_deadline),
                holding_duration=float(config.holding_duration),
                horizon=float(config.horizon),
                accessible_roster=acc_nodes,
                proposal_seed=config.master_seed + i_x * 100 + i_y,
                search_seed=config.master_seed + i_x * 100 + i_y + 1,
            )

            cell_rollouts: List[RolloutResult] = []
            cell_failures = {
                "weak_joint_shift": 0,
                "partial_joint_shift": 0,
                "target_not_sustained": 0,
                "budget_limited": 0,
                "unclassified": 0,
                "success": 0,
            }
            peak_alignments: List[float] = []
            joint_alignments: List[float] = []
            transformed_retention_flags: List[bool] = []
            target_occupancies: List[float] = []

            for o_idx, origin_state in enumerate(exploratory_states):
                orig_id = exploratory_origin_ids[o_idx]
                m_y0 = float(np.mean(origin_state.y[:, 0]))
                s_star = -1.0 if m_y0 > 0 else 1.0

                for fut_id in range(config.futures_per_origin):
                    sim_seed = config.master_seed + o_idx * 1000 + fut_id * 10 + 1
                    ctl_seed = config.master_seed + o_idx * 1000 + fut_id * 10 + 2

                    policy = ObservableSelectorPolicy(
                        selector=config.selector,
                        nominal_amplitude=float(config.nominal_amplitude),
                        duration=float(config.duration),
                        quota=float(config.quota),
                        is_feedback=bool(config.is_feedback),
                        component=int(config.component),
                        node_fraction=cur_node_fraction,
                    )
                    from src.control.observation import SampledPublicObservationModel
                    obs_model = SampledPublicObservationModel(
                        N=N,
                        accessible_roster=problem.accessible_roster,
                        rng=np.random.default_rng(ctl_seed),
                        **obs_kwargs,
                    )
                    evaluator = TargetEvaluator.create_for_state(
                        target_type=problem.target_type,
                        initial_state=origin_state,
                        holding_duration=problem.holding_duration,
                        horizon=problem.horizon,
                        release_deadline=problem.release_deadline,
                        problem_id=problem.problem_id,
                        target_params=problem.target_params,
                    )
                    rec = RolloutRecorder(RecorderConfig(
                        cadence=1.0,
                        record_scalar_trajectories=True,
                        record_action_history=True,
                        retention_threshold=config.target_threshold,
                    ))

                    res, _, ledger = run_controlled_simulation(
                        initial_state=origin_state.copy(),
                        problem=problem,
                        controller=policy,
                        observation_model=obs_model,
                        target_evaluator=evaluator,
                        recorder=rec,
                        sim_rng=np.random.default_rng(sim_seed),
                        controller_rng=np.random.default_rng(ctl_seed),
                        **physics,
                    )
                    cell_rollouts.append(res)
                    total_rollouts_done += 1

                    # Classify failure mechanism
                    mech = classify_rollout_failure_mechanism(
                        res,
                        target_threshold=config.target_threshold,
                        s_star=s_star,
                    )
                    cell_failures[mech] = cell_failures.get(mech, 0) + 1
                    overall_failure_counts[mech] = overall_failure_counts.get(mech, 0) + 1

                    # Target direction is defined by the focal initial orientation.
                    # Public-only alignment is retained for backward-compatible raw data;
                    # the plotted joint measure requires private and public movement.
                    if rec.scalar_points and res.is_eligible:
                        peak_alignments.append(max(s_star * p.m_y for p in rec.scalar_points))
                        joint_alignments.append(max(min(s_star * p.m_x, s_star * p.m_y)
                                                    for p in rec.scalar_points))
                        released_points = [p for p in rec.scalar_points if p.actual_release]
                        if released_points:
                            target_occupancies.append(float(np.mean([
                                min(s_star * p.m_x, s_star * p.m_y) >= config.target_threshold
                                for p in released_points
                            ])))
                    # The evaluator observes every state-changing event, including relapse.
                    # This is persistence of the transformed state through H after
                    # certification, distinct from retention of the initial state.
                    if res.is_eligible:
                        transformed_retention_flags.append(bool(res.success and not res.relapsed))

            # Cell aggregates
            n_tot = len(cell_rollouts)
            eligible = [r for r in cell_rollouts if r.is_eligible]
            n_eligible = len(eligible)
            n_succ = sum(bool(r.success) for r in eligible)
            p_hat = n_succ / n_eligible if n_eligible else None
            ci = wilson_interval(n_succ, n_eligible, confidence=0.95) if n_eligible else None
            mean_restr_t = float(np.mean([r.restricted_time for r in eligible])) if eligible else None
            mean_pk_align = float(np.mean(peak_alignments)) if peak_alignments else None
            mean_joint_align = float(np.mean(joint_alignments)) if joint_alignments else None
            transformed_retained = float(np.mean(transformed_retention_flags)) if transformed_retention_flags else None
            target_occupancy = float(np.mean(target_occupancies)) if target_occupancies else None

            cell_res = GridCellResult(
                coord_x=float(val_x),
                coord_y=float(val_y),
                coord_idx=(i_x, i_y),
                is_evaluated=True,
                status="evaluated",
                n_rollouts=n_tot,
                n_success=n_succ,
                n_eligible=n_eligible,
                success_probability=p_hat,
                success_ci=ci,
                mean_restricted_time=mean_restr_t,
                peak_target_alignment=mean_pk_align,
                peak_joint_alignment=mean_joint_align,
                transformed_state_retained_to_horizon=transformed_retained,
                post_release_target_occupancy=target_occupancy,
                failure_breakdown=cell_failures,
                rollout_summaries=[
                    {
                        "origin_id": r.origin_id,
                        "success": bool(r.success),
                        "is_eligible": bool(r.is_eligible),
                        "relapsed": bool(r.relapsed),
                        "transformed_state_retained_to_horizon": bool(r.success and not r.relapsed),
                        "restricted_time": r.restricted_time,
                        "completion_time": r.completion_time,
                    }
                    for r in cell_rollouts
                ],
            )
            cells.append(cell_res)
            raw_cell_data[coord_key] = cell_res.to_dict()

            if progress:
                print(f"  Grid cell ({i_x}, {i_y}) [{val_x}, {val_y}]: p={p_hat:.2f} ({n_succ}/{n_tot}), "
                      f"T_R={mean_restr_t:.1f}, failure_top={max(cell_failures, key=cell_failures.get)}")

    elapsed = time.perf_counter() - t_start_grid

    grid_result = ExploratoryGridResult(
        config=config,
        cells=cells,
        total_rollouts_executed=total_rollouts_done,
        compute_budget_exhausted=budget_exhausted,
        elapsed_seconds=elapsed,
        overall_failure_breakdown=overall_failure_counts,
        raw_data={
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cells": raw_cell_data,
            "axes": {
                "axis_x": config.axis_x.name,
                "values_x": list(config.axis_x.values),
                "axis_y": config.axis_y.name,
                "values_y": list(config.axis_y.values),
            },
            "exploratory_origins": exploratory_origin_ids,
            "preparation_attempts": preparation_attempts,
        },
    )

    if out_dir is not None:
        p_out = Path(out_dir) / "diagnostic_grid_data.json"
        grid_result.save_json(p_out)

    return grid_result


# Alias for compatibility
run_exploratory_response_grid = run_diagnostic_grid

