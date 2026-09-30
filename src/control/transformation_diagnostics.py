"""Small, paired feasibility and two-network diagnostics for the light campaign.

The positive control asks whether the simulator can sustain a reversal when every
focal node is reachable. The interaction pilot puts two established populations
in one physical simulation and varies their encounter eligibility. Only the
focal population enters the target evaluator and its trajectory recorder.
Neither experiment estimates confirmatory reliability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.control.campaign import _campaign_geometry, _stable_seed, load_origin_snapshot
from src.control.contracts import ControlProblem, TargetType
from src.control.controllers import ObservableSelectorPolicy, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import SampledPublicObservationModel
from src.control.recording import RecorderConfig, RolloutRecorder
from src.core.state import State
from src.execution.runner import NumpyEncoder


@dataclass(frozen=True)
class TransformationDiagnosticConfig:
    focal_origins: int = 2
    interaction_strengths: Tuple[float, ...] = (0.0, 0.05, 0.3)
    partner_orientations: Tuple[str, ...] = ("supportive", "opposing")
    master_seed: int = 99002

    def validate(self) -> None:
        if self.focal_origins < 1:
            raise ValueError("focal_origins must be positive")
        if not self.interaction_strengths or self.interaction_strengths[0] != 0.0:
            raise ValueError("interaction_strengths must begin with the isolated value 0")
        if any(value < 0 for value in self.interaction_strengths):
            raise ValueError("interaction strengths must be nonnegative")
        if set(self.partner_orientations) != {"supportive", "opposing"}:
            raise ValueError("compare both supportive and opposing partner orientations")


def _origin_states(campaign: Any) -> Dict[str, State]:
    prep = campaign.preparation_summary
    origin_ids = prep["origin_ids"]
    paths = prep["accepted_snapshot_paths"]
    if len(origin_ids) != len(paths) or any(not path for path in paths):
        raise ValueError("Diagnostic pilots require all accepted origin snapshots")
    return {origin_id: load_origin_snapshot(str(path)) for origin_id, path in zip(origin_ids, paths)}


def _combined_state(focal: State, partner: State, orientation: str) -> State:
    """Build two established populations with separate initial memory blocks."""
    if focal.D != partner.D or focal.state_mode != partner.state_mode:
        raise ValueError("Both networks must have the same state dimension and mode")
    nf, npart = focal.N, partner.N
    focal_sign = 1.0 if np.mean(focal.y[:, 0]) >= 0 else -1.0
    wanted = -focal_sign if orientation == "supportive" else focal_sign
    partner_sign = 1.0 if np.mean(partner.y[:, 0]) >= 0 else -1.0
    flip = wanted != partner_sign
    partner_x = -partner.x if flip else partner.x
    partner_y = -partner.y if flip else partner.y
    W = np.zeros((nf + npart, nf + npart), dtype=np.uint32)
    W[:nf, :nf] = focal.W
    W[nf:, nf:] = partner.W

    def trait(name: str, focal_default: Any, partner_default: Any) -> np.ndarray:
        left = getattr(focal, name)
        right = getattr(partner, name)
        return np.concatenate((left if left is not None else np.full_like(focal_default, focal_default),
                               right if right is not None else np.full_like(partner_default, partner_default)))

    # Default arrays matter only for legacy snapshots lacking heterogeneous traits.
    K = np.concatenate((focal.K if focal.K is not None else np.full(nf, 0.5),
                        partner.K if partner.K is not None else np.full(npart, 0.5)))
    tau_x = np.concatenate((focal.tau_x if focal.tau_x is not None else np.full(nf, 5.0),
                            partner.tau_x if partner.tau_x is not None else np.full(npart, 5.0)))
    nu = np.concatenate((focal.nu if focal.nu is not None else np.full(nf, 2.0),
                         partner.nu if partner.nu is not None else np.full(npart, 2.0)))
    g = np.concatenate((focal.g if focal.g is not None else np.ones(nf),
                        partner.g if partner.g is not None else np.ones(npart)))
    focal_b = focal.b if focal.b is not None else np.zeros((nf, focal.D))
    partner_b = partner.b if partner.b is not None else np.zeros((npart, partner.D))
    b = np.concatenate((focal_b, -partner_b if flip else partner_b), axis=0)
    row_caps = np.concatenate((focal.row_caps if focal.row_caps is not None else np.full(nf, 40),
                               partner.row_caps if partner.row_caps is not None else np.full(npart, 40)))
    groups = np.concatenate((np.zeros(nf, dtype=np.int64), np.ones(npart, dtype=np.int64)))
    combined = State(
        N=nf + npart, D=focal.D, state_mode=focal.state_mode,
        x=np.concatenate((focal.x, partner_x)), y=np.concatenate((focal.y, partner_y)),
        W=W, n=np.concatenate((focal.n, partner.n)), Q=np.concatenate((focal.Q, partner.Q)),
        t=0.0, K=K, tau_x=tau_x, nu=nu, g=g, b=b,
        row_caps=row_caps, opportunity_groups=groups, blocks=groups.copy(),
    )
    combined.validate()
    return combined


def _run_one(
    state: State, problem: ControlProblem, profile: Any,
    controller: Any, sim_seed: int, ctl_seed: int,
    between_eligibility: float = 0.05,
) -> Dict[str, Any]:
    evaluator = TargetEvaluator.create_for_state(
        target_type=problem.target_type, initial_state=state,
        holding_duration=problem.holding_duration, horizon=problem.horizon,
        release_deadline=problem.release_deadline, problem_id=problem.problem_id,
        target_params=problem.target_params,
    )
    observation = SampledPublicObservationModel(
        N=state.N, panel_fraction=profile.panel_fraction,
        edge_reveal_prob=profile.edge_reveal_prob,
        binary_flip_prob=profile.binary_flip_prob,
        delivery_delay=profile.delivery_delay_tau_public * profile.tau_public,
        accessible_roster=problem.accessible_roster,
        group_labels=state.opportunity_groups,
        rng=np.random.default_rng(ctl_seed),
    )
    recorder = RolloutRecorder(RecorderConfig(
        cadence=1.0, record_scalar_trajectories=True,
        record_action_history=True, record_network_frames=True,
        max_network_frames=80, retention_threshold=profile.target_threshold,
    ))
    result, final, _ = run_controlled_simulation(
        initial_state=state, problem=problem, controller=controller,
        observation_model=observation, target_evaluator=evaluator,
        recorder=recorder, sim_rng=np.random.default_rng(sim_seed),
        controller_rng=np.random.default_rng(ctl_seed),
        J=profile.J, T=profile.temperature, delta=profile.delta,
        epsilon=profile.exploration, closure=profile.closure,
        sample_interval=profile.sample_interval_tau_public * profile.tau_public,
        a=profile.prestige_global, c=profile.matching,
        tau_y=profile.tau_public, between_eligibility=between_eligibility,
        within_eligibility=1.0,
    )
    focal = problem.target_params.get("focal_indices")
    focal_nodes = list(focal) if focal is not None else list(range(state.N))
    other_nodes = [i for i in range(state.N) if i not in set(focal_nodes)]
    actuated = sorted({node for action in recorder.action_history for node in action.actuated_nodes
                       if node in focal_nodes})
    unactuated = [node for node in focal_nodes if node not in set(actuated)]
    release_frames = [frame for frame in recorder.network_frames
                      if frame.t >= problem.release_deadline - 1e-9]
    after = release_frames[0] if release_frames else recorder.network_frames[-1]

    def means(x: np.ndarray, y: np.ndarray, nodes: List[int]) -> Optional[Dict[str, float]]:
        if not nodes:
            return None
        return {"private": float(np.mean(x[nodes, 0])),
                "public": float(np.mean(y[nodes, 0]))}

    target_sign = evaluator.defn.s_star or 1.0
    target_points = [point for point in recorder.scalar_points if point.actual_release]
    return {
        "eligible": bool(result.is_eligible),
        "certified_once": bool(result.success),
        "sustained_to_horizon": bool(result.success and not result.relapsed),
        "relapsed": bool(result.relapsed),
        "completion_time": result.completion_time,
        "expenditure": result.expenditure,
        "distinct_nodes_used": result.distinct_nodes_used,
        "focal_initial": means(state.x, state.y, focal_nodes),
        "focal_after_release": means(after.x, after.y, focal_nodes),
        "focal_final": means(final.x, final.y, focal_nodes),
        "partner_initial": means(state.x, state.y, other_nodes),
        "partner_final": means(final.x, final.y, other_nodes),
        "targeted_at_release": means(after.x, after.y, actuated),
        "untargeted_at_release": means(after.x, after.y, unactuated),
        "targeted_final": means(final.x, final.y, actuated),
        "untargeted_final": means(final.x, final.y, unactuated),
        "peak_joint_target_alignment": max(
            (min(target_sign * p.m_x, target_sign * p.m_y) for p in recorder.scalar_points),
            default=None,
        ),
        "post_release_target_occupancy_sampled": (
            float(np.mean([p.target_predicate for p in target_points])) if target_points else None
        ),
        "cross_network_memory_final": int(final.W[np.ix_(focal_nodes, other_nodes)].sum()
                                          + final.W[np.ix_(other_nodes, focal_nodes)].sum()) if other_nodes else 0,
        "trajectory": [{"t": p.t, "private": p.m_x, "public": p.m_y,
                        "target": bool(p.target_predicate), "released": bool(p.actual_release)}
                       for p in recorder.scalar_points],
    }


def run_transformation_diagnostics(
    profile: Any, campaign: Any,
    config: Optional[TransformationDiagnosticConfig] = None,
    save_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Run a positive control and matched isolated/coupled network comparisons."""
    cfg = config or TransformationDiagnosticConfig()
    cfg.validate()
    if profile.target_type != "reversal":
        raise ValueError("These pilots currently require a reversal destination")
    states = _origin_states(campaign)
    pools = campaign.preparation_summary["pool_membership"]
    focal_ids = pools["evaluation"][:cfg.focal_origins]
    if not focal_ids:
        raise ValueError("No evaluation origins are available for the diagnostic pilots")
    partner_ids = pools.get("unassigned_spare", []) + pools["training"] + pools["selection"]
    if not partner_ids:
        raise ValueError("A separate established partner origin is required")
    geo = _campaign_geometry(profile, profile.N)
    positive: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    active = campaign.selected_feedback_candidate
    if active is None or active.nominal_amplitude <= 0:
        active = campaign.selected_static_candidate
    use_selected_active = bool(active is not None and active.nominal_amplitude > 0)

    def active_controller() -> Any:
        if use_selected_active:
            return active.to_controller()
        return ObservableSelectorPolicy(
            selector="reported_attention", nominal_amplitude=geo["amplitude_cap"],
            duration=profile.release_deadline, quota=1.0, is_feedback=True,
            node_fraction=profile.k_distinct_nodes_fraction,
        )

    for focal_index, focal_id in enumerate(focal_ids):
        focal = states[focal_id]
        sim_seed = _stable_seed(cfg.master_seed, focal_id, "sim")
        ctl_seed = _stable_seed(cfg.master_seed, focal_id, "ctl")
        # Diagnostic positive control is deliberately outside the constrained
        # access and expenditure geometry. It tests the simulator's ability to
        # reach and sustain the destination, not policy feasibility.
        positive_problem = ControlProblem(
            problem_id="diagnostic_positive_control", target_type=TargetType.REVERSAL,
            target_params={"threshold": profile.target_threshold}, N=focal.N, D=focal.D,
            k=focal.N, exposure_cap=focal.N * geo["amplitude_cap"] * profile.release_deadline,
            amplitude_cap=geo["amplitude_cap"], release_deadline=profile.release_deadline,
            holding_duration=profile.holding_duration, horizon=profile.horizon,
            accessible_roster=tuple(range(focal.N)),
        )
        for arm, controller in (
            ("zero", ZeroController()),
            ("broad_pulse", ObservableSelectorPolicy(
                selector="random", nominal_amplitude=geo["amplitude_cap"],
                duration=profile.release_deadline, quota=1.0,
                is_feedback=False, node_fraction=1.0,
            )),
        ):
            measured = _run_one(focal, positive_problem, profile, controller,
                                sim_seed, ctl_seed)
            positive.append({"origin_id": focal_id, "arm": arm, **measured})

        partner_id = partner_ids[focal_index % len(partner_ids)]
        partner = states[partner_id]
        for orientation in cfg.partner_orientations:
            combined = _combined_state(focal, partner, orientation)
            problem = ControlProblem(
                problem_id="diagnostic_network_interaction",
                target_type=TargetType.REVERSAL,
                target_params={"threshold": profile.target_threshold,
                               "focal_indices": list(range(focal.N))},
                N=combined.N, D=combined.D, k=geo["k"],
                exposure_cap=geo["exposure_cap"],
                amplitude_cap=geo["amplitude_cap"],
                release_deadline=profile.release_deadline,
                holding_duration=profile.holding_duration, horizon=profile.horizon,
                accessible_roster=geo["accessible_roster"],
            )
            for strength in cfg.interaction_strengths:
                for arm, controller in (("zero", ZeroController()),
                                        ("active", active_controller())):
                    measured = _run_one(combined, problem, profile, controller,
                                        sim_seed, ctl_seed,
                                        between_eligibility=strength)
                    interactions.append({
                        "focal_origin_id": focal_id,
                        "partner_origin_id": partner_id,
                        "partner_orientation": orientation,
                        "between_network_eligibility": strength,
                        "arm": arm, **measured,
                    })

    output = {
        "interpretation": (
            "Exploratory matched comparisons. The focal network alone defines success. "
            "The broad positive control relaxes access and budget. A transformation "
            "survives this finite experiment only if certified and never relapsed by H."
        ),
        "policy_source": active.candidate_id if use_selected_active else "diagnostic_active_fallback",
        "config": {"focal_origins": len(focal_ids),
                   "interaction_strengths": list(cfg.interaction_strengths),
                   "partner_orientations": list(cfg.partner_orientations),
                   "master_seed": cfg.master_seed},
        "positive_control": positive,
        "network_interaction": interactions,
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    return output


def run_tiered_positive_control(
    profile: Any,
    origin_provider: Any,
    *,
    n_origins: Optional[int] = None,
    master_seed: int = 190026,
    save_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Compare zero and broad pulses on independent constructed diagnostic origins.

    This relaxed-access experiment is diagnostic only. Its arms share initial
    states and simulation seeds; none is used for policy search or selection.
    """
    from src.experiment.ensembles import derive_origin_seed
    from src.control.campaign import _state_origin_hash

    count = int(n_origins if n_origins is not None else profile.diagnostic_origins)
    if count < 1:
        raise ValueError("Positive control needs at least one diagnostic origin")
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        seed = derive_origin_seed(int(profile.seed_base) + 100000, index)
        origin = origin_provider(profile, seed)
        origin_id = f"origin_diag_{index:03d}_{_state_origin_hash(origin)}"
        amplitude = float(profile.amplitude_cap_over_J) * float(profile.J)
        problem = ControlProblem(
            problem_id="tiered_broad_positive_control",
            target_type=TargetType.REVERSAL,
            target_params={"threshold": float(profile.target_threshold),
                           "component": int(profile.target_component),
                           "eligibility_mode": str(profile.eligibility_mode)},
            N=origin.N, D=origin.D, k=origin.N,
            exposure_cap=origin.N * amplitude * float(profile.release_deadline),
            amplitude_cap=amplitude,
            release_deadline=float(profile.release_deadline),
            holding_duration=float(profile.holding_duration),
            horizon=float(profile.horizon),
            accessible_roster=tuple(range(origin.N)),
        )
        sim_seed = int(derive_origin_seed(master_seed, index))
        ctl_seed = int(derive_origin_seed(master_seed + 1, index))
        opportunity = (origin.scenario_metadata or {}).get("opportunity", {})
        for arm, controller in (
            ("zero", ZeroController()),
            ("broad_pulse", ObservableSelectorPolicy(
                selector="random", nominal_amplitude=amplitude,
                duration=float(profile.release_deadline), quota=1.0,
                is_feedback=False, node_fraction=1.0,
            )),
        ):
            evaluator = TargetEvaluator.create_for_state(
                target_type=problem.target_type, initial_state=origin,
                holding_duration=problem.holding_duration,
                horizon=problem.horizon,
                release_deadline=problem.release_deadline,
                problem_id=problem.problem_id, target_params=problem.target_params,
            )
            observation = SampledPublicObservationModel(
                N=origin.N, panel_fraction=profile.panel_fraction,
                edge_reveal_prob=profile.edge_reveal_prob,
                binary_flip_prob=profile.binary_flip_prob,
                delivery_delay=profile.delivery_delay_tau_public * profile.tau_public,
                accessible_roster=problem.accessible_roster,
                group_labels=origin.opportunity_groups,
                rng=np.random.default_rng(ctl_seed),
            )
            recorder = RolloutRecorder(RecorderConfig(
                cadence=1.0, record_scalar_trajectories=True,
                record_action_history=True, record_network_frames=False,
                retention_threshold=float(profile.target_threshold),
            ))
            result, _, _ = run_controlled_simulation(
                initial_state=origin.copy(), problem=problem, controller=controller,
                observation_model=observation, target_evaluator=evaluator,
                recorder=recorder, sim_rng=np.random.default_rng(sim_seed),
                controller_rng=np.random.default_rng(ctl_seed),
                J=profile.J, T=profile.temperature, delta=profile.delta,
                epsilon=profile.exploration, closure=profile.closure,
                sample_interval=profile.sample_interval_tau_public * profile.tau_public,
                a=profile.prestige_global, c=profile.matching,
                tau_y=profile.tau_public,
                between_eligibility=float(opportunity.get("between", 0.05)),
                within_eligibility=float(opportunity.get("within", 1.0)),
            )
            rows.append({
                "origin_id": origin_id, "origin_seed": int(seed),
                "arm": arm, "sim_seed": sim_seed, "controller_seed": ctl_seed,
                "initial_public_mean": float(np.mean(origin.y[:, 0])),
                "initial_private_mean": float(np.mean(origin.x[:, 0])),
                "eligible": bool(result.is_eligible),
                "certified_once": bool(result.success),
                "sustained_to_horizon": bool(result.success and not result.relapsed),
                "relapsed": bool(result.relapsed),
                "completion_time": result.completion_time,
                "restricted_time": float(result.restricted_time),
                "expenditure": float(result.expenditure),
                "distinct_nodes_used": int(result.distinct_nodes_used),
                "actual_release_times": [float(a.decision_time) for a in recorder.action_history
                                         if a.shutoff_or_release in ("field_shutoff", "release_deadline", "explicit_release", "budget_exhaustion")],
                "trajectory": [{"t": float(p.t), "private": float(p.m_x),
                                "public": float(p.m_y), "target": bool(p.target_predicate),
                                "released": bool(p.actual_release)}
                               for p in recorder.scalar_points],
            })
    output = {
        "interpretation": "Independent exploratory origins; broad pulse relaxes access and expenditure. Neither arm is a constrained policy winner or confirmatory reliability estimate.",
        "origin_pool": "diagnostic_seed_base_plus_100000",
        "matched_pairs": count,
        "rows": rows,
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    return output
