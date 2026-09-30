"""Campaign orchestration for transformation control.

Defines the persistent contract (ControlCampaignSpec), origin preparation,
and result types, plus the full light-allocation campaign pipeline:

1. Preparation   — one declared ensemble of established origins, snapshots and
                   traits saved once, eligibility yield reported, split into
                   disjoint training / selection / evaluation sets.
2. Baselines     — no intervention, random targeting, observed-attention, all
                   under restricted observations on the fresh evaluation origins.
3. Search        — ObservablePolicySearchOptimizer  over training origins.
4. Selection     — top-K candidates re-evaluated on selection origins; the best
                   lexicographic score is declared the selected policy.
5. Evaluation    — selected policy on fresh origins with matched adaptive/frozen
                   pairing (identical simulator AND controller seeds per
                   origin/future; the frozen branch genuinely freezes W by
                   disabling encounters and decay: nu=0, delta=0).

Seeds are derived deterministically from (master_seed, role, origin_id,
future_id) so the whole campaign is reproducible. Hidden states are used only
for evaluation; restricted controllers see observations only.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.control.contracts import ControlProblem, RolloutResult, TargetType
from src.control.evaluator import TargetEvaluator
from src.control.optimizers.search import (
    ObservablePolicyCandidate,
    ObservablePolicySearchOptimizer,
    compute_candidate_score,
    candidate_family,
    rank_candidates_by_family,
    rank_candidates_global,
    summarize_node_counts,
)


@dataclass(frozen=True)
class ControlCampaignSpec:
    """The declared target contract persisted everywhere. Immutable after a run starts."""
    target_type: str
    target_params: Dict[str, Any]
    measured_observables: List[str]
    tolerances: Dict[str, float]
    holding_duration: float
    horizon: float
    release_deadline: float
    reliability_target: float
    task_information: str


@dataclass
class PreparedOrigin:
    """A prepared baseline state snapshot."""
    origin_id: str
    seed: int
    snapshot_path: str
    traits: Dict[str, Any]
    is_eligible: bool
    eligibility_reason: str


@dataclass
class ControlCampaignResult:
    """Overall outcome of a control campaign."""
    spec: ControlCampaignSpec
    preparation_summary: Dict[str, Any]
    baseline_results: Dict[str, List[RolloutResult]]
    search_result: Any  # OptimizationResult
    selection_summary: Dict[str, Any]
    evaluation_results: Dict[str, List[RolloutResult]]
    adaptive_vs_frozen_pairs: List[Any]
    manifest: Dict[str, Any]
    selected_candidate: Optional[ObservablePolicyCandidate] = None
    selected_static_candidate: Optional[ObservablePolicyCandidate] = None
    selected_feedback_candidate: Optional[ObservablePolicyCandidate] = None
    paired_comparisons: Dict[str, Any] = field(default_factory=dict)
    reliability_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bundle_path: Optional[str] = None
    diagnostic_trajectories: Dict[str, Any] = field(default_factory=dict)
    action_histories: Dict[str, Any] = field(default_factory=dict)
    network_frames: Dict[str, Any] = field(default_factory=dict)
    retention_summaries: Dict[str, Any] = field(default_factory=dict)
    preparation_attempts: List[Dict[str, Any]] = field(default_factory=list)
    phase_timings: Dict[str, Any] = field(default_factory=dict)
    workload_projection: Dict[str, Any] = field(default_factory=dict)
    work_registry: Any = None


@dataclass
class CampaignPreparation:
    """Prepared origin ensemble split into disjoint campaign phases."""
    prepared: List[PreparedOrigin]
    origins: List[Any]  # ReferenceOrigin
    training: List[Any]
    selection: List[Any]
    evaluation: List[Any]
    summary: Dict[str, Any]
    attempts: List["PreparationAttempt"] = field(default_factory=list)


@dataclass
class PreparationAttempt:
    """One establishment attempt, retained whether accepted or rejected.

    Rejected attempts were previously discarded, which made the reported
    eligibility yield use the wrong denominator (accepted/accepted instead of
    accepted/attempted, e.g. the Colab1 run reported 8/8 while the true yield
    was 8/75 over accepted indices ending at 74).
    """
    index: int
    seed: int
    accepted: bool
    reason: str
    m_y0: float
    m_x0: float
    origin_id: Optional[str] = None
    snapshot_path: str = ""
    pool: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


class PreparationExhausted(RuntimeError):
    """Raised when the preparation attempt cap is hit before enough eligible origins.

    Exhausted preparation means insufficient eligible origins were found within
    the declared attempt cap — it is NOT evidence that the declared target is
    structurally unreachable. Carries partial preparation diagnostics so the
    failure can still be reported.
    """

    def __init__(self, message: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.diagnostics: Dict[str, Any] = diagnostics or {}


def preparation_attempt_summary(attempts: Sequence[Any]) -> Dict[str, Any]:
    """Attempt/rejection accounting over PreparationAttempt records (or dicts).

    Pure accounting helper kept separate from ``prepare_origins`` so rejection
    fixtures (e.g. the historical 8-accepted-in-75-attempts pattern) can be
    checked without running any simulation.
    """
    def _get(a: Any, key: str) -> Any:
        return a[key] if isinstance(a, dict) else getattr(a, key)

    attempted = len(attempts)
    accepted = sum(1 for a in attempts if bool(_get(a, "accepted")))
    rejected = attempted - accepted
    return {
        "n_attempted": attempted,
        "n_accepted": accepted,
        "n_rejected": rejected,
        "eligibility_yield": (accepted / attempted) if attempted else 0.0,
        "eligibility_yield_denominator": "attempted",
    }


def classify_reliability_status(
    successes: int,
    trials: int,
    p_hat: float,
    exact_low: float,
    reliability_target: float,
) -> str:
    """Explicit outcome status for one policy arm (R1 reporting contract).

    - ``no_eligible_trials``: nothing could be evaluated.
    - ``no_feasible_policy_found``: zero successes among eligible trials.
    - ``reliability_certified``: the declared one-sided lower bound meets the
      reliability target (exact binomial for one future per origin; conservative
      origin-level bound for nested futures).
    - ``estimated_feasible``: point estimate meets the target but the lower
      bound does not (feasibility is estimated, not certified).
    - ``best_failing_candidate``: successes exist but the target is not met.

    Ranking and the success definition are unchanged; this only labels the
    outcome honestly.
    """
    if trials <= 0:
        return "no_eligible_trials"
    if successes == 0:
        return "no_feasible_policy_found"
    if exact_low >= reliability_target - 1e-12:
        return "reliability_certified"
    if p_hat >= reliability_target - 1e-12:
        return "estimated_feasible"
    return "best_failing_candidate"


def describe_selection_tie_break(
    best_score: Tuple[Any, ...],
    runner_up_score: Optional[Tuple[Any, ...]] = None,
) -> Dict[str, Any]:
    """Explain which lexicographic component decided the selection.

    The score layout comes from ``compute_candidate_score`` (kept unchanged).
    Position 1 is ``mean_restricted_time`` in the feasible tier and
    ``success_probability`` in the infeasible tier, so the applicable label is
    selected by the tier of the best score.
    """
    if runner_up_score is None:
        return {
            "decided_by": "single_candidate",
            "tied_components": [],
            "description": "only one selection candidate was ranked",
        }
    tier = "feasible" if int(best_score[0]) == 0 else "infeasible"
    if int(best_score[0]) != int(runner_up_score[0]):
        return {
            "decided_by": "tier",
            "tied_components": [],
            "tier": tier,
            "description": "feasible tier beats infeasible tier",
        }
    labels = ["tier"] + (
        ["mean_restricted_time", "success_probability", "terminal_target_violation",
         "mean_exposure", "distinct_nodes"]
        if tier == "feasible"
        else ["success_probability", "mean_restricted_time", "terminal_target_violation",
              "mean_exposure", "distinct_nodes"]
    ) + ["candidate_id"]
    tied: List[str] = []
    for pos, (a, b) in enumerate(zip(best_score, runner_up_score)):
        decided = labels[pos] if pos < len(labels) else f"component_{pos}"
        if a != b:
            if decided == "candidate_id":
                description = "all scored components tied; decided by candidate id"
            else:
                description = (
                    f"decided by {decided} after ties on: "
                    f"{', '.join(tied) if tied else 'nothing'}"
                )
            return {
                "decided_by": decided,
                "tied_components": tied,
                "tier": tier,
                "description": description,
            }
        tied.append(decided)
    return {
        "decided_by": "candidate_id",
        "tied_components": tied,
        "tier": tier,
        "description": "all components tied; decided by candidate id",
    }


def _stable_seed(*parts: Any) -> int:
    """Deterministic non-negative seed from arbitrary role/context parts."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _state_origin_hash(state: Any) -> str:
    """Identify the complete constructed initial state, including effective traits."""
    if getattr(state, "scenario_metadata", None) is None:
        payload = (np.ascontiguousarray(state.W).tobytes()
                   + np.ascontiguousarray(state.y).tobytes()
                   + np.ascontiguousarray(state.x).tobytes())
        return hashlib.sha256(payload).hexdigest()[:12]
    digest = hashlib.sha256()
    for name in ("x", "y", "W", "n", "Q", "blocks", "K", "tau_x", "nu",
                 "g", "b", "row_caps", "opportunity_groups", "opportunity_matrix"):
        value = getattr(state, name, None)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    scalar = {"N": state.N, "D": state.D, "state_mode": str(state.state_mode),
              "t": state.t, "scenario_metadata": getattr(state, "scenario_metadata", None)}
    digest.update(json.dumps(scalar, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()[:12]


def _scenario_origin_descriptives(state: State) -> Dict[str, Any]:
    metadata = getattr(state, "scenario_metadata", None) or {}
    if metadata.get("source") != "tiered_constructed":
        return {}
    def grouped(labels: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for label in dict.fromkeys(labels):
            indices = [i for i, item in enumerate(labels) if item == label]
            output[str(label)] = {
                "n": len(indices),
                "public_plus_share": float(np.mean(state.y[indices, 0] > 0)),
                "private_plus_share": float(np.mean(state.x[indices, 0] > 0)),
                "public_mean": float(np.mean(state.y[indices, 0])),
                "private_mean": float(np.mean(state.x[indices, 0])),
            }
        return output
    core = np.asarray(metadata["tier_assignments"]) <= 3
    total = int(state.W.sum())
    return {
        "origin_source": metadata["source"],
        "t": float(state.t),
        "public_plus_share": float(np.mean(state.y[:, 0] > 0)),
        "private_plus_share": float(np.mean(state.x[:, 0] > 0)),
        "public_private_disagreement_share": float(np.mean(state.x[:, 0] != state.y[:, 0])),
        "core_attention_share_realized": float(state.W[:, core].sum() / total) if total else None,
        "core_attention_share_target": metadata["attention_target_core_share"],
        "tier": grouped(metadata["tier_assignments"]),
        "jurisdiction": grouped(metadata["jurisdictions"]),
        "role": grouped(metadata["roles"]),
        "encounter_matrix_nonzero_fraction": float(np.mean(state.opportunity_matrix > 0)) if state.opportunity_matrix is not None else None,
    }


def build_campaign_spec(profile: Any) -> ControlCampaignSpec:
    """Declare the target contract from a ControlProfile (contract blocks)."""
    return ControlCampaignSpec(
        target_type=str(profile.target_type),
        target_params={
            "threshold": float(profile.target_threshold),
            "component": getattr(profile, "target_component", 0),
            "eligibility_mode": getattr(profile, "eligibility_mode", "legacy_strong")
        },
        measured_observables=["public_means", "private_means"],
        tolerances={"alignment": float(profile.target_threshold)},
        holding_duration=float(profile.holding_duration),
        horizon=float(profile.horizon),
        release_deadline=float(profile.release_deadline),
        reliability_target=float(profile.reliability_target),
        task_information=(
            f"{profile.target_type} targeting opposite of the initial orientation; "
            "thresholds, horizon, and persistence duration declared before optimization"
        ),
    )


_PREPARATION_PROFILE_FIELDS = (
    "N", "D", "state_mode", "establishment_time", "seed_base",
    "J", "K_nominal", "tau_public", "tau_private_nominal",
    "temperature", "nu_nominal", "delta", "exploration",
    "prestige_global", "matching", "closure", "memory_capacity_per_row",
    "sample_interval", "animation_interval",
)


def _origin_source_identity(profile: Any, origin_provider: Optional[Callable] = None) -> Dict[str, Any]:
    """Stable source identity used by snapshots, registry and manifest."""
    identity: Dict[str, Any] = {"mode": "scenario" if hasattr(profile, "scenario_spec_path") else "legacy"}
    if origin_provider is not None:
        identity["provider"] = f"{origin_provider.__module__}.{origin_provider.__qualname__}"
        import inspect
        source_file = inspect.getsourcefile(origin_provider)
        if source_file is not None:
            from pathlib import Path
            identity["provider_source_sha256"] = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if identity["mode"] == "scenario":
        from pathlib import Path
        path = Path(profile.scenario_spec_path)
        identity["scenario_spec_path"] = str(path)
        identity["scenario_spec_path_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _preparation_fingerprint(profile: Any, origin_provider: Optional[Callable] = None) -> str:
    """Identify origin settings and source contents, not merely the path names."""
    settings = {name: getattr(profile, name) for name in _PREPARATION_PROFILE_FIELDS}
    is_scenario = hasattr(profile, "scenario_spec_path")
    settings["snapshot_schema"] = 4 if is_scenario else 2
    if is_scenario or origin_provider is not None:
        settings["origin_source"] = _origin_source_identity(profile, origin_provider)
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _validate_scenario_state(state: State, profile: Any) -> None:
    """Reject a provider or cached snapshot from another scenario contract."""
    if not hasattr(profile, "scenario_spec_path"):
        return
    from src.config.scenarios import load_tiered_spec
    from pathlib import Path
    spec = load_tiered_spec(profile.scenario_spec_path)
    expected_spec = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = getattr(state, "scenario_metadata", None) or {}
    if (metadata.get("source") != "tiered_constructed"
            or metadata.get("spec_sha256") != expected_spec
            or state.t != 0.0 or state.N != int(profile.N) or state.D != int(profile.D)):
        raise ValueError("Origin provider or snapshot does not match the declared Tiered-population scenario and t=0 contract")


def save_origin_snapshot(
    state: Any, path: str, *, profile_fingerprint: Optional[str] = None,
    origin_seed: Optional[int] = None,
) -> str:
    """Save one complete established origin with optional provenance metadata."""
    payload: Dict[str, Any] = {
        "N": np.int64(state.N),
        "D": np.int64(state.D),
        "state_mode": np.array(str(getattr(state.state_mode, "value", state.state_mode))),
        "t": np.float64(state.t),
        "x": np.asarray(state.x),
        "y": np.asarray(state.y),
        "W": np.asarray(state.W),
        "n": np.asarray(state.n),
        "Q": np.asarray(state.Q),
    }
    for name in ("K", "tau_x", "nu", "g", "b", "row_caps", "opportunity_groups", "opportunity_matrix", "blocks"):
        value = getattr(state, name)
        payload[name] = np.asarray(value) if value is not None else np.array(np.nan)
    if getattr(state, "scenario_metadata", None) is not None:
        payload["scenario_metadata_json"] = np.array(json.dumps(state.scenario_metadata, sort_keys=True))
    if profile_fingerprint is not None:
        payload["preparation_fingerprint"] = np.array(profile_fingerprint)
    if origin_seed is not None:
        payload["origin_seed"] = np.int64(origin_seed)
    np.savez_compressed(path, **payload)
    return path


def load_origin_snapshot(path: str) -> Any:
    """Reconstruct a saved established origin state."""
    from src.core.state import State

    data = np.load(path, allow_pickle=False)
    trait_arrays: Dict[str, Any] = {}
    for name in ("K", "tau_x", "nu", "g", "b", "row_caps", "opportunity_groups", "opportunity_matrix", "blocks"):
        if name not in data:
            trait_arrays[name] = None
            continue
        arr = data[name]
        trait_arrays[name] = None if (arr.ndim == 0 and np.issubdtype(arr.dtype, np.floating) and np.isnan(arr)) else arr
    metadata = json.loads(str(data["scenario_metadata_json"])) if "scenario_metadata_json" in data else None
    return State(
        N=int(data["N"]),
        D=int(data["D"]),
        state_mode=str(data["state_mode"]),
        x=data["x"],
        y=data["y"],
        W=data["W"],
        n=data["n"],
        Q=data["Q"],
        t=float(data["t"]),
        scenario_metadata=metadata,
        **trait_arrays,
    )


def prepare_origins(
    profile: Any,
    snapshot_dir: Optional[str] = None,
    progress: bool = False,
    registry: Optional[Any] = None,
    origin_provider: Optional[Callable[[Any, int], State]] = None,
) -> CampaignPreparation:
    """Prepare the declared ensemble of established origins once.

    Each origin is established independently via build_realism_runner with a
    deterministic per-origin seed, saved completely (state + traits), checked
    for declared-target eligibility, and split into disjoint training /
    selection / evaluation sets. Establishment is preparation, not research.
    """
    from src.config.control_profiles import preflight_control_profile
    from src.config.profiles import build_realism_runner
    from src.experiment.baselines.ensembles import ReferenceOrigin
    from src.experiment.ensembles import derive_origin_seed
    import os
    from pathlib import Path

    # Cheap allocation/geometry preflight before any simulation. This also
    # catches invalid model_copy(update=...) profiles that bypass Pydantic
    # validation (e.g. an extended allocation with too few prepared origins).
    preflight_control_profile(profile)
    is_scenario = hasattr(profile, "scenario_spec_path")
    if is_scenario and origin_provider is None:
        raise ValueError("Tiered-population scenario campaign requires an explicit scenario origin_provider")

    spec = build_campaign_spec(profile)
    n_prepare = int(profile.n_origins)
    n_needed = (
        int(profile.training_origins)
        + int(profile.selection_origins)
        + int(profile.evaluation_origins)
    )
    if n_needed > n_prepare:
        raise ValueError(
            f"profile allocation requires {n_needed} origins "
            f"(training+selection+evaluation) but n_origins={n_prepare}"
        )

    ensemble_id = f"prepared_{profile.topic}_{profile.tier}_seed{profile.seed_base}"
    preparation_fingerprint = _preparation_fingerprint(profile, origin_provider)
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
    prepared: List[PreparedOrigin] = []
    origins: List[Any] = []
    attempt_records: List[PreparationAttempt] = []

    max_attempts = n_prepare * 10
    attempts = 0
    idx = 0
    
    while len(prepared) < n_prepare:
        if attempts >= max_attempts:
            partial = preparation_attempt_summary(attempt_records)
            diagnostics = {
                "n_prepare_target": n_prepare,
                "n_prepared": len(prepared),
                "attempt_cap": max_attempts,
                "attempts_used": attempts,
                "attempts": [a.to_record() for a in attempt_records],
                "partial_attempt_summary": partial,
                "interpretation": (
                    "Insufficient eligible origins were found within the declared "
                    "attempt cap. This is NOT proof that reversal is unreachable; "
                    "it bounds only this preparation procedure at this cap."
                ),
            }
            raise PreparationExhausted(
                f"Preparation exhausted the attempt cap ({max_attempts}) with only "
                f"{len(prepared)}/{n_prepare} eligible origins established. "
                f"Accepted {partial['n_accepted']} of {partial['n_attempted']} attempts. "
                "This means insufficient eligible origins were found within the cap — "
                "it is NOT evidence that the declared target is structurally unreachable.",
                diagnostics=diagnostics,
            )

        seed = derive_origin_seed(profile.seed_base, idx)
        candidate_snap_file = None
        if snapshot_dir is not None:
            candidate_snap_file = (
                Path(snapshot_dir) / f"origin_{idx:03d}_{preparation_fingerprint}.npz"
            )

        # Check if an accepted snapshot is already available on disk from previous execution
        if candidate_snap_file is not None and candidate_snap_file.is_file():
            with np.load(candidate_snap_file, allow_pickle=False) as snapshot:
                if (
                    "preparation_fingerprint" not in snapshot
                    or str(snapshot["preparation_fingerprint"]) != preparation_fingerprint
                    or "origin_seed" not in snapshot
                    or int(snapshot["origin_seed"]) != seed
                ):
                    raise ValueError(
                        f"Origin snapshot {candidate_snap_file} has incompatible "
                        "preparation settings or seed; keep it for inspection "
                        "and use a fresh snapshot directory."
                    )
            state = load_origin_snapshot(str(candidate_snap_file))
            if state.N != int(profile.N) or state.D != int(profile.D):
                raise ValueError(
                    f"Origin snapshot {candidate_snap_file} has incompatible dimensions"
                )
            _validate_scenario_state(state, profile)
            elapsed_prep = 0.0
            snapshot_path = str(candidate_snap_file)
            m_y0 = float(np.mean(state.y[:, 0]))
            m_x0 = float(np.mean(state.x[:, 0]))
            evaluator = TargetEvaluator.create_for_state(
                target_type=TargetType(spec.target_type),
                initial_state=state,
                holding_duration=spec.holding_duration,
                horizon=spec.horizon,
                release_deadline=spec.release_deadline,
                target_params=spec.target_params,
            )
            is_eligible = bool(evaluator.defn.is_eligible)
            reason = evaluator.defn.ineligibility_reason or "eligible"
        else:
            t0_origin = _time.perf_counter()
            if origin_provider is not None:
                state = origin_provider(profile, seed)
            else:
                runner = build_realism_runner(profile, origin_seed=seed)
                traj = runner.run(horizon=profile.establishment_time)
                state = traj.final_state
            _validate_scenario_state(state, profile)
            elapsed_prep = _time.perf_counter() - t0_origin
            m_y0 = float(np.mean(state.y[:, 0]))
            m_x0 = float(np.mean(state.x[:, 0]))

            evaluator = TargetEvaluator.create_for_state(
                target_type=TargetType(spec.target_type),
                initial_state=state,
                holding_duration=spec.holding_duration,
                horizon=spec.horizon,
                release_deadline=spec.release_deadline,
                target_params=spec.target_params,
            )
            is_eligible = bool(evaluator.defn.is_eligible)
            reason = evaluator.defn.ineligibility_reason or "eligible"

            snapshot_path = ""
            if (is_eligible or is_scenario) and snapshot_dir is not None:
                snapshot_path = save_origin_snapshot(
                    state, str(candidate_snap_file),
                    profile_fingerprint=preparation_fingerprint, origin_seed=seed,
                )

        if not is_eligible and not is_scenario:
            attempt_records.append(PreparationAttempt(
                index=idx, seed=seed, accepted=False, reason=reason,
                m_y0=m_y0, m_x0=m_x0,
            ))
            if registry is not None:
                registry.record_preparation_progress(
                    attempts_made=attempts + 1,
                    accepted_count=len(prepared),
                    rejected_count=(attempts + 1 - len(prepared)),
                    remaining_units=max(0, n_prepare - len(prepared)),
                )
            idx += 1
            attempts += 1
            continue

        origin_id = f"origin_{idx:03d}_{_state_origin_hash(state)}"
        traits = {
            "index": idx,
            "m_y0": m_y0,
            "m_x0": m_x0,
            "total_memory": int(state.W.sum()),
            "public_majority_margin": abs(m_y0),
            "is_eligible": is_eligible,
            "scenario_descriptives": _scenario_origin_descriptives(state),
        }

        prepared.append(PreparedOrigin(
            origin_id=origin_id,
            seed=seed,
            snapshot_path=snapshot_path,
            traits=traits,
            is_eligible=is_eligible,
            eligibility_reason=reason,
        ))
        attempt_records.append(PreparationAttempt(
            index=idx, seed=seed, accepted=True, reason=reason,
            m_y0=m_y0, m_x0=m_x0, origin_id=origin_id, snapshot_path=snapshot_path,
        ))
        origins.append(ReferenceOrigin(
            origin_id=origin_id,
            ensemble_id=ensemble_id,
            source="tiered_constructed" if is_scenario else "prepared_establishment",
            state=state,
            metadata={"index": idx, "seed": seed, "snapshot_path": snapshot_path},
            between_eligibility=float((getattr(state, "scenario_metadata", None) or {}).get("opportunity", {}).get("between", 0.05)),
            within_eligibility=float((getattr(state, "scenario_metadata", None) or {}).get("opportunity", {}).get("within", 1.0)),
        ))
        if registry is not None:
            snap_hash = hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest() if (snapshot_path and Path(snapshot_path).is_file()) else ""
            registry.record_snapshot(origin_id, seed, idx, snapshot_path, snap_hash, traits, is_eligible, elapsed_seconds=elapsed_prep)
            registry.record_preparation_progress(
                attempts_made=attempts + 1,
                accepted_count=len(prepared),
                rejected_count=(attempts + 1 - len(prepared)),
                remaining_units=max(0, n_prepare - len(prepared)),
            )

        if progress:
            print(
                f"  origin {idx}: seed={seed}, m_y={prepared[-1].traits['m_y0']:+.3f}, "
                f"eligible={is_eligible} (attempt {attempts + 1})"
            )
            
        idx += 1
        attempts += 1

    n_train = int(profile.training_origins)
    n_sel = int(profile.selection_origins)
    origins_split = {
        "training": origins[:n_train],
        "selection": origins[n_train:n_train + n_sel],
        "evaluation": origins[n_train + n_sel:n_needed],
    }
    # Explicit pool membership per accepted origin (declared, persisted).
    pools: Dict[str, Optional[str]] = {}
    for pos, p in enumerate(prepared):
        if pos < n_train:
            pools[p.origin_id] = "training"
        elif pos < n_train + n_sel:
            pools[p.origin_id] = "selection"
        elif pos < n_needed:
            pools[p.origin_id] = "evaluation"
        else:
            pools[p.origin_id] = None  # retained but not assigned to a campaign pool
    for a in attempt_records:
        if a.accepted:
            a.pool = pools.get(a.origin_id)

    attempt_counts = preparation_attempt_summary(attempt_records)
    n_assigned = sum(1 for pool in pools.values() if pool is not None)
    summary = {
        "ensemble_id": ensemble_id,
        "n_prepared": len(prepared),
        "n_eligible": sum(p.is_eligible for p in prepared),
        "n_ineligible": sum(not p.is_eligible for p in prepared),
        **attempt_counts,
        "n_retained": len(prepared),
        "n_assigned": n_assigned,
        "eligibility_yield_definition": "eligible/constructed" if is_scenario else "accepted/attempted (rejected attempts included)",
        "eligibility_rate": sum(p.is_eligible for p in prepared) / len(prepared) if prepared else 0.0,
        "establishment_time": float(profile.establishment_time),
        "saved_snapshots": sum(1 for p in prepared if p.snapshot_path),
        "origin_ids": [p.origin_id for p in prepared],
        "accepted_origin_seeds": [p.seed for p in prepared],
        "accepted_snapshot_paths": [p.snapshot_path for p in prepared],
        "pool_membership": {
            "training": [p.origin_id for p in prepared[:n_train]],
            "selection": [p.origin_id for p in prepared[n_train:n_train + n_sel]],
            "evaluation": [p.origin_id for p in prepared[n_train + n_sel:n_needed]],
            "unassigned_spare": [p.origin_id for p in prepared[n_needed:]],
        },
        "attempts": [a.to_record() for a in attempt_records],
        "eligibility": [
            {"origin_id": p.origin_id, "is_eligible": p.is_eligible, "reason": p.eligibility_reason}
            for p in prepared
        ],
        "origin_descriptives": {p.origin_id: p.traits.get("scenario_descriptives") for p in prepared if p.traits.get("scenario_descriptives")},
    }
    return CampaignPreparation(
        prepared=prepared,
        origins=origins,
        training=origins_split["training"],
        selection=origins_split["selection"],
        evaluation=origins_split["evaluation"],
        summary=summary,
        attempts=attempt_records,
    )


def _campaign_geometry(profile: Any, N: int) -> Dict[str, Any]:
    """Derive the declared intervention geometry (roster, k, caps) for a profile.

    Explicit conventions (pilot assumptions, editable in the profile):
    - k = round(k_distinct_nodes_fraction * N), at least 1.
    - amplitude_cap = amplitude_cap_over_J * J.
    - exposure_cap = exposure_cap_normalized * N * amplitude_cap.
    - accessible roster: round(accessible_fraction * N) nodes drawn from a
      deterministic master-seeded RNG (declared once, fixed across origins).
    """
    J = float(profile.J)
    k = max(1, int(round(float(profile.k_distinct_nodes_fraction) * N)))
    amplitude_cap = float(profile.amplitude_cap_over_J) * J
    exposure_cap = float(profile.exposure_cap_normalized) * N * amplitude_cap
    rng = np.random.default_rng(_stable_seed(profile.seed_base, "accessible_roster"))
    n_acc = max(1, int(round(float(profile.accessible_fraction) * N)))
    roster = tuple(sorted(rng.choice(N, size=min(n_acc, N), replace=False).tolist()))
    return {
        "k": k,
        "amplitude_cap": float(amplitude_cap),
        "exposure_cap": float(exposure_cap),
        "accessible_roster": roster,
    }


def _run_observable_rollouts(
    controller_factory: Callable[[], Any],
    problem: ControlProblem,
    origins: Sequence[Any],
    futures_per_origin: int,
    policy_id: str,
    dynamics: str,
    sim_seed_fn: Callable[[str, int], int],
    ctl_seed_fn: Callable[[str, int], int],
    physics: Dict[str, Any],
    obs_kwargs: Dict[str, Any],
    recorder_factory: Optional[Callable[[str, str, int, str], Any]] = None,
    phase: str = "baselines",
    registry: Optional[Any] = None,
) -> List[RolloutResult]:
    """Execute one policy across origins/futures under restricted observations.

    The simulator and controller RNG streams are strictly independent; seeds
    are deterministic functions of (role, origin_id, future_id) so adaptive and
    frozen branches — and different policies — share matched seeds per
    origin/future (paired comparisons, matched observation panels).
    """
    from src.control.engine import run_controlled_simulation
    from src.control.observation import SampledPublicObservationModel

    results: List[RolloutResult] = []
    for origin in origins:
        for future_id in range(futures_per_origin):
            # Check registry cache for previously completed rollout
            if registry is not None and registry.is_completed(phase, policy_id, origin.origin_id, future_id, dynamics):
                cached_res = registry.get_rollout(phase, policy_id, origin.origin_id, future_id, dynamics)
                if cached_res is not None:
                    results.append(cached_res)
                    continue

            t_unit_start = _time.perf_counter()
            sim_seed = int(sim_seed_fn(origin.origin_id, future_id))
            ctl_seed = int(ctl_seed_fn(origin.origin_id, future_id))

            controller = controller_factory()
            observation_model = SampledPublicObservationModel(
                N=problem.N,
                accessible_roster=problem.accessible_roster,
                rng=np.random.default_rng(ctl_seed),
                **obs_kwargs,
            )
            evaluator = TargetEvaluator.create_for_state(
                target_type=problem.target_type,
                initial_state=origin.state,
                holding_duration=problem.holding_duration,
                horizon=problem.horizon,
                release_deadline=problem.release_deadline,
                problem_id=problem.problem_id,
                policy_id=policy_id,
                origin_id=origin.origin_id,
                future_id=future_id,
                target_params=problem.target_params,
            )
            rec = None
            if recorder_factory is not None:
                rec = recorder_factory(policy_id, origin.origin_id, future_id, dynamics)

            res, _, _ = run_controlled_simulation(
                initial_state=origin.branch(),
                problem=problem,
                controller=controller,
                observation_model=observation_model,
                target_evaluator=evaluator,
                J=physics["J"],
                T=physics["T"],
                delta=physics["delta"],
                epsilon=physics["epsilon"],
                closure=physics["closure"],
                sample_interval=physics["sample_interval"],
                adaptive_memory=(dynamics == "adaptive"),
                between_eligibility=origin.between_eligibility,
                within_eligibility=origin.within_eligibility,
                a=physics["a"],
                c=physics["c"],
                tau_y=physics["tau_y"],
                sim_rng=np.random.default_rng(sim_seed),
                controller_rng=np.random.default_rng(ctl_seed),
                recorder=rec,
            )
            res.summaries.update({
                "policy_id": policy_id,
                "dynamics": dynamics,
                "sim_seed": sim_seed,
                "ctl_seed": ctl_seed,
            })
            # F1c: report initial / peak-active / cumulative-distinct node
            # counts separately instead of only the initial deployment.
            res.summaries["node_count_decomposition"] = summarize_node_counts(res)
            elapsed_unit = _time.perf_counter() - t_unit_start
            if registry is not None:
                registry.record_rollout(phase, policy_id, origin.origin_id, future_id, dynamics, res, elapsed_unit)
            results.append(res)
    return results


def summarize_rollout_reliability(
    results: Sequence[RolloutResult],
    reliability_target: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Aggregate rollouts into a reliability summary (eligible trials only).

    Uses the reference statistics: Wilson interval and one-sided exact
    Clopper-Pearson lower bound for the success probability.
    """
    from src.experiment.baselines.reference import (
        exact_binomial_lower_bound,
        wilson_interval,
    )

    valid = [r for r in results if r.is_valid]
    eligible = [r for r in valid if r.is_eligible]
    trials = len(eligible)
    successes = sum(bool(r.success) for r in eligible)
    failures = trials - successes
    p_hat = successes / trials if trials else 0.0

    # Origin clustering check
    by_origin: Dict[str, List[RolloutResult]] = {}
    for r in eligible:
        by_origin.setdefault(r.origin_id, []).append(r)
    n_origins = len(by_origin)
    max_futures_per_origin = max(len(v) for v in by_origin.values()) if by_origin else 0
    if max_futures_per_origin > 1:
        p_hat = float(np.mean([
            np.mean([bool(r.success) for r in rs]) for rs in by_origin.values()
        ]))
    certification_method = "exact_binomial" if max_futures_per_origin <= 1 else "origin_hoeffding"

    # F1d: origin-level success counts (uncertainty summary), not a point
    # estimate alone. Origin-level counts make the small-sample limitation
    # explicit (e.g. 0/2 never supports an impossibility claim) and expose
    # which independent origins carried successes.
    origin_level_success_counts: Dict[str, Dict[str, int]] = {
        origin_id: {
            "trials": len(rs),
            "successes": sum(bool(r.success) for r in rs),
            "eligible_trials": len(rs),
        }
        for origin_id, rs in sorted(by_origin.items())
    }
    n_origins_with_success = sum(
        1 for counts in origin_level_success_counts.values() if counts["successes"] > 0
    )

    if trials == 0:
        low, high, exact_low = float("nan"), float("nan"), float("nan")
        uncertainty_method = "no_eligible_trials"
    elif max_futures_per_origin <= 1:
        # Simple binomial intervals for one future per independent origin
        low, high = wilson_interval(successes, trials)
        exact_low = exact_binomial_lower_bound(successes, trials, alpha)
        uncertainty_method = "binomial_exact_and_wilson"
    else:
        # Bootstrap percentiles describe variability. Certification instead uses
        # an independent-origin bounded-mean inequality.
        origin_means = np.array([
            np.mean([bool(r.success) for r in rs]) for rs in by_origin.values()
        ], dtype=float)
        boot_rng = np.random.default_rng(42)
        p_boot = np.mean(
            boot_rng.choice(origin_means, size=(2000, n_origins), replace=True), axis=1
        )
        low = float(np.percentile(p_boot, 100 * (alpha / 2)))
        high = float(np.percentile(p_boot, 100 * (1.0 - alpha / 2)))
        exact_low = max(0.0, p_hat - float(np.sqrt(np.log(1.0 / alpha) / (2 * n_origins))))
        uncertainty_method = "cluster_bootstrap_origin_level"

    mean_time = (
        float(np.mean([r.restricted_time for r in eligible])) if eligible else float("nan")
    )
    # Restricted time labeling: without successes every eligible rollout was
    # censored at the horizon, so mean_restricted_time is the failure penalty
    # (H), not a measured completion time. Success-conditional completion time
    # is reported separately and is unavailable when there are no successes.
    success_times = [float(r.restricted_time) for r in eligible if r.success]
    mean_time_on_successes = (
        float(np.mean(success_times)) if success_times else None
    )
    if trials == 0:
        time_interpretation = "no_eligible_trials"
    elif successes == 0:
        time_interpretation = "failure_penalty_horizon"
    elif failures == 0:
        time_interpretation = "successful_completions"
    else:
        time_interpretation = "mixed_completion_and_failure_penalty"
    mean_exposure = (
        float(np.mean([r.expenditure for r in eligible])) if eligible else float("nan")
    )
    sustained_successes = sum(bool(r.success and not r.relapsed) for r in eligible)
    sustained_probability = sustained_successes / trials if trials else 0.0
    if max_futures_per_origin > 1:
        sustained_probability = float(np.mean([
            np.mean([bool(r.success and not r.relapsed) for r in rs])
            for rs in by_origin.values()
        ]))
    if trials and max_futures_per_origin <= 1:
        sustained_low, sustained_high = wilson_interval(sustained_successes, trials)
        sustained_exact_low = exact_binomial_lower_bound(sustained_successes, trials, alpha)
    elif trials > 0:
        origin_means = np.array([
            np.mean([bool(r.success and not r.relapsed) for r in rs])
            for rs in by_origin.values()
        ], dtype=float)
        boot_rng = np.random.default_rng(42)
        p_boot = np.mean(
            boot_rng.choice(origin_means, size=(2000, n_origins), replace=True), axis=1
        )
        sustained_low = float(np.percentile(p_boot, 100 * (alpha / 2)))
        sustained_high = float(np.percentile(p_boot, 100 * (1.0 - alpha / 2)))
        sustained_exact_low = max(
            0.0, sustained_probability - float(np.sqrt(np.log(1.0 / alpha) / (2 * n_origins)))
        )
    else:
        sustained_low = sustained_high = sustained_exact_low = float("nan")
    sustained_status = classify_reliability_status(
        sustained_successes, trials, sustained_probability, sustained_exact_low,
        float(reliability_target),
    )
    relapsed_count = sum(bool(r.relapsed) for r in eligible)
    relapse_rate = (
        float(np.mean([bool(r.relapsed) for r in eligible])) if eligible else float("nan")
    )
    # Relapse conditional on certified successes: relapse can only follow a
    # success, so the eligible denominator for the conditional rate is the
    # number of certified successes. Zero successes => N/A (None).
    relapse_conditional = (
        relapsed_count / successes if successes else None
    )
    status = classify_reliability_status(
        successes, trials, p_hat, exact_low, float(reliability_target)
    )
    return {
        "policy_ids": sorted({r.policy_id for r in results}),
        "total_rollouts": len(results),
        "eligible_trials": trials,
        "ineligible_rollouts": len(valid) - trials,
        "successes": successes,
        "failures": failures,
        "failure_fraction": (failures / trials) if trials else float("nan"),
        "success_probability": p_hat,
        "sustained_successes": sustained_successes,
        "sustained_success_probability": sustained_probability,
        "sustained_wilson_low": sustained_low,
        "sustained_wilson_high": sustained_high,
        "sustained_status": sustained_status,
        "sustained_outcome_definition": "certified reversal with no later relapse through horizon H",
        "wilson_low": low,
        "wilson_high": high,
        "descriptive_interval_low": low,
        "descriptive_interval_high": high,
        "exact_one_sided_lower": exact_low if max_futures_per_origin <= 1 else None,
        "certification_lower_bound": exact_low,
        "sustained_certification_lower_bound": sustained_exact_low,
        "certification_method": certification_method,
        "estimand": "equal-origin mean future success probability" if max_futures_per_origin > 1 else "independent-origin success probability",
        "uncertainty_method": uncertainty_method,
        "n_origins": n_origins,
        "max_futures_per_origin": max_futures_per_origin,
        # F1d: origin-level uncertainty summary, not a point estimate alone.
        "origin_level_success_counts": origin_level_success_counts,
        "n_origins_with_success": n_origins_with_success,
        "n_origins_without_success": n_origins - n_origins_with_success,
        "origin_level_interpretation": (
            "Success counts per independent origin; small origin counts make "
            "all-failure sets uninformative (0/2 is not evidence of impossibility)."
        ),
        "reliability_target": reliability_target,
        "meets_point_target": bool(trials > 0 and p_hat >= reliability_target),
        "status": status,
        "mean_restricted_time": mean_time,
        "restricted_time_interpretation": time_interpretation,
        "mean_restricted_time_on_successes": mean_time_on_successes,
        "success_conditional_completion_time_available": bool(successes),
        "mean_expenditure": mean_exposure,
        "relapse_count": relapsed_count,
        "relapse_rate": relapse_rate,
        "relapse_numerator": relapsed_count,
        "relapse_denominator": trials,
        "relapse_conditional_numerator": relapsed_count,
        "relapse_conditional_denominator": successes,
        "relapse_rate_conditional_on_success": relapse_conditional,
        "relapse_conditional_available": bool(successes),
    }


def summarize_paired_adaptive_frozen_comparison(
    pairs: Sequence[Dict[str, Any]],
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired differences and origin-clustered uncertainty for adaptive vs frozen branches.

    Resamples origin clusters with replacement, preserving pairing of futures
    between adaptive and frozen branches within each resampled origin.
    """
    if len(pairs) == 0:
        return {"n_pairs": 0, "status": "no_pairs"}

    # Group pairs by origin_id
    by_origin: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_origin.setdefault(p["origin_id"], []).append(p)

    origin_keys = list(by_origin.keys())
    n_origins = len(origin_keys)

    # Compute observed paired differences
    diff_succ_list: List[float] = []
    diff_time_list: List[float] = []
    diff_exp_list: List[float] = []

    for p in pairs:
        ra, rf = p.get("adaptive"), p.get("frozen")
        if ra is not None and rf is not None and ra.is_eligible and rf.is_eligible:
            diff_succ_list.append(float(bool(ra.success)) - float(bool(rf.success)))
            diff_time_list.append(float(ra.restricted_time) - float(rf.restricted_time))
            diff_exp_list.append(float(ra.expenditure) - float(rf.expenditure))

    n_valid_pairs = len(diff_succ_list)
    if n_valid_pairs == 0:
        return {"n_pairs": len(pairs), "n_valid_pairs": 0, "status": "no_valid_pairs"}

    mean_diff_succ = float(np.mean(diff_succ_list))
    mean_diff_time = float(np.mean(diff_time_list))
    mean_diff_exp = float(np.mean(diff_exp_list))

    # Cluster bootstrap resampling origins
    boot_rng = np.random.default_rng(seed)
    boot_diff_succ = np.empty(n_boot, dtype=np.float64)
    boot_diff_time = np.empty(n_boot, dtype=np.float64)
    boot_diff_exp = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_origins = boot_rng.choice(origin_keys, size=n_origins, replace=True)
        b_succ: List[float] = []
        b_time: List[float] = []
        b_exp: List[float] = []
        for o_key in sampled_origins:
            for p in by_origin[o_key]:
                ra, rf = p.get("adaptive"), p.get("frozen")
                if ra is not None and rf is not None and ra.is_eligible and rf.is_eligible:
                    b_succ.append(float(bool(ra.success)) - float(bool(rf.success)))
                    b_time.append(float(ra.restricted_time) - float(rf.restricted_time))
                    b_exp.append(float(ra.expenditure) - float(rf.expenditure))
        boot_diff_succ[b] = float(np.mean(b_succ)) if b_succ else 0.0
        boot_diff_time[b] = float(np.mean(b_time)) if b_time else 0.0
        boot_diff_exp[b] = float(np.mean(b_exp)) if b_exp else 0.0

    return {
        "n_pairs": len(pairs),
        "n_valid_pairs": n_valid_pairs,
        "n_origins": n_origins,
        "mean_difference_success": mean_diff_succ,
        "ci_success": (
            float(np.percentile(boot_diff_succ, 100 * (alpha / 2))),
            float(np.percentile(boot_diff_succ, 100 * (1.0 - alpha / 2))),
        ),
        "mean_difference_time": mean_diff_time,
        "ci_time": (
            float(np.percentile(boot_diff_time, 100 * (alpha / 2))),
            float(np.percentile(boot_diff_time, 100 * (1.0 - alpha / 2))),
        ),
        "mean_difference_expenditure": mean_diff_exp,
        "ci_expenditure": (
            float(np.percentile(boot_diff_exp, 100 * (alpha / 2))),
            float(np.percentile(boot_diff_exp, 100 * (1.0 - alpha / 2))),
        ),
        "uncertainty_method": "cluster_bootstrap_paired_origins",
    }


def run_campaign(
    profile: Any,
    campaign_id: str = "control_campaign",
    master_seed: int = 99001,
    out_dir: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    progress: bool = False,
    record_cadence: float = 1.0,
    resume: bool = True,
    origin_provider: Optional[Callable[[Any, int], State]] = None,
) -> ControlCampaignResult:
    """Run the full campaign pipeline at the profile's declared allocation.

    Preparation -> baselines -> observable search -> selection -> fresh-origin
    evaluation with adaptive/frozen pairing. When `out_dir` is given, one
    unified results bundle is established and exported via export_control_bundle.
    """
    from src.control.controllers import ObservableSelectorPolicy, ZeroController
    from src.control.recording import RecorderConfig, RolloutRecorder, evaluate_original_state_retention

    from src.config.control_profiles import preflight_control_profile

    # Allocation/geometry preflight before any simulation (catches invalid
    # model_copy(update=...) profiles that bypass Pydantic validation).
    preflight_control_profile(profile)
    if hasattr(profile, "scenario_spec_path") and origin_provider is None:
        raise ValueError("Tiered-population scenario campaign requires an explicit scenario origin_provider")
    origin_source = _origin_source_identity(profile, origin_provider)
    registry_profile = ({"profile": profile.to_manifest(), "origin_source": origin_source}
                        if hasattr(profile, "scenario_spec_path") else profile)

    # Scenario resumption must name one exact existing bundle, never guess which
    # of several historical campaigns to pick from a parent directory.
    if hasattr(profile, "scenario_spec_path") and resume and out_dir is not None:
        from pathlib import Path
        if not Path(out_dir).name.startswith(f"control_{campaign_id}_"):
            raise ValueError("Tiered-population resume requires the exact existing bundle path")
    if not resume and snapshot_dir is not None:
        from pathlib import Path
        if any(Path(snapshot_dir).glob("*.npz")):
            raise ValueError("Fresh campaign snapshot directory already contains origins; use an empty directory")

    # Establish unified bundle directory once at campaign start if out_dir given
    bundle_path = None
    if out_dir is not None:
        from src.io.results import make_campaign_bundle_dir
        bundle_path = make_campaign_bundle_dir(out_dir, campaign_id, resume=resume)
        if snapshot_dir is None:
            snapshot_dir = str(bundle_path / "snapshots")

    spec = build_campaign_spec(profile)

    # Initialize work registry for persistence and resumption
    registry = None
    if bundle_path is not None:
        from src.control.registry import CampaignWorkRegistry, _stable_hash
        if resume:
            registry = CampaignWorkRegistry.load_or_create(bundle_path, spec, registry_profile, verify_checksums=True)
            
            manifest_path = bundle_path / "manifest.json"
            if manifest_path.exists():
                import json
                from src.io.results import _executable_provenance
                from src.control.checkpointing import IncompatibleConfigurationError
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                prov = _executable_provenance()
                
                if "git_head" in manifest_data and manifest_data["git_head"] != prov.get("git_head"):
                    raise IncompatibleConfigurationError(
                        f"Resume rejected: git_head mismatch (bundle={manifest_data['git_head']}, current={prov.get('git_head')})"
                    )
                if "source_digest" in manifest_data and manifest_data["source_digest"] != prov.get("source_digest"):
                    raise IncompatibleConfigurationError(
                        f"Resume rejected: source_digest mismatch (bundle={manifest_data['source_digest']}, current={prov.get('source_digest')})"
                    )
        else:
            spec_h = _stable_hash(asdict(spec))
            prof_h = _stable_hash(registry_profile.to_manifest() if hasattr(registry_profile, "to_manifest") else registry_profile)
            registry = CampaignWorkRegistry(bundle_path, spec_hash=spec_h, profile_hash=prof_h)

    t_prep_start = _time.perf_counter()
    prep = prepare_origins(profile, snapshot_dir=snapshot_dir, progress=progress, registry=registry, origin_provider=origin_provider)
    if registry is not None:
        registry.record_phase_timing("preparation", _time.perf_counter() - t_prep_start, status="completed")

    geometry = _campaign_geometry(profile, int(profile.N))
    problem = ControlProblem(
        problem_id=campaign_id,
        target_type=TargetType(spec.target_type),
        target_params=dict(spec.target_params),
        N=int(profile.N),
        D=int(profile.D),
        k=geometry["k"],
        exposure_cap=geometry["exposure_cap"],
        amplitude_cap=geometry["amplitude_cap"],
        release_deadline=spec.release_deadline,
        holding_duration=spec.holding_duration,
        horizon=spec.horizon,
        accessible_roster=geometry["accessible_roster"],
        proposal_seed=_stable_seed(master_seed, "proposal"),
        search_seed=_stable_seed(master_seed, "search"),
        config_version="control-campaign-m4-1.0",
    )

    tau_y = float(profile.tau_public)
    physics = {
        "J": float(profile.J),
        "T": float(profile.temperature),
        "delta": float(profile.delta),
        "epsilon": float(profile.exploration),
        "closure": float(profile.closure),
        "a": float(profile.prestige_global),
        "c": float(profile.matching),
        "tau_y": tau_y,
        "sample_interval": float(profile.sample_interval_tau_public) * tau_y,
    }
    obs_kwargs = {
        "panel_fraction": float(profile.panel_fraction),
        "edge_reveal_prob": float(profile.edge_reveal_prob),
        "binary_flip_prob": float(profile.binary_flip_prob),
        "delivery_delay": float(profile.delivery_delay_tau_public) * tau_y,
    }
    sim_seed_fn = lambda oid, f: _stable_seed(master_seed, "sim", oid, f)  # noqa: E731
    ctl_seed_fn = lambda oid, f: _stable_seed(master_seed, "ctl", oid, f)  # noqa: E731

    # --- Baselines on the fresh evaluation origins (adaptive branch) ---
    baseline_amp = min(
        float(np.median(profile.nominal_amplitudes_over_J)) * profile.J,
        geometry["amplitude_cap"],
    )
    baseline_dur = min(
        float(np.median(profile.durations_tau_public)) * tau_y,
        spec.release_deadline,
    )
    default_node_frac = (
        float(profile.candidate_node_fractions[0])
        if hasattr(profile, "candidate_node_fractions") and profile.candidate_node_fractions
        else 0.1
    )
    # Matched comparator: baseline_attention and baseline_random share the exact same
    # accessible/observed pool (quota=1.0), roster, node count, amplitude, duration, budget, and timing.
    baseline_specs: List[Tuple[str, Callable[[], Any]]] = [
        ("baseline_zero", lambda: ZeroController()),
        ("baseline_random", lambda: ObservableSelectorPolicy(
            selector="random", nominal_amplitude=baseline_amp,
            duration=baseline_dur, quota=1.0, is_feedback=False, component=0,
            node_fraction=default_node_frac,
        )),
        ("baseline_attention", lambda: ObservableSelectorPolicy(
            selector="reported_attention", nominal_amplitude=baseline_amp,
            duration=baseline_dur, quota=1.0, is_feedback=False, component=0,
            node_fraction=default_node_frac,
        )),
    ]
    if getattr(profile, "include_all_accessible_baseline", False):
        # Labeled separately as an all-accessible random comparator
        baseline_specs.append((
            "baseline_random_all_accessible", lambda: ObservableSelectorPolicy(
                selector="random", nominal_amplitude=baseline_amp,
                duration=baseline_dur, quota=0.0, is_feedback=False, component=0,
                node_fraction=default_node_frac,
            ),
        ))
    cadence = float(record_cadence)
    rep_origin_id = prep.evaluation[0].origin_id if prep.evaluation else None

    def make_baseline_recorder(p_id: str, o_id: str, f_id: int, dyn: str) -> RolloutRecorder:
        is_rep = (p_id == "baseline_zero" and o_id == rep_origin_id and f_id == 0)
        cfg = RecorderConfig(
            cadence=cadence,
            record_scalar_trajectories=True,
            record_action_history=True,
            record_network_frames=is_rep,
            max_network_frames=100,
            retention_threshold=float(profile.target_threshold),
        )
        return RolloutRecorder(cfg)

    t_base_start = _time.perf_counter()
    baseline_results: Dict[str, List[RolloutResult]] = {}
    for name, factory in baseline_specs:
        baseline_results[name] = _run_observable_rollouts(
            controller_factory=factory,
            problem=problem,
            origins=prep.evaluation,
            futures_per_origin=int(profile.evaluation_futures_per_origin),
            policy_id=name,
            dynamics="adaptive",
            sim_seed_fn=sim_seed_fn,
            ctl_seed_fn=ctl_seed_fn,
            physics=physics,
            obs_kwargs=obs_kwargs,
            recorder_factory=make_baseline_recorder,
            phase="baselines",
            registry=registry,
        )
        if progress:
            print(f"  baseline done: {name} ({len(baseline_results[name])} rollouts)")
    if registry is not None:
        registry.record_phase_timing("baselines", _time.perf_counter() - t_base_start, status="completed")

    # --- Observable policy search over training origins (optimizer) ---
    t_srch_start = _time.perf_counter()
    optimizer = ObservablePolicySearchOptimizer(
        problem=problem,
        candidate_cap=int(profile.candidate_cap),
        futures_per_candidate=int(profile.training_futures_per_origin),
        p0=float(spec.reliability_target),
        search_seed=problem.search_seed,
        selectors=tuple(profile.selectors),
        observed_slot_quotas=tuple(profile.observed_slot_quotas),
        feedback_choices=tuple(profile.feedback_choices),
        nominal_amplitudes=[a * profile.J for a in profile.nominal_amplitudes_over_J],
        durations=[d * tau_y for d in profile.durations_tau_public],
        candidate_node_fractions=tuple(profile.candidate_node_fractions) if hasattr(profile, "candidate_node_fractions") else (0.05, 0.1),
    )
    search_result: OptimizationResult = optimizer.optimize(
        training_origins=prep.training,
        J=physics["J"],
        T=physics["T"],
        delta=physics["delta"],
        epsilon=physics["epsilon"],
        closure=physics["closure"],
        sample_interval=physics["sample_interval"],
        adaptive_memory=True,
        a=physics["a"],
        c=physics["c"],
        tau_y=physics["tau_y"],
        observation_model_kwargs=dict(obs_kwargs),
        registry=registry,
    )
    if registry is not None:
        registry.record_phase_timing("search", _time.perf_counter() - t_srch_start, status="completed")
    if progress:
        print(f"  search done: {search_result.total_rollouts} rollouts, "
              f"incumbent={search_result.incumbent.candidate_id}")

    # --- Selection: independently select top static and top feedback candidates ---
    t_sel_start = _time.perf_counter()
    static_cands = [
        (cid, score) for cid, score in search_result.candidate_scores.items()
        if not search_result.search_state.candidate_registry[cid].feedback
    ]
    feedback_cands = [
        (cid, score) for cid, score in search_result.candidate_scores.items()
        if search_result.search_state.candidate_registry[cid].feedback
    ]
    ranked_static = sorted(static_cands, key=lambda kv: kv[1])
    ranked_feedback = sorted(feedback_cands, key=lambda kv: kv[1])

    n_sel = max(1, int(profile.selection_candidates))
    top_static = ranked_static[:n_sel]
    top_feedback = ranked_feedback[:n_sel]

    candidates_to_evaluate: List[Tuple[str, str]] = []
    for cid, _ in top_static:
        if cid not in [c[0] for c in candidates_to_evaluate]:
            candidates_to_evaluate.append((cid, "static"))
    for cid, _ in top_feedback:
        if cid not in [c[0] for c in candidates_to_evaluate]:
            candidates_to_evaluate.append((cid, "feedback"))

    selection_scores: Dict[str, Tuple[Any, ...]] = {}
    selection_summaries: Dict[str, Dict[str, Any]] = {}
    selection_reliability: Dict[str, Dict[str, Any]] = {}
    for cid, _family in candidates_to_evaluate:
        candidate = search_result.search_state.candidate_registry[cid]
        rollouts = _run_observable_rollouts(
            controller_factory=candidate.to_controller,
            problem=problem,
            origins=prep.selection,
            futures_per_origin=int(profile.selection_futures_per_origin),
            policy_id=f"selection:{cid}",
            dynamics="adaptive",
            sim_seed_fn=sim_seed_fn,
            ctl_seed_fn=ctl_seed_fn,
            physics=physics,
            obs_kwargs=obs_kwargs,
            phase="selection",
            registry=registry,
        )
        score, summary = compute_candidate_score(
            candidate, rollouts, p0=float(spec.reliability_target)
        )
        selection_scores[cid] = score
        selection_summaries[cid] = summary
        selection_reliability[cid] = summarize_rollout_reliability(
            rollouts, float(spec.reliability_target)
        )
    if registry is not None:
        registry.record_phase_timing("selection", _time.perf_counter() - t_sel_start, status="completed")

    # Independent selection of family winners
    static_sel_ranked = sorted(
        [(cid, selection_scores[cid]) for cid, _ in top_static if cid in selection_scores],
        key=lambda kv: kv[1],
    )
    feedback_sel_ranked = sorted(
        [(cid, selection_scores[cid]) for cid, _ in top_feedback if cid in selection_scores],
        key=lambda kv: kv[1],
    )

    best_static_cid = static_sel_ranked[0][0] if static_sel_ranked else None
    best_feedback_cid = feedback_sel_ranked[0][0] if feedback_sel_ranked else None

    # Overall winner
    ranked_selection = sorted(selection_scores.items(), key=lambda kv: kv[1])
    best_cid = ranked_selection[0][0]
    runner_up_cid = ranked_selection[1][0] if len(ranked_selection) > 1 else None

    selected = search_result.search_state.candidate_registry[best_cid]
    selected_static = search_result.search_state.candidate_registry[best_static_cid] if best_static_cid else None
    selected_feedback = search_result.search_state.candidate_registry[best_feedback_cid] if best_feedback_cid else None

    # Honest rankings (audit F1b). ranked_static/ranked_feedback are SEPARATE
    # per-family rankings over the search (training-origin) scores; their
    # concatenation was previously exported under the misleading name
    # 'ranked_search_candidates'. Keep that key as a deprecated alias for
    # bundle-reader backward compatibility, and export the semantics plus the
    # per-family and true-global rankings explicitly.
    family_ranked = rank_candidates_by_family(
        search_result.candidate_scores, search_result.search_state.candidate_registry
    )
    global_ranked = rank_candidates_global(search_result.candidate_scores)

    selection_summary = {
        "search_incumbent": search_result.incumbent.candidate_id,
        "search_static_incumbent": getattr(search_result, "static_incumbent", None).candidate_id if getattr(search_result, "static_incumbent", None) else None,
        "search_feedback_incumbent": getattr(search_result, "feedback_incumbent", None).candidate_id if getattr(search_result, "feedback_incumbent", None) else None,
        # DEPRECATED alias: concatenation of separately ranked families,
        # NOT a global ranking. Retained only for bundle-reader compatibility.
        "ranked_search_candidates": [cid for cid, _ in ranked_static + ranked_feedback],
        "ranking_semantics": (
            "ranked_search_candidates is the concatenation of separately "
            "ranked static and feedback family rankings (training-origin "
            "scores), NOT a global ranking. See family_ranked_candidates for "
            "the per-family orderings and global_ranked_candidates for a "
            "true cross-family ordering."
        ),
        "family_ranked_candidates": {
            fam: [cid for cid, _ in ranked]
            for fam, ranked in family_ranked.items()
        },
        "family_ranked_scores": {
            fam: {cid: list(score) for cid, score in ranked}
            for fam, ranked in family_ranked.items()
        },
        "global_ranked_candidates": [cid for cid, _ in global_ranked],
        "search_coverage": getattr(search_result, "search_coverage", None),
        "phase_pools": {
            "training": {
                "origin_ids": [o.origin_id for o in prep.training],
                "futures_per_origin": int(profile.training_futures_per_origin),
            },
            "selection": {
                "origin_ids": [o.origin_id for o in prep.selection],
                "futures_per_origin": int(profile.selection_futures_per_origin),
            },
            "evaluation": {
                "origin_ids": [o.origin_id for o in prep.evaluation],
                "futures_per_origin": int(profile.evaluation_futures_per_origin),
            },
        },
        "selection_candidate_ids": list(selection_scores.keys()),
        "selection_scores": {cid: list(score) for cid, score in selection_scores.items()},
        "selection_summaries": selection_summaries,
        "selection_reliability": selection_reliability,
        "selected_candidate_id": best_cid,
        "selected_static_candidate_id": best_static_cid,
        "selected_feedback_candidate_id": best_feedback_cid,
        "status": selection_reliability[best_cid]["sustained_status"],
        "certification_status": selection_reliability[best_cid]["status"],
        "ranking_outcome": "certified_and_no_relapse_through_horizon",
        "static_status": selection_reliability[best_static_cid]["sustained_status"] if best_static_cid else None,
        "feedback_status": selection_reliability[best_feedback_cid]["sustained_status"] if best_feedback_cid else None,
        "status_definitions": {
            "no_feasible_policy_found": "zero successes among eligible trials",
            "best_failing_candidate": "successes exist but below the reliability target",
            "estimated_feasible": "point estimate meets the target; lower bound does not",
            "reliability_certified": "declared independent-origin one-sided lower bound meets the target",
        },
        "runner_up_candidate_id": runner_up_cid,
        "selection_tie_break": describe_selection_tie_break(
            ranked_selection[0][1],
            ranked_selection[1][1] if len(ranked_selection) > 1 else None,
        ),
        "selected_candidate_params": {
            "selector": selected.selector,
            "node_fraction": getattr(selected, "node_fraction", None),
            "observed_slot_quota": selected.observed_slot_quota,
            "feedback": selected.feedback,
            "initial_recruitment_fraction": selected.initial_recruitment_fraction,
            "nominal_amplitude": selected.nominal_amplitude,
            "duration": selected.duration,
            "component": selected.component,
        },
    }
    if progress:
        print(f"  selection done: selected={best_cid} (static={best_static_cid}, feedback={best_feedback_cid})")

    # --- Fresh-origin evaluation: evaluate both static and feedback policy families ---
    policies_to_evaluate: List[Tuple[str, ObservablePolicyCandidate, str]] = []
    if selected_static is not None and selected_feedback is not None:
        policies_to_evaluate.append((f"static:{selected_static.candidate_id}", selected_static, "static"))
        policies_to_evaluate.append((f"feedback:{selected_feedback.candidate_id}", selected_feedback, "feedback"))
    else:
        policies_to_evaluate.append((f"selected:{selected.candidate_id}", selected, "selected"))

    representative_origin_id = prep.evaluation[0].origin_id if prep.evaluation else None

    def make_evaluation_recorder(p_id: str, o_id: str, f_id: int, dyn: str) -> RolloutRecorder:
        is_rep = (o_id == representative_origin_id and f_id == 0)
        cfg = RecorderConfig(
            cadence=cadence,
            record_scalar_trajectories=True,
            record_action_history=True,
            record_network_frames=is_rep,
            max_network_frames=100,
            retention_threshold=float(profile.target_threshold),
        )
        return RolloutRecorder(cfg)

    t_eval_start = _time.perf_counter()
    evaluation_results: Dict[str, List[RolloutResult]] = {}
    adaptive_vs_frozen_pairs: List[Dict[str, Any]] = []
    paired_comparisons: Dict[str, Any] = {}

    for pol_id, pol_cand, family in policies_to_evaluate:
        adaptive_rollouts = _run_observable_rollouts(
            controller_factory=pol_cand.to_controller,
            problem=problem,
            origins=prep.evaluation,
            futures_per_origin=int(profile.evaluation_futures_per_origin),
            policy_id=pol_id,
            dynamics="adaptive",
            sim_seed_fn=sim_seed_fn,
            ctl_seed_fn=ctl_seed_fn,
            physics=physics,
            obs_kwargs=obs_kwargs,
            recorder_factory=make_evaluation_recorder,
            phase="evaluation",
            registry=registry,
        )
        frozen_rollouts = _run_observable_rollouts(
            controller_factory=pol_cand.to_controller,
            problem=problem,
            origins=prep.evaluation,
            futures_per_origin=int(profile.evaluation_futures_per_origin),
            policy_id=pol_id,
            dynamics="frozen",
            sim_seed_fn=sim_seed_fn,
            ctl_seed_fn=ctl_seed_fn,
            physics=physics,
            obs_kwargs=obs_kwargs,
            recorder_factory=make_evaluation_recorder,
            phase="evaluation",
            registry=registry,
        )
        evaluation_results[f"{pol_id}|adaptive"] = adaptive_rollouts
        evaluation_results[f"{pol_id}|frozen"] = frozen_rollouts

        # Matched adaptive/frozen pairs on identical fresh origins and seeds
        adaptive_by_key = {(r.origin_id, r.future_id): r for r in adaptive_rollouts}
        policy_pairs: List[Dict[str, Any]] = []
        for r_frozen in frozen_rollouts:
            key = (r_frozen.origin_id, r_frozen.future_id)
            r_adaptive = adaptive_by_key.get(key)
            pair_dict = {
                "origin_id": r_frozen.origin_id,
                "future_id": r_frozen.future_id,
                "sim_seed": r_frozen.summaries["sim_seed"],
                "ctl_seed": r_frozen.summaries["ctl_seed"],
                "policy_id": pol_id,
                "family": family,
                "adaptive": r_adaptive,
                "frozen": r_frozen,
            }
            policy_pairs.append(pair_dict)
            adaptive_vs_frozen_pairs.append(pair_dict)

        paired_comparisons[family] = summarize_paired_adaptive_frozen_comparison(policy_pairs)
    if registry is not None:
        registry.record_phase_timing("evaluation", _time.perf_counter() - t_eval_start, status="completed")

    # --- Reliability summaries (baselines adaptive; evaluated policies both branches) ---
    reliability_summaries: Dict[str, Dict[str, Any]] = {}
    for name, rollouts in baseline_results.items():
        reliability_summaries[f"{name}|adaptive"] = summarize_rollout_reliability(
            rollouts, float(spec.reliability_target)
        )
    for key, rollouts in evaluation_results.items():
        reliability_summaries[key] = summarize_rollout_reliability(
            rollouts, float(spec.reliability_target)
        )

    # Collect telemetry for result container
    diagnostic_trajectories: Dict[str, Any] = {}
    action_histories: Dict[str, Any] = {}
    network_frames_dict: Dict[str, Any] = {}

    all_arms = [("baseline", baseline_results), ("evaluation", evaluation_results)]
    for group_name, arms in all_arms:
        for arm_name, rollouts in arms.items():
            for r in rollouts:
                rk = f"{group_name}__{r.policy_id}__{r.origin_id}__fut{r.future_id}__{r.summaries.get('dynamics', 'adaptive')}"
                rec = getattr(r, "recorder", None)
                if rec is not None:
                    if rec.scalar_points:
                        diagnostic_trajectories[rk] = {
                            "t": [p.t for p in rec.scalar_points],
                            "m_x": [p.m_x for p in rec.scalar_points],
                            "m_y": [p.m_y for p in rec.scalar_points],
                            "distance_to_target": [p.distance_to_target for p in rec.scalar_points],
                            "target_predicate": [p.target_predicate for p in rec.scalar_points],
                            "hold_progress": [p.hold_progress for p in rec.scalar_points],
                            "actual_release": [p.actual_release for p in rec.scalar_points],
                            "exposure": [p.exposure for p in rec.scalar_points],
                            "spending_rate": [p.spending_rate for p in rec.scalar_points],
                            "distinct_nodes_used": [p.distinct_nodes_used for p in rec.scalar_points],
                        }
                    if rec.action_history:
                        action_histories[rk] = [a.to_dict() for a in rec.action_history]
                    if rec.network_frames:
                        network_frames_dict[rk] = [f.to_dict() for f in rec.network_frames]

    # Original-state retention evaluation on uncontrolled baseline
    retention_results = []
    zero_rollouts = baseline_results.get("baseline_zero", [])
    for r in zero_rollouts:
        rec = getattr(r, "recorder", None)
        if rec is not None:
            ret = rec.evaluate_retention(threshold=float(profile.target_threshold))
            retention_results.append(ret.to_dict())

    zero_eligible = [r for r in zero_rollouts if r.is_eligible]
    eval_states = {origin.origin_id: origin.state for origin in prep.evaluation}
    initial_joint_aligned = 0
    sampled_majority_retained: List[bool] = []
    for result in zero_rollouts:
        initial = eval_states.get(result.origin_id)
        if initial is None:
            continue
        m_y0 = float(np.mean(initial.y[:, 0]))
        m_x0 = float(np.mean(initial.x[:, 0]))
        if abs(m_y0) > 1e-12:
            sign = 1.0 if m_y0 > 0 else -1.0
            initial_joint_aligned += int(
                sign * m_y0 >= float(profile.target_threshold)
                and sign * m_x0 >= float(profile.target_threshold)
            )
            recorder = getattr(result, "recorder", None)
            if recorder is not None and recorder.scalar_points:
                sampled_majority_retained.append(all(
                    sign * point.m_y > 0 for point in recorder.scalar_points
                ))
    if retention_results:
        uncontrolled_persistence = {
            "n_rollouts": len(zero_rollouts),
            "n_eligible": len(zero_eligible),
            "n_success": sum(bool(r.success) for r in zero_rollouts),
            "n_relapsed": sum(bool(r.relapsed) for r in zero_rollouts),
            "status": "measured",
            "original_state_persistence_measured": True,
            "retention_threshold": float(profile.target_threshold),
            "n_initial_joint_threshold_aligned": initial_joint_aligned,
            "public_majority_retention_sampled": {
                "n_observed": len(sampled_majority_retained),
                "n_retained": sum(sampled_majority_retained),
                "rate": float(np.mean(sampled_majority_retained)) if sampled_majority_retained else None,
                "interpretation": "Public majority sign at recorded samples; this is not event-exact majority persistence",
            },
            "retention_window": [0.0, float(profile.horizon)],
            "total_trials": len(retention_results),
            "n_retained": sum(1 for r in retention_results if r["is_retained"]),
            "retention_rate": float(np.mean([r["is_retained"] for r in retention_results])),
            "mean_min_alignment": float(np.mean([r["min_alignment"] for r in retention_results])),
            "mean_max_excursion": float(np.mean([r["max_excursion"] for r in retention_results])),
            "interpretation": (
                "Evaluates whether uncontrolled dynamics retain original orientation "
                "(both means aligned >= threshold) throughout the horizon. "
                "Distinct from reversal benchmark."
            ),
        }
    else:
        uncontrolled_persistence = {
            "n_rollouts": len(zero_rollouts),
            "n_eligible": len(zero_eligible),
            "n_success": sum(bool(r.success) for r in zero_rollouts),
            "n_relapsed": sum(bool(r.relapsed) for r in zero_rollouts),
            "status": "unmeasured",
            "original_state_persistence_measured": False,
            "interpretation": "No uncontrolled baseline rollouts were evaluated.",
        }

    from src.control.controllers import audit_policy_timing
    timing_audits = {
        f"duration_{d:.1f}": audit_policy_timing(
            delivery_delay=float(profile.delivery_delay_tau_public) * tau_y,
            duration=float(d) * tau_y,
            sample_interval=float(profile.sample_interval_tau_public) * tau_y,
            release_deadline=spec.release_deadline,
        )
        for d in profile.durations_tau_public
    }

    n_base_rollouts = len(baseline_specs) * len(prep.evaluation) * int(profile.evaluation_futures_per_origin)
    n_srch_rollouts = int(profile.candidate_cap) * len(prep.training) * int(profile.training_futures_per_origin)
    n_sel_rollouts = len(candidates_to_evaluate) * len(prep.selection) * int(profile.selection_futures_per_origin)
    n_eval_rollouts = len(policies_to_evaluate) * 2 * len(prep.evaluation) * int(profile.evaluation_futures_per_origin)
    total_planned = n_base_rollouts + n_srch_rollouts + n_sel_rollouts + n_eval_rollouts

    workload_projection = registry.compute_runtime_projection(total_planned) if registry is not None else {}

    manifest: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "tier": str(profile.tier),
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "master_seed": int(master_seed),
        "spec": asdict(spec),
        "profile": profile.to_manifest(),
        "origin_source": origin_source,
        "preparation_fingerprint": _preparation_fingerprint(profile, origin_provider),
        "problem": {
            "problem_id": problem.problem_id,
            "model_hash": problem.model_hash,
            "proposal_seed": problem.proposal_seed,
            "search_seed": problem.search_seed,
            "config_version": problem.config_version,
            "k": problem.k,
            "exposure_cap": problem.exposure_cap,
            "amplitude_cap": problem.amplitude_cap,
            "release_deadline": problem.release_deadline,
            "holding_duration": problem.holding_duration,
            "horizon": problem.horizon,
            "accessible_roster": list(problem.accessible_roster),
        },
        "geometry": {
            "k": geometry["k"],
            "amplitude_cap": geometry["amplitude_cap"],
            "exposure_cap": geometry["exposure_cap"],
            "accessible_roster": list(geometry["accessible_roster"]),
        },
        "physics": dict(physics),
        "pool_memberships": {
            "training": [o.origin_id for o in prep.training],
            "selection": [o.origin_id for o in prep.selection],
            "evaluation": [o.origin_id for o in prep.evaluation],
        },
        # F1d: development/held-out separation. The three pools are positional,
        # disjoint slices of one constructed origin ensemble (each accepted
        # origin belongs to at most one pool), so no evaluation origin/future
        # is ever used for training or calibration, and vice versa. Seeds are
        # derived exactly from (master_seed, role, origin_id, future_id), so
        # the (origin, future, seed) triples differ across pools by
        # construction. Origins whose evaluations were inspected in earlier
        # development work are henceforth development evidence, not held-out
        # evidence.
        "pool_pairing": {
            "disjoint_by_construction": {
                "training_vs_selection": not (set(o.origin_id for o in prep.training) & set(o.origin_id for o in prep.selection)),
                "training_vs_evaluation": not (set(o.origin_id for o in prep.training) & set(o.origin_id for o in prep.evaluation)),
                "selection_vs_evaluation": not (set(o.origin_id for o in prep.selection) & set(o.origin_id for o in prep.evaluation)),
            },
            "future_ids": {
                "training": list(range(int(profile.training_futures_per_origin))),
                "selection": list(range(int(profile.selection_futures_per_origin))),
                "evaluation": list(range(int(profile.evaluation_futures_per_origin))),
            },
            "seed_formula": (
                "sim_seed = stable(master_seed, 'sim', origin_id, future_id); "
                "ctl_seed = stable(master_seed, 'ctl', origin_id, future_id). "
                "Training-phase search seeds use (search_seed|origin_id|future_id) "
                "and (proposal_seed|origin_id|future_id) analogously."
            ),
            "development_evidence_note": (
                "Evaluation origins inspected before this design freeze (e.g. "
                "the supplied light bundle's origins 000-003) are treated as "
                "development evidence going forward; held-out claims must use "
                "fresh origins prepared after design changes."
            ),
        },
        "preparation_summary": prep.summary,
        "baseline_schedule": {
            "nominal_amplitude": baseline_amp,
            "duration": baseline_dur,
            "node_fraction": default_node_frac,
            "baseline_attention_pool": "observed_accessible",
            "baseline_random_matched_pool": "observed_accessible",
            "convention": "median profile nominal amplitude/duration, clamped to caps; matched observed accessible pool",
        },
        "selection_summary": selection_summary,
        "transformation_interpretation": (
            "A certified reversal held for the declared duration after release; "
            "sustained success additionally requires no relapse through the finite horizon H. "
            "Neither measure proves literal permanence beyond H."
        ),
        "uncontrolled_persistence": uncontrolled_persistence,
        "timing_audits": timing_audits,
        "policy_families": {
            "static_winner": best_static_cid,
            "feedback_winner": best_feedback_cid,
            "overall_winner": best_cid,
            "evaluated_families": [f for _, _, f in policies_to_evaluate],
        },
        "workload_tracking": {
            "phase_timings": registry.phase_timings if registry is not None else {},
            "preparation_progress": registry.preparation_progress if registry is not None else {},
            "workload_projection": workload_projection,
        },
        "limitations": [
            "Reported quantities are best found by the declared heuristic search, "
            "not proven global optima.",
            "Pilot values are assumptions, not empirical calibration.",
            "Restricted completion time is the failure penalty (horizon H) whenever "
            "an arm has no successes; success-conditional completion time is "
            "reported separately and is unavailable without successes.",
        ],
    }

    result = ControlCampaignResult(
        spec=spec,
        preparation_summary=prep.summary,
        baseline_results=baseline_results,
        search_result=search_result,
        selection_summary=selection_summary,
        evaluation_results=evaluation_results,
        adaptive_vs_frozen_pairs=adaptive_vs_frozen_pairs,
        manifest=manifest,
        selected_candidate=selected,
        selected_static_candidate=selected_static,
        selected_feedback_candidate=selected_feedback,
        paired_comparisons=paired_comparisons,
        reliability_summaries=reliability_summaries,
        bundle_path=str(bundle_path) if bundle_path is not None else None,
        diagnostic_trajectories=diagnostic_trajectories,
        action_histories=action_histories,
        network_frames=network_frames_dict,
        retention_summaries={"baseline_zero": retention_results},
        preparation_attempts=[a.to_record() for a in prep.attempts],
        phase_timings=registry.phase_timings if registry is not None else {},
        workload_projection=workload_projection,
        work_registry=registry,
    )

    if out_dir is not None:
        from src.io.results import export_control_bundle, update_bundle_manifest

        t_exp_start = _time.perf_counter()
        result.bundle_path = export_control_bundle(result, str(bundle_path))
        if registry is not None:
            registry.record_phase_timing("export", _time.perf_counter() - t_exp_start, status="completed")
            update_bundle_manifest(bundle_path)
    return result
