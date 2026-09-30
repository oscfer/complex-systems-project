"""Campaign Work Registry and Resumption Engine.

Provides:
- Persistent campaign work registry keyed by configuration/version, phase,
  candidate, origin, future, and dynamics.
- Atomic persistence of completed work units (accepted origin snapshots, baseline
  rollouts, training search rollouts, selection rollouts, evaluation rollouts).
- Exact reuse of accepted snapshots and completed rollout records without work duplication.
- SHA-256 checksum and configuration compatibility verification: detects and clearly
  rejects incompatible configurations or corrupted checkpoints.
- Phase timing tracking and empirical workload runtime projection with uncertainty.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.control.checkpointing import (
    CheckpointError,
    CorruptedCheckpointError,
    IncompatibleConfigurationError,
)
from src.control.contracts import Action, ControlProblem, RolloutResult, TargetType
from src.control.recording import (
    ActionRecord,
    NetworkFrame,
    RecorderConfig,
    RolloutRecorder,
    ScalarTrajectoryPoint,
)
from src.execution.runner import NumpyEncoder, _atomic_write_json, _canonical_json_bytes, _json_safe


REGISTRY_SCHEMA_VERSION = 1


def _stable_hash(obj: Any) -> str:
    """Generate a stable deterministic hex digest of a JSON-serializable object."""
    data = _json_safe(obj)
    return hashlib.sha256(_canonical_json_bytes(data)).hexdigest()[:16]


# -----------------------------------------------------------------------------
# RolloutResult Serialization Helpers
# -----------------------------------------------------------------------------

def serialize_rollout_result(res: RolloutResult) -> Dict[str, Any]:
    """Serialize RolloutResult into a JSON-safe dictionary."""
    rec_d = None
    if getattr(res, "recorder", None) is not None:
        rec = res.recorder
        rec_d = {
            "config": asdict(rec.config),
            "scalar_points": [p.to_dict() for p in rec.scalar_points],
            "action_history": [a.to_dict() for a in rec.action_history],
            "network_frames": [
                {
                    "t": float(f.t),
                    "x": f.x.tolist(),
                    "y": f.y.tolist(),
                    "W": f.W.tolist(),
                    "controlled_mask": f.controlled_mask.tolist(),
                    "influence_inputs": f.influence_inputs.tolist(),
                }
                for f in rec.network_frames
            ],
            "_initial_sign": float(rec._initial_sign) if rec._initial_sign is not None else None,
            "_event_exact_min_mx": float(rec._event_exact_min_mx),
            "_event_exact_min_my": float(rec._event_exact_min_my),
            "_event_exact_max_exc": float(rec._event_exact_max_exc),
            "_event_exact_retained": bool(rec._event_exact_retained),
            "_event_exact_loss_time": float(rec._event_exact_loss_time) if rec._event_exact_loss_time is not None else None,
            "_next_record_time": float(rec._next_record_time),
            "_last_sampled_time": float(rec._last_sampled_time) if rec._last_sampled_time is not None else None,
        }

    return {
        "problem_id": str(res.problem_id),
        "policy_id": str(res.policy_id),
        "origin_id": str(res.origin_id),
        "future_id": int(res.future_id),
        "is_valid": bool(res.is_valid),
        "is_eligible": bool(res.is_eligible),
        "success": bool(res.success),
        "completion_time": float(res.completion_time) if res.completion_time is not None else None,
        "restricted_time": float(res.restricted_time),
        "expenditure": float(res.expenditure),
        "distinct_nodes_used": int(res.distinct_nodes_used),
        "holding_start_time": float(res.holding_start_time) if res.holding_start_time is not None else None,
        "relapsed": bool(res.relapsed),
        "terminal_target_violation": float(res.terminal_target_violation),
        "event_counts": dict(res.event_counts),
        "summaries": _json_safe(res.summaries),
        "error_message": res.error_message,
        "structural_modifications": int(getattr(res, "structural_modifications", 0)),
        "recorder": rec_d,
    }


def deserialize_rollout_result(d: Dict[str, Any]) -> RolloutResult:
    """Reconstruct RolloutResult from a dictionary."""
    rec = None
    rec_d = d.get("recorder")
    if rec_d is not None:
        cfg = RecorderConfig(**rec_d["config"])
        rec = RolloutRecorder(cfg)
        for sp in rec_d.get("scalar_points", []):
            rec.scalar_points.append(ScalarTrajectoryPoint(
                t=float(sp["t"]),
                m_x=float(sp["m_x"]),
                m_y=float(sp["m_y"]),
                m_x_all=[float(v) for v in sp["m_x_all"]],
                m_y_all=[float(v) for v in sp["m_y_all"]],
                target_direction=float(sp["target_direction"]),
                distance_to_target=float(sp["distance_to_target"]),
                target_predicate=bool(sp["target_predicate"]),
                hold_progress=float(sp["hold_progress"]),
                actual_release=bool(sp["actual_release"]),
                exposure=float(sp["exposure"]),
                spending_rate=float(sp["spending_rate"]),
                distinct_nodes_used=int(sp["distinct_nodes_used"]),
            ))
        for ar in rec_d.get("action_history", []):
            rec.action_history.append(ActionRecord(
                decision_time=float(ar["decision_time"]),
                report_sample_time=float(ar["report_sample_time"]) if ar.get("report_sample_time") is not None else None,
                actuated_nodes=[int(n) for n in ar["actuated_nodes"]],
                field_entries=list(ar["field_entries"]),
                requested_end=float(ar["requested_end"]) if ar.get("requested_end") is not None else None,
                effective_end=float(ar["effective_end"]) if ar.get("effective_end") is not None else None,
                budget_clamped=bool(ar["budget_clamped"]),
                shutoff_or_release=str(ar["shutoff_or_release"]),
                reason=str(ar["reason"]),
                spending_rate=float(ar["spending_rate"]),
                cumulative_exposure=float(ar["cumulative_exposure"]),
            ))
        for nf in rec_d.get("network_frames", []):
            rec.network_frames.append(NetworkFrame(
                t=float(nf["t"]),
                x=np.asarray(nf["x"], dtype=np.float64),
                y=np.asarray(nf["y"], dtype=np.float64),
                W=np.asarray(nf["W"], dtype=np.uint32),
                controlled_mask=np.asarray(nf["controlled_mask"], dtype=bool),
                influence_inputs=np.asarray(nf["influence_inputs"], dtype=np.float64),
            ))
        rec._initial_sign = float(rec_d["_initial_sign"]) if rec_d.get("_initial_sign") is not None else None
        rec._event_exact_min_mx = float(rec_d.get("_event_exact_min_mx", float("inf")))
        rec._event_exact_min_my = float(rec_d.get("_event_exact_min_my", float("inf")))
        rec._event_exact_max_exc = float(rec_d.get("_event_exact_max_exc", 0.0))
        rec._event_exact_retained = bool(rec_d.get("_event_exact_retained", True))
        rec._event_exact_loss_time = float(rec_d["_event_exact_loss_time"]) if rec_d.get("_event_exact_loss_time") is not None else None
        rec._next_record_time = float(rec_d.get("_next_record_time", 0.0))
        rec._last_sampled_time = float(rec_d["_last_sampled_time"]) if rec_d.get("_last_sampled_time") is not None else None

    return RolloutResult(
        problem_id=str(d["problem_id"]),
        policy_id=str(d["policy_id"]),
        origin_id=str(d["origin_id"]),
        future_id=int(d["future_id"]),
        is_valid=bool(d["is_valid"]),
        is_eligible=bool(d["is_eligible"]),
        success=bool(d["success"]),
        completion_time=float(d["completion_time"]) if d.get("completion_time") is not None else None,
        restricted_time=float(d["restricted_time"]),
        expenditure=float(d["expenditure"]),
        distinct_nodes_used=int(d["distinct_nodes_used"]),
        holding_start_time=float(d["holding_start_time"]) if d.get("holding_start_time") is not None else None,
        relapsed=bool(d.get("relapsed", False)),
        terminal_target_violation=float(d.get("terminal_target_violation", 0.0)),
        event_counts=dict(d.get("event_counts", {})),
        summaries=dict(d.get("summaries", {})),
        error_message=d.get("error_message"),
        structural_modifications=int(d.get("structural_modifications", 0)),
        recorder=rec,
    )


# -----------------------------------------------------------------------------
# Work Unit Key and Record
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkUnitKey:
    """Unique key identifying a discrete unit of campaign work."""
    phase: str           # "preparation", "baseline", "search", "selection", "evaluation"
    candidate_id: str    # policy arm or candidate id
    origin_id: str       # origin ID or "origin_none"
    future_id: int       # future index
    dynamics: str        # "adaptive", "frozen", "n/a"
    config_version: str  # version string

    def to_string_key(self) -> str:
        return f"{self.phase}__{self.candidate_id}__{self.origin_id}__fut{self.future_id}__{self.dynamics}"


@dataclass
class WorkUnitRecord:
    """Metadata record of a completed work unit in the campaign registry."""
    key: str
    phase: str
    candidate_id: str
    origin_id: str
    future_id: int
    dynamics: str
    status: str          # "completed"
    file_path: str       # Path relative to bundle_dir
    sha256: str          # SHA-256 of file on disk
    completed_at: str    # Timestamp
    elapsed_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "phase": self.phase,
            "candidate_id": self.candidate_id,
            "origin_id": self.origin_id,
            "future_id": self.future_id,
            "dynamics": self.dynamics,
            "status": self.status,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "completed_at": self.completed_at,
            "elapsed_seconds": float(self.elapsed_seconds),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorkUnitRecord:
        return cls(
            key=str(d["key"]),
            phase=str(d["phase"]),
            candidate_id=str(d["candidate_id"]),
            origin_id=str(d["origin_id"]),
            future_id=int(d["future_id"]),
            dynamics=str(d["dynamics"]),
            status=str(d["status"]),
            file_path=str(d["file_path"]),
            sha256=str(d["sha256"]),
            completed_at=str(d["completed_at"]),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            metrics=dict(d.get("metrics", {})),
        )


# -----------------------------------------------------------------------------
# Campaign Work Registry
# -----------------------------------------------------------------------------

class CampaignWorkRegistry:
    """Persisted registry tracking discrete campaign work units and phase timings."""

    @staticmethod
    def compute_spec_hash(spec: Any) -> str:
        if spec is None:
            return ""
        manifest = spec.to_manifest() if hasattr(spec, "to_manifest") else dict(spec)
        return _stable_hash(manifest)

    @staticmethod
    def compute_profile_hash(profile: Any) -> str:
        if profile is None:
            return ""
        manifest = profile.to_manifest() if hasattr(profile, "to_manifest") else dict(profile)
        return _stable_hash(manifest)

    def __init__(
        self,
        bundle_dir: Path,
        config_version: str = "control-campaign-m4-1.0",
        spec_hash: str = "",
        profile_hash: str = "",
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.config_version = config_version
        self.spec_hash = spec_hash
        self.profile_hash = profile_hash

        self.units: Dict[str, WorkUnitRecord] = {}
        self.phase_timings: Dict[str, Dict[str, Any]] = {
            "preparation": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
            "baselines": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
            "search": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
            "selection": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
            "evaluation": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
            "export": {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"},
        }
        self.preparation_progress: Dict[str, Any] = {
            "attempts_made": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "acceptance_rate": 0.0,
            "completed_units": 0,
            "remaining_units": 0,
        }
        self.unit_runtimes: List[float] = []
        self._cached_results: Dict[str, RolloutResult] = {}

    @property
    def registry_file_path(self) -> Path:
        return self.bundle_dir / "work_registry.json"

    def is_completed(
        self,
        phase: str,
        candidate_id: str,
        origin_id: str,
        future_id: int,
        dynamics: str = "adaptive",
    ) -> bool:
        """Check whether a specific work unit has been completed and registered."""
        k = f"{phase}__{candidate_id}__{origin_id}__fut{future_id}__{dynamics}"
        return k in self.units and self.units[k].status == "completed"

    def get_rollout(
        self,
        phase: str,
        candidate_id: str,
        origin_id: str,
        future_id: int,
        dynamics: str = "adaptive",
    ) -> Optional[RolloutResult]:
        """Retrieve completed RolloutResult from cache or disk, verifying checksum."""
        k = f"{phase}__{candidate_id}__{origin_id}__fut{future_id}__{dynamics}"
        if k in self._cached_results:
            return self._cached_results[k]

        record = self.units.get(k)
        if record is None:
            return None

        file_p = self.bundle_dir / record.file_path
        if not file_p.is_file():
            raise CorruptedCheckpointError(f"Missing work unit file on disk: {file_p}")

        with open(file_p, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)

        payload = envelope["payload"]
        chk = envelope["sha256"]
        actual_chk = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if actual_chk != chk:
            raise CorruptedCheckpointError(
                f"Work unit checksum mismatch for {k}: expected {chk}, got {actual_chk}"
            )

        res = deserialize_rollout_result(payload)
        self._cached_results[k] = res
        return res

    def record_rollout(
        self,
        phase: str,
        candidate_id: str,
        origin_id: str,
        future_id: int,
        dynamics: str,
        result: RolloutResult,
        elapsed_seconds: float,
    ) -> WorkUnitRecord:
        """Atomically persist a completed rollout to disk and register it."""
        k = f"{phase}__{candidate_id}__{origin_id}__fut{future_id}__{dynamics}"
        units_dir = self.bundle_dir / "work_units"
        units_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"work_units/{k}.json"
        dest_path = self.bundle_dir / rel_path

        payload = serialize_rollout_result(result)
        checksum = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        envelope = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "key": k,
            "sha256": checksum,
            "payload": payload,
        }
        _atomic_write_json(dest_path, envelope)

        rec = WorkUnitRecord(
            key=k,
            phase=phase,
            candidate_id=candidate_id,
            origin_id=origin_id,
            future_id=future_id,
            dynamics=dynamics,
            status="completed",
            file_path=rel_path,
            sha256=checksum,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=float(elapsed_seconds),
            metrics={
                "success": bool(result.success),
                "completion_time": result.completion_time,
                "restricted_time": result.restricted_time,
                "expenditure": result.expenditure,
            },
        )
        self.units[k] = rec
        self._cached_results[k] = result
        self.unit_runtimes.append(float(elapsed_seconds))
        if phase in self.phase_timings:
            self.phase_timings[phase]["completed_units"] = self.phase_timings[phase].get("completed_units", 0) + 1
            self.phase_timings[phase]["wall_clock_seconds"] = self.phase_timings[phase].get("wall_clock_seconds", 0.0) + float(elapsed_seconds)

        self.save()
        return rec

    def record_snapshot(
        self,
        origin_id: str,
        seed: int,
        index: int,
        snapshot_rel_path: str,
        sha256_hash: str,
        traits: Dict[str, Any],
        is_eligible: bool,
        elapsed_seconds: float = 0.0,
    ) -> WorkUnitRecord:
        """Register an accepted origin snapshot in the registry."""
        k = f"preparation__{origin_id}____{index}__n/a"
        rec = WorkUnitRecord(
            key=k,
            phase="preparation",
            candidate_id=origin_id,
            origin_id=origin_id,
            future_id=index,
            dynamics="n/a",
            status="completed",
            file_path=snapshot_rel_path,
            sha256=sha256_hash,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=float(elapsed_seconds),
            metrics={
                "seed": seed,
                "index": index,
                "is_eligible": is_eligible,
                "traits": traits,
            },
        )
        self.units[k] = rec
        self.preparation_progress["completed_units"] = self.preparation_progress.get("completed_units", 0) + 1
        if "preparation" in self.phase_timings:
            self.phase_timings["preparation"]["completed_units"] = self.phase_timings["preparation"].get("completed_units", 0) + 1
            self.phase_timings["preparation"]["wall_clock_seconds"] = self.phase_timings["preparation"].get("wall_clock_seconds", 0.0) + float(elapsed_seconds)
        self.save()
        return rec

    def record_phase_timing(
        self,
        phase: str,
        wall_clock_seconds: float,
        status: str = "completed",
    ) -> None:
        """Explicitly record total wall-clock timing and status for a campaign phase."""
        if phase not in self.phase_timings:
            self.phase_timings[phase] = {"wall_clock_seconds": 0.0, "completed_units": 0, "status": "pending"}
        self.phase_timings[phase]["wall_clock_seconds"] = float(wall_clock_seconds)
        self.phase_timings[phase]["status"] = str(status)
        self.save()

    def record_preparation_progress(
        self,
        attempts_made: int,
        accepted_count: int,
        rejected_count: int,
        remaining_units: int = 0,
    ) -> None:
        """Track preparation origin acceptance progress and completed/remaining work counts."""
        rate = (accepted_count / attempts_made) if attempts_made > 0 else 0.0
        self.preparation_progress.update({
            "attempts_made": int(attempts_made),
            "accepted_count": int(accepted_count),
            "rejected_count": int(rejected_count),
            "acceptance_rate": float(rate),
            "completed_units": int(accepted_count),
            "remaining_units": int(remaining_units),
        })
        self.save()

    def compute_runtime_projection(self, total_planned_rollouts: int) -> Dict[str, Any]:
        """Project runtime with empirical uncertainty based on measured completed work."""
        completed = len(self.unit_runtimes)
        remaining = max(0, total_planned_rollouts - completed)

        if completed > 0:
            times = np.array(self.unit_runtimes, dtype=np.float64)
            mean_t = float(np.mean(times))
            std_t = float(np.std(times, ddof=1)) if completed > 1 else mean_t * 0.25
            se_t = std_t / np.sqrt(completed) if completed > 1 else std_t

            proj_remaining = mean_t * remaining
            half_width = 1.96 * se_t * remaining
            ci_lower = max(0.0, proj_remaining - half_width)
            ci_upper = proj_remaining + half_width

            return {
                "measured_completed_rollouts": completed,
                "remaining_planned_rollouts": remaining,
                "total_planned_rollouts": total_planned_rollouts,
                "mean_rollout_seconds": mean_t,
                "std_rollout_seconds": std_t,
                "projected_remaining_seconds": proj_remaining,
                "projected_remaining_ci_95": [ci_lower, ci_upper],
                "confidence_level": 0.95,
                "status": "empirical_projection",
                "methodology": "Point projection with 95% standard error bounds from measured units.",
                "disclaimer": "Runtime projection from measured local work with uncertainty; no unsupported promises.",
            }
        else:
            return {
                "measured_completed_rollouts": 0,
                "remaining_planned_rollouts": total_planned_rollouts,
                "total_planned_rollouts": total_planned_rollouts,
                "status": "unmeasured_pending",
                "methodology": "No rollouts completed yet to establish measured unit baseline.",
                "disclaimer": "Runtime projection from measured local work with uncertainty; no unsupported promises.",
            }

    def save(self) -> None:
        """Write work_registry.json atomically to disk with SHA-256 digest envelope."""
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "config_version": self.config_version,
            "spec_hash": self.spec_hash,
            "profile_hash": self.profile_hash,
            "units": {k: u.to_dict() for k, u in self.units.items()},
            "phase_timings": self.phase_timings,
            "preparation_progress": self.preparation_progress,
            "unit_runtimes": [float(t) for t in self.unit_runtimes],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        checksum = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        envelope = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "sha256": checksum,
            "payload": payload,
        }
        _atomic_write_json(self.registry_file_path, envelope)

    @classmethod
    def load_or_create(
        cls,
        bundle_dir: Path,
        spec: Any,
        profile: Any,
        verify_checksums: bool = True,
    ) -> CampaignWorkRegistry:
        """Load an existing work registry, verify config and checksums, or create a new one.

        Raises:
            CorruptedCheckpointError: If registry or work files have checksum mismatches or corruption.
            IncompatibleConfigurationError: If spec or profile hash does not match existing bundle.
        """
        bundle_dir = Path(bundle_dir)
        reg_file = bundle_dir / "work_registry.json"

        if spec is not None:
            spec_h = _stable_hash(asdict(spec) if hasattr(spec, "__dataclass_fields__") else dict(spec))
        else:
            spec_h = ""
        if profile is not None:
            prof_manifest = profile.to_manifest() if hasattr(profile, "to_manifest") else dict(profile)
            prof_h = _stable_hash(prof_manifest)
        else:
            prof_h = ""

        if not reg_file.is_file():
            # Fresh registry
            reg = cls(
                bundle_dir=bundle_dir,
                config_version="control-campaign-m4-1.0",
                spec_hash=spec_h,
                profile_hash=prof_h,
            )
            reg.save()
            return reg

        # Load existing registry
        try:
            with open(reg_file, "r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except Exception as exc:
            raise CorruptedCheckpointError(f"Failed to read registry JSON at {reg_file}: {exc}") from exc

        if not isinstance(envelope, dict) or "payload" not in envelope or "sha256" not in envelope:
            raise CorruptedCheckpointError(f"Work registry envelope at {reg_file} is malformed")

        payload = envelope["payload"]
        expected_sha = envelope["sha256"]
        actual_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if actual_sha != expected_sha:
            raise CorruptedCheckpointError(
                f"Work registry checksum mismatch at {reg_file}: expected {expected_sha}, got {actual_sha}"
            )

        # Config compatibility check
        reg_spec_h = payload.get("spec_hash", "")
        reg_prof_h = payload.get("profile_hash", "")
        if reg_spec_h and reg_spec_h != spec_h:
            raise IncompatibleConfigurationError(
                f"Incompatible campaign spec in existing bundle {bundle_dir}: "
                f"registry spec_hash={reg_spec_h}, requested spec_hash={spec_h}. "
                "Refusing to mix incompatible campaign data."
            )
        if reg_prof_h and reg_prof_h != prof_h:
            raise IncompatibleConfigurationError(
                f"Incompatible campaign profile in existing bundle {bundle_dir}: "
                f"registry profile_hash={reg_prof_h}, requested profile_hash={prof_h}. "
                "Refusing to mix incompatible campaign data."
            )

        reg = cls(
            bundle_dir=bundle_dir,
            config_version=payload.get("config_version", "control-campaign-m4-1.0"),
            spec_hash=spec_h,
            profile_hash=prof_h,
        )
        reg.phase_timings = dict(payload.get("phase_timings", reg.phase_timings))
        reg.preparation_progress = dict(payload.get("preparation_progress", reg.preparation_progress))
        reg.unit_runtimes = [float(t) for t in payload.get("unit_runtimes", [])]

        raw_units = payload.get("units", {})
        for k, u_dict in raw_units.items():
            record = WorkUnitRecord.from_dict(u_dict)
            if verify_checksums:
                if (bundle_dir / record.file_path).is_file():
                    target_file = bundle_dir / record.file_path
                elif Path(record.file_path).is_file():
                    target_file = Path(record.file_path)
                else:
                    raise CorruptedCheckpointError(f"Registered work unit file missing on disk: {bundle_dir / record.file_path}")
                # For npz or json files, verify raw file sha256 or envelope
                if target_file.suffix == ".json":
                    try:
                        with open(target_file, "r", encoding="utf-8") as f_u:
                            env = json.load(f_u)
                        act_chk = hashlib.sha256(_canonical_json_bytes(env["payload"])).hexdigest()
                        if act_chk != record.sha256:
                            raise CorruptedCheckpointError(
                                f"Work unit {k} file checksum mismatch: expected {record.sha256}, got {act_chk}"
                            )
                    except Exception as exc:
                        if isinstance(exc, CorruptedCheckpointError):
                            raise
                        raise CorruptedCheckpointError(f"Failed to verify work unit {target_file}: {exc}") from exc
                elif target_file.suffix == ".npz":
                    with open(target_file, "rb") as f_npz:
                        act_chk = hashlib.sha256(f_npz.read()).hexdigest()
                    if act_chk != record.sha256:
                        raise CorruptedCheckpointError(
                            f"Origin snapshot {k} checksum mismatch: expected {record.sha256}, got {act_chk}"
                        )
            reg.units[k] = record

        return reg
