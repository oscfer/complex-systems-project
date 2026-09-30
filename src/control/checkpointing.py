"""Controlled simulation rollout checkpointing and mid-rollout continuation.

Supports exact bit-for-bit mid-rollout resumption of closed-loop controlled simulations:
- Complete state preservation: microstate (x, y, W, n, Q, traits), memory manager,
  PoissonScheduler (including residual stochastic clock and deterministic min-heap queue),
  simulator and controller RNG bit_generator states, observation model (panel indices, mask,
  RNG state, and pending observations awaiting delivery), controller internal state,
  BudgetLedger (actuation, release status, remaining exposure, spending rate),
  TargetEvaluator (holding timer, success certification, relapse status, history),
  scheduled holding deadlines, and RolloutRecorder position.
- Explicit callback reconstruction: reconstructed functional dispatch without pickling
  unpickleable closure objects.
- Invariant: Interrupted rollout resumed from checkpoint matches uninterrupted seeded
  rollout bit-for-bit, including pending-delivery and holding-window cases.
- Integrity verification: checksum validation via SHA-256 envelope and strict configuration
  compatibility checks; corrupted or incompatible checkpoints are rejected clearly.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.control.contracts import Action, ActionEntry, ControlProblem, Observation, TargetType
from src.control.controllers import (
    Controller,
    ObservableSelectorPolicy,
    OpenLoopPulseController,
    ZeroController,
)
from src.control.evaluator import TargetDefinition, TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import (
    ObservationModel,
    SampledPublicObservationModel,
    CompletePublicObservationModel,
)
from src.control.recording import (
    ActionRecord,
    NetworkFrame,
    RecorderConfig,
    RolloutRecorder,
    ScalarTrajectoryPoint,
)
from src.core.fields import compute_single_public_field
from src.core.heat_baths import sample_binary_glauber_scalar, sample_continuous_inverse_cdf_scalar
from src.core.memory import MemoryManager
from src.core.scheduler import (
    DeterministicEvent,
    PRIORITY_HOLDING,
    PRIORITY_HORIZON,
    PRIORITY_INTERVENTION,
    PRIORITY_RESTORE,
    PRIORITY_SNAPSHOT,
    PoissonScheduler,
    SchedulerStats,
)
from src.core.state import State, StateMode
from src.execution.runner import NumpyEncoder, _canonical_json_bytes, _json_safe


CONTROL_CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(Exception):
    """Base exception for rollout checkpointing errors."""
    pass


class CorruptedCheckpointError(CheckpointError, ValueError):
    """Raised when a checkpoint file or envelope has been tampered with or corrupted."""
    pass


class IncompatibleConfigurationError(CheckpointError, ValueError):
    """Raised when a checkpoint configuration does not match the requested simulation."""
    pass


# -----------------------------------------------------------------------------
# Serialization Helpers
# -----------------------------------------------------------------------------

def serialize_state(state: State) -> Dict[str, Any]:
    """Serialize State into a JSON-safe dictionary."""
    d: Dict[str, Any] = {
        "N": int(state.N),
        "D": int(state.D),
        "state_mode": getattr(state.state_mode, "value", str(state.state_mode)),
        "t": float(state.t),
        "x": state.x.tolist(),
        "y": state.y.tolist(),
        "W": state.W.tolist(),
        "n": state.n.tolist(),
        "Q": state.Q.tolist(),
        "holding_active": bool(getattr(state, "holding_active", False)),
        "t_start_hold": float(state.t_start_hold) if getattr(state, "t_start_hold", None) is not None else None,
        "t_deadline": float(state.t_deadline) if getattr(state, "t_deadline", None) is not None else None,
        "transition_occurred": bool(getattr(state, "transition_occurred", False)),
        "t_transition": float(state.t_transition) if getattr(state, "t_transition", None) is not None else None,
        "pulse_active": bool(getattr(state, "pulse_active", False)),
        "pulse_val": float(getattr(state, "pulse_val", 0.0)),
    }
    for attr in ("K", "tau_x", "nu", "g", "b", "row_caps", "opportunity_groups", "opportunity_matrix"):
        val = getattr(state, attr, None)
        if isinstance(val, np.ndarray):
            d[attr] = val.tolist()
        elif val is not None:
            d[attr] = val
        else:
            d[attr] = None
    d["blocks"] = state.blocks.tolist() if state.blocks is not None else None
    d["scenario_metadata"] = state.scenario_metadata
    return d


def deserialize_state(d: Dict[str, Any]) -> State:
    """Reconstruct State from a dictionary."""
    trait_kwargs: Dict[str, Any] = {}
    for attr in ("K", "tau_x", "nu", "g", "b", "row_caps", "opportunity_groups", "opportunity_matrix"):
        val = d.get(attr)
        if val is not None:
            if attr in ("K", "g", "b", "opportunity_matrix"):
                trait_kwargs[attr] = np.asarray(val, dtype=np.float64)
            elif attr in ("row_caps", "opportunity_groups"):
                trait_kwargs[attr] = np.asarray(val, dtype=np.int64 if attr == "row_caps" else int)
            elif attr in ("tau_x", "nu"):
                trait_kwargs[attr] = np.asarray(val, dtype=np.float64) if isinstance(val, list) else float(val)
        else:
            trait_kwargs[attr] = None

    return State(
        N=int(d["N"]),
        D=int(d["D"]),
        state_mode=StateMode(d["state_mode"]),
        x=np.asarray(d["x"], dtype=np.float64),
        y=np.asarray(d["y"], dtype=np.float64),
        W=np.asarray(d["W"], dtype=np.uint32),
        n=np.asarray(d["n"], dtype=np.uint8),
        Q=np.asarray(d["Q"], dtype=np.float64),
        t=float(d["t"]),
        holding_active=bool(d.get("holding_active", False)),
        t_start_hold=float(d["t_start_hold"]) if d.get("t_start_hold") is not None else None,
        t_deadline=float(d["t_deadline"]) if d.get("t_deadline") is not None else None,
        transition_occurred=bool(d.get("transition_occurred", False)),
        t_transition=float(d["t_transition"]) if d.get("t_transition") is not None else None,
        pulse_active=bool(d.get("pulse_active", False)),
        pulse_val=float(d.get("pulse_val", 0.0)),
        blocks=np.asarray(d["blocks"], dtype=np.int64) if d.get("blocks") is not None else None,
        scenario_metadata=d.get("scenario_metadata"),
        **trait_kwargs,
    )


def serialize_action(action: Optional[Action]) -> Optional[Dict[str, Any]]:
    """Serialize Action into a dictionary."""
    if action is None:
        return None
    entries = [
        {"node": int(e.node), "component": int(e.component), "field": float(e.field)}
        for e in action.entries
    ]
    struct_entries = []
    if hasattr(action, "structural_entries") and action.structural_entries:
        for se in action.structural_entries:
            struct_entries.append({
                "source": int(se.source),
                "target": int(se.target),
                "operation": getattr(se.operation, "value", str(se.operation)),
                "count": int(se.count),
                "rewire_target": int(se.rewire_target) if se.rewire_target is not None else None,
            })
    return {
        "entries": entries,
        "intended_end_time": float(action.intended_end_time) if action.intended_end_time is not None else None,
        "release": bool(action.release),
        "structural_entries": struct_entries,
    }


def deserialize_action(d: Optional[Dict[str, Any]]) -> Optional[Action]:
    """Reconstruct Action from a dictionary."""
    if d is None:
        return None
    from src.control.contracts import StructuralActionEntry, StructuralOp
    entries = tuple(
        ActionEntry(node=int(e["node"]), component=int(e["component"]), field=float(e["field"]))
        for e in d.get("entries", [])
    )
    struct_entries = tuple(
        StructuralActionEntry(
            source=int(se["source"]),
            target=int(se["target"]),
            operation=StructuralOp(se.get("operation", "force")),
            count=int(se.get("count", 1)),
            rewire_target=int(se["rewire_target"]) if se.get("rewire_target") is not None else None,
        )
        for se in d.get("structural_entries", [])
    )
    return Action(
        entries=entries,
        intended_end_time=float(d["intended_end_time"]) if d.get("intended_end_time") is not None else None,
        release=bool(d.get("release", False)),
        structural_entries=struct_entries,
    )


def serialize_observation(obs: Observation) -> Dict[str, Any]:
    """Serialize Observation dataclass into a JSON-safe dictionary."""
    edges_dict = {f"{i},{j}": int(cnt) for (i, j), cnt in obs.observed_edges.items()}
    return {
        "time": float(obs.time),
        "public_states": obs.public_states.tolist() if obs.public_states is not None else None,
        "panel_mask": obs.panel_mask.tolist() if obs.panel_mask is not None else None,
        "observed_edges": edges_dict,
        "accessible_roster": list(obs.accessible_roster) if obs.accessible_roster is not None else None,
        "group_labels": obs.group_labels.tolist() if obs.group_labels is not None else None,
        "sample_time": float(obs.sample_time),
        "delivery_time": float(obs.delivery_time),
        "remaining_exposure": float(obs.remaining_exposure),
        "remaining_distinct_nodes": int(obs.remaining_distinct_nodes),
        "current_spending_rate": float(obs.current_spending_rate),
        "noise_model_version": str(obs.noise_model_version),
        "is_oracle": bool(obs.is_oracle),
    }


def deserialize_observation(d: Dict[str, Any]) -> Observation:
    """Reconstruct Observation dataclass from dictionary."""
    pub = np.asarray(d["public_states"], dtype=np.float64) if d.get("public_states") is not None else None
    mask = np.asarray(d["panel_mask"], dtype=bool) if d.get("panel_mask") is not None else None
    edges = {}
    for k, cnt in d.get("observed_edges", {}).items():
        parts = k.split(",")
        edges[(int(parts[0]), int(parts[1]))] = int(cnt)
    roster = tuple(d["accessible_roster"]) if d.get("accessible_roster") is not None else None
    groups = np.asarray(d["group_labels"], dtype=int) if d.get("group_labels") is not None else None

    return Observation(
        time=float(d["time"]),
        public_states=pub,
        panel_mask=mask,
        observed_edges=edges,
        accessible_roster=roster,
        group_labels=groups,
        sample_time=float(d["sample_time"]),
        delivery_time=float(d["delivery_time"]),
        remaining_exposure=float(d.get("remaining_exposure", 0.0)),
        remaining_distinct_nodes=int(d.get("remaining_distinct_nodes", 0)),
        current_spending_rate=float(d.get("current_spending_rate", 0.0)),
        noise_model_version=str(d.get("noise_model_version", "sampled_public_v1")),
        is_oracle=bool(d.get("is_oracle", False)),
    )


# -----------------------------------------------------------------------------
# Checkpoint Data Container
# -----------------------------------------------------------------------------

@dataclass
class ControlledRolloutCheckpoint:
    """Comprehensive checkpoint container for a controlled simulation rollout."""
    checkpoint_time: float
    schema_version: int = CONTROL_CHECKPOINT_SCHEMA_VERSION
    config_version: str = "control-checkpoint-v1"
    problem_hash: str = ""
    problem_dict: Dict[str, Any] = field(default_factory=dict)
    physics_dict: Dict[str, Any] = field(default_factory=dict)
    state_dict: Dict[str, Any] = field(default_factory=dict)
    memory_dict: Dict[str, Any] = field(default_factory=dict)
    scheduler_dict: Dict[str, Any] = field(default_factory=dict)
    sim_rng_state: Dict[str, Any] = field(default_factory=dict)
    controller_rng_state: Dict[str, Any] = field(default_factory=dict)
    observation_model_dict: Dict[str, Any] = field(default_factory=dict)
    pending_observations: Dict[str, Any] = field(default_factory=dict)
    scheduled_holding_deadlines: List[float] = field(default_factory=list)
    controller_dict: Dict[str, Any] = field(default_factory=dict)
    ledger_dict: Dict[str, Any] = field(default_factory=dict)
    target_evaluator_dict: Dict[str, Any] = field(default_factory=dict)
    recorder_dict: Optional[Dict[str, Any]] = None

    @property
    def observation_model_state(self) -> Dict[str, Any]:
        d = dict(self.observation_model_dict)
        d["pending_deliveries"] = self.pending_observations
        return d

    @property
    def target_evaluator_state(self) -> Dict[str, Any]:
        d = dict(self.target_evaluator_dict)
        d["in_holding"] = (self.target_evaluator_dict.get("t_start_hold") is not None)
        d["holding_start_time"] = self.target_evaluator_dict.get("t_start_hold")
        d["certified"] = self.target_evaluator_dict.get("is_success", False)
        return d

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to a fully JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "checkpoint_time": float(self.checkpoint_time),
            "config_version": self.config_version,
            "problem_hash": self.problem_hash,
            "problem_dict": self.problem_dict,
            "physics_dict": self.physics_dict,
            "state_dict": self.state_dict,
            "memory_dict": self.memory_dict,
            "scheduler_dict": self.scheduler_dict,
            "sim_rng_state": self.sim_rng_state,
            "controller_rng_state": self.controller_rng_state,
            "observation_model_dict": self.observation_model_dict,
            "pending_observations": self.pending_observations,
            "scheduled_holding_deadlines": [float(t) for t in self.scheduled_holding_deadlines],
            "controller_dict": self.controller_dict,
            "ledger_dict": self.ledger_dict,
            "target_evaluator_dict": self.target_evaluator_dict,
            "recorder_dict": self.recorder_dict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ControlledRolloutCheckpoint:
        """Construct checkpoint from dictionary."""
        return cls(
            checkpoint_time=float(data["checkpoint_time"]),
            schema_version=int(data.get("schema_version", CONTROL_CHECKPOINT_SCHEMA_VERSION)),
            config_version=str(data.get("config_version", "")),
            problem_hash=str(data.get("problem_hash", "")),
            problem_dict=dict(data.get("problem_dict", {})),
            physics_dict=dict(data.get("physics_dict", {})),
            state_dict=dict(data.get("state_dict", {})),
            memory_dict=dict(data.get("memory_dict", {})),
            scheduler_dict=dict(data.get("scheduler_dict", {})),
            sim_rng_state=dict(data.get("sim_rng_state", {})),
            controller_rng_state=dict(data.get("controller_rng_state", {})),
            observation_model_dict=dict(data.get("observation_model_dict", {})),
            pending_observations=dict(data.get("pending_observations", {})),
            scheduled_holding_deadlines=[float(t) for t in data.get("scheduled_holding_deadlines", [])],
            controller_dict=dict(data.get("controller_dict", {})),
            ledger_dict=dict(data.get("ledger_dict", {})),
            target_evaluator_dict=dict(data.get("target_evaluator_dict", {})),
            recorder_dict=data.get("recorder_dict"),
        )


# -----------------------------------------------------------------------------
# Disk Persistence with Checksum & Compatibility Verification
# -----------------------------------------------------------------------------

def save_controlled_checkpoint(
    checkpoint: ControlledRolloutCheckpoint,
    filepath: Union[str, Path],
) -> str:
    """Save ControlledRolloutCheckpoint atomically to disk with SHA-256 envelope."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_safe(checkpoint.to_dict())
    checksum = hashlib.sha256(_canonical_json_bytes(serializable)).hexdigest()
    envelope = {
        "schema_version": CONTROL_CHECKPOINT_SCHEMA_VERSION,
        "sha256": checksum,
        "payload": serializable,
    }

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, cls=NumpyEncoder, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return str(path)


def load_controlled_checkpoint(
    filepath: Union[str, Path],
    expected_problem: Optional[ControlProblem] = None,
    expected_model_hash: Optional[str] = None,
) -> ControlledRolloutCheckpoint:
    """Load, verify, and deserialize a ControlledRolloutCheckpoint from disk.

    Raises:
        CorruptedCheckpointError: If envelope is missing, malformed, or sha256 checksum fails.
        IncompatibleConfigurationError: If schema version or problem configuration does not match.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except Exception as exc:
        raise CorruptedCheckpointError(f"Failed to read/parse checkpoint JSON from {path}: {exc}") from exc

    if not isinstance(envelope, dict) or "payload" not in envelope or "sha256" not in envelope:
        raise CorruptedCheckpointError(f"Checkpoint envelope in {path} is incomplete or missing sha256/payload")

    payload = envelope["payload"]
    expected_sha256 = envelope["sha256"]
    actual_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual_sha256 != expected_sha256:
        raise CorruptedCheckpointError(
            f"Checkpoint checksum mismatch in {path}: expected {expected_sha256}, got {actual_sha256}"
        )

    if payload.get("schema_version") != CONTROL_CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleConfigurationError(
            f"Unsupported checkpoint schema version {payload.get('schema_version')} (expected {CONTROL_CHECKPOINT_SCHEMA_VERSION})"
        )

    checkpoint = ControlledRolloutCheckpoint.from_dict(payload)

    # Check compatibility if expected problem / hash provided
    if expected_model_hash is not None and checkpoint.problem_hash:
        if checkpoint.problem_hash != expected_model_hash:
            raise IncompatibleConfigurationError(
                f"Checkpoint problem hash mismatch: expected {expected_model_hash}, got {checkpoint.problem_hash}"
            )

    if expected_problem is not None:
        p_dict = checkpoint.problem_dict
        for key in ("N", "D", "horizon", "release_deadline", "holding_duration", "k"):
            if key in p_dict:
                exp_val = getattr(expected_problem, key)
                chk_val = p_dict[key]
                if exp_val != chk_val:
                    raise IncompatibleConfigurationError(
                        f"Incompatible problem parameter '{key}': expected {exp_val}, got {chk_val}"
                    )
        if "target_type" in p_dict:
            exp_tt = getattr(expected_problem.target_type, "value", str(expected_problem.target_type))
            if p_dict["target_type"] != exp_tt:
                raise IncompatibleConfigurationError(
                    f"Incompatible target_type: expected {exp_tt}, got {p_dict['target_type']}"
                )

    return checkpoint


# -----------------------------------------------------------------------------
# State Serialization & Callback Reconstruction
# -----------------------------------------------------------------------------

def capture_rollout_checkpoint(
    current_time: float,
    problem: ControlProblem,
    physics: Dict[str, Any],
    state: State,
    memory_mgr: MemoryManager,
    scheduler: PoissonScheduler,
    ledger: BudgetLedger,
    observation_model: ObservationModel,
    pending_observations: Dict[float, Observation],
    controller: Controller,
    target_evaluator: TargetEvaluator,
    sim_rng: np.random.Generator,
    controller_rng: np.random.Generator,
    scheduled_holding_deadlines: Set[float],
    recorder: Optional[RolloutRecorder] = None,
) -> ControlledRolloutCheckpoint:
    """Capture complete state of a running controlled simulation."""
    # 1. Problem & Physics
    p_dict = {
        "problem_id": problem.problem_id,
        "target_type": getattr(problem.target_type, "value", str(problem.target_type)),
        "target_params": problem.target_params,
        "N": problem.N,
        "D": problem.D,
        "k": problem.k,
        "exposure_cap": problem.exposure_cap,
        "amplitude_cap": problem.amplitude_cap,
        "release_deadline": problem.release_deadline,
        "holding_duration": problem.holding_duration,
        "horizon": problem.horizon,
        "accessible_roster": list(problem.accessible_roster) if problem.accessible_roster else None,
        "config_version": problem.config_version,
    }

    # 2. State
    state_d = serialize_state(state)

    # 3. Memory Manager
    memory_d = {
        "total_memory": int(memory_mgr.total_memory),
        "reciprocal": bool(memory_mgr.reciprocal),
        "epsilon": float(memory_mgr.epsilon),
        "a": float(memory_mgr.a),
        "c": float(memory_mgr.c),
        "chi": float(memory_mgr.chi),
        "between_eligibility": float(physics.get("between_eligibility", 0.05)),
        "within_eligibility": float(physics.get("within_eligibility", 1.0)),
        "C_matrix": memory_mgr.C_matrix.tolist() if memory_mgr.C_matrix is not None else None,
    }

    # 4. Scheduler
    det_events = [
        {
            "time": float(e.time),
            "priority": int(e.priority),
            "seq_id": int(e.seq_id),
            "event_type": str(e.event_type),
            "payload": dict(e.payload),
        }
        for e in scheduler.pq
    ]
    scheduler_d = {
        "time": float(scheduler.time),
        "stats": {
            "total_stochastic_events": int(scheduler.stats.total_stochastic_events),
            "private_updates": int(scheduler.stats.private_updates),
            "public_updates": int(scheduler.stats.public_updates),
            "encounters": int(scheduler.stats.encounters),
            "memory_decays": int(scheduler.stats.memory_decays),
            "participation_updates": int(scheduler.stats.participation_updates),
            "deterministic_events": int(scheduler.stats.deterministic_events),
            "preemptions": int(scheduler.stats.preemptions),
        },
        "pending_clock": scheduler.pending_clock_state(),
        "seq_counter": int(scheduler.seq_counter),
        "deterministic_events": det_events,
    }

    # 5. PRNG states
    sim_rng_d = sim_rng.bit_generator.state
    ctl_rng_d = controller_rng.bit_generator.state

    # 6. Observation Model
    if isinstance(observation_model, SampledPublicObservationModel):
        obs_d = {
            "type": "SampledPublicObservationModel",
            "N": int(observation_model.N),
            "panel_fraction": float(observation_model.panel_fraction),
            "binary_flip_prob": float(observation_model.binary_flip_prob),
            "continuous_noise_sd": float(observation_model.continuous_noise_sd),
            "edge_reveal_prob": float(observation_model.edge_reveal_prob),
            "delivery_delay": float(observation_model.delivery_delay),
            "accessible_roster": list(observation_model.accessible_roster) if observation_model.accessible_roster else None,
            "panel_indices": observation_model.panel_indices.tolist(),
            "panel_mask": observation_model.panel_mask.tolist(),
            "group_labels": observation_model.group_labels.tolist() if observation_model.group_labels is not None else None,
            "rng_state": observation_model.rng.bit_generator.state if hasattr(observation_model.rng, "bit_generator") else None,
        }
    else:
        obs_d = {
            "type": observation_model.__class__.__name__,
            "N": getattr(observation_model, "N", problem.N),
        }

    # 7. Pending Observations
    pending_d = {
        str(t_del): serialize_observation(obs)
        for t_del, obs in pending_observations.items()
    }

    # 8. Controller
    if isinstance(controller, ObservableSelectorPolicy):
        ctl_d = {
            "type": "ObservableSelectorPolicy",
            "roster_selection": controller.roster_selection,
            "broadcast_amplitude": controller.broadcast_amplitude,
            "selector": controller.selector,
            "nominal_amplitude": float(controller.nominal_amplitude),
            "duration": float(controller.duration),
            "quota": float(controller.quota),
            "is_feedback": bool(controller.is_feedback),
            "initial_recruitment_fraction": float(controller.initial_recruitment_fraction),
            "component": int(controller.component),
            "seed": int(controller.seed),
            "node_fraction": float(controller.node_fraction) if controller.node_fraction is not None else None,
            "s_star": float(controller.s_star) if controller.s_star is not None else None,
            "committed_end_time": float(controller.committed_end_time) if controller.committed_end_time is not None else None,
            "initial_actuation_done": bool(controller.initial_actuation_done),
            "used_nodes": sorted(list(controller.used_nodes)),
            "fallback_records": copy.deepcopy(controller.fallback_records),
            "feedback_opportunities_encountered": int(controller.feedback_opportunities_encountered),
            "feedback_updates_performed": int(controller.feedback_updates_performed),
            "requested_settings": copy.deepcopy(controller.requested_settings),
            "effective_settings": copy.deepcopy(controller.effective_settings),
        }
    elif isinstance(controller, OpenLoopPulseController):
        ctl_d = {
            "type": "OpenLoopPulseController",
            "target_nodes": list(controller.target_nodes),
            "nominal_amplitude": float(controller.nominal_amplitude),
            "duration": float(controller.duration),
            "signs": {str(k): float(v) for k, v in controller.signs.items()},
            "component": int(controller.component),
            "actuated": bool(controller.actuated),
            "pulse_end_time": float(controller.pulse_end_time) if controller.pulse_end_time is not None else None,
        }
    else:
        ctl_d = {
            "type": controller.__class__.__name__,
        }

    # 9. Budget Ledger
    ledger_d = {
        "current_time": float(ledger.current_time),
        "integrated_exposure": float(ledger.integrated_exposure),
        "distinct_nodes_actuated": sorted(list(ledger.distinct_nodes_actuated)),
        "is_released": bool(ledger.is_released),
        "release_time": float(ledger.release_time) if ledger.release_time is not None else None,
        "is_exhausted": bool(ledger.is_exhausted),
        "is_structural_exhausted": bool(ledger.is_structural_exhausted),
        "integrated_structural_modifications": int(ledger.integrated_structural_modifications),
        "current_field": ledger.current_field.tolist(),
        "active_action": serialize_action(ledger.active_action),
        "active_action_end_time": float(ledger.active_action_end_time) if ledger.active_action_end_time is not None else None,
    }

    # 10. Target Evaluator
    eval_d = {
        "problem_id": target_evaluator.problem_id,
        "policy_id": target_evaluator.policy_id,
        "origin_id": target_evaluator.origin_id,
        "future_id": int(target_evaluator.future_id),
        "t_start_hold": float(target_evaluator.t_start_hold) if target_evaluator.t_start_hold is not None else None,
        "completion_time": float(target_evaluator.completion_time) if target_evaluator.completion_time is not None else None,
        "is_success": bool(target_evaluator.is_success),
        "relapsed": bool(target_evaluator.relapsed),
        "last_condition_value": bool(target_evaluator.last_condition_value),
        "is_released": bool(target_evaluator.is_released),
        "terminal_violation": float(target_evaluator.terminal_violation),
        "history_events": [[float(t), bool(c)] for t, c in target_evaluator.history_events],
        "defn": {
            "target_type": getattr(target_evaluator.defn.target_type, "value", str(target_evaluator.defn.target_type)),
            "holding_duration": float(target_evaluator.defn.holding_duration),
            "horizon": float(target_evaluator.defn.horizon),
            "release_deadline": float(target_evaluator.defn.release_deadline),
            "s_star": float(target_evaluator.defn.s_star) if target_evaluator.defn.s_star is not None else None,
            "threshold": float(target_evaluator.defn.threshold),
            "focal_indices": list(target_evaluator.defn.focal_indices) if target_evaluator.defn.focal_indices is not None else None,
            "C_N_0": float(target_evaluator.defn.C_N_0) if target_evaluator.defn.C_N_0 is not None else None,
            "L_0": sorted(list(target_evaluator.defn.L_0)) if target_evaluator.defn.L_0 is not None else None,
            "initial_metrics": dict(target_evaluator.defn.initial_metrics),
            "is_eligible": bool(target_evaluator.defn.is_eligible),
            "ineligibility_reason": target_evaluator.defn.ineligibility_reason,
            "is_already_target": bool(target_evaluator.defn.is_already_target),
        },
    }

    # 11. Rollout Recorder
    recorder_d = None
    if recorder is not None:
        recorder_d = {
            "config": asdict(recorder.config),
            "scalar_points": [p.to_dict() for p in recorder.scalar_points],
            "action_history": [a.to_dict() for a in recorder.action_history],
            "network_frames": [
                {
                    "t": float(f.t),
                    "x": f.x.tolist(),
                    "y": f.y.tolist(),
                    "W": f.W.tolist(),
                    "controlled_mask": f.controlled_mask.tolist(),
                    "influence_inputs": f.influence_inputs.tolist(),
                }
                for f in recorder.network_frames
            ],
            "_initial_sign": float(recorder._initial_sign) if recorder._initial_sign is not None else None,
            "_event_exact_min_mx": float(recorder._event_exact_min_mx),
            "_event_exact_min_my": float(recorder._event_exact_min_my),
            "_event_exact_max_exc": float(recorder._event_exact_max_exc),
            "_event_exact_retained": bool(recorder._event_exact_retained),
            "_event_exact_loss_time": float(recorder._event_exact_loss_time) if recorder._event_exact_loss_time is not None else None,
            "_next_record_time": float(recorder._next_record_time),
            "_last_sampled_time": float(recorder._last_sampled_time) if recorder._last_sampled_time is not None else None,
        }

    return ControlledRolloutCheckpoint(
        checkpoint_time=float(current_time),
        schema_version=CONTROL_CHECKPOINT_SCHEMA_VERSION,
        config_version=problem.config_version,
        problem_hash=problem.model_hash,
        problem_dict=p_dict,
        physics_dict=dict(physics),
        state_dict=state_d,
        memory_dict=memory_d,
        scheduler_dict=scheduler_d,
        sim_rng_state=sim_rng_d,
        controller_rng_state=ctl_rng_d,
        observation_model_dict=obs_d,
        pending_observations=pending_d,
        scheduled_holding_deadlines=sorted(list(scheduled_holding_deadlines)),
        controller_dict=ctl_d,
        ledger_dict=ledger_d,
        target_evaluator_dict=eval_d,
        recorder_dict=recorder_d,
    )


def restore_rollout_state(
    checkpoint: ControlledRolloutCheckpoint,
    problem: Optional[ControlProblem] = None,
) -> Tuple[
    State,
    MemoryManager,
    PoissonScheduler,
    BudgetLedger,
    ObservationModel,
    Controller,
    TargetEvaluator,
    Optional[RolloutRecorder],
    np.random.Generator,
    np.random.Generator,
    Dict[float, Observation],
    Set[float],
]:
    """Reconstruct all simulation objects and PRNG streams from a checkpoint."""
    # 1. State
    state = deserialize_state(checkpoint.state_dict)
    N, D = state.N, state.D

    # 2. PRNG Generators
    sim_rng = np.random.default_rng()
    sim_rng.bit_generator.state = checkpoint.sim_rng_state

    controller_rng = np.random.default_rng()
    controller_rng.bit_generator.state = checkpoint.controller_rng_state

    # 3. Memory Manager
    mem_d = checkpoint.memory_dict
    memory_mgr = MemoryManager(
        N=N,
        reciprocal=bool(mem_d.get("reciprocal", False)),
        epsilon=float(mem_d.get("epsilon", 0.1)),
        a=float(mem_d.get("a", 0.0)),
        c=float(mem_d.get("c", 1.0)),
        chi=float(mem_d.get("chi", 0.3)),
        initial_W=state.W.copy(),
        capacity=state.row_caps,
        opportunity_groups=state.opportunity_groups,
        C_matrix=np.asarray(mem_d["C_matrix"], dtype=np.float64) if mem_d.get("C_matrix") is not None else state.opportunity_matrix,
        between_eligibility=float(mem_d.get("between_eligibility", 0.05)),
        within_eligibility=float(mem_d.get("within_eligibility", 1.0)),
        rng=sim_rng,
    )
    # Ensure memory matrix exactly matches state W
    memory_mgr.W = state.W.copy()
    memory_mgr.total_memory = int(mem_d.get("total_memory", state.W.sum()))

    # 4. Budget Ledger
    p_dict = checkpoint.problem_dict
    k_val = p_dict.get("k", problem.k if problem else 1)
    exp_cap = float(p_dict.get("exposure_cap", problem.exposure_cap if problem else 10.0))
    amp_cap = float(p_dict.get("amplitude_cap", problem.amplitude_cap if problem else 4.0))
    rel_dead = float(p_dict.get("release_deadline", problem.release_deadline if problem else 20.0))
    acc_roster = set(p_dict["accessible_roster"]) if p_dict.get("accessible_roster") else None

    ledger = BudgetLedger(
        N=N,
        D=D,
        k=k_val,
        exposure_cap=exp_cap,
        amplitude_cap=amp_cap,
        release_deadline=rel_dead,
        accessible_roster=acc_roster,
    )
    led_d = checkpoint.ledger_dict
    ledger.current_time = float(led_d["current_time"])
    ledger.integrated_exposure = float(led_d["integrated_exposure"])
    ledger.distinct_nodes_actuated = set(led_d["distinct_nodes_actuated"])
    ledger.is_released = bool(led_d["is_released"])
    ledger.release_time = float(led_d["release_time"]) if led_d.get("release_time") is not None else None
    ledger.is_exhausted = bool(led_d["is_exhausted"])
    ledger.is_structural_exhausted = bool(led_d.get("is_structural_exhausted", False))
    ledger.integrated_structural_modifications = int(led_d.get("integrated_structural_modifications", 0))
    ledger.current_field = np.asarray(led_d["current_field"], dtype=np.float64)
    ledger.active_action = deserialize_action(led_d.get("active_action"))
    ledger.active_action_end_time = float(led_d["active_action_end_time"]) if led_d.get("active_action_end_time") is not None else None

    # 5. Observation Model
    obs_d = checkpoint.observation_model_dict
    obs_type = obs_d.get("type", "SampledPublicObservationModel")
    if obs_type == "SampledPublicObservationModel":
        obs_rng = np.random.default_rng()
        if obs_d.get("rng_state"):
            obs_rng.bit_generator.state = obs_d["rng_state"]
        observation_model = SampledPublicObservationModel(
            N=N,
            panel_fraction=float(obs_d.get("panel_fraction", 0.5)),
            binary_flip_prob=float(obs_d.get("binary_flip_prob", 0.05)),
            continuous_noise_sd=float(obs_d.get("continuous_noise_sd", 0.1)),
            edge_reveal_prob=float(obs_d.get("edge_reveal_prob", 0.5)),
            delivery_delay=float(obs_d.get("delivery_delay", 1.0)),
            accessible_roster=tuple(obs_d["accessible_roster"]) if obs_d.get("accessible_roster") else None,
            group_labels=np.asarray(obs_d["group_labels"], dtype=int) if obs_d.get("group_labels") is not None else None,
            rng=obs_rng,
        )
        observation_model.panel_indices = np.asarray(obs_d["panel_indices"], dtype=int)
        observation_model.panel_mask = np.asarray(obs_d["panel_mask"], dtype=bool)
    else:
        observation_model = CompletePublicObservationModel(N=N)

    # 6. Pending Observations
    pending_observations: Dict[float, Observation] = {}
    for t_str, o_dict in checkpoint.pending_observations.items():
        pending_observations[float(t_str)] = deserialize_observation(o_dict)

    # 7. Scheduled holding deadlines
    scheduled_holding_deadlines = set(float(t) for t in checkpoint.scheduled_holding_deadlines)

    # 8. Controller
    ctl_d = checkpoint.controller_dict
    ctl_type = ctl_d.get("type", "ZeroController")
    if ctl_type == "ObservableSelectorPolicy":
        controller = ObservableSelectorPolicy(
            roster_selection=bool(ctl_d.get("roster_selection", False)),
            broadcast_amplitude=float(ctl_d.get("broadcast_amplitude", 0.0)),
            selector=ctl_d["selector"],
            nominal_amplitude=float(ctl_d["nominal_amplitude"]),
            duration=float(ctl_d["duration"]),
            quota=float(ctl_d["quota"]),
            is_feedback=bool(ctl_d["is_feedback"]),
            initial_recruitment_fraction=float(ctl_d["initial_recruitment_fraction"]),
            component=int(ctl_d["component"]),
            seed=int(ctl_d["seed"]),
            node_fraction=float(ctl_d["node_fraction"]) if ctl_d.get("node_fraction") is not None else None,
        )
        controller.rng = controller_rng
        controller.s_star = float(ctl_d["s_star"]) if ctl_d.get("s_star") is not None else None
        controller.committed_end_time = float(ctl_d["committed_end_time"]) if ctl_d.get("committed_end_time") is not None else None
        controller.initial_actuation_done = bool(ctl_d.get("initial_actuation_done", False))
        controller.used_nodes = set(int(n) for n in ctl_d.get("used_nodes", []))
        controller.fallback_records = copy.deepcopy(ctl_d.get("fallback_records", []))
        controller.feedback_opportunities_encountered = int(ctl_d.get("feedback_opportunities_encountered", 0))
        controller.feedback_updates_performed = int(ctl_d.get("feedback_updates_performed", 0))
        controller.requested_settings = copy.deepcopy(ctl_d.get("requested_settings", {}))
        controller.effective_settings = copy.deepcopy(ctl_d.get("effective_settings", {}))
        if problem is not None:
            controller.problem = problem
    elif ctl_type == "OpenLoopPulseController":
        controller = OpenLoopPulseController(
            target_nodes=[int(n) for n in ctl_d.get("target_nodes", [])],
            nominal_amplitude=float(ctl_d["nominal_amplitude"]),
            duration=float(ctl_d["duration"]),
            signs={int(k): float(v) for k, v in ctl_d.get("signs", {}).items()},
            component=int(ctl_d.get("component", 0)),
        )
        controller.actuated = bool(ctl_d.get("actuated", False))
        controller.pulse_end_time = float(ctl_d["pulse_end_time"]) if ctl_d.get("pulse_end_time") is not None else None
        if problem is not None:
            controller.problem = problem
    else:
        controller = ZeroController()
        if problem is not None:
            controller.problem = problem

    # 9. Target Evaluator
    eval_d = checkpoint.target_evaluator_dict
    defn_d = eval_d["defn"]
    t_defn = TargetDefinition(
        target_type=TargetType(defn_d["target_type"]),
        holding_duration=float(defn_d["holding_duration"]),
        horizon=float(defn_d["horizon"]),
        release_deadline=float(defn_d["release_deadline"]),
        s_star=float(defn_d["s_star"]) if defn_d.get("s_star") is not None else None,
        threshold=float(defn_d.get("threshold", 0.5)),
        focal_indices=tuple(int(i) for i in defn_d["focal_indices"]) if defn_d.get("focal_indices") is not None else None,
        C_N_0=float(defn_d["C_N_0"]) if defn_d.get("C_N_0") is not None else None,
        L_0=set(int(x) for x in defn_d["L_0"]) if defn_d.get("L_0") is not None else None,
        initial_metrics=dict(defn_d.get("initial_metrics", {})),
        is_eligible=bool(defn_d.get("is_eligible", True)),
        ineligibility_reason=defn_d.get("ineligibility_reason"),
        is_already_target=bool(defn_d.get("is_already_target", False)),
    )
    target_evaluator = TargetEvaluator(
        target_definition=t_defn,
        problem_id=eval_d.get("problem_id", "problem_default"),
        policy_id=eval_d.get("policy_id", "policy_default"),
        origin_id=eval_d.get("origin_id", "origin_default"),
        future_id=int(eval_d.get("future_id", 0)),
    )
    target_evaluator.t_start_hold = float(eval_d["t_start_hold"]) if eval_d.get("t_start_hold") is not None else None
    target_evaluator.completion_time = float(eval_d["completion_time"]) if eval_d.get("completion_time") is not None else None
    target_evaluator.is_success = bool(eval_d.get("is_success", False))
    target_evaluator.relapsed = bool(eval_d.get("relapsed", False))
    target_evaluator.last_condition_value = bool(eval_d.get("last_condition_value", False))
    target_evaluator.is_released = bool(eval_d.get("is_released", False))
    target_evaluator.terminal_violation = float(eval_d.get("terminal_violation", 0.0))
    target_evaluator.history_events = [(float(t), bool(c)) for t, c in eval_d.get("history_events", [])]

    # 10. Scheduler
    sched_d = checkpoint.scheduler_dict
    phys_d = checkpoint.physics_dict
    tau_x_param = state.tau_x if state.tau_x is not None else 5.0
    nu_param = state.nu if state.nu is not None else 2.0
    adaptive_mem = phys_d.get("adaptive_memory", True)
    if not adaptive_mem:
        nu_param = 0.0
    scheduler_delta = float(phys_d.get("delta", 0.1)) if adaptive_mem else 0.0
    tau_y = float(phys_d.get("tau_y", 1.0))

    scheduler = PoissonScheduler(
        N=N,
        D=D,
        tau_x=tau_x_param,
        tau_y=tau_y,
        nu=nu_param,
        delta=scheduler_delta,
        rng=sim_rng,
    )
    scheduler.time = float(sched_d["time"])
    scheduler.set_total_memory(int(memory_mgr.total_memory))
    scheduler.seq_counter = int(sched_d.get("seq_counter", 0))

    stats_d = sched_d.get("stats", {})
    scheduler.stats.private_updates = int(stats_d.get("private_updates", 0))
    scheduler.stats.public_updates = int(stats_d.get("public_updates", 0))
    scheduler.stats.encounters = int(stats_d.get("encounters", 0))
    scheduler.stats.null_encounters = int(stats_d.get("null_encounters", 0))
    scheduler.stats.memory_decays = int(stats_d.get("memory_decays", 0))
    scheduler.stats.participation_updates = int(stats_d.get("participation_updates", 0))
    scheduler.stats.deterministic_events = int(stats_d.get("deterministic_events", 0))
    scheduler.stats.preemptions = int(stats_d.get("preemptions", 0))

    scheduler.set_pending_clock_state(sched_d.get("pending_clock"))

    # Populate raw deterministic events in pq (callbacks reconstructed in next step)
    for ev_d in sched_d.get("deterministic_events", []):
        event = DeterministicEvent(
            time=float(ev_d["time"]),
            priority=int(ev_d["priority"]),
            seq_id=int(ev_d["seq_id"]),
            event_type=str(ev_d["event_type"]),
            payload=dict(ev_d.get("payload", {})),
            callback=None,
        )
        heapq.heappush(scheduler.pq, event)

    # 11. Rollout Recorder
    recorder = None
    rec_d = checkpoint.recorder_dict
    if rec_d is not None:
        cfg = RecorderConfig(**rec_d["config"])
        recorder = RolloutRecorder(cfg)
        for sp in rec_d.get("scalar_points", []):
            recorder.scalar_points.append(ScalarTrajectoryPoint(
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
            recorder.action_history.append(ActionRecord(
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
            recorder.network_frames.append(NetworkFrame(
                t=float(nf["t"]),
                x=np.asarray(nf["x"], dtype=np.float64),
                y=np.asarray(nf["y"], dtype=np.float64),
                W=np.asarray(nf["W"], dtype=np.uint32),
                controlled_mask=np.asarray(nf["controlled_mask"], dtype=bool),
                influence_inputs=np.asarray(nf["influence_inputs"], dtype=np.float64),
            ))
        recorder._initial_sign = float(rec_d["_initial_sign"]) if rec_d.get("_initial_sign") is not None else None
        recorder._event_exact_min_mx = float(rec_d.get("_event_exact_min_mx", float("inf")))
        recorder._event_exact_min_my = float(rec_d.get("_event_exact_min_my", float("inf")))
        recorder._event_exact_max_exc = float(rec_d.get("_event_exact_max_exc", 0.0))
        recorder._event_exact_retained = bool(rec_d.get("_event_exact_retained", True))
        recorder._event_exact_loss_time = float(rec_d["_event_exact_loss_time"]) if rec_d.get("_event_exact_loss_time") is not None else None
        recorder._next_record_time = float(rec_d.get("_next_record_time", 0.0))
        recorder._last_sampled_time = float(rec_d["_last_sampled_time"]) if rec_d.get("_last_sampled_time") is not None else None

    return (
        state,
        memory_mgr,
        scheduler,
        ledger,
        observation_model,
        controller,
        target_evaluator,
        recorder,
        sim_rng,
        controller_rng,
        pending_observations,
        scheduled_holding_deadlines,
    )


def reconstruct_engine_callbacks(
    scheduler: PoissonScheduler,
    state: State,
    memory_mgr: MemoryManager,
    ledger: BudgetLedger,
    target_evaluator: TargetEvaluator,
    observation_model: ObservationModel,
    controller: Controller,
    problem: ControlProblem,
    physics: Dict[str, Any],
    sim_rng: np.random.Generator,
    pending_observations: Dict[float, Observation],
    scheduled_holding_deadlines: Set[float],
    recorder: Optional[RolloutRecorder] = None,
) -> None:
    """Explicitly wire stochastic and deterministic event callbacks without closures."""
    N, D = state.N, state.D
    J = float(physics.get("J", 1.0))
    T = float(physics.get("T", 0.3))
    K_arr = state.K if state.K is not None else np.full(N, 0.5)
    g_arr = state.g if state.g is not None else np.ones(N)
    b_arr = state.b if state.b is not None else np.zeros((N, D))

    from src.control.advanced.structural import StructuralActuator
    structural_actuator = StructuralActuator(rng=sim_rng)

    def check_and_schedule_holding_deadline(current_t: float) -> None:
        if (
            ledger.is_released
            and target_evaluator.t_start_hold is not None
            and not target_evaluator.is_success
        ):
            t_comp = target_evaluator.t_start_hold + problem.holding_duration
            if (
                t_comp <= problem.horizon
                and t_comp not in scheduled_holding_deadlines
                and t_comp > current_t - 1e-12
            ):
                scheduled_holding_deadlines.add(t_comp)
                scheduler.schedule_deterministic_event(
                    time=t_comp,
                    priority=PRIORITY_HOLDING,
                    event_type="holding_deadline",
                    callback=lambda s, e: (
                        target_evaluator.process_holding_check(s.time, ledger.is_released),
                        check_and_schedule_holding_deadline(s.time),
                    ),
                )

    B_mat: Optional[np.ndarray] = None
    if D == 3:
        try:
            from src.core.couplings import get_coupling_matrix
            B_mat = get_coupling_matrix("B+", D=3)
        except Exception:
            B_mat = None
    L_int = 1.0

    def cb_private(node_idx: int) -> None:
        ledger.advance_to(scheduler.time)
        state.t = scheduler.time
        i = node_idx
        alpha = int(sim_rng.integers(0, D))
        f_x = float(K_arr[i] * state.y[i, alpha] + b_arr[i, alpha])
        if B_mat is not None and D > 1:
            f_x += L_int * float(np.dot(B_mat[alpha], state.x[i, :]))
        kappa = f_x / T
        if state.state_mode == StateMode.BINARY:
            new_x = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
        else:
            new_x = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))
        state.x[i, alpha] = new_x
        target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
        if recorder is not None:
            recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
        check_and_schedule_holding_deadline(scheduler.time)

    def cb_public(node_idx: int) -> None:
        ledger.advance_to(scheduler.time)
        state.t = scheduler.time
        i = node_idx
        W_row = memory_mgr.W[i, :]
        alpha = int(sim_rng.integers(0, D))
        u_val = float(ledger.current_field[i, alpha])
        f_y = compute_single_public_field(
            W_row=W_row,
            y_col=state.y[:, alpha],
            x_val=float(state.x[i, alpha]),
            J=J,
            K=float(K_arr[i]),
            h_val=0.0,
            g_val=float(g_arr[i]),
            u_val=u_val,
        )
        kappa = f_y / T
        if state.state_mode == StateMode.BINARY:
            new_y = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
        else:
            new_y = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))
        state.y[i, alpha] = new_y
        target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
        if recorder is not None:
            recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
        check_and_schedule_holding_deadline(scheduler.time)

    def cb_encounter(initiator_idx: int) -> None:
        ledger.advance_to(scheduler.time)
        state.t = scheduler.time
        res = memory_mgr.sample_and_apply_encounter(
            i=initiator_idx,
            y=state.y,
            rng=sim_rng,
        )
        if not res.is_null:
            scheduler.set_total_memory(int(memory_mgr.total_memory))
            state.W = memory_mgr.W
            target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(scheduler.time)

    def cb_decay() -> None:
        ledger.advance_to(scheduler.time)
        state.t = scheduler.time
        i_dec, j_dec = memory_mgr.sample_and_apply_decay(rng=sim_rng)
        if i_dec != -1:
            scheduler.set_total_memory(int(memory_mgr.total_memory))
            state.W = memory_mgr.W
            target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(scheduler.time)

    scheduler.register_callbacks(
        cb_private=cb_private,
        cb_public=cb_public,
        cb_encounter=cb_encounter,
        cb_decay=cb_decay,
    )

    def make_snapshot_callback(t_snap: float):
        def _cb(sched: PoissonScheduler, evt: Any = None) -> None:
            ledger.advance_to(sched.time)
            obs = observation_model.sample_observation(
                state=state,
                current_time=sched.time,
                remaining_exposure=ledger.remaining_exposure,
                remaining_distinct_nodes=ledger.remaining_distinct_nodes,
                current_spending_rate=ledger.current_spending_rate,
            )
            delivery_t = obs.delivery_time
            pending_observations[delivery_t] = obs

            if delivery_t <= problem.horizon:
                sched.schedule_deterministic_event(
                    time=delivery_t,
                    priority=PRIORITY_INTERVENTION,
                    event_type="observation_delivery",
                    payload={"delivery_time": delivery_t},
                    callback=make_delivery_callback(delivery_t),
                )
        return _cb

    def make_delivery_callback(t_del: float):
        def _cb(sched: PoissonScheduler, evt: Any) -> None:
            ledger.advance_to(sched.time)
            obs = pending_observations.pop(t_del, None)
            if obs is not None and not ledger.is_released and sched.time < problem.release_deadline:
                action = controller.act(obs, ledger)
                eff_end = ledger.apply_action(action, current_time=sched.time)
                if recorder is not None:
                    recorder.record_action(
                        action=action,
                        decision_time=sched.time,
                        report_sample_time=obs.sample_time if hasattr(obs, "sample_time") else sched.time,
                        effective_end=eff_end,
                        ledger=ledger,
                        reason="observation_delivery",
                    )
                if action.structural_entries:
                    structural_actuator.apply_action(
                        action=action,
                        state=state,
                        memory_mgr=memory_mgr,
                        scheduler=sched,
                        ledger=ledger,
                    )
                if eff_end is not None and eff_end > sched.time and eff_end <= problem.horizon:
                    def _shutoff_delivery(s: PoissonScheduler, e: Any) -> None:
                        ledger.advance_to(s.time)
                        if recorder is not None:
                            recorder.record_shutoff(s.time, ledger, reason="field_shutoff")
                        target_evaluator.process_state_update(state, s.time, ledger.is_released)
                        if recorder is not None:
                            recorder.on_event_state(state, s.time, ledger, target_evaluator)
                        check_and_schedule_holding_deadline(s.time)

                    sched.schedule_deterministic_event(
                        time=eff_end,
                        priority=PRIORITY_RESTORE,
                        event_type="field_shutoff",
                        callback=_shutoff_delivery,
                    )
                target_evaluator.process_state_update(state, sched.time, ledger.is_released)
                if recorder is not None:
                    recorder.on_event_state(state, sched.time, ledger, target_evaluator)
                check_and_schedule_holding_deadline(sched.time)
        return _cb

    def on_release_deadline(s: PoissonScheduler, e: Any) -> None:
        ledger.release_control(s.time)
        if recorder is not None:
            recorder.record_shutoff(s.time, ledger, reason="release_deadline")
        target_evaluator.process_state_update(state, s.time, ledger.is_released)
        if recorder is not None:
            recorder.on_event_state(state, s.time, ledger, target_evaluator)
        check_and_schedule_holding_deadline(s.time)

    def _make_field_shutoff_cb():
        def _cb(s: PoissonScheduler, e: Any) -> None:
            ledger.advance_to(s.time)
            if recorder is not None:
                recorder.record_shutoff(s.time, ledger, reason="field_shutoff")
            target_evaluator.process_state_update(state, s.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, s.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(s.time)
        return _cb

    # Reconstruct callbacks on all events in min-heap pq
    for ev in scheduler.pq:
        if ev.event_type == "observation_snapshot":
            ev.callback = make_snapshot_callback(ev.time)
        elif ev.event_type == "observation_delivery":
            del_t = float(ev.payload.get("delivery_time", ev.time))
            ev.callback = make_delivery_callback(del_t)
        elif ev.event_type == "field_shutoff":
            ev.callback = _make_field_shutoff_cb()
        elif ev.event_type == "release_deadline":
            ev.callback = on_release_deadline
        elif ev.event_type == "holding_check":
            ev.callback = lambda s, e: (
                target_evaluator.process_holding_check(s.time, ledger.is_released),
                check_and_schedule_holding_deadline(s.time),
            )
        elif ev.event_type == "holding_deadline":
            ev.callback = lambda s, e: (
                target_evaluator.process_holding_check(s.time, ledger.is_released),
                check_and_schedule_holding_deadline(s.time),
            )
        elif ev.event_type == "horizon":
            ev.callback = lambda s, e: None
