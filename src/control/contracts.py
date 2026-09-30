"""Typed contracts and ownership interfaces for optimal transformation control.

Defines:
- ControlProblem: Immutable specification of the control task, target, and constraints.
- Action: Sorted unique (node, component, field) tuples and structural edge operations.
- ActionEntry: Single thermodynamic field intervention entry.
- StructuralOp: Types of structural modifications (force, sever, rewire).
- StructuralActionEntry: Direct network edge manipulation entry.
- Observation: Copy/read-only restricted or oracle views of the physical system.
- RolloutResult: Standardized execution outcome record with holding and persistence certification.
- SearchState: Deterministic candidate tracking and optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


class TargetType(str, Enum):
    """Supported transformation target endpoints."""
    REVERSAL = "reversal"
    DECENTRALIZATION = "decentralization"
    TURNOVER = "turnover"


class ObservationMode(str, Enum):
    """Observation visibility regimes."""
    SAMPLED_PUBLIC = "sampled_public"
    COMPLETE_PUBLIC = "complete_public"
    ORACLE = "oracle"


class ActuatorClass(str, Enum):
    """Intervention actuator classes."""
    FIELD = "field"
    SUPPRESS_ENCOUNTERS = "suppress_encounters"
    DELETE_MEMORY = "delete_memory"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


class StructuralOp(str, Enum):
    """Structural intervention operation types."""
    FORCE = "force"       # Force/reinforce direct memory tie between source and target
    SEVER = "sever"       # Sever/decay direct memory tie between source and target
    REWIRE = "rewire"     # Disconnect (source, target) and reconnect (source, rewire_target)


@dataclass(frozen=True, order=True)
class StructuralActionEntry:
    """Single structural actuator entry for direct network edge manipulation.

    Invariants:
    - Self-loops are strictly prohibited (source != target).
    - Modification count must be a positive integer (count >= 1).
    - For REWIRE operations, rewire_target must be defined and distinct from source.
    """
    source: int
    target: int
    operation: StructuralOp = StructuralOp.FORCE
    count: int = 1
    rewire_target: Optional[int] = None

    def __post_init__(self) -> None:
        if self.source < 0:
            raise ValueError(f"Structural source node must be non-negative, got {self.source}")
        if self.target < 0:
            raise ValueError(f"Structural target node must be non-negative, got {self.target}")
        if self.source == self.target:
            raise ValueError(f"Self-loops are strictly prohibited: ({self.source}, {self.target})")
        if self.count <= 0:
            raise ValueError(f"Structural operation count must be >= 1, got {self.count}")
        if self.operation == StructuralOp.REWIRE:
            if self.rewire_target is None:
                raise ValueError("rewire_target must be specified for REWIRE operations")
            if self.rewire_target < 0:
                raise ValueError(f"rewire_target must be non-negative, got {self.rewire_target}")
            if self.source == self.rewire_target:
                raise ValueError(f"Rewire target cannot form a self-loop with source: ({self.source}, {self.rewire_target})")
            if self.target == self.rewire_target:
                raise ValueError(f"Rewire target cannot be identical to current target: ({self.target})")
        elif self.rewire_target is not None:
            raise ValueError("rewire_target can only be specified for REWIRE operations")


@dataclass(frozen=True, order=True)
class ActionEntry:
    """Single actuator entry for node and belief component."""
    node: int
    component: int
    field: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.field):
            raise ValueError(f"Action field must be finite, got {self.field}")


@dataclass(frozen=True)
class Action:
    """Action returned by controller at an actuation or delivery time.

    Invariants:
    - Field entries are sorted unique by (node, component).
    - Fields replace previous field vector; they do not accumulate.
    - Nonfinite fields and out-of-bounds nodes/components are rejected.
    - Structural entries represent discrete edge manipulations.
    """
    entries: Tuple[ActionEntry, ...] = ()
    intended_end_time: Optional[float] = None
    release: bool = False
    structural_entries: Tuple[StructuralActionEntry, ...] = ()

    def __post_init__(self) -> None:
        # Enforce unique, sorted field entries
        seen: Set[Tuple[int, int]] = set()
        for e in self.entries:
            key = (e.node, e.component)
            if key in seen:
                raise ValueError(f"Duplicate (node, component) in action entries: {key}")
            seen.add(key)
        sorted_entries = tuple(sorted(self.entries, key=lambda e: (e.node, e.component)))
        if sorted_entries != self.entries:
            object.__setattr__(self, "entries", sorted_entries)

        # Enforce unique, sorted structural entries
        if self.structural_entries:
            struct_seen: Set[Tuple[int, int, str, Optional[int]]] = set()
            for se in self.structural_entries:
                s_key = (se.source, se.target, se.operation.value, se.rewire_target)
                if s_key in struct_seen:
                    raise ValueError(f"Duplicate structural action entry: {s_key}")
                struct_seen.add(s_key)
            sorted_struct = tuple(
                sorted(
                    self.structural_entries,
                    key=lambda se: (
                        se.source,
                        se.target,
                        se.operation.value,
                        se.rewire_target if se.rewire_target is not None else -1,
                    ),
                )
            )
            if sorted_struct != self.structural_entries:
                object.__setattr__(self, "structural_entries", sorted_struct)

    @classmethod
    def from_dict(
        cls,
        fields_dict: Dict[Tuple[int, int], float],
        intended_end_time: Optional[float] = None,
        release: bool = False,
        structural_entries: Sequence[StructuralActionEntry] = (),
    ) -> Action:
        """Construct Action from a dictionary mapping (node, component) -> field value."""
        entries = [
            ActionEntry(node=n, component=c, field=float(val))
            for (n, c), val in fields_dict.items()
            if abs(float(val)) > 1e-15
        ]
        return cls(entries=tuple(sorted(entries, key=lambda e: (e.node, e.component))),
                   intended_end_time=intended_end_time,
                   release=release,
                   structural_entries=tuple(structural_entries))

    @classmethod
    def from_array(
        cls,
        field_arr: np.ndarray,
        intended_end_time: Optional[float] = None,
        release: bool = False,
        structural_entries: Sequence[StructuralActionEntry] = (),
    ) -> Action:
        """Construct Action from a dense array of shape (N,) or (N, D)."""
        arr = np.asarray(field_arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        N, D = arr.shape
        entries: List[ActionEntry] = []
        nonzero_idx = np.argwhere(np.abs(arr) > 1e-15)
        for i, alpha in nonzero_idx:
            entries.append(ActionEntry(node=int(i), component=int(alpha), field=float(arr[i, alpha])))
        return cls(entries=tuple(sorted(entries, key=lambda e: (e.node, e.component))),
                   intended_end_time=intended_end_time,
                   release=release,
                   structural_entries=tuple(structural_entries))

    @classmethod
    def structural(
        cls,
        entries: Sequence[StructuralActionEntry],
        intended_end_time: Optional[float] = None,
        release: bool = False,
    ) -> Action:
        """Construct Action containing purely structural edge manipulations."""
        return cls(
            entries=(),
            intended_end_time=intended_end_time,
            release=release,
            structural_entries=tuple(entries),
        )

    def to_array(self, N: int, D: int = 1) -> np.ndarray:
        """Convert action to dense NumPy array of shape (N, D)."""
        arr = np.zeros((N, D), dtype=np.float64)
        for e in self.entries:
            if 0 <= e.node < N and 0 <= e.component < D:
                arr[e.node, e.component] = e.field
        return arr

    def actuated_nodes(self) -> Set[int]:
        """Return set of distinct node indices that receive nonzero field."""
        return {e.node for e in self.entries if abs(e.field) > 1e-15}

    def structural_nodes(self) -> Set[int]:
        """Return set of distinct node indices touched by structural entries."""
        nodes: Set[int] = set()
        for se in self.structural_entries:
            nodes.add(se.source)
            nodes.add(se.target)
            if se.rewire_target is not None:
                nodes.add(se.rewire_target)
        return nodes

    @property
    def spending_rate(self) -> float:
        """Instantaneous exposure spending rate c_u = sum |u_{i, alpha}|."""
        return sum(abs(e.field) for e in self.entries)

    @property
    def structural_spending(self) -> int:
        """Total structural modifications requested in this action."""
        total = 0
        for se in self.structural_entries:
            if se.operation == StructuralOp.REWIRE:
                total += 2 * se.count
            else:
                total += se.count
        return total

    def validate(
        self,
        N: int,
        D: int,
        amplitude_cap: Optional[float] = None,
        accessible_roster: Optional[Set[int]] = None,
    ) -> None:
        """Validate action bounds, finiteness, and accessibility."""
        node_total_amp: Dict[int, float] = {}
        for e in self.entries:
            if not np.isfinite(e.field):
                raise ValueError(f"Non-finite field {e.field} on node {e.node}")
            if e.node < 0 or e.node >= N:
                raise ValueError(f"Node index {e.node} out of bounds [0, {N})")
            if e.component < 0 or e.component >= D:
                raise ValueError(f"Component index {e.component} out of bounds [0, {D})")
            if amplitude_cap is not None and abs(e.field) > amplitude_cap + 1e-10:
                raise ValueError(
                    f"Field amplitude {abs(e.field)} exceeds cap {amplitude_cap} on node {e.node}"
                )
            if accessible_roster is not None and e.node not in accessible_roster and abs(e.field) > 1e-15:
                raise ValueError(f"Node {e.node} is not in the accessible roster")
            node_total_amp[e.node] = node_total_amp.get(e.node, 0.0) + abs(e.field)

        if amplitude_cap is not None:
            for node, tot_amp in node_total_amp.items():
                if tot_amp > amplitude_cap + 1e-10:
                    raise ValueError(
                        f"Total per-node field amplitude {tot_amp:.4f} exceeds per-node amplitude cap "
                        f"{amplitude_cap} on node {node}"
                    )

        # Validate structural entries
        for se in self.structural_entries:
            if se.source < 0 or se.source >= N:
                raise ValueError(f"Structural source node {se.source} out of bounds [0, {N})")
            if se.target < 0 or se.target >= N:
                raise ValueError(f"Structural target node {se.target} out of bounds [0, {N})")
            if se.rewire_target is not None and (se.rewire_target < 0 or se.rewire_target >= N):
                raise ValueError(f"Structural rewire_target {se.rewire_target} out of bounds [0, {N})")
            if accessible_roster is not None and se.source not in accessible_roster:
                raise ValueError(f"Structural source node {se.source} is not in the accessible roster")


@dataclass(frozen=True)
class Observation:
    """Restricted or oracle observation delivered to a controller.

    Restricted views are isolated copies with no backward references to mutable physical state.
    """
    time: float
    public_states: np.ndarray  # (N, D), NaN for unobserved nodes in sampled mode
    panel_mask: np.ndarray     # (N,), bool: True if node in observation panel
    observed_edges: Dict[Tuple[int, int], int]  # (i, j) -> revealed W_ij count
    accessible_roster: Tuple[int, ...]          # Roster of intervention-eligible nodes
    group_labels: Optional[np.ndarray] = None   # (N,), opportunity groups
    sample_time: float = 0.0
    delivery_time: float = 0.0
    remaining_exposure: float = 0.0
    remaining_distinct_nodes: int = 0
    current_spending_rate: float = 0.0
    noise_model_version: str = "1.0"
    is_oracle: bool = False
    private_states: Optional[np.ndarray] = None # None in non-oracle mode
    true_memory: Optional[np.ndarray] = None    # None in non-oracle mode

    def is_node_observed(self, node: int) -> bool:
        """Whether node's public state is revealed in this observation."""
        return bool(self.panel_mask[node])

    def is_node_accessible(self, node: int) -> bool:
        """Whether node is in the intervention accessible roster."""
        return node in self.accessible_roster

    def observed_public_mean(self, component: int = 0) -> Optional[float]:
        """Mean public state over observed panel nodes.

        Missing-data semantics (audit finding F3): if the report contains no
        finite observed values (empty panel, or all panel members unobserved),
        this returns ``None`` — the report is *unavailable*, and a majority of
        0.0 is NEVER fabricated for it. Callers must treat ``None`` as "wait
        for the next report", not as "the population mean is zero".

        A tied panel (finite values averaging exactly 0.0) is a legitimate
        observed value and is returned as ``0.0``. Whether a tie suffices to
        fix a target *direction* is a controller-level decision (see
        ObservableSelectorPolicy._observed_target_mean for the explicit tie
        rule: a tie is ambiguous and does not lock a direction).
        """
        observed_vals = self.public_states[self.panel_mask, component]
        valid = observed_vals[np.isfinite(observed_vals)]
        if len(valid) == 0:
            return None
        return float(np.mean(valid))


@dataclass(frozen=True)
class ControlProblem:
    """Complete specification of a control problem instance. Immutable once started."""
    problem_id: str
    target_type: TargetType
    target_params: Dict[str, Any]
    N: int
    D: int = 1
    k: int = 1                         # Max distinct nodes ever actuated
    exposure_cap: float = 10.0         # Cal B
    amplitude_cap: float = 4.0         # u_max
    release_deadline: float = 20.0     # T_c
    holding_duration: float = 50.0     # Delta
    horizon: float = 270.0             # H
    accessible_roster: Tuple[int, ...] = ()
    actuator_class: ActuatorClass = ActuatorClass.FIELD
    observation_mode: ObservationMode = ObservationMode.SAMPLED_PUBLIC
    proposal_seed: int = 42
    search_seed: int = 42
    config_version: str = "realism-control-proposal-2.0"
    model_hash: str = ""
    structural_budget: Optional[float] = None
    max_structural_per_step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if self.k < 1 or self.k > self.N:
            raise ValueError(f"k must be in [1, N], got {self.k} (N={self.N})")
        if self.exposure_cap < 0.0:
            raise ValueError(f"exposure_cap cannot be negative, got {self.exposure_cap}")
        if self.release_deadline < 0.0:
            raise ValueError(f"release_deadline must be >= 0, got {self.release_deadline}")
        if self.holding_duration <= 0.0:
            raise ValueError(f"holding_duration must be > 0, got {self.holding_duration}")
        if self.horizon <= self.release_deadline:
            raise ValueError(f"horizon {self.horizon} must be > release_deadline {self.release_deadline}")
        if self.structural_budget is not None and self.structural_budget < 0.0:
            raise ValueError(f"structural_budget cannot be negative, got {self.structural_budget}")
        if self.max_structural_per_step is not None and self.max_structural_per_step < 1:
            raise ValueError(f"max_structural_per_step must be >= 1, got {self.max_structural_per_step}")

        if not self.model_hash:
            # Generate deterministic hash of problem configuration
            repr_dict = {
                "problem_id": self.problem_id,
                "target_type": self.target_type.value,
                "target_params": self.target_params,
                "N": self.N,
                "D": self.D,
                "k": self.k,
                "exposure_cap": self.exposure_cap,
                "amplitude_cap": self.amplitude_cap,
                "release_deadline": self.release_deadline,
                "holding_duration": self.holding_duration,
                "horizon": self.horizon,
                "accessible_roster": list(self.accessible_roster),
                "config_version": self.config_version,
            }
            if self.structural_budget is not None:
                repr_dict["structural_budget"] = self.structural_budget
            if self.max_structural_per_step is not None:
                repr_dict["max_structural_per_step"] = self.max_structural_per_step
            h = hashlib.sha256(json.dumps(repr_dict, sort_keys=True).encode("utf-8")).hexdigest()[:16]
            object.__setattr__(self, "model_hash", h)


@dataclass
class RolloutResult:
    """Standardized result of a single rollout execution."""
    problem_id: str
    policy_id: str
    origin_id: str
    future_id: int
    is_valid: bool
    is_eligible: bool
    success: bool
    completion_time: Optional[float]       # T_G (or None if censored)
    restricted_time: float                 # min(T_G, H), failing rollouts assign H
    expenditure: float                     # Cumulative exposure spent
    distinct_nodes_used: int               # Distinct nodes actuated
    holding_start_time: Optional[float] = None
    relapsed: bool = False
    terminal_target_violation: float = 0.0 # d_G for plateau tie-breaking
    event_counts: Dict[str, int] = field(default_factory=dict)
    summaries: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    structural_modifications: int = 0
    recorder: Optional[Any] = None


@dataclass
class SearchState:
    """Deterministic policy search registry and incumbent state."""
    candidate_registry: Dict[str, Any] = field(default_factory=dict)
    completed_evaluations: List[RolloutResult] = field(default_factory=list)
    incumbent_policy_id: Optional[str] = None
    incumbent_score: Optional[Tuple[Any, ...]] = None
    evaluation_budget: int = 100
    evaluations_used: int = 0
    proposal_step: int = 0
