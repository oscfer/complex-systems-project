"""Unified State Container for Emergent & Collective Order Simulator (Stages 0–4).

Provides the State dataclass holding private beliefs (x), public expressions (y),
interaction memory (W), collective action participation (n), Q-value tables (Q),
simulation time (t), institutional holding clocks, and serialization routines.

Simulation Stage Mapping:
    - Stage 0: Benchmark runs (state allocation, memory profiling, copy performance).
    - Stage 1: Core emergence (beliefs x, y in binary/continuous, memory matrix W).
    - Stage 2: Extension campaigns (participation states n, institutional transitions).
    - Stage 3: Adaptive learning (Q-tables for experience accumulation).
    - Stage 4: Diagnostics & analytics (discrepancy property d_2 calculation).
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


class StateMode(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass
class State:
    """Simulation state container for a population of N agents across D belief dimensions (Stages 0–4).

    Attributes:
        N: Population size (int >= 1).
        D: Belief dimensionality (int in {1, 3}).
        state_mode: StateMode.BINARY or StateMode.CONTINUOUS.
        x: Private belief array of shape (N, D), dtype float64.
           Binary mode: values strictly in {-1.0, +1.0}.
           Continuous mode: values strictly in [-1.0, 1.0].
        y: Expressed belief array of shape (N, D), dtype float64.
           Binary mode: values strictly in {-1.0, +1.0}.
           Continuous mode: values strictly in [-1.0, 1.0].
        W: Interaction memory count matrix of shape (N, N), dtype uint32.
           W[i, j] records encounters of agent i attending to agent j.
           Invariant: W[i, i] == 0 for all i, W[i, j] >= 0.
        n: Participation action state array of shape (N,), dtype uint8.
           Values strictly in {0, 1}.
        Q: Q-value table for expressed action updates of shape (N, D, 2), dtype float64.
           Action index 0 corresponds to s = -1; index 1 corresponds to s = +1.
        t: Current continuous simulation time, float64 >= 0.0.

        holding_active: Whether an institutional threshold holding interval is active.
        t_start_hold: Simulation time when threshold was crossed, or None.
        t_deadline: Scheduled institutional transition deadline, or None.
        deadline_generation: Monotonic generation counter for lazy deadline cancellation.
        transition_occurred: Whether an institutional transition has occurred.
        t_transition: Timestamp of recorded institutional transition, or None.

        pulse_active: Whether an external field pulse is currently active.
        pulse_val: External field amplitude currently applied during pulse.
        pulse_component: Belief component index target of pulse (0-indexed).
        pulse_targets: Optional boolean mask of shape (N,) or index array of pulsed nodes.
    """

    N: int
    D: int
    state_mode: StateMode
    x: np.ndarray
    y: np.ndarray
    W: np.ndarray
    n: np.ndarray
    Q: np.ndarray
    t: float = 0.0

    holding_active: bool = False
    t_start_hold: Optional[float] = None
    t_deadline: Optional[float] = None
    deadline_generation: int = 0
    transition_occurred: bool = False
    t_transition: Optional[float] = None

    pulse_active: bool = False
    pulse_val: float = 0.0
    pulse_component: int = 0
    pulse_targets: Optional[np.ndarray] = None

    blocks: Optional[np.ndarray] = None
    hub_idx: Optional[int] = None

    # Quenched heterogeneity traits & bounded memory caps (Realism Physics Revision)
    K: Optional[np.ndarray] = None
    tau_x: Optional[np.ndarray] = None
    nu: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    row_caps: Optional[np.ndarray] = None
    opportunity_groups: Optional[np.ndarray] = None
    opportunity_matrix: Optional[np.ndarray] = None
    scenario_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.state_mode, str):
            self.state_mode = StateMode(self.state_mode.lower())
        if hasattr(self, "x") and self.x is not None:
            if isinstance(self.x, np.ndarray) and self.x.dtype == bool:
                self.x = np.where(self.x, 1.0, -1.0)
            elif isinstance(self.x, (list, tuple)) and len(self.x) > 0 and isinstance(self.x[0], bool):
                self.x = np.where(np.array(self.x, dtype=bool), 1.0, -1.0)
            elif not isinstance(self.x, np.ndarray):
                self.x = np.array(self.x, dtype=np.float64)
            elif self.x.dtype != np.float64:
                self.x = self.x.astype(np.float64)
            if self.x.ndim == 1:
                if self.D == 1 and self.x.size == self.N:
                    self.x = self.x.reshape(self.N, 1)
                elif self.x.size == self.N * self.D:
                    self.x = self.x.reshape(self.N, self.D)
        if hasattr(self, "y") and self.y is not None:
            if isinstance(self.y, np.ndarray) and self.y.dtype == bool:
                self.y = np.where(self.y, 1.0, -1.0)
            elif isinstance(self.y, (list, tuple)) and len(self.y) > 0 and isinstance(self.y[0], bool):
                self.y = np.where(np.array(self.y, dtype=bool), 1.0, -1.0)
            elif not isinstance(self.y, np.ndarray):
                self.y = np.array(self.y, dtype=np.float64)
            elif self.y.dtype != np.float64:
                self.y = self.y.astype(np.float64)
            if self.y.ndim == 1:
                if self.D == 1 and self.y.size == self.N:
                    self.y = self.y.reshape(self.N, 1)
                elif self.y.size == self.N * self.D:
                    self.y = self.y.reshape(self.N, self.D)
        if hasattr(self, "W") and self.W is not None:
            if not isinstance(self.W, np.ndarray):
                self.W = np.array(self.W, dtype=np.uint32)
            elif self.W.dtype != np.uint32:
                self.W = self.W.astype(np.uint32)
            if self.W.ndim == 1 and self.W.size == self.N * self.N:
                self.W = self.W.reshape(self.N, self.N)
        if hasattr(self, "n") and self.n is not None:
            if not isinstance(self.n, np.ndarray):
                self.n = np.array(self.n, dtype=np.uint8)
            elif self.n.dtype != np.uint8:
                self.n = self.n.astype(np.uint8)
        if hasattr(self, "Q") and self.Q is not None:
            if not isinstance(self.Q, np.ndarray):
                self.Q = np.array(self.Q, dtype=np.float64)
            elif self.Q.dtype != np.float64:
                self.Q = self.Q.astype(np.float64)
        if hasattr(self, "blocks") and self.blocks is not None:
            if not isinstance(self.blocks, np.ndarray):
                self.blocks = np.array(self.blocks, dtype=np.int64)
            elif self.blocks.dtype != np.int64:
                self.blocks = self.blocks.astype(np.int64)
        if hasattr(self, "K") and self.K is not None:
            if not isinstance(self.K, np.ndarray):
                self.K = np.array(self.K, dtype=np.float64)
            elif self.K.dtype != np.float64:
                self.K = self.K.astype(np.float64)
        if hasattr(self, "tau_x") and self.tau_x is not None:
            if not isinstance(self.tau_x, np.ndarray):
                self.tau_x = np.array(self.tau_x, dtype=np.float64)
            elif self.tau_x.dtype != np.float64:
                self.tau_x = self.tau_x.astype(np.float64)
        if hasattr(self, "nu") and self.nu is not None:
            if not isinstance(self.nu, np.ndarray):
                self.nu = np.array(self.nu, dtype=np.float64)
            elif self.nu.dtype != np.float64:
                self.nu = self.nu.astype(np.float64)
        if hasattr(self, "g") and self.g is not None:
            if not isinstance(self.g, np.ndarray):
                self.g = np.array(self.g, dtype=np.float64)
            elif self.g.dtype != np.float64:
                self.g = self.g.astype(np.float64)
        if hasattr(self, "b") and self.b is not None:
            if not isinstance(self.b, np.ndarray):
                self.b = np.array(self.b, dtype=np.float64)
            elif self.b.dtype != np.float64:
                self.b = self.b.astype(np.float64)
            if self.b.ndim == 1 and self.b.size == self.N * self.D:
                self.b = self.b.reshape(self.N, self.D)
        if hasattr(self, "row_caps") and self.row_caps is not None:
            if not isinstance(self.row_caps, np.ndarray):
                self.row_caps = np.array(self.row_caps, dtype=np.uint32)
            elif self.row_caps.dtype != np.uint32:
                self.row_caps = self.row_caps.astype(np.uint32)
        if hasattr(self, "opportunity_groups") and self.opportunity_groups is not None:
            if not isinstance(self.opportunity_groups, np.ndarray):
                self.opportunity_groups = np.array(self.opportunity_groups, dtype=np.int64)
            elif self.opportunity_groups.dtype != np.int64:
                self.opportunity_groups = self.opportunity_groups.astype(np.int64)
            if self.blocks is None:
                self.blocks = self.opportunity_groups
        elif hasattr(self, "blocks") and self.blocks is not None:
            if self.opportunity_groups is None:
                self.opportunity_groups = self.blocks
        if self.opportunity_matrix is not None:
            self.opportunity_matrix = np.asarray(self.opportunity_matrix, dtype=np.float64)


    @classmethod
    def create_empty(
        cls,
        N: int,
        D: int = 1,
        state_mode: Union[StateMode, str] = StateMode.BINARY,
        default_spin: Optional[float] = None,
        dtype: np.dtype = np.float64,
    ) -> State:
        """Create an empty state with zero memory and valid default beliefs.

        Args:
            N: Population size (int >= 1).
            D: Belief dimensionality (int in {1, 3}).
            state_mode: StateMode.BINARY or StateMode.CONTINUOUS.
            default_spin: Optional scalar value to populate private states `x` and public
                states `y`. For StateMode.BINARY, defaults to -1.0 if None, ensuring the
                returned state satisfies all invariant constraints under `validate()`.
                Pass `default_spin=0.0` to explicitly allocate an unpopulated buffer shell
                (which can be verified using `validate(allow_uninitialized=True)`).
                For StateMode.CONTINUOUS, defaults to 0.0 (neutral baseline) if None.
            dtype: Floating-point NumPy dtype for belief arrays.

        Returns:
            State: Initialized simulation state instance.
        """
        mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode
        if default_spin is not None:
            fill_val = float(default_spin)
        elif mode == StateMode.BINARY:
            fill_val = -1.0
        else:
            fill_val = 0.0

        x = np.full((N, D), fill_val, dtype=dtype)
        y = np.full((N, D), fill_val, dtype=dtype)
        W = np.zeros((N, N), dtype=np.uint32)
        n = np.zeros(N, dtype=np.uint8)
        Q = np.zeros((N, D, 2), dtype=np.float64)
        return cls(
            N=N,
            D=D,
            state_mode=mode,
            x=x,
            y=y,
            W=W,
            n=n,
            Q=Q,
            t=0.0,
        )

    @classmethod
    def create_unbiased_emergence(
        cls,
        N: int,
        D: int = 1,
        state_mode: Union[StateMode, str] = StateMode.BINARY,
        rng: Optional[np.random.Generator] = None,
    ) -> State:
        """Create an initial unbiased emergence state with W = 0 and y(0) = x(0)."""
        if rng is None:
            rng = np.random.default_rng()
        mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode

        if mode == StateMode.BINARY:
            raw_spins = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(N, D))
            x = raw_spins
            y = raw_spins.copy()
        elif mode == StateMode.CONTINUOUS:
            raw_vals = rng.uniform(-1.0, 1.0, size=(N, D))
            x = raw_vals
            y = raw_vals.copy()
        else:
            raise ValueError(f"Unsupported state mode: {mode}")

        W = np.zeros((N, N), dtype=np.uint32)
        n = np.zeros(N, dtype=np.uint8)
        Q = np.zeros((N, D, 2), dtype=np.float64)

        return cls(
            N=N,
            D=D,
            state_mode=mode,
            x=x,
            y=y,
            W=W,
            n=n,
            Q=Q,
            t=0.0,
        )

    @classmethod
    def create_endpoint_continuous(
        cls,
        N: int,
        D: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> State:
        """Create a continuous state initialized to discrete endpoints {-1.0, +1.0}."""
        if rng is None:
            rng = np.random.default_rng()
        spins = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(N, D))
        W = np.zeros((N, N), dtype=np.uint32)
        n = np.zeros(N, dtype=np.uint8)
        Q = np.zeros((N, D, 2), dtype=np.float64)
        return cls(
            N=N,
            D=D,
            state_mode=StateMode.CONTINUOUS,
            x=spins,
            y=spins.copy(),
            W=W,
            n=n,
            Q=Q,
            t=0.0,
        )

    def copy(self, deep: bool = True) -> State:

        """Return a copy of the state.

        If deep=True (default), deep copies all underlying NumPy buffers to guarantee
        branch isolation and prevent shared memory bugs.
        """
        if deep:
            return State(
                N=self.N,
                D=self.D,
                state_mode=self.state_mode,
                x=self.x.copy(),
                y=self.y.copy(),
                W=self.W.copy(),
                n=self.n.copy(),
                Q=self.Q.copy(),
                t=float(self.t),
                holding_active=self.holding_active,
                t_start_hold=self.t_start_hold,
                t_deadline=self.t_deadline,
                deadline_generation=self.deadline_generation,
                transition_occurred=self.transition_occurred,
                t_transition=self.t_transition,
                pulse_active=self.pulse_active,
                pulse_val=self.pulse_val,
                pulse_component=self.pulse_component,
                pulse_targets=self.pulse_targets.copy() if self.pulse_targets is not None else None,
                blocks=self.blocks.copy() if self.blocks is not None else None,
                hub_idx=self.hub_idx,
                K=self.K.copy() if self.K is not None else None,
                tau_x=self.tau_x.copy() if self.tau_x is not None else None,
                nu=self.nu.copy() if self.nu is not None else None,
                g=self.g.copy() if self.g is not None else None,
                b=self.b.copy() if self.b is not None else None,
                row_caps=self.row_caps.copy() if self.row_caps is not None else None,
                opportunity_groups=self.opportunity_groups.copy() if self.opportunity_groups is not None else None,
                opportunity_matrix=self.opportunity_matrix.copy() if self.opportunity_matrix is not None else None,
                scenario_metadata=copy.deepcopy(self.scenario_metadata),
            )
        return State(
            N=self.N,
            D=self.D,
            state_mode=self.state_mode,
            x=self.x,
            y=self.y,
            W=self.W,
            n=self.n,
            Q=self.Q,
            t=self.t,
            holding_active=self.holding_active,
            t_start_hold=self.t_start_hold,
            t_deadline=self.t_deadline,
            deadline_generation=self.deadline_generation,
            transition_occurred=self.transition_occurred,
            t_transition=self.t_transition,
            pulse_active=self.pulse_active,
            pulse_val=self.pulse_val,
            pulse_component=self.pulse_component,
            pulse_targets=self.pulse_targets,
            blocks=self.blocks,
            hub_idx=self.hub_idx,
            K=self.K,
            tau_x=self.tau_x,
            nu=self.nu,
            g=self.g,
            b=self.b,
            row_caps=self.row_caps,
            opportunity_groups=self.opportunity_groups,
            opportunity_matrix=self.opportunity_matrix,
            scenario_metadata=self.scenario_metadata,
        )

    def validate(self, allow_uninitialized: bool = False) -> None:
        """Validate all invariant constraints of the simulation state.

        Args:
            allow_uninitialized: If True, allow 0.0 values in binary state arrays (x, y)
                representing an unpopulated buffer shell. Default False requires strict
                spins in {-1.0, +1.0}.

        Raises:
            ValueError: If shapes, values, diagonal ties, or timers are invalid.
        """
        if self.N < 1:
            raise ValueError(f"Population size N must be >= 1, got {self.N}")
        if self.D not in (1, 3):
            raise ValueError(f"Belief dimensionality D must be 1 or 3, got {self.D}")

        if self.x.shape != (self.N, self.D):
            raise ValueError(f"x shape mismatch: expected ({self.N}, {self.D}), got {self.x.shape}")
        if self.y.shape != (self.N, self.D):
            raise ValueError(f"y shape mismatch: expected ({self.N}, {self.D}), got {self.y.shape}")
        if self.W.shape != (self.N, self.N):
            raise ValueError(f"W shape mismatch: expected ({self.N}, {self.N}), got {self.W.shape}")
        if self.n.shape != (self.N,):
            raise ValueError(f"n shape mismatch: expected ({self.N},), got {self.n.shape}")
        if self.Q.shape != (self.N, self.D, 2):
            raise ValueError(f"Q shape mismatch: expected ({self.N}, {self.D}, 2), got {self.Q.shape}")

        if np.any(np.diag(self.W) != 0):
            raise ValueError("Self-ties forbidden: diag(W) must be strictly 0")
        if np.any(self.W < 0):
            raise ValueError("Memory counts W cannot be negative")

        if np.any(~np.isin(self.n, [0, 1])):
            raise ValueError("Participation n must be in {0, 1}")

        if self.t < 0.0:
            raise ValueError(f"Simulation time t cannot be negative, got {self.t}")

        if self.blocks is not None:
            if self.blocks.shape != (self.N,):
                raise ValueError(f"blocks shape mismatch: expected ({self.N},), got {self.blocks.shape}")
            if np.any(self.blocks < 0):
                raise ValueError("blocks community identifiers must be non-negative integers")

        if self.hub_idx is not None:
            if self.hub_idx < 0 or self.hub_idx >= self.N:
                raise ValueError(f"hub_idx {self.hub_idx} out of range [0, {self.N - 1}]")

        if self.K is not None:
            if self.K.shape != (self.N,):
                raise ValueError(f"K shape mismatch: expected ({self.N},), got {self.K.shape}")
            if np.any(self.K < 0.0):
                raise ValueError("Coupling K entries must be non-negative")

        if self.tau_x is not None:
            if self.tau_x.shape != (self.N,):
                raise ValueError(f"tau_x shape mismatch: expected ({self.N},), got {self.tau_x.shape}")
            if np.any(self.tau_x <= 0.0):
                raise ValueError("tau_x entries must be strictly positive")

        if self.nu is not None:
            if self.nu.shape != (self.N,):
                raise ValueError(f"nu shape mismatch: expected ({self.N},), got {self.nu.shape}")
            if np.any(self.nu < 0.0):
                raise ValueError("nu entries must be non-negative")

        if self.g is not None:
            if self.g.shape != (self.N,):
                raise ValueError(f"g shape mismatch: expected ({self.N},), got {self.g.shape}")
            if np.any(self.g < 0.0):
                raise ValueError("g entries must be non-negative")

        if self.b is not None:
            if self.b.shape != (self.N, self.D):
                raise ValueError(f"b shape mismatch: expected ({self.N}, {self.D}), got {self.b.shape}")

        if self.row_caps is not None:
            if self.row_caps.shape != (self.N,):
                raise ValueError(f"row_caps shape mismatch: expected ({self.N},), got {self.row_caps.shape}")
            if np.any(self.row_caps < 1):
                raise ValueError("row_caps entries must be >= 1")
            row_sums = np.sum(self.W, axis=1)
            if np.any(row_sums > self.row_caps):
                raise ValueError("Row memory counts in W exceed directed row capacity caps")

        if self.opportunity_groups is not None:
            if self.opportunity_groups.shape != (self.N,):
                raise ValueError(f"opportunity_groups shape mismatch: expected ({self.N},), got {self.opportunity_groups.shape}")
        if self.opportunity_matrix is not None:
            if self.opportunity_matrix.shape != (self.N, self.N):
                raise ValueError("opportunity_matrix shape must be (N, N)")
            if not np.all(np.isfinite(self.opportunity_matrix)) or np.any(self.opportunity_matrix < 0):
                raise ValueError("opportunity_matrix must be finite and nonnegative")
            if np.any(np.diag(self.opportunity_matrix) != 0):
                raise ValueError("opportunity_matrix must have zero diagonal")

        if self.state_mode == StateMode.BINARY:
            valid_set = [-1.0, 0.0, 1.0] if allow_uninitialized else [-1.0, 1.0]
            if np.any(~np.isin(self.x, valid_set)):
                raise ValueError("Binary private states x must be strictly in {-1, +1}")
            if np.any(~np.isin(self.y, valid_set)):
                raise ValueError("Binary expressed states y must be strictly in {-1, +1}")
        elif self.state_mode == StateMode.CONTINUOUS:
            if np.any(self.x < -1.0) or np.any(self.x > 1.0):
                raise ValueError("Continuous private states x must be strictly in [-1.0, 1.0]")
            if np.any(self.y < -1.0) or np.any(self.y > 1.0):
                raise ValueError("Continuous expressed states y must be strictly in [-1.0, 1.0]")
        else:
            raise ValueError(f"Unsupported state_mode: {self.state_mode}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary of serializable primitives and NumPy arrays."""
        return {
            "N": self.N,
            "D": self.D,
            "state_mode": self.state_mode.value,
            "x": self.x.copy(),
            "y": self.y.copy(),
            "W": self.W.copy(),
            "n": self.n.copy(),
            "Q": self.Q.copy(),
            "t": float(self.t),
            "holding_active": bool(self.holding_active),
            "t_start_hold": self.t_start_hold if self.t_start_hold is not None else np.nan,
            "t_deadline": self.t_deadline if self.t_deadline is not None else np.nan,
            "deadline_generation": int(self.deadline_generation),
            "transition_occurred": bool(self.transition_occurred),
            "t_transition": self.t_transition if self.t_transition is not None else np.nan,
            "pulse_active": bool(self.pulse_active),
            "pulse_val": float(self.pulse_val),
            "pulse_component": int(self.pulse_component),
            "has_pulse_targets": self.pulse_targets is not None,
            "pulse_targets": self.pulse_targets.copy() if self.pulse_targets is not None else np.array([], dtype=bool),
            "has_blocks": self.blocks is not None,
            "blocks": self.blocks.copy() if self.blocks is not None else np.array([], dtype=np.int64),
            "has_hub_idx": self.hub_idx is not None,
            "hub_idx": int(self.hub_idx) if self.hub_idx is not None else -1,
            "has_K": self.K is not None,
            "K": self.K.copy() if self.K is not None else np.array([], dtype=np.float64),
            "has_tau_x": self.tau_x is not None,
            "tau_x": self.tau_x.copy() if self.tau_x is not None else np.array([], dtype=np.float64),
            "has_nu": self.nu is not None,
            "nu": self.nu.copy() if self.nu is not None else np.array([], dtype=np.float64),
            "has_g": self.g is not None,
            "g": self.g.copy() if self.g is not None else np.array([], dtype=np.float64),
            "has_b": self.b is not None,
            "b": self.b.copy() if self.b is not None else np.array([], dtype=np.float64),
            "has_row_caps": self.row_caps is not None,
            "row_caps": self.row_caps.copy() if self.row_caps is not None else np.array([], dtype=np.uint32),
            "has_opportunity_groups": self.opportunity_groups is not None,
            "opportunity_groups": self.opportunity_groups.copy() if self.opportunity_groups is not None else np.array([], dtype=np.int64),
            "has_opportunity_matrix": self.opportunity_matrix is not None,
            "opportunity_matrix": self.opportunity_matrix.copy() if self.opportunity_matrix is not None else np.array([], dtype=np.float64),
            "scenario_metadata_json": json.dumps(self.scenario_metadata, sort_keys=True) if self.scenario_metadata is not None else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> State:
        """Construct State from a dictionary."""
        t_start_hold = data.get("t_start_hold")
        if t_start_hold is not None and (np.isnan(t_start_hold) or t_start_hold is None):
            t_start_hold = None
        else:
            t_start_hold = float(t_start_hold) if t_start_hold is not None else None

        t_deadline = data.get("t_deadline")
        if t_deadline is not None and (np.isnan(t_deadline) or t_deadline is None):
            t_deadline = None
        else:
            t_deadline = float(t_deadline) if t_deadline is not None else None

        t_transition = data.get("t_transition")
        if t_transition is not None and (np.isnan(t_transition) or t_transition is None):
            t_transition = None
        else:
            t_transition = float(t_transition) if t_transition is not None else None

        has_pulse_targets = bool(data.get("has_pulse_targets", False))
        pulse_targets = None
        if has_pulse_targets:
            pulse_targets = np.array(data["pulse_targets"], dtype=bool)

        has_blocks = bool(data.get("has_blocks", False))
        blocks = np.array(data["blocks"], dtype=np.int64) if has_blocks else None

        has_hub_idx = bool(data.get("has_hub_idx", False))
        hub_idx = int(data["hub_idx"]) if has_hub_idx else None

        has_K = bool(data.get("has_K", False))
        K = np.array(data["K"], dtype=np.float64) if has_K else None

        has_tau_x = bool(data.get("has_tau_x", False))
        tau_x = np.array(data["tau_x"], dtype=np.float64) if has_tau_x else None

        has_nu = bool(data.get("has_nu", False))
        nu = np.array(data["nu"], dtype=np.float64) if has_nu else None

        has_g = bool(data.get("has_g", False))
        g = np.array(data["g"], dtype=np.float64) if has_g else None

        has_b = bool(data.get("has_b", False))
        b = np.array(data["b"], dtype=np.float64) if has_b else None

        has_row_caps = bool(data.get("has_row_caps", False))
        row_caps = np.array(data["row_caps"], dtype=np.uint32) if has_row_caps else None

        has_opportunity_groups = bool(data.get("has_opportunity_groups", False))
        opportunity_groups = np.array(data["opportunity_groups"], dtype=np.int64) if has_opportunity_groups else None
        opportunity_matrix = np.array(data["opportunity_matrix"], dtype=np.float64) if bool(data.get("has_opportunity_matrix", False)) else None
        metadata_json = data.get("scenario_metadata_json", "")
        scenario_metadata = json.loads(str(metadata_json)) if metadata_json else None

        return cls(
            N=int(data["N"]),
            D=int(data["D"]),
            state_mode=StateMode(data["state_mode"]),
            x=np.array(data["x"], dtype=np.float64),
            y=np.array(data["y"], dtype=np.float64),
            W=np.array(data["W"], dtype=np.uint32),
            n=np.array(data["n"], dtype=np.uint8),
            Q=np.array(data["Q"], dtype=np.float64),
            t=float(data["t"]),
            holding_active=bool(data.get("holding_active", False)),
            t_start_hold=t_start_hold,
            t_deadline=t_deadline,
            deadline_generation=int(data.get("deadline_generation", 0)),
            transition_occurred=bool(data.get("transition_occurred", False)),
            t_transition=t_transition,
            pulse_active=bool(data.get("pulse_active", False)),
            pulse_val=float(data.get("pulse_val", 0.0)),
            pulse_component=int(data.get("pulse_component", 0)),
            pulse_targets=pulse_targets,
            blocks=blocks,
            hub_idx=hub_idx,
            K=K,
            tau_x=tau_x,
            nu=nu,
            g=g,
            b=b,
            row_caps=row_caps,
            opportunity_groups=opportunity_groups,
            opportunity_matrix=opportunity_matrix,
            scenario_metadata=scenario_metadata,
        )

    @classmethod
    def create_realism_state(
        cls,
        N: int,
        D: int = 1,
        state_mode: Union[StateMode, str] = StateMode.BINARY,
        J: float = 1.0,
        K_nominal: float = 0.5,
        tau_x_nominal: float = 5.0,
        nu_nominal: float = 2.0,
        enable_heterogeneity: bool = True,
        row_cap: Optional[int] = 40,
        enable_opportunity_groups: bool = True,
        num_groups: int = 4,
        between_eligibility: float = 0.05,
        within_eligibility: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> State:
        """Create an initial realism simulation state (Realism Physics Revision).

        Initializes:
        1. Quenched heterogeneity: independent clipped lognormal multipliers for K_i, tau_{x,i}, nu_i,
           uniform susceptibility g_i in [0.5, 1.5], and truncated normal biases b_{i, alpha}.
        2. Directed row memory capacity caps (default C_i = 40).
        3. Constrained encounter opportunity groups (default 4 equal-sized groups, cross-eligibility 0.05).
        4. Emergence initial stance W = 0 and y(0) = x(0).

        Args:
            N: Population size.
            D: Belief dimensionality (1 or 3).
            state_mode: 'binary' or 'continuous'.
            J: Social alignment coupling scale.
            K_nominal: Nominal private-public coupling.
            tau_x_nominal: Nominal private relaxation timescale.
            nu_nominal: Nominal encounter rate.
            enable_heterogeneity: Whether to draw heterogeneous agent traits.
            row_cap: Memory units cap per directed row (default 40, None for uncapped).
            enable_opportunity_groups: Whether to assign 4 equal opportunity groups.
            num_groups: Number of opportunity groups (default 4).
            between_eligibility: Cross-group encounter eligibility (default 0.05).
            within_eligibility: Within-group encounter eligibility (default 1.0).
            rng: NumPy random Generator.

        Returns:
            State: Initialized realism state.
        """
        if rng is None:
            rng = np.random.default_rng()

        mode = StateMode(state_mode.lower()) if isinstance(state_mode, str) else state_mode

        # Emergence stance initialization: W = 0, y(0) = x(0)
        if mode == StateMode.BINARY:
            raw_spins = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(N, D))
            x = raw_spins
            y = raw_spins.copy()
        elif mode == StateMode.CONTINUOUS:
            raw_vals = rng.uniform(-1.0, 1.0, size=(N, D))
            x = raw_vals
            y = raw_vals.copy()
        else:
            raise ValueError(f"Unsupported state mode: {mode}")

        W = np.zeros((N, N), dtype=np.uint32)
        n = np.zeros(N, dtype=np.uint8)
        Q = np.zeros((N, D, 2), dtype=np.float64)

        # Quenched heterogeneity
        if enable_heterogeneity:
            from src.core.heterogeneity import generate_quenched_heterogeneity
            qh = generate_quenched_heterogeneity(
                N=N,
                D=D,
                J=J,
                K_nominal=K_nominal,
                tau_x_nominal=tau_x_nominal,
                nu_nominal=nu_nominal,
                rng=rng,
            )
            K = qh.K
            tau_x = qh.tau_x
            nu = qh.nu
            g = qh.g
            b = qh.b
        else:
            K = np.full(N, K_nominal, dtype=np.float64)
            tau_x = np.full(N, tau_x_nominal, dtype=np.float64)
            nu = np.full(N, nu_nominal, dtype=np.float64)
            g = np.ones(N, dtype=np.float64)
            b = np.zeros((N, D), dtype=np.float64)

        # Row memory caps
        if row_cap is not None:
            c_val = float(row_cap)
            if c_val < 1.0 or not c_val.is_integer():
                raise ValueError(f"row_cap must be an integer >= 1, got {row_cap}")
            row_caps = np.full(N, int(c_val), dtype=np.uint32)
        else:
            row_caps = None

        # Opportunity groups
        if enable_opportunity_groups:
            from src.core.heterogeneity import generate_opportunity_groups
            groups, _, _ = generate_opportunity_groups(
                N=N,
                num_groups=num_groups,
                between_eligibility=between_eligibility,
                within_eligibility=within_eligibility,
                randomize=True,
                rng=rng,
            )
        else:
            groups = None

        state = cls(
            N=N,
            D=D,
            state_mode=mode,
            x=x,
            y=y,
            W=W,
            n=n,
            Q=Q,
            t=0.0,
            blocks=groups,
            K=K,
            tau_x=tau_x,
            nu=nu,
            g=g,
            b=b,
            row_caps=row_caps,
            opportunity_groups=groups,
        )
        state.validate()
        return state

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save state to compressed .npz file."""
        d = self.to_dict()
        np.savez_compressed(filepath, **d)

    def save_npz(self, filepath: Union[str, Path]) -> None:
        """Alias for save_checkpoint."""
        self.save_checkpoint(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> State:
        """Load state from compressed .npz file."""
        with np.load(filepath, allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in loaded.files}
            # Extract scalar types stored in 0d arrays
            parsed: Dict[str, Any] = {}
            for k, v in data.items():
                if v.ndim == 0:
                    parsed[k] = v.item()
                else:
                    parsed[k] = v
            return cls.from_dict(parsed)

    @classmethod
    def load_npz(cls, filepath: Union[str, Path]) -> State:
        """Alias for load_checkpoint."""
        return cls.load_checkpoint(filepath)

    # Derived Observables & Helper Properties
    @property
    def row_sums(self) -> np.ndarray:
        """Row sums of memory matrix W, shape (N,)."""
        return np.sum(self.W, axis=1, dtype=np.uint32)

    @property
    def total_memory(self) -> int:
        """Total remembered memory units S(t) = sum_{ij} W_{ij}."""
        return int(np.sum(self.W))

    @property
    def active_agents_count(self) -> int:
        """Count of agents with non-empty outgoing memory (row sum > 0)."""
        return int(np.sum(self.row_sums > 0))

    @property
    def participation_fraction(self) -> float:
        """Instantaneous participation fraction M_n(t) = (1/N) sum_i n_i."""
        return float(np.mean(self.n))

    @property
    def mean_expressed_belief(self) -> np.ndarray:
        """Mean expressed stance m^y = (1/N) sum_i y_i, shape (D,)."""
        return np.mean(self.y, axis=0)

    @property
    def mean_private_belief(self) -> np.ndarray:
        """Mean private belief m^x = (1/N) sum_i x_i, shape (D,)."""
        return np.mean(self.x, axis=0)

    @property
    def discrepancy(self) -> float:
        """Instantaneous population discrepancy d_2:

        d_2 = (1 / (4 * N * D)) * sum_{i, alpha} (x_{i, alpha} - y_{i, alpha})^2 in [0, 1].
        """
        diff = self.x - self.y
        return float(np.sum(diff * diff) / (4.0 * self.N * self.D))
