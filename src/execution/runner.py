"""Trajectory runner and checkpointing engine.

Strictly decouples execution orchestration from underlying physics (src/core/)
and controllers (src/control/). Provides:
- TrajectoryRunner: Single-trajectory continuous-time simulator with bit-for-bit restart reproducibility.
- save_checkpoint, load_checkpoint, branch_checkpoint: Deterministic state & PRNG preservation.
- Trajectory: Container holding final simulation State and observable timeseries.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import os
import platform
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.core.fields import (
    compute_single_private_field,
    compute_single_public_field,
)
from src.core.heat_baths import (
    sample_binary_glauber_scalar,
    sample_continuous_inverse_cdf_scalar,
)
from src.core.memory import MemoryManager
from src.core.participation import ParticipationParams, ParticipationTracker
from src.core.q_learning import QLearningParams, step_q_learning
from src.core.scheduler import PoissonScheduler
from src.core.state import State, StateMode


@dataclass
class Trajectory:
    """Simulation trajectory output container.

    Attributes:
        final_state: The final State of the simulation.
        data: Additional trajectory telemetry (e.g. timeseries, metrics, events).
    """

    final_state: State
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> State:
        """Alias to final_state for API convenience."""
        return self.final_state

    @property
    def timeseries(self) -> List[Dict[str, Any]]:
        """Extracted timeseries list if present in data."""
        return self.data.get("timeseries", [])


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder handling NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


CHECKPOINT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite sentinels to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        cls=NumpyEncoder,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Flush a JSON artifact and atomically replace its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, cls=NumpyEncoder, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    filepath: Optional[Union[str, Path]] = None,
    *,
    state: Optional[State] = None,
    rng: Optional[np.random.Generator] = None,
    t: float = 0.0,
    scheduled_events: Optional[List[Any]] = None,
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Save simulation state, PRNG state, and scheduler context into a checkpoint.

    Args:
        filepath: Optional filesystem path to write serialized JSON checkpoint.
        state: Core State container.
        rng: NumPy Generator whose bit_generator state will be saved.
        t: Current simulation timestamp.
        scheduled_events: Queue of scheduled deterministic events.
        config: Optional SimulationConfig or dictionary of parameters.
        seed: Random seed associated with this trajectory.
        extra: Additional arbitrary metadata.
        kwargs: Additional key-value checkpoint parameters.

    Returns:
        Dictionary representation of the checkpoint.
    """
    chk: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "t": float(t),
        "seed": seed,
    }

    if rng is not None and hasattr(rng, "bit_generator"):
        chk["rng_state"] = rng.bit_generator.state
        chk["rng"] = rng

    if state is not None:
        if hasattr(state, "to_dict"):
            chk["state"] = state.to_dict()
        else:
            chk["state"] = state

    if scheduled_events is not None:
        chk["scheduled_events"] = copy.deepcopy(scheduled_events)
    else:
        chk["scheduled_events"] = []

    if config is not None:
        if hasattr(config, "model_dump"):
            chk["config"] = config.model_dump()
        elif hasattr(config, "__dict__"):
            chk["config"] = config.__dict__
        else:
            chk["config"] = config

    chk.update(kwargs)
    if extra:
        chk.update(extra)

    if filepath is not None:
        path = Path(filepath)
        serializable = {}
        for k, v in chk.items():
            if k == "rng":
                continue
            serializable[k] = v
        checksum = hashlib.sha256(_canonical_json_bytes(serializable)).hexdigest()
        _atomic_write_json(path, {"payload": serializable, "sha256": checksum})

    return chk


def load_checkpoint(checkpoint: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and restore a checkpoint dictionary from disk or in-memory dict.

    Args:
        checkpoint: Filepath (str/Path) or checkpoint dictionary.

    Returns:
        Restored checkpoint dictionary with reconstructed PRNG generator.

    Raises:
        ValueError: If checkpoint file or data is corrupted/invalid.
    """
    data: Dict[str, Any]
    if isinstance(checkpoint, (str, Path)):
        p = Path(checkpoint)
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, dict) or "payload" not in raw:
                    raise ValueError("on-disk checkpoint requires a checksummed envelope")
                payload = raw.get("payload")
                expected = raw.get("sha256")
                if not isinstance(payload, dict) or not isinstance(expected, str):
                    raise ValueError("checkpoint envelope is incomplete")
                actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
                if actual != expected:
                    raise ValueError("checkpoint checksum mismatch")
                if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                    raise ValueError("unsupported checkpoint schema version")
                data = payload
            except Exception as exc:
                raise ValueError(f"Failed to parse checkpoint JSON file {p}: {exc}") from exc
        else:
            try:
                data = json.loads(str(checkpoint))
            except Exception as exc:
                raise ValueError(f"Corrupted checkpoint string or non-existent file: {checkpoint}") from exc
    elif isinstance(checkpoint, dict):
        data = dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint must resolve to a dictionary, got {type(data)}")

    # Reconstruct PRNG Generator if rng_state is present
    if "rng_state" in data and data["rng_state"] is not None:
        try:
            restored_rng = np.random.Generator(np.random.PCG64())
            restored_rng.bit_generator.state = data["rng_state"]
            data["rng"] = restored_rng
        except Exception as exc:
            raise ValueError(f"Invalid checkpoint RNG state: {exc}") from exc

    return data


def branch_checkpoint(
    chk: Dict[str, Any],
    arm: str = "default",
    seed_offset: int = 1,
) -> Dict[str, Any]:
    """Branch from an existing checkpoint with isolated seed and arm identifier.

    Args:
        chk: Base checkpoint dictionary.
        arm: Branch / condition arm identifier (e.g. 'perturbed', 'unperturbed').
        seed_offset: Offset added to base seed for PRNG isolation.

    Returns:
        Branched checkpoint dictionary.
    """
    branched = copy.deepcopy(chk)
    base_seed = chk.get("seed", 0)
    if base_seed is None:
        base_seed = 0
    identity = {"base_seed": int(base_seed), "arm": str(arm), "seed_offset": int(seed_offset)}
    digest = hashlib.sha256(_canonical_json_bytes(identity)).digest()
    entropy = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
    new_seed = int.from_bytes(digest[:8], "big")
    branched["seed"] = new_seed
    branched["arm"] = arm

    new_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    branched["rng"] = new_rng
    branched["rng_state"] = new_rng.bit_generator.state
    return branched


class TrajectoryRunner:
    """Master single-trajectory simulation runner.

    Encapsulates simulation state, Poisson jump scheduler, memory manager,
    interventions, and deterministic event priority queue. Strictly compliant
    with the execution/physics separation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        *,
        state: Optional[State] = None,
        W: Optional[np.ndarray] = None,
        blocks: Optional[np.ndarray] = None,
        hub: Optional[int] = None,
        N: Optional[int] = None,
        D: int = 1,
        T: float = 0.3,
        delta: float = 0.1,
        nu: Optional[Union[float, np.ndarray]] = None,
        J: float = 1.0,
        K: Optional[Union[float, np.ndarray]] = None,
        L: float = 1.0,
        b: Optional[Union[float, np.ndarray]] = None,
        tau_x: Optional[Union[float, np.ndarray]] = None,
        tau_y: float = 1.0,
        tau_n: Optional[float] = None,
        epsilon: float = 0.05,
        a: float = 0.0,
        c: float = 0.0,
        chi: float = 0.0,
        reciprocal: bool = False,
        state_mode: str = "binary",
        seed: Optional[int] = None,
        interventions: Optional[Sequence[Any]] = None,
        interface_schedule: Optional[Any] = None,
        enable_q_learning: bool = False,
        q_eta: float = 0.1,
        q_temperature: float = 1.0,
        sample_interval: float = 1.0,
        record_network_frames: bool = False,
        animation_interval: float = 1.0,
        g: Optional[np.ndarray] = None,
        num_populations: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **extra_kwargs: Any,
    ) -> None:
        # 1. Unpack config if provided
        if config is not None:
            if hasattr(config, "experiment") and config.experiment is not None:
                exp = config.experiment
                N = getattr(exp, "N", N)
                D = getattr(exp, "D", D)
                T = getattr(exp, "T", T)
                delta = getattr(exp, "delta", delta)
                nu = getattr(exp, "nu", nu)
                J = getattr(exp, "J", J)
                K = getattr(exp, "K", K)
                L = getattr(exp, "L", L)
                tau_x = getattr(exp, "tau_x", tau_x)
                tau_y = getattr(exp, "tau_y", tau_y)
                tau_n = getattr(exp, "tau_n", tau_n)
                epsilon = getattr(exp, "epsilon", epsilon)
                a = getattr(exp, "a", a)
                c = getattr(exp, "c", c)
                chi = getattr(exp, "chi", chi)
                reciprocal = getattr(exp, "reciprocal", reciprocal)
                state_mode = getattr(exp, "state_mode", state_mode)
                seed = getattr(exp, "seed", seed)
            elif hasattr(config, "N"):
                N = getattr(config, "N", N)
                D = getattr(config, "D", D)
                T = getattr(config, "T", T)
                delta = getattr(config, "delta", delta)
                nu = getattr(config, "nu", nu)
                J = getattr(config, "J", J)
                K = getattr(config, "K", K)
                L = getattr(config, "L", L)
                tau_x = getattr(config, "tau_x", tau_x)
                tau_y = getattr(config, "tau_y", tau_y)
                tau_n = getattr(config, "tau_n", tau_n)
                epsilon = getattr(config, "epsilon", epsilon)
                a = getattr(config, "a", a)
                c = getattr(config, "c", c)
                chi = getattr(config, "chi", chi)
                reciprocal = getattr(config, "reciprocal", reciprocal)
                state_mode = getattr(config, "state_mode", state_mode)
                seed = getattr(config, "seed", seed)
            elif isinstance(config, dict):
                N = config.get("N", N)
                D = config.get("D", D)
                T = config.get("T", T)
                delta = config.get("delta", delta)
                nu = config.get("nu", nu)
                J = config.get("J", J)
                K = config.get("K", K)
                L = config.get("L", L)
                tau_x = config.get("tau_x", tau_x)
                tau_y = config.get("tau_y", tau_y)
                tau_n = config.get("tau_n", tau_n)
                epsilon = config.get("epsilon", epsilon)
                a = config.get("a", a)
                c = config.get("c", c)
                chi = config.get("chi", chi)
                reciprocal = config.get("reciprocal", reciprocal)
                state_mode = config.get("state_mode", state_mode)
                seed = config.get("seed", seed)

        # 2. PRNG initialization
        self.seed = seed
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # 3. State initialization
        if state is not None:
            self.state = state.copy()
            self.N = self.state.N
            self.D = self.state.D
            self.state_mode = str(self.state.state_mode.value if hasattr(self.state.state_mode, "value") else self.state.state_mode)
        elif W is not None:
            W_arr = np.asarray(W, dtype=np.uint32)
            self.N = W_arr.shape[0]
            self.D = D
            self.state_mode = state_mode
            mode_enum = StateMode.CONTINUOUS if state_mode.lower() == "continuous" else StateMode.BINARY
            x_init = np.ones((self.N, self.D), dtype=np.float64)
            y_init = np.ones((self.N, self.D), dtype=np.float64)
            n_init = np.zeros(self.N, dtype=np.uint8)
            Q_init = np.zeros((self.N, self.D, 2), dtype=np.float64)
            self.state = State(
                N=self.N,
                D=self.D,
                state_mode=mode_enum,
                x=x_init,
                y=y_init,
                W=W_arr.copy(),
                n=n_init,
                Q=Q_init,
                blocks=blocks.copy() if blocks is not None else None,
                hub_idx=hub,
            )
        else:
            self.N = N if N is not None else 10
            self.D = D
            self.state_mode = state_mode
            mode_enum = StateMode.CONTINUOUS if state_mode.lower() == "continuous" else StateMode.BINARY
            # Use deterministic generator derived from seed for initial spin assignment
            init_rng = np.random.default_rng(self.seed if self.seed is not None else 0)
            self.state = State.create_unbiased_emergence(N=self.N, D=self.D, state_mode=mode_enum, rng=init_rng)

        if blocks is not None and self.state.blocks is None:
            self.state.blocks = np.asarray(blocks).copy()
        if hub is not None and self.state.hub_idx is None:
            self.state.hub_idx = hub

        # 4. Parameters
        self.T = float(T)
        self.delta = float(delta)
        self.nu = nu if nu is not None else 2.0
        self.J = float(J)
        self.K = K if K is not None else 0.5
        self.L = float(L)
        self.b = b
        self.tau_x = tau_x if tau_x is not None else 1.0
        self.tau_y = float(tau_y)
        self.tau_n = float(tau_n) if tau_n is not None else None
        self.epsilon = float(epsilon)
        self.a = float(a)
        self.c = float(c)
        self.chi = float(chi)
        self.reciprocal = bool(reciprocal)
        self.sample_interval = float(sample_interval)
        self.record_network_frames = bool(record_network_frames)
        self.animation_interval = float(animation_interval)
        self.num_populations = num_populations

        # Adopt quenched realism traits carried by the state unless explicitly
        # overridden by constructor arguments (audit D2).
        if state is not None:
            if state.K is not None and K is None:
                self.K = np.asarray(state.K, dtype=np.float64)
            if state.tau_x is not None and tau_x is None:
                self.tau_x = np.asarray(state.tau_x, dtype=np.float64)
            if state.nu is not None and nu is None:
                self.nu = np.asarray(state.nu, dtype=np.float64)
            if state.g is not None:
                self.g = np.asarray(state.g, dtype=np.float64)
            if state.b is not None and b is None:
                self.b = np.asarray(state.b, dtype=np.float64)

        # Commanded intervention channel u_{i alpha}(t) applied as g_i * u
        # (charge commanded u; susceptibility does not enter accounting).
        self.u_command: Optional[np.ndarray] = None
        self.u_exposure: float = 0.0

        self.K = self.K if isinstance(self.K, np.ndarray) else float(self.K)
        self.tau_x = self.tau_x if isinstance(self.tau_x, np.ndarray) else float(self.tau_x)
        self.nu = self.nu if isinstance(self.nu, np.ndarray) else float(self.nu)
        self.g = np.asarray(g, dtype=np.float64) if g is not None else getattr(self, "g", None)

        # Adaptive learning (Q-learning)
        self.enable_q_learning = bool(enable_q_learning)
        self.q_eta = float(q_eta)
        self.q_temperature = float(q_temperature)
        self.q_params = QLearningParams(eta=float(q_eta), T_Q=float(q_temperature))

        # Interventions
        self.interventions: List[Any] = list(interventions) if interventions is not None else []
        self.interface_schedule = interface_schedule

        # 5. Memory Manager
        if self.interface_schedule is not None:
            num_pops = self.num_populations
            if num_pops is None and hasattr(self.interface_schedule, 'num_populations'):
                num_pops = self.interface_schedule.num_populations
            if num_pops is not None and num_pops > 1 and self.state.opportunity_groups is None:
                pop_size = self.N // num_pops
                rem = self.N % num_pops
                groups = []
                for p_idx in range(num_pops):
                    count = pop_size + (1 if p_idx < rem else 0)
                    groups.extend([p_idx] * count)
                self.state.opportunity_groups = np.array(groups, dtype=np.int32)

        opp_groups = self.state.opportunity_groups
        if opp_groups is None and self.state.blocks is not None:
            opp_groups = self.state.blocks
        self.memory_mgr = MemoryManager(
            N=self.N,
            reciprocal=self.reciprocal,
            epsilon=self.epsilon,
            a=self.a,
            c=self.c,
            chi=self.chi,
            initial_W=self.state.W,
            capacity=self.state.row_caps,
            opportunity_groups=opp_groups,
            C_matrix=self.state.opportunity_matrix,
            rng=self.rng,
        )

        # 6. Poisson Jump Scheduler
        self.scheduler = PoissonScheduler(
            N=self.N,
            D=self.D,
            tau_x=self.tau_x,
            tau_y=self.tau_y,
            nu=self.nu,
            delta=self.delta,
            tau_n=self.tau_n,
            reciprocal=self.reciprocal,
            rng=self.rng,
        )
        self.scheduler.set_total_memory(int(self.memory_mgr.total_memory))
        self.scheduler.time = float(self.state.t)

        # Institutional transition tracker if participation enabled
        self.transition_tracker: Optional[ParticipationTracker] = None
        if self.tau_n is not None:
            theta = extra_kwargs.get("theta", 0.5)
            t_hold = extra_kwargs.get("t_hold", 3.0)
            self.transition_tracker = ParticipationTracker(
                N=self.N,
                params=ParticipationParams(
                    I=int(extra_kwargs.get("participation_I", 1)),
                    b=float(extra_kwargs.get("participation_b", 1.0)),
                    g=float(extra_kwargs.get("participation_g", 0.0)),
                    c=float(extra_kwargs.get("participation_c", 0.5)),
                    T_n=float(extra_kwargs.get("participation_T", 0.1)),
                    tau_n=self.tau_n,
                    theta=float(theta),
                    t_hold=float(t_hold),
                ),
                initial_n=self.state.n,
            )

        # When a requested horizon falls between events, keep the prior event
        # time so a resumed run can reuse the untouched waiting-time draw.
        self._deferred_time_origin: Optional[float] = None

        # 7. Register simulation step callbacks
        self._register_channel_callbacks()

        # 8. Schedule deterministic intervention events
        self._scheduled_event_records: List[Dict[str, Any]] = []
        self._schedule_interventions()

        # Telemetry storage
        self.timeseries: List[Dict[str, Any]] = []
        self._next_network_frame_t: float = 0.0
        self._record_snapshot(self.scheduler.time)

    def _register_channel_callbacks(self) -> None:
        """Connect physics update logic to stochastic jump channels."""
        runner_self = self
        state = self.state
        N, D = self.N, self.D
        T = self.T
        J = self.J
        K = self.K
        L = self.L
        b = self.b
        memory_mgr = self.memory_mgr
        scheduler = self.scheduler

        B_mat: Optional[np.ndarray] = None
        if D == 3:
            try:
                from src.core.couplings import get_coupling_matrix
                B_mat = get_coupling_matrix("B+", D=3)
            except Exception:
                B_mat = None

        def cb_private(node_idx: int) -> None:
            state.t = scheduler.time
            i = node_idx
            x_row = state.x[i, :]
            active_rng = runner_self.rng
            # Spec: an event changes exactly ONE component. The channel rate is
            # D * sum_i 1/tau_x,i, so per-component attempt rate is 1/tau_x,i
            # only when each event redraws a single uniformly chosen component.
            alpha = int(active_rng.integers(0, D))
            y_val = float(state.y[i, alpha])
            f_x = compute_single_private_field(
                y_val=y_val,
                x_row=x_row,
                alpha=alpha,
                B=B_mat,
                K=float(K[i]) if isinstance(K, np.ndarray) else float(K),
                L=L,
                b_val=float(b[i, alpha]) if isinstance(b, np.ndarray) else (float(b) if b is not None else 0.0),
            )
            kappa = f_x / T
            if state.state_mode == StateMode.BINARY:
                new_x = float(sample_binary_glauber_scalar(kappa, rng=active_rng))
            else:
                new_x = float(sample_continuous_inverse_cdf_scalar(kappa, active_rng.random()))
            state.x[i, alpha] = new_x

        def cb_public(node_idx: int) -> None:
            state.t = scheduler.time
            i = node_idx
            W_row = memory_mgr.W[i, :]
            active_rng = runner_self.rng

            h_val = 0.0
            if state.pulse_active:
                if state.pulse_targets is None or state.pulse_targets[i]:
                    h_val = state.pulse_val

            # One component per event (see cb_private note): per-component
            # attempt rate 1/tau_y exactly.
            alpha = int(active_rng.integers(0, D))
            h_comp = h_val if (not state.pulse_active or state.pulse_component == alpha) else 0.0
            u_val = float(runner_self.u_command[i, alpha]) if runner_self.u_command is not None else 0.0
            g_val = float(runner_self.g[i]) if runner_self.g is not None else 1.0
            f_y = compute_single_public_field(
                W_row=W_row,
                y_col=state.y[:, alpha],
                x_val=float(state.x[i, alpha]),
                J=J,
                K=float(K[i]) if isinstance(K, np.ndarray) else float(K),
                h_val=h_comp,
                g_val=g_val,
                u_val=u_val,
            )
            kappa = f_y / T
            if runner_self.enable_q_learning:
                step_q_learning(
                    i=i,
                    alpha=alpha,
                    Q_table=state.Q,
                    y=state.y,
                    field=f_y,
                    params=runner_self.q_params,
                    rng=active_rng,
                )
            else:
                if state.state_mode == StateMode.BINARY:
                    new_y = float(sample_binary_glauber_scalar(kappa, rng=active_rng))
                else:
                    new_y = float(sample_continuous_inverse_cdf_scalar(kappa, active_rng.random()))
                state.y[i, alpha] = new_y

        def cb_encounter(initiator_idx: int) -> None:
            state.t = scheduler.time
            active_rng = runner_self.rng
            if runner_self.interface_schedule is not None:
                E_mat = runner_self.interface_schedule.get_matrix(scheduler.time)
                if state.opportunity_groups is not None:
                    memory_mgr.C_matrix = MemoryManager.create_eligibility_matrix(E_mat, state.opportunity_groups)
                elif E_mat.shape == (N, N):
                    memory_mgr.C_matrix = E_mat

            res = memory_mgr.sample_and_apply_encounter(
                i=initiator_idx,
                y=state.y,
                rng=active_rng,
            )
            if not res.is_null:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W

        def cb_decay() -> None:
            state.t = scheduler.time
            active_rng = runner_self.rng
            i_dec, j_dec = memory_mgr.sample_and_apply_decay(rng=active_rng)
            if i_dec != -1:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W

        def cb_participation() -> None:
            state.t = scheduler.time
            if runner_self.transition_tracker is not None:
                node_idx = int(runner_self.rng.integers(0, N))
                runner_self.transition_tracker.update_agent(
                    node_idx,
                    float(state.x[node_idx, 0]),
                    memory_mgr.A[node_idx, :],
                    scheduler.time,
                    runner_self.rng,
                    scheduler,
                )
                def _handle_deadline(sched: Any, evt: Any) -> None:
                    runner_self.transition_tracker.handle_deadline_event(sched.time, evt.payload)
                    state.holding_active = runner_self.transition_tracker.holding_active
                    state.transition_occurred = runner_self.transition_tracker.transition_occurred

                for event in scheduler.pq:
                    if event.event_type == "holding_deadline" and event.callback is None:
                        event.callback = _handle_deadline
                state.n = runner_self.transition_tracker.n.copy()
                state.holding_active = runner_self.transition_tracker.holding_active
                state.transition_occurred = runner_self.transition_tracker.transition_occurred

        scheduler.register_callbacks(
            cb_private=cb_private,
            cb_public=cb_public,
            cb_encounter=cb_encounter,
            cb_decay=cb_decay,
            cb_participation=cb_participation,
        )

    def _schedule_interventions(self) -> None:
        """Register deterministic pulse/disruption/schedule events on the scheduler."""
        for interv in self.interventions:
            if hasattr(interv, "disruption_time"):
                interv.schedule_on(self.scheduler, self.state, self.memory_mgr)
                self._scheduled_event_records.append({
                    "type": type(interv).__name__,
                    "disruption_time": float(interv.disruption_time),
                    "fraction": float(getattr(interv, "fraction", 0.3)),
                    "reciprocal": getattr(interv, "reciprocal", None),
                    "seed": getattr(interv, "seed", None),
                })
            elif hasattr(interv, "schedule_on"):
                # Commanded-field interventions need the runner (u channel + g_i);
                # prescribed pulses only need the state.
                if type(interv).__name__ == "CommandedFieldIntervention":
                    interv.schedule_on(self.scheduler, self)
                    record = {"type": "CommandedFieldIntervention", "start_time": float(interv.start_time)}
                    for name in ("duration", "amplitude", "component", "u_max"):
                        if hasattr(interv, name):
                            record[name] = getattr(interv, name)
                    if getattr(interv, "target_mask", None) is not None:
                        record["target_mask"] = np.asarray(interv.target_mask, dtype=bool)
                else:
                    interv.schedule_on(self.scheduler, self.state)
                    record = {"type": type(interv).__name__, "start_time": getattr(interv, "start_time", 0.0)}
                    for name in ("duration", "amplitude", "component"):
                        if hasattr(interv, name):
                            record[name] = getattr(interv, name)
                    if getattr(interv, "target_mask", None) is not None:
                        record["target_mask"] = np.asarray(interv.target_mask, dtype=bool)
                self._scheduled_event_records.append(record)

    def _record_snapshot(self, t: float) -> None:
        """Store compact macro-observables snapshot in timeseries.

        Full network copies (W, y) are only attached when `record_network_frames`
        is enabled and t falls on an animation grid point (every
        `animation_interval` time units) or at the first/last snapshot. This
        keeps dense macro-observables cheap for extended campaigns.
        """
        include_network = False
        if self.record_network_frames:
            # Attach full network state only at animation grid points.
            if t >= self._next_network_frame_t - 1e-9 or not self.timeseries:
                include_network = True
                self._next_network_frame_t = t + max(self.animation_interval, 1e-9)
        snapshot = {
            "t": float(t),
            "m_x": np.mean(self.state.x, axis=0).tolist(),
            "m_y": np.mean(self.state.y, axis=0).tolist(),
            "d2": float(self.state.discrepancy),
            "total_memory": int(self.memory_mgr.total_memory),
            "f_active": float(np.mean(np.sum(self.memory_mgr.W, axis=1) > 0)),
        }
        if include_network:
            snapshot["W"] = self.memory_mgr.W.copy()
            snapshot["y"] = self.state.y.copy()
        self.timeseries.append(snapshot)

    def run(self, horizon: float) -> Trajectory:
        """Advance trajectory simulation continuously to the specified horizon.

        Args:
            horizon: Termination simulation time.

        Returns:
            Trajectory container holding final state and timeseries.
        """
        horizon_val = float(horizon)
        if horizon_val < float(self.state.t) - 1e-12:
            raise ValueError("horizon cannot precede the current trajectory time")
        if self._deferred_time_origin is not None:
            self.scheduler.time = float(self._deferred_time_origin)
            self.state.t = float(self._deferred_time_origin)
            self._deferred_time_origin = None
        next_sample_t = self.scheduler.time + self.sample_interval

        while self.scheduler.time < horizon_val:
            # Pending-clock convention: the drawn stochastic clock is preserved
            # across boundaries (never re-drawn), so deferred boundaries do not
            # alter the seeded path. Events beyond the horizon are deferred.
            next_event_time = self.scheduler.peek_next_event_time()
            if next_event_time is None or next_event_time > horizon_val:
                self._deferred_time_origin = float(self.scheduler.time)
                self.scheduler.time = horizon_val
                break
            advanced = self.scheduler.step()
            if not advanced:
                break

            if self.scheduler.time >= next_sample_t:
                self._record_snapshot(self.scheduler.time)
                next_sample_t = self.scheduler.time + self.sample_interval

        self.state.t = self.scheduler.time
        self._record_snapshot(self.scheduler.time)

        return Trajectory(
            final_state=self.state,
            data={
                "timeseries": self.timeseries,
                "horizon": horizon_val,
                "stats": {
                    "total_events": self.scheduler.stats.total_events,
                    "preemptions": self.scheduler.stats.preemptions,
                    "deterministic_events": self.scheduler.stats.deterministic_events,
                },
            },
        )

    def get_checkpoint(
        self,
        filepath: Optional[Union[str, Path]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate checkpoint capturing complete simulation state and PRNG state."""
        supported_event_types = {"pulse_start", "pulse_end", "tie_disruption", "holding_deadline"}
        unsupported = sorted({
            event.event_type
            for event in self.scheduler.pq
            if event.time > self.scheduler.time + 1e-12 and event.event_type not in supported_event_types
        })
        if unsupported:
            raise ValueError(f"Cannot checkpoint unsupported pending scheduler events: {unsupported}")
        pending_event_queue = [
            {
                "time": event.time,
                "priority": event.priority,
                "seq_id": event.seq_id,
                "event_type": event.event_type,
                "payload": event.payload,
            }
            for event in sorted(self.scheduler.pq)
            if event.time > self.scheduler.time + 1e-12
        ]
        params = {
            "N": self.N,
            "D": self.D,
            "T": self.T,
            "delta": self.delta,
            "nu": self.nu,
            "J": self.J,
            "K": self.K,
            "L": self.L,
            "b": self.b,
            "tau_x": self.tau_x,
            "tau_y": self.tau_y,
            "tau_n": self.tau_n,
            "epsilon": self.epsilon,
            "a": self.a,
            "c": self.c,
            "chi": self.chi,
            "reciprocal": self.reciprocal,
            "state_mode": self.state_mode,
            "sample_interval": self.sample_interval,
            "enable_q_learning": self.enable_q_learning,
            "q_eta": self.q_eta,
            "q_temperature": self.q_temperature,
            "num_populations": self.num_populations,
        }
        interface_schedule_data = None
        if self.interface_schedule is not None:
            interface_schedule_data = {
                "num_populations": int(self.interface_schedule.num_populations),
                "global_transitions": list(self.interface_schedule._global_transitions),
                "pair_transitions": [
                    {"pair": list(pair), "transitions": list(transitions)}
                    for pair, transitions in sorted(self.interface_schedule._pair_transitions.items())
                ],
            }
        return save_checkpoint(
            filepath=filepath,
            state=self.state,
            rng=self.scheduler.rng,
            t=self.scheduler.time,
            scheduled_events=self._scheduled_event_records,
            seed=self.seed,
            params=params,
            timeseries=self.timeseries,
            deferred_time_origin=self._deferred_time_origin,
            scheduler_stats=copy.deepcopy(self.scheduler.stats.__dict__),
            participation_tracker=(self.transition_tracker.to_dict() if self.transition_tracker is not None else None),
            participation_params=(
                dict(self.transition_tracker.params.__dict__) if self.transition_tracker is not None else None
            ),
            interface_schedule=interface_schedule_data,
            runtime={"python": platform.python_version(), "numpy": np.__version__, "runner_schema": CHECKPOINT_SCHEMA_VERSION},
            pending_event_queue=pending_event_queue,
            scheduler_pending_clock=self.scheduler.pending_clock_state(),
            extra=extra,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Union[Dict[str, Any], str, Path]) -> TrajectoryRunner:
        """Reconstruct TrajectoryRunner from saved checkpoint with bit-for-bit continuity."""
        data = load_checkpoint(checkpoint)
        t_chk = float(data.get("t", 0.0))
        seed_chk = data.get("seed")
        rng_obj = data.get("rng")
        if rng_obj is None and "rng_state" in data and data["rng_state"] is not None:
            rng_obj = np.random.default_rng()
            rng_obj.bit_generator.state = data["rng_state"]

        state_data = data.get("state")
        if state_data is not None:
            if isinstance(state_data, State):
                restored_state = state_data.copy()
            else:
                restored_state = State.from_dict(state_data)
        else:
            raise ValueError("Checkpoint is missing State data")

        restored_state.t = t_chk

        params = data.get("params", {})
        interventions: List[Any] = []
        if data.get("scheduled_events"):
            from src.experiment.interventions import FieldPulseIntervention, TieDisruptionIntervention
            for record in data["scheduled_events"]:
                kind = record.get("type")
                if kind == "FieldPulseIntervention":
                    mask = record.get("target_mask")
                    interventions.append(FieldPulseIntervention(
                        start_time=float(record.get("start_time", 0.0)),
                        duration=float(record.get("duration", 0.0)),
                        amplitude=float(record.get("amplitude", 0.0)),
                        component=int(record.get("component", 0)),
                        target_mask=(np.asarray(mask, dtype=bool) if mask is not None else None),
                    ))
                elif kind == "TieDisruptionIntervention":
                    interventions.append(TieDisruptionIntervention(
                        disruption_time=float(record.get("disruption_time", record.get("time", 0.0))),
                        fraction=float(record.get("fraction", 0.3)),
                        reciprocal=record.get("reciprocal"),
                        seed=record.get("seed"),
                    ))
        interface_schedule = None
        if data.get("interface_schedule") is not None:
            from src.experiment.interventions import InterfaceSchedule
            schedule_data = data["interface_schedule"]
            interface_schedule = InterfaceSchedule(int(schedule_data["num_populations"]))
            interface_schedule._global_transitions = [tuple(item) for item in schedule_data["global_transitions"]]
            interface_schedule._pair_transitions = {
                tuple(item["pair"]): [tuple(transition) for transition in item["transitions"]]
                for item in schedule_data["pair_transitions"]
            }

        K_value = params.get("K")
        if isinstance(K_value, list):
            K_value = np.asarray(K_value, dtype=np.float64)
        b_value = params.get("b")
        if isinstance(b_value, list):
            b_value = np.asarray(b_value, dtype=np.float64)

        runner = cls(
            state=restored_state,
            seed=seed_chk,
            rng=rng_obj,
            N=params.get("N", restored_state.N),
            D=params.get("D", restored_state.D),
            T=params.get("T", 0.3),
            delta=params.get("delta", 0.1),
            nu=params.get("nu"),
            J=params.get("J", 1.0),
            K=K_value,
            L=params.get("L", 1.0),
            b=b_value,
            tau_x=params.get("tau_x"),
            tau_y=params.get("tau_y", 1.0),
            tau_n=params.get("tau_n"),
            epsilon=params.get("epsilon", 0.05),
            a=params.get("a", 0.0),
            c=params.get("c", 0.0),
            chi=params.get("chi", 0.0),
            reciprocal=params.get("reciprocal", False),
            state_mode=params.get("state_mode", "binary"),
            sample_interval=params.get("sample_interval", 1.0),
            enable_q_learning=params.get("enable_q_learning", False),
            q_eta=params.get("q_eta", 0.1),
            q_temperature=params.get("q_temperature", 1.0),
            interventions=interventions,
            interface_schedule=interface_schedule,
            num_populations=params.get("num_populations"),
        )

        runner.scheduler.time = t_chk
        runner.scheduler.pq = [event for event in runner.scheduler.pq if event.time > t_chk + 1e-12]
        heapq.heapify(runner.scheduler.pq)
        runner.scheduler.rng = rng_obj
        runner.memory_mgr.rng = rng_obj
        runner.rng = rng_obj
        runner.scheduler.set_total_memory(int(runner.memory_mgr.total_memory))
        # Restore the pending stochastic clock AFTER rates are rebuilt, so the
        # residual integrated hazard carries across the checkpoint boundary and
        # resumption is bit-for-bit continuous with the interrupted run.
        runner.scheduler.set_pending_clock_state(data.get("scheduler_pending_clock"))
        runner._deferred_time_origin = data.get("deferred_time_origin")
        if data.get("scheduler_stats"):
            for key, value in data["scheduler_stats"].items():
                if hasattr(runner.scheduler.stats, key):
                    setattr(runner.scheduler.stats, key, value)
        if runner.transition_tracker is not None and data.get("participation_tracker") is not None:
            if data.get("participation_params") is not None:
                runner.transition_tracker.params = ParticipationParams(**data["participation_params"])
            runner.transition_tracker.from_dict(data["participation_tracker"])
            runner.state.n = runner.transition_tracker.n.copy()
            deadline = runner.transition_tracker.t_deadline
            if deadline is not None and deadline > t_chk and not runner.transition_tracker.transition_occurred:
                generation = runner.transition_tracker.deadline_generation
                runner.scheduler.schedule_deterministic_event(
                    time=deadline,
                    priority=2,
                    event_type="holding_deadline",
                    payload={"generation": generation},
                    callback=lambda sched, event: (
                        runner.transition_tracker.handle_deadline_event(sched.time, event.payload),
                        setattr(runner.state, "holding_active", runner.transition_tracker.holding_active),
                        setattr(runner.state, "transition_occurred", runner.transition_tracker.transition_occurred),
                    ),
                )
        if "timeseries" in data:
            runner.timeseries = copy.deepcopy(data["timeseries"])

        return runner
