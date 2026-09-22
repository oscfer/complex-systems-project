"""Experimental interventions for emergent order simulations.

Implements:
1. Scheduled applied field pulses (targeted elite intervention, uniform broadcast, opposing majority counter-pulses).
2. Structural tie disruptions (random edge deletions and memory mass tracking).
3. Population interface schedules (closed, simultaneous, and sequential openings).
4. Mechanism speed rescaling (scaling nu and delta while preserving memory scale nu / delta).

Simulation Stage Mapping:
    - Stage 2: Extension campaigns driving controlled structural and dynamic perturbations
      across the 6 experimental scenario families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.state import State


@dataclass
class FieldPulseIntervention:
    """Scheduled external field pulse intervention.

    Applies external driving field pulse h(t) of given amplitude to targeted agents
    during the interval [start_time, start_time + duration).
    """

    start_time: float = 0.0
    duration: float = 0.0
    amplitude: float = 0.0
    component: int = 0
    target_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.duration < 0.0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")
        if self.component < 0:
            raise ValueError(f"component must be non-negative, got {self.component}")

    @property
    def end_time(self) -> float:
        """Timestamp when pulse terminates."""
        return self.start_time + self.duration

    def is_active(self, t: float) -> bool:
        """Check if pulse is currently active at simulation time t."""
        if self.duration <= 0.0 or self.amplitude == 0.0:
            return False
        return self.start_time <= t < self.end_time

    def get_field(self, t: float) -> float:
        """Return scalar external field amplitude applied at time t."""
        if self.is_active(t):
            return float(self.amplitude)
        return 0.0

    def get_fields_vector(self, t: float, N: int) -> np.ndarray:
        """Return array of shape (N,) with external field applied to each agent at time t."""
        if not self.is_active(t):
            return np.zeros(N, dtype=np.float64)

        val = float(self.amplitude)
        if self.target_mask is None:
            return np.full(N, val, dtype=np.float64)

        mask = np.asarray(self.target_mask, dtype=bool)
        if len(mask) != N:
            raise ValueError(f"target_mask length {len(mask)} does not match population size N={N}")
        fields = np.zeros(N, dtype=np.float64)
        fields[mask] = val
        return fields

    def apply(self, state: State, t: float) -> None:
        """Apply intervention update to state at time t."""
        active = self.is_active(t)
        state.pulse_active = active
        state.pulse_val = float(self.amplitude) if active else 0.0
        state.pulse_component = self.component
        if self.target_mask is not None:
            state.pulse_targets = self.target_mask.copy()
        else:
            state.pulse_targets = None

    def schedule_on(self, scheduler: Any, state: Optional[State] = None) -> None:
        """Schedule deterministic pulse onset and offset events on scheduler."""
        scheduler.add_deterministic_event(
            time=self.start_time,
            priority=1,
            event_type="pulse_start",
            callback=(lambda *args: self.apply(state, self.start_time)) if state is not None else None,
        )
        scheduler.add_deterministic_event(
            time=self.end_time,
            priority=0,
            event_type="pulse_end",
            callback=(lambda *args: self.apply(state, self.end_time)) if state is not None else None,
        )


def create_counter_pulse(
    pre_pulse_mean: float,
    amplitude: Optional[float] = None,
    amplitude_ratio: Optional[float] = None,
    J: float = 1.0,
    start_time: float = 0.0,
    duration: float = 5.0,
    component: int = 0,
    target_mask: Optional[np.ndarray] = None,
) -> FieldPulseIntervention:
    """Create a field pulse opposing the sign of pre-pulse public stance mean.

    Direction opposes pre_pulse_mean:
        If pre_pulse_mean > 0: amplitude is negative.
        If pre_pulse_mean < 0: amplitude is positive.
        If pre_pulse_mean == 0: deterministic negative direction.

    Args:
        pre_pulse_mean: Mean expressed stance m^y before pulse.
        amplitude: Explicit absolute amplitude |h|.
        amplitude_ratio: Normalized ratio |h| / J.
        J: Social coupling strength.
        start_time: Scheduled start time.
        duration: Pulse duration.
        component: Target belief component.
        target_mask: Optional boolean mask of target nodes.

    Returns:
        FieldPulseIntervention configured with opposing amplitude.
    """
    if amplitude is not None:
        mag = abs(float(amplitude))
    elif amplitude_ratio is not None:
        mag = abs(float(amplitude_ratio) * float(J))
    else:
        mag = 1.0

    if pre_pulse_mean > 0.0:
        signed_amplitude = -mag
    elif pre_pulse_mean < 0.0:
        signed_amplitude = mag
    else:
        signed_amplitude = -mag

    return FieldPulseIntervention(
        start_time=start_time,
        duration=duration,
        amplitude=signed_amplitude,
        component=component,
        target_mask=target_mask,
    )


def create_targeted_pulse(
    ranking: np.ndarray,
    fraction: float = 0.10,
    amplitude: float = 1.0,
    start_time: float = 0.0,
    duration: float = 5.0,
    component: int = 0,
) -> FieldPulseIntervention:
    """Create a field pulse targeted to top fraction of nodes by ranking metric.

    Args:
        ranking: 1D array of node rankings (e.g. in-degree or attention metric r).
        fraction: Top fraction of nodes to target in (0.0, 1.0].
        amplitude: Field amplitude to apply.
        start_time: Start time.
        duration: Pulse duration.
        component: Target belief component.

    Returns:
        FieldPulseIntervention targeting top nodes.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"Targeting fraction must be in [0.0, 1.0], got {fraction}")
    r = np.nan_to_num(np.asarray(ranking, dtype=np.float64), nan=-np.inf)
    N = len(r)
    k = int(round(N * fraction))
    if k < 1 and fraction > 0.0 and N > 0:
        k = 1
    if k > N:
        k = N

    target_mask = np.zeros(N, dtype=bool)
    if k > 0:
        top_indices = np.argsort(r)[-k:]
        target_mask[top_indices] = True

    return FieldPulseIntervention(
        start_time=start_time,
        duration=duration,
        amplitude=amplitude,
        component=component,
        target_mask=target_mask,
    )


def create_uniform_pulse(
    N: int,
    fraction: float = 0.10,
    amplitude: float = 1.0,
    start_time: float = 0.0,
    duration: float = 5.0,
    component: int = 0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> FieldPulseIntervention:
    """Create a field pulse applied uniformly at random to a fraction of nodes.

    Args:
        N: Population size.
        fraction: Fraction of nodes to target.
        amplitude: Field amplitude to apply.
        start_time: Start time.
        duration: Pulse duration.
        component: Target belief component.
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        FieldPulseIntervention with random subset mask.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"Targeting fraction must be in [0.0, 1.0], got {fraction}")

    if rng is None:
        rng = np.random.default_rng(seed)

    k = int(round(N * fraction))
    if k < 1 and fraction > 0.0 and N > 0:
        k = 1
    if k > N:
        k = N

    target_mask = np.zeros(N, dtype=bool)
    if k > 0:
        chosen = rng.choice(N, size=k, replace=False)
        target_mask[chosen] = True

    return FieldPulseIntervention(
        start_time=start_time,
        duration=duration,
        amplitude=amplitude,
        component=component,
        target_mask=target_mask,
    )


def apply_tie_disruption(
    W: np.ndarray,
    fraction: float,
    reciprocal: Optional[bool] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform structural tie disruption by deleting a fraction of active ties.

    For each deleted tie (i, j), W_ij is set to 0 (full memory erasure).
    In reciprocal mode (or when reciprocal=None and W is symmetric), ties are deleted
    as pairs to preserve symmetry and prevent uint32 underflow during decay.

    Args:
        W: Interaction memory matrix of shape (N, N).
        fraction: Fraction of active ties (W_ij > 0) to delete, in [0.0, 1.0].
        reciprocal: Whether to enforce paired reciprocal deletions. If None, auto-detected
            from whether W is non-empty and symmetric.
        seed: Optional PRNG seed.
        rng: Optional NumPy random Generator.

    Returns:
        Tuple of:
            W_after: Disrupted memory matrix with deleted ties set to 0.
            stats: Dictionary with "edges_deleted" and "memory_mass_deleted".
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"Disruption fraction must be in [0.0, 1.0], got {fraction}")

    W_arr = np.asarray(W)
    if W_arr.ndim != 2 or W_arr.shape[0] != W_arr.shape[1]:
        raise ValueError(f"W must be a square 2D matrix, got shape {W_arr.shape}")
    if np.any(W_arr < 0):
        raise ValueError("Memory matrix W cannot contain negative entries")

    target_dtype = W_arr.dtype if W_arr.dtype in (np.uint32, np.int32, np.int64) else np.uint32
    W_after = np.array(W_arr, dtype=target_dtype).copy()
    if reciprocal is None:
        reciprocal = bool(np.sum(W_after) > 0 and np.array_equal(W_after, W_after.T))

    if rng is None:
        rng = np.random.default_rng(seed)

    if reciprocal:
        upper_coords = np.argwhere(np.triu(W_after, k=1) > 0)
        num_pairs = len(upper_coords)
        if num_pairs == 0 or fraction <= 0.0:
            return W_after, {"edges_deleted": 0, "memory_mass_deleted": 0}

        k = int(round(num_pairs * fraction))
        if k > num_pairs:
            k = num_pairs
        if k <= 0:
            return W_after, {"edges_deleted": 0, "memory_mass_deleted": 0}

        selected_indices = rng.choice(num_pairs, size=k, replace=False)
        mass_before = int(np.sum(W_after))
        for idx in selected_indices:
            r, c = upper_coords[idx]
            W_after[r, c] = 0
            W_after[c, r] = 0
        mass_after = int(np.sum(W_after))
        return W_after, {
            "edges_deleted": 2 * k,
            "memory_mass_deleted": mass_before - mass_after,
        }
    else:
        active_coords = np.argwhere(W_after > 0)
        num_active = len(active_coords)

        if num_active == 0 or fraction <= 0.0:
            return W_after, {"edges_deleted": 0, "memory_mass_deleted": 0}

        k = int(round(num_active * fraction))
        if k > num_active:
            k = num_active
        if k <= 0:
            return W_after, {"edges_deleted": 0, "memory_mass_deleted": 0}

        selected_indices = rng.choice(num_active, size=k, replace=False)
        mass_before = int(np.sum(W_after))

        for idx in selected_indices:
            r, c = active_coords[idx]
            W_after[r, c] = 0

        mass_after = int(np.sum(W_after))
        return W_after, {
            "edges_deleted": k,
            "memory_mass_deleted": mass_before - mass_after,
        }


@dataclass
class CommandedFieldIntervention:
    """Commanded intervention field u_{i,alpha}(t) under the control contract.

    Unlike FieldPulseIntervention (a prescribed background field h with no
    susceptibility), a commanded field enters the public local field as
    g_i * u_{i,alpha}, is bounded by u_max, and is charged as commanded u:
    exposure = integral of sum_{i,alpha} |u_{i,alpha}| dt.
    """

    start_time: float = 0.0
    duration: float = 0.0
    amplitude: float = 0.0
    component: int = 0
    target_mask: Optional[np.ndarray] = None
    u_max: float = 4.0

    def __post_init__(self) -> None:
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.duration < 0.0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")
        if self.component < 0:
            raise ValueError(f"component must be non-negative, got {self.component}")
        if abs(self.amplitude) > self.u_max:
            raise ValueError(
                f"commanded amplitude |u|={abs(self.amplitude)} exceeds cap u_max={self.u_max}"
            )

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def is_active(self, t: float) -> bool:
        if self.duration <= 0.0 or self.amplitude == 0.0:
            return False
        return self.start_time <= t < self.end_time

    def exposure(self) -> float:
        """Commanded charge over the full pulse: sum |u| * duration over targets."""
        if self.duration <= 0.0 or self.amplitude == 0.0:
            return 0.0
        count = getattr(self, "N_targets", 0)
        return float(abs(self.amplitude) * self.duration * count)

    def schedule_on(self, scheduler: Any, runner: Any) -> None:
        """Install commanded-field entries on the runner at start/end events."""
        N = getattr(runner, "N", None)
        if N is None:
            raise ValueError("CommandedFieldIntervention requires a runner with attribute N")
        D = runner.D
        self.N_targets = int(np.sum(np.asarray(self.target_mask, dtype=bool))) if self.target_mask is not None else N

        runner_self = runner

        def _install(sched: Any, event: Any = None) -> None:
            if runner_self.u_command is None:
                runner_self.u_command = np.zeros((N, D), dtype=np.float64)
            if self.target_mask is None:
                runner_self.u_command[:, self.component] = float(self.amplitude)
            else:
                mask = np.asarray(self.target_mask, dtype=bool)
                runner_self.u_command[mask, self.component] = float(self.amplitude)
            runner_self._u_active = True

        def _release(sched: Any, event: Any = None) -> None:
            # Charge commanded u over the active interval, not g_i * u.
            active_nodes = (
                np.asarray(self.target_mask, dtype=bool).sum()
                if self.target_mask is not None
                else N
            )
            runner_self.u_exposure += float(abs(self.amplitude)) * float(self.duration) * int(active_nodes)
            if runner_self.u_command is not None:
                if self.target_mask is None:
                    runner_self.u_command[:, self.component] = 0.0
                else:
                    mask = np.asarray(self.target_mask, dtype=bool)
                    runner_self.u_command[mask, self.component] = 0.0

        t_start = float(self.start_time)
        t_end = float(self.end_time)
        if self.is_active(t_start + 1e-12):
            scheduler.schedule_deterministic_event(
                time=t_start, priority=1, event_type="commanded_field_start", callback=_install
            )
        if self.duration > 0.0:
            scheduler.schedule_deterministic_event(
                time=t_start + float(self.duration),
                priority=0,
                event_type="commanded_field_end",
                callback=_release,
            )


@dataclass
class TieDisruptionIntervention:
    """Scheduled tie disruption intervention wrapper."""

    disruption_time: float = 0.0
    fraction: float = 0.30
    reciprocal: Optional[bool] = None
    seed: Optional[int] = None
    time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.time is not None:
            self.disruption_time = self.time
        else:
            self.time = self.disruption_time
        if self.disruption_time < 0.0:
            raise ValueError(f"disruption_time must be non-negative, got {self.disruption_time}")
        if not (0.0 <= self.fraction <= 1.0):
            raise ValueError(f"fraction must be in [0.0, 1.0], got {self.fraction}")

    def apply(
        self,
        state: State,
        rng: Optional[np.random.Generator] = None,
        memory_manager: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Apply tie disruption to state memory matrix and optionally sync memory manager and scheduler."""
        W_after, stats = apply_tie_disruption(
            state.W,
            self.fraction,
            reciprocal=self.reciprocal,
            seed=self.seed,
            rng=rng,
        )
        state.W = W_after
        if memory_manager is not None:
            memory_manager.set_memory(W_after)
        if scheduler is not None:
            if hasattr(scheduler, "set_total_memory"):
                scheduler.set_total_memory(int(np.sum(W_after)))
            elif hasattr(scheduler, "scheduler") and hasattr(scheduler.scheduler, "set_total_memory"):
                scheduler.scheduler.set_total_memory(int(np.sum(W_after)))
        return stats

    def schedule_on(
        self,
        scheduler: Any,
        state: Optional[State] = None,
        memory_manager: Optional[Any] = None,
    ) -> None:
        """Schedule deterministic tie disruption event on scheduler."""
        scheduler.add_deterministic_event(
            time=self.disruption_time,
            priority=1,
            event_type="tie_disruption",
            callback=(lambda *args: self.apply(state, memory_manager=memory_manager, scheduler=scheduler)) if state is not None else None,
        )


class InterfaceSchedule:
    """Multi-population contact eligibility schedule.

    Controls the K x K encounter eligibility matrix E(t) where E_{ab}(t) in [0, 1]
    determines whether agents in population a can encounter agents in population b.
    Internal encounters within a population are always eligible (E_{aa} = 1.0).
    """

    def __init__(self, num_populations: int = 1):
        if num_populations < 1:
            raise ValueError(f"num_populations must be >= 1, got {num_populations}")
        self.num_populations = int(num_populations)
        # Pair-specific transitions: key (a, b) -> list of (t, state)
        self._pair_transitions: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        # Global transitions affecting all cross-population pairs
        self._global_transitions: List[Tuple[float, float]] = []

    def set_closed(self) -> None:
        """Set all cross-population encounters to closed (0.0)."""
        self._global_transitions = [(0.0, 0.0)]
        self._pair_transitions.clear()

    def add_transition(
        self,
        t: float,
        state: float,
        pair: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add an interface state transition at time t."""
        if t < 0.0:
            raise ValueError(f"Transition time t must be non-negative, got {t}")
        if not (0.0 <= state <= 1.0):
            raise ValueError(f"Transition state must be in [0.0, 1.0], got {state}")
        if pair is not None:
            a, b = pair
            if a < 0 or a >= self.num_populations or b < 0 or b >= self.num_populations or a == b:
                raise ValueError(
                    f"Invalid pair ({a}, {b}) for num_populations={self.num_populations}. "
                    "Must be distinct indices in [0, num_populations - 1]."
                )
            p1 = (min(a, b), max(a, b))
            p2 = (max(a, b), min(a, b))
            if p1 not in self._pair_transitions:
                self._pair_transitions[p1] = []
                self._pair_transitions[p2] = []
            self._pair_transitions[p1].append((float(t), float(state)))
            self._pair_transitions[p2].append((float(t), float(state)))
            self._pair_transitions[p1].sort(key=lambda x: x[0])
            self._pair_transitions[p2].sort(key=lambda x: x[0])
        else:
            self._global_transitions.append((float(t), float(state)))
            self._global_transitions.sort(key=lambda x: x[0])

    def schedule_simultaneous_opening(self, t_open: float = 0.0) -> None:
        """Schedule all cross-population interfaces to open simultaneously at t_open."""
        if t_open < 0.0:
            raise ValueError(f"Opening time t_open must be non-negative, got {t_open}")
        self._global_transitions = [(0.0, 0.0), (float(t_open), 1.0)]
        self._global_transitions.sort(key=lambda x: x[0])
        self._pair_transitions.clear()

    def schedule_sequential_opening(
        self,
        t_start: float,
        interval: float,
    ) -> None:
        """Schedule pairs to open sequentially one by one every interval.

        Pair (0, 1) opens at t_start.
        Pair (1, 2) opens at t_start + interval.
        Pair (p, p+1) opens at t_start + p * interval.
        """
        if t_start < 0.0:
            raise ValueError(f"Start time t_start must be non-negative, got {t_start}")
        if interval <= 0.0:
            raise ValueError(f"Opening interval must be strictly positive, got {interval}")
        self._global_transitions = [(0.0, 0.0)]
        self._pair_transitions.clear()
        for p in range(self.num_populations - 1):
            t_p = float(t_start + p * interval)
            self.add_transition(t=t_p, state=1.0, pair=(p, p + 1))

    def get_matrix(self, t: float) -> np.ndarray:
        """Return K x K eligibility matrix at simulation time t."""
        K = self.num_populations
        E = np.eye(K, dtype=np.float64)

        if K == 1:
            return E

        # Determine default global cross state at time t
        cross_val = 0.0
        for t_trans, st in self._global_transitions:
            if t >= t_trans:
                cross_val = st
            else:
                break

        for a in range(K):
            for b in range(K):
                if a == b:
                    continue
                # Check pair-specific transition
                pair = (a, b)
                if pair in self._pair_transitions:
                    p_val = cross_val
                    for t_trans, st in self._pair_transitions[pair]:
                        if t >= t_trans:
                            p_val = st
                        else:
                            break
                    E[a, b] = p_val
                else:
                    E[a, b] = cross_val

        return E

    def get_eligibility_matrix(
        self,
        t: float,
        population_origins: np.ndarray,
    ) -> np.ndarray:
        """Return full N x N agent-level eligibility matrix C at simulation time t."""
        from src.core.memory import MemoryManager
        E = self.get_matrix(t)
        return MemoryManager.create_eligibility_matrix(E, population_origins)

    def schedule_on(
        self,
        scheduler: Any,
        memory_manager: Optional[Any] = None,
        population_origins: Optional[np.ndarray] = None,
    ) -> None:
        """Schedule deterministic interface transition events on scheduler."""
        unique_times: Dict[float, List[str]] = {}
        for t_trans, _ in self._global_transitions:
            unique_times.setdefault(t_trans, []).append("interface_global")
        for pair, transitions in self._pair_transitions.items():
            if pair[0] < pair[1]:
                for t_trans, _ in transitions:
                    unique_times.setdefault(t_trans, []).append(f"interface_pair_{pair[0]}_{pair[1]}")

        for t_trans in sorted(unique_times.keys()):
            ev_label = "+".join(unique_times[t_trans])

            def make_callback(t_ev: float):
                if memory_manager is None:
                    return None

                def cb(*args):
                    E = self.get_matrix(t_ev)
                    if hasattr(memory_manager, "set_population_interfaces"):
                        if population_origins is not None:
                            memory_manager.set_population_interfaces(population_origins, E)
                        elif getattr(memory_manager, "N", None) == self.num_populations:
                            memory_manager.set_eligibility_matrix(E)
                        else:
                            N = memory_manager.N
                            K = self.num_populations
                            block_sizes = [N // K + (1 if b < N % K else 0) for b in range(K)]
                            pop = np.repeat(np.arange(K, dtype=np.int64), block_sizes)
                            memory_manager.set_population_interfaces(pop, E)
                    elif hasattr(memory_manager, "set_eligibility_matrix"):
                        if getattr(memory_manager, "N", None) == self.num_populations:
                            memory_manager.set_eligibility_matrix(E)
                        else:
                            from src.core.memory import MemoryManager
                            if population_origins is not None:
                                pop = population_origins
                            else:
                                N = getattr(memory_manager, "N", len(E))
                                K = self.num_populations
                                block_sizes = [N // K + (1 if b < N % K else 0) for b in range(K)]
                                pop = np.repeat(np.arange(K, dtype=np.int64), block_sizes)
                            C = MemoryManager.create_eligibility_matrix(E, pop)
                            memory_manager.set_eligibility_matrix(C)

                return cb

            scheduler.add_deterministic_event(
                time=t_trans,
                priority=1,
                event_type=ev_label,
                callback=make_callback(t_trans) if memory_manager is not None else None,
            )


def rescale_mechanism_speed(
    nu: float,
    delta: float,
    speed_multiplier: float,
) -> Tuple[float, float]:
    """Rescale mechanism rates: (nu, delta) <- (lambda * nu, lambda * delta).

    Preserves the memory scale invariant nu / delta.

    Args:
        nu: Per-agent encounter rate.
        delta: Memory decay rate.
        speed_multiplier: Scaling factor lambda > 0.

    Returns:
        Tuple of (rescaled_nu, rescaled_delta).
    """
    if speed_multiplier <= 0.0:
        raise ValueError(f"Speed multiplier must be strictly positive, got {speed_multiplier}")
    if nu < 0.0 or delta < 0.0:
        raise ValueError("Rates nu and delta must be non-negative")
    return float(nu) * float(speed_multiplier), float(delta) * float(speed_multiplier)


class MechanismSpeedManager:
    """Manages dynamic mechanism speed rescaling during simulation."""

    def __init__(self, base_nu: float, base_delta: float):
        if base_nu < 0.0 or base_delta < 0.0:
            raise ValueError("Base rates base_nu and base_delta must be non-negative")
        self.base_nu = float(base_nu)
        self.base_delta = float(base_delta)
        self.current_nu = float(base_nu)
        self.current_delta = float(base_delta)
        self.speed_multiplier = 1.0

    def apply_multiplier(self, multiplier: float, scheduler: Optional[Any] = None) -> Tuple[float, float]:
        """Apply speed multiplier lambda to rates and optionally update scheduler."""
        self.speed_multiplier = float(multiplier)
        self.current_nu, self.current_delta = rescale_mechanism_speed(
            self.base_nu, self.base_delta, self.speed_multiplier
        )
        if scheduler is not None:
            if hasattr(scheduler, "nu"):
                scheduler.nu = self.current_nu
            if hasattr(scheduler, "delta"):
                scheduler.delta = self.current_delta
            if hasattr(scheduler, "scheduler"):
                inner = scheduler.scheduler
                if hasattr(inner, "nu"):
                    inner.nu = self.current_nu
                if hasattr(inner, "delta"):
                    inner.delta = self.current_delta
        return self.current_nu, self.current_delta

    def restore_baseline(self, scheduler: Optional[Any] = None) -> Tuple[float, float]:
        """Restore rates to baseline (lambda = 1.0)."""
        return self.apply_multiplier(1.0, scheduler=scheduler)
