"""Master continuous-time Poisson jump event scheduler with deterministic preemption (Stages 0–3).

Combines exact stochastic Markov jump processes across 5 discrete event channels:
1. Private state attempt channel (R_x = N * D / tau_x).
2. Public state attempt channel (R_y = N * D / tau_y).
3. Encounter attempt channel (R_enc = N * nu, or N * nu / 2 in reciprocal mode).
4. Memory unit decay channel (R_forget = delta * S(t), or delta * S(t) / 2 in reciprocal mode).
5. Participation redraw channel (R_n = N / tau_n).

Integrated with a deterministic priority queue (min-heap) providing exact-timestamp
preemption for scheduled field pulses, structural tie disruptions, interface openings,
institutional holding deadlines, and observation snapshots.

Simulation Stage Mapping:
    - Stage 0: Event rate calibration, timing benchmarks, and analytical attempt count verification.
    - Stage 1: Core trajectory evolution across private/public states and encounter accumulation.
    - Stage 2: Scheduled interventions (preemption for field pulses, disruptions, interface opening).
    - Stage 3: Q-learning step event scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any, Callable, Dict, List, Optional
import numpy as np


# Priority constants for deterministic events occurring at identical timestamps
PRIORITY_RESTORE = 0        # e.g., restoring field h = 0 at pulse termination
PRIORITY_INTERVENTION = 1   # e.g., field injection, edge disruption, interface toggle
PRIORITY_HOLDING = 2        # institutional transition holding deadlines
PRIORITY_SNAPSHOT = 3       # data recording and observable harvesting
PRIORITY_HORIZON = 4        # simulation horizon termination


@dataclass(order=True)
class DeterministicEvent:
    """Scheduled deterministic event in the min-heap."""

    time: float
    priority: int
    seq_id: int
    event_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Optional[Callable[[PoissonScheduler, DeterministicEvent], None]] = field(
        compare=False, default=None
    )


def _invoke_callback(cb: Callable[..., Any], scheduler: Any, event_or_time: Any) -> None:
    """Safely invoke callback handling 0, 1, or 2 positional arguments."""
    import inspect
    try:
        sig = inspect.signature(cb)
        params = list(sig.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs:
            cb(scheduler, event_or_time)
            return
        num_pos = sum(
            1
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        )
        if num_pos == 0:
            cb()
        elif num_pos == 1:
            cb(scheduler)
        else:
            cb(scheduler, event_or_time)
    except (ValueError, TypeError):
        try:
            cb()
        except TypeError:
            try:
                cb(scheduler)
            except TypeError:
                cb(scheduler, event_or_time)


def _make_channel_invoker(cb: Optional[Callable[..., Any]]) -> Optional[Callable[[int], None]]:
    """Create optimized channel callback wrapper to avoid per-event inspect overhead."""
    if cb is None:
        return None
    import inspect
    try:
        sig = inspect.signature(cb)
        params = list(sig.parameters.values())
        if len(params) == 0:
            return lambda _: cb()
        else:
            return lambda node: cb(node)
    except (ValueError, TypeError):
        def fallback(node: int) -> None:
            try:
                cb(node)
            except TypeError:
                cb()
        return fallback


def _invoke_channel_cb(cb: Callable[..., Any], node_idx: int) -> None:
    """Safely invoke channel callback, forwarding node_idx if accepted."""
    import inspect
    try:
        sig = inspect.signature(cb)
        params = list(sig.parameters.values())
        if len(params) == 0:
            cb()
        else:
            cb(node_idx)
    except (ValueError, TypeError):
        try:
            cb()
        except TypeError:
            cb(node_idx)


@dataclass
class SchedulerStats:
    """Counters tracking event executions and preemptions."""

    private_updates: int = 0
    public_updates: int = 0
    encounters: int = 0
    null_encounters: int = 0
    memory_decays: int = 0
    participation_updates: int = 0
    deterministic_events: int = 0
    preemptions: int = 0

    @property
    def total_stochastic_events(self) -> int:
        """Total stochastic Poisson events fired."""
        return (
            self.private_updates
            + self.public_updates
            + self.encounters
            + self.null_encounters
            + self.memory_decays
            + self.participation_updates
        )

    @property
    def total_events(self) -> int:
        """Total events executed (stochastic + deterministic)."""
        return self.total_stochastic_events + self.deterministic_events


class PoissonScheduler:
    """Master Poisson jump scheduler with priority-queue deterministic preemption.

    Attributes:
        N: Population size.
        D: Belief dimensionality.
        tau_x: Private belief update timescale.
        tau_y: Public expression update timescale.
        nu: Encounter attempt rate per agent.
        delta: Memory unit decay rate.
        tau_n: Participation update timescale (None if participation inactive).
        reciprocal: Whether reciprocal memory mode is enabled.
        time: Current simulation time.
        stats: SchedulerStats counter object.
    """

    def __init__(
        self,
        N: int = 10,
        D: int = 1,
        tau_x: Union[float, np.ndarray] = 1.0,
        tau_y: float = 1.0,
        nu: Union[float, np.ndarray] = 1.0,
        delta: float = 0.1,
        tau_n: Optional[float] = None,
        reciprocal: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        if tau_y <= 0.0:
            raise ValueError("Time scales tau_x, tau_y must be strictly positive")
        if delta < 0.0:
            raise ValueError("Rates nu and delta cannot be negative")
        if tau_n is not None and tau_n <= 0.0:
            raise ValueError(f"Participation timescale tau_n must be strictly positive, got {tau_n}")

        self.N = N
        self.D = D
        self.tau_y = float(tau_y)
        self._delta = float(delta)
        self.tau_n = float(tau_n) if tau_n is not None else None
        self.reciprocal = bool(reciprocal)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.time: float = 0.0
        self.pq: List[DeterministicEvent] = []
        self.seq_counter: int = 0
        self.stats = SchedulerStats()

        # Pending stochastic clock: absolute time of the next stochastic event,
        # drawn once and preserved across deterministic-event boundaries. The
        # residual integrated hazard is conserved when channel rates change
        # between the draw and the event, so scheduled boundaries (snapshots,
        # deliveries, release deadlines, holding checks, checkpoints) never
        # alter the seeded path. None = no clock drawn yet.
        self._pending_time: Optional[float] = None
        self._r_at_draw: Optional[float] = None

        # Precomputed probability arrays for weighted node sampling
        self._p_tau_x: Optional[np.ndarray] = None
        self._p_nu: Optional[np.ndarray] = None

        # Handle tau_x (scalar or quenched array)
        self.tau_x_arr: Optional[np.ndarray] = None
        if isinstance(tau_x, (int, float, np.floating, np.integer)):
            if tau_x <= 0.0:
                raise ValueError("Time scales tau_x, tau_y must be strictly positive")
            self._tau_x = float(tau_x)
            self.R_x = float(N * D) / self._tau_x
        else:
            arr_x = np.asarray(tau_x, dtype=np.float64)
            if arr_x.shape != (N,):
                raise ValueError(f"tau_x shape mismatch: expected ({N},), got {arr_x.shape}")
            if np.any(arr_x <= 0.0):
                raise ValueError("Time scales tau_x, tau_y must be strictly positive")
            self.tau_x_arr = arr_x
            self._tau_x = float(np.mean(arr_x))
            inv = 1.0 / self.tau_x_arr
            inv_sum = float(np.sum(inv))
            self._p_tau_x = inv / inv_sum
            self.R_x = float(D) * inv_sum

        # Handle nu (scalar or quenched array)
        self.nu_arr: Optional[np.ndarray] = None
        if isinstance(nu, (int, float, np.floating, np.integer)):
            if nu < 0.0:
                raise ValueError("Rates nu and delta cannot be negative")
            self._nu = float(nu)
            self.R_enc = (float(N) * self._nu / 2.0) if self.reciprocal else (float(N) * self._nu)
        else:
            arr_nu = np.asarray(nu, dtype=np.float64)
            if arr_nu.shape != (N,):
                raise ValueError(f"nu shape mismatch: expected ({N},), got {arr_nu.shape}")
            if np.any(arr_nu < 0.0):
                raise ValueError("Rates nu and delta cannot be negative")
            self.nu_arr = arr_nu
            self._nu = float(np.mean(arr_nu))
            sum_nu = float(np.sum(self.nu_arr))
            self._p_nu = (self.nu_arr / sum_nu) if sum_nu > 0.0 else None
            self.R_enc = (sum_nu / 2.0) if self.reciprocal else sum_nu

        # Public and participation propensities
        self.R_y = float(N * D) / self.tau_y
        self.R_n = (float(N) / self.tau_n) if self.tau_n is not None else 0.0

        # Dynamic memory propensity
        self.total_memory: int = 0
        self.R_forget: float = 0.0

        # Registered channel callbacks and fast invokers
        self.cb_private: Optional[Callable[..., None]] = None
        self.cb_public: Optional[Callable[..., None]] = None
        self.cb_encounter: Optional[Callable[..., None]] = None
        self.cb_decay: Optional[Callable[..., None]] = None
        self.cb_participation: Optional[Callable[..., None]] = None
        self._invoker_private: Optional[Callable[[int], None]] = None
        self._invoker_public: Optional[Callable[[int], None]] = None
        self._invoker_encounter: Optional[Callable[[int], None]] = None

    @property
    def nu(self) -> Union[float, np.ndarray]:
        """Per-agent encounter rate."""
        return self.nu_arr if self.nu_arr is not None else self._nu

    @nu.setter
    def nu(self, val: Union[float, np.ndarray]) -> None:
        if isinstance(val, (int, float, np.floating, np.integer)):
            if val < 0.0:
                raise ValueError("Rate nu cannot be negative")
            self._nu = float(val)
            self.nu_arr = None
            self._p_nu = None
            self.R_enc = (float(self.N) * self._nu / 2.0) if self.reciprocal else (float(self.N) * self._nu)
        else:
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape != (self.N,):
                raise ValueError(f"nu shape mismatch: expected ({self.N},), got {arr.shape}")
            if np.any(arr < 0.0):
                raise ValueError("Rates nu and delta cannot be negative")
            self.nu_arr = arr
            self._nu = float(np.mean(arr))
            sum_nu = float(np.sum(self.nu_arr))
            self._p_nu = (self.nu_arr / sum_nu) if sum_nu > 0.0 else None
            self.R_enc = (sum_nu / 2.0) if self.reciprocal else sum_nu

    @property
    def delta(self) -> float:
        """Memory unit decay rate."""
        return self._delta

    @delta.setter
    def delta(self, val: float) -> None:
        if val < 0.0:
            raise ValueError("Rate delta cannot be negative")
        self._delta = float(val)
        if self.reciprocal:
            self.R_forget = self._delta * (float(self.total_memory) / 2.0)
        else:
            self.R_forget = self._delta * float(self.total_memory)

    @property
    def tau_x(self) -> Union[float, np.ndarray]:
        """Private belief update timescale."""
        return self.tau_x_arr if self.tau_x_arr is not None else self._tau_x

    @tau_x.setter
    def tau_x(self, val: Union[float, np.ndarray]) -> None:
        if isinstance(val, (int, float, np.floating, np.integer)):
            if val <= 0.0:
                raise ValueError("Time scales tau_x, tau_y must be strictly positive")
            self._tau_x = float(val)
            self.tau_x_arr = None
            self._p_tau_x = None
            self.R_x = float(self.N * self.D) / self._tau_x
        else:
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape != (self.N,):
                raise ValueError(f"tau_x shape mismatch: expected ({self.N},), got {arr.shape}")
            if np.any(arr <= 0.0):
                raise ValueError("Time scales tau_x, tau_y must be strictly positive")
            self.tau_x_arr = arr
            self._tau_x = float(np.mean(arr))
            inv = 1.0 / self.tau_x_arr
            inv_sum = float(np.sum(inv))
            self._p_tau_x = inv / inv_sum
            self.R_x = float(self.D) * inv_sum

    def sample_private_node(self) -> int:
        """Sample node i for private state update proportional to 1 / tau_{x,i}."""
        if self._p_tau_x is not None:
            return int(self.rng.choice(self.N, p=self._p_tau_x))
        return int(self.rng.integers(0, self.N))

    def sample_encounter_initiator(self) -> int:
        """Sample initiator node i for encounter attempt proportional to nu_i."""
        if self._p_nu is not None:
            return int(self.rng.choice(self.N, p=self._p_nu))
        return int(self.rng.integers(0, self.N))

    def sample_public_node(self) -> int:
        """Sample node i for public state update uniformly."""
        return int(self.rng.integers(0, self.N))

    @property
    def R_total(self) -> float:
        """Total Poisson jump propensity."""
        return self.R_x + self.R_y + self.R_enc + self.R_forget + self.R_n

    def _draw_pending_clock(self) -> None:
        """Draw the next stochastic event time once (Exp(R_total) from self.time).

        The draw is preserved until the event fires; it is never discarded by
        deterministic boundaries.
        """
        r_tot = self.R_total
        if r_tot > 0.0:
            u = self.rng.random()
            if u <= 0.0:
                u = np.nextafter(0.0, 1.0)
            self._pending_time = self.time - np.log(u) / r_tot
            self._r_at_draw = r_tot
        else:
            self._pending_time = None
            self._r_at_draw = None

    def _sync_pending_clock(self) -> None:
        """Rescale the pending clock after a channel-rate change.

        Residual integrated hazard to the pending event is conserved:
        (pending - time) * r_old == (pending' - time) * r_new. If the total
        rate crosses zero, the pending clock is voided and redrawn on demand.
        """
        r_now = self.R_total
        if self._pending_time is not None:
            r_old = self._r_at_draw
            if r_old is not None and r_old != r_now:
                if r_old > 0.0 and r_now > 0.0:
                    residual = (self._pending_time - self.time) * r_old
                    self._pending_time = self.time + residual / r_now
                else:
                    self._pending_time = None
        self._r_at_draw = r_now

    @property
    def pending_time(self) -> Optional[float]:
        """Absolute time of the pending stochastic event (None if not drawn)."""
        return self._pending_time

    def pending_clock_state(self) -> Dict[str, Optional[float]]:
        """Serializable pending-clock state for checkpointing."""
        return {"pending_time": self._pending_time, "r_at_draw": self._r_at_draw}

    def set_pending_clock_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore pending-clock state from a checkpoint (no-op if absent)."""
        if not state:
            return
        pt = state.get("pending_time")
        rad = state.get("r_at_draw")
        self._pending_time = float(pt) if pt is not None else None
        self._r_at_draw = float(rad) if rad is not None else None

    def peek_next_event_time(self) -> Optional[float]:
        """Earliest upcoming event time across the deterministic queue and the pending clock.

        Draws the pending stochastic clock if not yet drawn; the draw is
        preserved (never discarded), so peeking does not change the seeded path.
        """
        if self._pending_time is None and self.R_total > 0.0:
            self._draw_pending_clock()
        candidates = []
        if self.pq:
            candidates.append(self.pq[0].time)
        if self._pending_time is not None:
            candidates.append(self._pending_time)
        return min(candidates) if candidates else None

    def set_total_memory(self, S: int) -> None:
        """Update total remembered token count S(t) and memory decay propensity R_forget."""
        self.total_memory = int(S)
        if self.reciprocal:
            self.R_forget = self._delta * (float(self.total_memory) / 2.0)
        else:
            self.R_forget = self._delta * float(self.total_memory)

    def register_callbacks(
        self,
        cb_private: Optional[Callable[[], None]] = None,
        cb_public: Optional[Callable[[], None]] = None,
        cb_encounter: Optional[Callable[[], None]] = None,
        cb_decay: Optional[Callable[[], None]] = None,
        cb_participation: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register handler callbacks for the 5 stochastic event channels."""
        if cb_private is not None:
            self.cb_private = cb_private
            self._invoker_private = _make_channel_invoker(cb_private)
        if cb_public is not None:
            self.cb_public = cb_public
            self._invoker_public = _make_channel_invoker(cb_public)
        if cb_encounter is not None:
            self.cb_encounter = cb_encounter
            self._invoker_encounter = _make_channel_invoker(cb_encounter)
        if cb_decay is not None:
            self.cb_decay = cb_decay
        if cb_participation is not None:
            self.cb_participation = cb_participation

    def schedule_deterministic_event(
        self,
        time: float,
        priority: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[PoissonScheduler, DeterministicEvent], None]] = None,
    ) -> int:
        """Schedule a deterministic event in the min-heap priority queue.

        Args:
            time: Floating-point timestamp when the event must execute.
            priority: Integer tie-breaking priority (lower values fire first).
            event_type: Descriptive name of event (e.g. 'holding_deadline', 'pulse_end').
            payload: Optional metadata dictionary.
            callback: Optional callable executed when event is dispatched.

        Returns:
            Monotonic sequence ID assigned to this event.
        """
        self.seq_counter += 1
        event = DeterministicEvent(
            time=float(time),
            priority=int(priority),
            seq_id=self.seq_counter,
            event_type=event_type,
            payload=payload if payload is not None else {},
            callback=callback,
        )
        heapq.heappush(self.pq, event)
        return self.seq_counter

    def add_deterministic_event(
        self,
        time: Optional[float] = None,
        priority: int = 10,
        event_type: str = "deterministic",
        payload: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[..., None]] = None,
        t_event: Optional[float] = None,
    ) -> int:
        """Alias for schedule_deterministic_event supporting time or t_event."""
        t = time if time is not None else t_event
        if t is None:
            raise ValueError("Must provide either time or t_event")
        return self.schedule_deterministic_event(t, priority, event_type, payload, callback)


    def peek_next_deterministic_time(self) -> Optional[float]:
        """Return the timestamp of the next deterministic event in queue, or None."""
        return self.pq[0].time if self.pq else None

    def step(self) -> bool:
        """Advance simulation by exactly one event (deterministic or stochastic jump).

        The pending stochastic clock is drawn once and preserved across
        deterministic-event boundaries; its residual integrated hazard is
        rescaled when channel rates change between the draw and the event, so
        scheduled boundaries never alter the seeded path. Deterministic events
        due strictly before the pending stochastic event preempt; a stochastic
        event exactly at a deterministic event's timestamp fires first, so a
        violating event at a holding deadline is processed before the deadline
        certification (conservative holding convention).

        Returns:
            True if an event was processed.
            False if simulation cannot advance (no pending clock and no deterministic events).
        """
        if self._pending_time is None and self.R_total > 0.0:
            self._draw_pending_clock()
        self._sync_pending_clock()

        # Deterministic event due strictly before the pending stochastic event.
        if self.pq and (self._pending_time is None or self.pq[0].time < self._pending_time):
            event = heapq.heappop(self.pq)
            self.time = event.time
            self.stats.deterministic_events += 1
            if self._pending_time is not None:
                self.stats.preemptions += 1
            if event.callback is not None:
                _invoke_callback(event.callback, self, event)
            return True

        if self._pending_time is None:
            # No pending stochastic clock and no deterministic events queued
            return False

        # Fire the pending stochastic event at its preserved timestamp.
        self.time = self._pending_time
        self._pending_time = None
        self._r_at_draw = None

        # Sample channel proportional to [R_x, R_y, R_enc, R_forget, R_n]
        r_tot = self.R_total
        draw = self.rng.random() * r_tot

        # Channel 1: Private state attempt
        cum = self.R_x
        if draw < cum:
            self.stats.private_updates += 1
            if self.cb_private is not None:
                node = self.sample_private_node()
                if self._invoker_private is not None:
                    self._invoker_private(node)
                else:
                    _invoke_channel_cb(self.cb_private, node)
            return True

        # Channel 2: Public state attempt
        cum += self.R_y
        if draw < cum:
            self.stats.public_updates += 1
            if self.cb_public is not None:
                node = self.sample_public_node()
                if self._invoker_public is not None:
                    self._invoker_public(node)
                else:
                    _invoke_channel_cb(self.cb_public, node)
            return True

        # Channel 3: Encounter attempt
        cum += self.R_enc
        if draw < cum:
            self.stats.encounters += 1
            if self.cb_encounter is not None:
                node = self.sample_encounter_initiator()
                if self._invoker_encounter is not None:
                    self._invoker_encounter(node)
                else:
                    _invoke_channel_cb(self.cb_encounter, node)
            return True

        # Channel 4: Memory unit decay
        cum += self.R_forget
        if draw < cum:
            self.stats.memory_decays += 1
            if self.cb_decay is not None:
                self.cb_decay()
            return True
        elif self.R_n > 0.0:
            # Channel 5: Participation redraw
            self.stats.participation_updates += 1
            if self.cb_participation is not None:
                self.cb_participation()
            return True
        else:
            # Floating-point boundary fallthrough when R_n == 0.0:
            # Attribute boundary draw to last active channel with positive propensity.
            if self.R_forget > 0.0:
                self.stats.memory_decays += 1
                if self.cb_decay is not None:
                    self.cb_decay()
            elif self.R_enc > 0.0:
                self.stats.encounters += 1
                if self.cb_encounter is not None:
                    node = self.sample_encounter_initiator()
                    if self._invoker_encounter is not None:
                        self._invoker_encounter(node)
                    else:
                        _invoke_channel_cb(self.cb_encounter, node)
            elif self.R_y > 0.0:
                self.stats.public_updates += 1
                if self.cb_public is not None:
                    node = self.sample_public_node()
                    if self._invoker_public is not None:
                        self._invoker_public(node)
                    else:
                        _invoke_channel_cb(self.cb_public, node)
            else:
                self.stats.private_updates += 1
                if self.cb_private is not None:
                    node = self.sample_private_node()
                    if self._invoker_private is not None:
                        self._invoker_private(node)
                    else:
                        _invoke_channel_cb(self.cb_private, node)
            return True

    def run_until(self, horizon: float) -> None:
        """Run continuous-time simulation up to specified horizon time.

        Args:
            horizon: Simulation end time H.
        """
        self.schedule_deterministic_event(
            time=horizon,
            priority=PRIORITY_HORIZON,
            event_type="horizon",
        )
        while self.time < horizon:
            if not self.step():
                break

    def advance_step(
        self,
        current_time: float,
        stochastic_dt: float,
    ) -> Tuple[float, bool]:
        """Advance time toward current_time + stochastic_dt, preempting if deterministic event occurs.

        Returns:
            Tuple of (effective_time, event_fired_boolean).
        """
        target_time = current_time + stochastic_dt
        if self.pq and self.pq[0].time <= target_time:
            first_time = self.pq[0].time
            self.time = first_time
            while self.pq and self.pq[0].time <= first_time + 1e-9:
                ev = heapq.heappop(self.pq)
                self.stats.deterministic_events += 1
                self.stats.preemptions += 1
                if ev.callback is not None:
                    _invoke_callback(ev.callback, self, ev)
            return (first_time, True)
        self.time = target_time
        return (target_time, False)


class MasterScheduler:
    """Master scheduler coordinating deterministic event queues and stochastic steps."""

    def __init__(
        self,
        N: int = 10,
        D: int = 1,
        tau_x: Union[float, np.ndarray] = 1.0,
        tau_y: float = 1.0,
        nu: Union[float, np.ndarray] = 1.0,
        delta: float = 0.1,
        tau_n: Optional[float] = None,
        reciprocal: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        if tau_n is not None and tau_n <= 0.0:
            raise ValueError(f"Participation timescale tau_n must be strictly positive, got {tau_n}")
        self.scheduler = PoissonScheduler(
            N=N,
            D=D,
            tau_x=tau_x,
            tau_y=tau_y,
            nu=nu,
            delta=delta,
            tau_n=tau_n,
            reciprocal=reciprocal,
            rng=rng,
        )
        self.events: List[Tuple[float, int, int, Callable[..., None], str]] = []
        self._seq: int = 0
        self.current_time: float = 0.0

    @property
    def nu(self) -> Union[float, np.ndarray]:
        return self.scheduler.nu

    @nu.setter
    def nu(self, val: Union[float, np.ndarray]) -> None:
        self.scheduler.nu = val

    @property
    def delta(self) -> float:
        return self.scheduler.delta

    @delta.setter
    def delta(self, val: float) -> None:
        self.scheduler.delta = val

    @property
    def time(self) -> float:
        return self.current_time

    @time.setter
    def time(self, val: float) -> None:
        self.current_time = float(val)
        self.scheduler.time = float(val)

    @property
    def total_memory(self) -> int:
        return self.scheduler.total_memory

    @total_memory.setter
    def total_memory(self, val: int) -> None:
        self.scheduler.set_total_memory(val)

    @property
    def R_total(self) -> float:
        return self.scheduler.R_total

    @property
    def R_x(self) -> float:
        return self.scheduler.R_x

    @property
    def R_y(self) -> float:
        return self.scheduler.R_y

    @property
    def R_enc(self) -> float:
        return self.scheduler.R_enc

    @property
    def R_forget(self) -> float:
        return self.scheduler.R_forget

    @property
    def R_n(self) -> float:
        return self.scheduler.R_n

    @property
    def stats(self) -> SchedulerStats:
        return self.scheduler.stats

    @property
    def N(self) -> int:
        return self.scheduler.N

    @property
    def D(self) -> int:
        return self.scheduler.D

    @property
    def tau_x(self) -> Union[float, np.ndarray]:
        return self.scheduler.tau_x

    @property
    def tau_y(self) -> float:
        return self.scheduler.tau_y

    @property
    def tau_n(self) -> Optional[float]:
        return self.scheduler.tau_n

    @property
    def reciprocal(self) -> bool:
        return self.scheduler.reciprocal

    def set_total_memory(self, S: int) -> None:
        """Forward total memory update to underlying scheduler."""
        self.scheduler.set_total_memory(S)

    def update_mechanism_rates(
        self,
        nu: Optional[Union[float, np.ndarray]] = None,
        delta: Optional[float] = None,
    ) -> None:
        """Forward mechanism rate updates to underlying scheduler."""
        self.scheduler.update_mechanism_rates(nu=nu, delta=delta)

    def peek_next_deterministic_time(self) -> Optional[float]:
        """Return the timestamp of the next deterministic event in queue, or None."""
        return self.events[0][0] if self.events else None

    def add_deterministic_event(
        self,
        t_event: Optional[float] = None,
        callback: Optional[Callable[..., None]] = None,
        time: Optional[float] = None,
        priority: int = 10,
        event_type: str = "deterministic",
    ) -> int:
        t_val = time if time is not None else t_event
        if t_val is None:
            raise ValueError("Must provide either time or t_event")
        t = float(t_val)
        self._seq += 1
        heapq.heappush(self.events, (t, int(priority), self._seq, callback, event_type))
        return self._seq

    def schedule_deterministic_event(
        self,
        time: float,
        priority: int = 10,
        event_type: str = "deterministic",
        payload: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[..., None]] = None,
    ) -> int:
        """Schedule a deterministic event, matching PoissonScheduler signature."""
        return self.add_deterministic_event(
            time=time,
            priority=priority,
            event_type=event_type,
            callback=callback,
        )

    def advance_step(
        self,
        current_time: float,
        stochastic_dt: float,
    ) -> Tuple[float, bool]:
        """Advance time toward current_time + stochastic_dt, preempting if deterministic event occurs.

        Returns:
            Tuple of (effective_time, event_fired_boolean).
        """
        target_time = current_time + stochastic_dt
        if self.events and self.events[0][0] <= target_time:
            first_time = self.events[0][0]
            self.time = first_time
            while self.events and self.events[0][0] <= first_time + 1e-9:
                t_ev, prio, seq, cb, ev_type = heapq.heappop(self.events)
                if cb is not None:
                    _invoke_callback(cb, self, t_ev)
            return (first_time, True)
        self.time = target_time
        return (target_time, False)



def calculate_total_event_rate(
    N: int,
    D: int = 1,
    tau_x: Union[float, np.ndarray] = 1.0,
    tau_y: float = 1.0,
    nu: Union[float, np.ndarray] = 1.0,
    delta: float = 0.1,
    total_memory: int = 0,
    tau_n: Optional[float] = None,
    reciprocal: bool = False,
) -> Dict[str, float]:
    """Calculate channel-by-channel and total Poisson event rates.

    Supports both scalar parameters and quenched heterogeneous arrays.

    Args:
        N: Population size.
        D: Belief dimensionality.
        tau_x: Private belief relaxation timescale (scalar or array of shape (N,)).
        tau_y: Public stance relaxation timescale.
        nu: Per-agent encounter rate (scalar or array of shape (N,)).
        delta: Memory decay rate.
        total_memory: Total memory count S.
        tau_n: Participation update timescale.
        reciprocal: Whether reciprocal pair memory mode is active.

    Returns:
        Dictionary with rate breakdown and total rate.
    """
    if N < 1:
        raise ValueError(f"Population size N must be >= 1, got {N}")
    if D < 1:
        raise ValueError(f"Belief dimensionality D must be >= 1, got {D}")
    if tau_y <= 0.0:
        raise ValueError("Timescales tau_x and tau_y must be strictly positive")
    if delta < 0.0 or total_memory < 0:
        raise ValueError("Rates nu, delta, and total_memory must be non-negative")
    if tau_n is not None and tau_n <= 0.0:
        raise ValueError(f"Participation timescale tau_n must be strictly positive, got {tau_n}")

    if isinstance(tau_x, (int, float, np.floating, np.integer)):
        if tau_x <= 0.0:
            raise ValueError("Timescales tau_x and tau_y must be strictly positive")
        r_x = float(N * D) / float(tau_x)
    else:
        arr_x = np.asarray(tau_x, dtype=np.float64)
        if arr_x.shape != (N,):
            raise ValueError(f"tau_x shape mismatch: expected ({N},), got {arr_x.shape}")
        if np.any(arr_x <= 0.0):
            raise ValueError("Timescales tau_x and tau_y must be strictly positive")
        r_x = float(D) * float(np.sum(1.0 / arr_x))

    r_y = float(N * D) / float(tau_y)

    if isinstance(nu, (int, float, np.floating, np.integer)):
        if nu < 0.0:
            raise ValueError("Rates nu, delta, and total_memory must be non-negative")
        r_enc = (float(N) * float(nu) / 2.0) if reciprocal else (float(N) * float(nu))
    else:
        arr_nu = np.asarray(nu, dtype=np.float64)
        if arr_nu.shape != (N,):
            raise ValueError(f"nu shape mismatch: expected ({N},), got {arr_nu.shape}")
        if np.any(arr_nu < 0.0):
            raise ValueError("Rates nu, delta, and total_memory must be non-negative")
        sum_nu = float(np.sum(arr_nu))
        r_enc = (sum_nu / 2.0) if reciprocal else sum_nu

    r_forget = (float(delta) * float(total_memory) / 2.0) if reciprocal else (float(delta) * float(total_memory))
    r_n = (float(N) / float(tau_n)) if tau_n is not None and tau_n > 0 else 0.0
    total = r_x + r_y + r_enc + r_forget + r_n
    return {
        "private": r_x,
        "public": r_y,
        "encounter": r_enc,
        "encounters": r_enc,
        "forget": r_forget,
        "decay": r_forget,
        "participation": r_n,
        "total": total,
    }


def sample_waiting_time(
    total_rate: float,
    u: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample stochastic waiting time delta_t ~ Exp(total_rate)."""
    if total_rate <= 0.0:
        return float("inf")
    if u is not None:
        draw = float(u)
    elif rng is not None:
        draw = float(rng.random())
    else:
        draw = float(np.random.default_rng().random())
    if draw <= 0.0:
        draw = np.nextafter(0.0, 1.0)
    return -np.log(draw) / float(total_rate)


def calculate_channel_probabilities(propensities: Dict[str, float]) -> Dict[str, float]:
    """Calculate normalized channel selection probabilities."""
    total = sum(propensities.values())
    if total <= 0.0:
        return {k: 0.0 for k in propensities}
    return {k: v / total for k, v in propensities.items()}

