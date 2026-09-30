"""Non-invasive simulation recorder, action history, and original-state retention.

Provides:
1. RecorderConfig: Declarative configuration for diagnostic recording.
2. ScalarTrajectoryPoint: Diagnostic scalar observation at a physical timestamp.
3. ActionRecord: Realized intervention action history entry.
4. NetworkFrame: Bounded snapshot of network topology, node states, and intervention mask.
5. RetentionResult & evaluate_original_state_retention: Formal original-state retention metric
   evaluating whether the established state remains in its initial orientation basin throughout
   a declared window, distinct from the reversal target.
6. RolloutRecorder: Non-invasive recorder integrating into run_controlled_simulation.
   Crucial invariant: NEVER draws from simulator, controller, or observation RNGs.
   Preserves engine RNG streams and event ordering bit-for-bit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.contracts import Action, ControlProblem, TargetType
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.core.state import State


@dataclass(frozen=True)
class RecorderConfig:
    """Configuration for diagnostic rollout recording."""
    cadence: float = 1.0  # Diagnostic sampling cadence (tau units)
    record_scalar_trajectories: bool = True
    record_action_history: bool = True
    record_network_frames: bool = False
    max_network_frames: int = 100
    retention_threshold: float = 0.5
    retention_window: Optional[Tuple[float, float]] = None


@dataclass
class ScalarTrajectoryPoint:
    """Scalar observation at a single physical timestamp."""
    t: float
    m_x: float  # Component 0 private mean
    m_y: float  # Component 0 public mean
    m_x_all: List[float]  # All component private means
    m_y_all: List[float]  # All component public means
    target_direction: float
    distance_to_target: float
    target_predicate: bool
    hold_progress: float
    actual_release: bool
    exposure: float
    spending_rate: float
    distinct_nodes_used: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionRecord:
    """Realized intervention schedule entry."""
    decision_time: float
    report_sample_time: Optional[float]
    actuated_nodes: List[int]
    field_entries: List[Dict[str, Any]]
    requested_end: Optional[float]
    effective_end: Optional[float]
    budget_clamped: bool
    shutoff_or_release: str  # "actuation", "field_shutoff", "release_deadline", "explicit_release", "budget_exhaustion", "zero_action"
    reason: str
    spending_rate: float
    cumulative_exposure: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkFrame:
    """Bounded network snapshot for topology and spatial belief visualization."""
    t: float
    x: np.ndarray  # (N, D) copied
    y: np.ndarray  # (N, D) copied
    W: np.ndarray  # (N, N) copied
    controlled_mask: np.ndarray  # (N,) bool
    influence_inputs: np.ndarray  # (N, D) copied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": float(self.t),
            "x": self.x.copy(),
            "y": self.y.copy(),
            "W": self.W.copy(),
            "controlled_mask": self.controlled_mask.copy(),
            "influence_inputs": self.influence_inputs.copy(),
        }


@dataclass
class RetentionResult:
    """Outcome of original-state retention evaluation."""
    is_retained: bool
    initial_sign: float
    threshold: float
    window_start: float
    window_end: float
    min_m_y_alignment: float
    min_m_x_alignment: float
    min_alignment: float
    max_excursion: float
    loss_time: Optional[float]
    loss_duration: float
    evaluation_mode: str  # "sampled" or "event_exact"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_original_state_retention(
    trajectory: Sequence[Any],
    initial_state: Optional[State] = None,
    threshold: float = 0.5,
    window: Optional[Tuple[float, float]] = None,
    mode: str = "sampled",
    initial_m_y: Optional[float] = None,
    initial_m_x: Optional[float] = None,
) -> RetentionResult:
    """Evaluate whether established state retains its initial orientation throughout a window.

    Original-state retention is defined separately from reversal:
    1. Initial sign s_0 is fixed from origin initial public mean sign(m_y(0)).
    2. Retention condition: s_0 * m_y(t) >= threshold AND s_0 * m_x(t) >= threshold simultaneously.
    3. Evaluated across declared window [t_start, t_end]. If window is None, spans [0, H].
    4. Tracks minima, excursions, first loss time, and cumulative loss duration.
    5. A trajectory that loses alignment without reversing (e.g. relaxing to 0.2 < threshold)
       is truthfully certified as nonpersistent (is_retained=False).
    """
    # Determine initial orientation sign
    if initial_state is not None:
        m_y0 = float(np.mean(initial_state.y[:, 0]))
        m_x0 = float(np.mean(initial_state.x[:, 0]))
    elif initial_m_y is not None:
        m_y0 = float(initial_m_y)
        m_x0 = float(initial_m_x) if initial_m_x is not None else m_y0
    elif len(trajectory) > 0:
        first = trajectory[0]
        m_y0 = float(first["m_y"] if isinstance(first, dict) else getattr(first, "m_y"))
        m_x0 = float(first["m_x"] if isinstance(first, dict) else getattr(first, "m_x"))
    else:
        return RetentionResult(
            is_retained=False,
            initial_sign=1.0,
            threshold=float(threshold),
            window_start=0.0,
            window_end=0.0,
            min_m_y_alignment=float("nan"),
            min_m_x_alignment=float("nan"),
            min_alignment=float("nan"),
            max_excursion=float("nan"),
            loss_time=None,
            loss_duration=0.0,
            evaluation_mode=mode,
        )

    s_0 = 1.0 if m_y0 >= 0.0 else -1.0
    tol = 1e-12

    # Determine window
    if window is not None:
        t_start, t_end = float(window[0]), float(window[1])
    else:
        t_start = 0.0
        if len(trajectory) > 0:
            last = trajectory[-1]
            t_end = float(last["t"] if isinstance(last, dict) else getattr(last, "t"))
        else:
            t_end = 0.0

    min_my = float("inf")
    min_mx = float("inf")
    max_exc = 0.0
    is_retained = True
    loss_time: Optional[float] = None
    loss_duration = 0.0
    prev_t: Optional[float] = None
    prev_lost = False

    for pt in trajectory:
        t = float(pt["t"] if isinstance(pt, dict) else getattr(pt, "t"))
        if t < t_start - tol or t > t_end + tol:
            continue

        my = float(pt["m_y"] if isinstance(pt, dict) else getattr(pt, "m_y"))
        mx = float(pt["m_x"] if isinstance(pt, dict) else getattr(pt, "m_x"))

        align_y = s_0 * my
        align_x = s_0 * mx
        min_my = min(min_my, align_y)
        min_mx = min(min_mx, align_x)

        exc_y = max(0.0, threshold - align_y)
        exc_x = max(0.0, threshold - align_x)
        exc = max(exc_y, exc_x)
        max_exc = max(max_exc, exc)

        point_lost = bool((align_y < threshold - tol) or (align_x < threshold - tol))
        if point_lost:
            is_retained = False
            if loss_time is None:
                loss_time = t

        if prev_t is not None and prev_lost:
            dt = t - prev_t
            if dt > 0.0:
                loss_duration += dt

        prev_t = t
        prev_lost = point_lost

    if np.isinf(min_my):
        min_my = 0.0
    if np.isinf(min_mx):
        min_mx = 0.0

    min_align = min(min_my, min_mx)

    return RetentionResult(
        is_retained=is_retained,
        initial_sign=s_0,
        threshold=float(threshold),
        window_start=t_start,
        window_end=t_end,
        min_m_y_alignment=float(min_my),
        min_m_x_alignment=float(min_mx),
        min_alignment=float(min_align),
        max_excursion=float(max_exc),
        loss_time=loss_time,
        loss_duration=float(loss_duration),
        evaluation_mode=mode,
    )


class RolloutRecorder:
    """Non-invasive diagnostic recorder for a single controlled rollout.

    CRITICAL INVARIANT:
    This class never accesses or draws from any random number generator.
    Its operations are strictly passive observers that copy arrays and evaluate
    deterministic scalar metrics, preserving stochastic simulation ordering bit-for-bit.
    """

    def __init__(self, config: Optional[RecorderConfig] = None) -> None:
        self.config = config or RecorderConfig()
        self.scalar_points: List[ScalarTrajectoryPoint] = []
        self.action_history: List[ActionRecord] = []
        self.network_frames: List[NetworkFrame] = []

        # Event-exact retention tracking
        self._initial_sign: Optional[float] = None
        self._event_exact_min_mx: float = float("inf")
        self._event_exact_min_my: float = float("inf")
        self._event_exact_max_exc: float = 0.0
        self._event_exact_retained: bool = True
        self._event_exact_loss_time: Optional[float] = None

        # Sampling cadence progress
        self._next_record_time: float = 0.0
        self._last_sampled_time: Optional[float] = None

    @staticmethod
    def _target_nodes(state: State, target_evaluator: TargetEvaluator):
        indices = target_evaluator.defn.focal_indices
        return list(indices) if indices is not None else slice(None)

    def initialize(
        self,
        state: State,
        ledger: BudgetLedger,
        target_evaluator: TargetEvaluator,
        problem: ControlProblem,
    ) -> None:
        """Capture initial t=0.0 state and initialize retention tracking."""
        nodes = self._target_nodes(state, target_evaluator)
        m_y0 = float(np.mean(state.y[nodes, 0]))
        m_x0 = float(np.mean(state.x[nodes, 0]))
        self._initial_sign = 1.0 if m_y0 >= 0.0 else -1.0

        # Initial event-exact tracking
        align_y = self._initial_sign * m_y0
        align_x = self._initial_sign * m_x0
        self._event_exact_min_my = align_y
        self._event_exact_min_mx = align_x
        thresh = self.config.retention_threshold
        exc = max(0.0, thresh - align_y, thresh - align_x)
        self._event_exact_max_exc = exc
        if (align_y < thresh - 1e-12) or (align_x < thresh - 1e-12):
            self._event_exact_retained = False
            self._event_exact_loss_time = 0.0

        # Record t=0 point
        self._record_scalar_point(state, ledger, target_evaluator, problem, t=0.0)
        if self.config.record_network_frames and self.config.max_network_frames > 0:
            self._record_network_frame(state, ledger, t=0.0)

        self._next_record_time = self.config.cadence
        self._last_sampled_time = 0.0

    def on_event_state(
        self,
        state: State,
        current_time: float,
        ledger: BudgetLedger,
        target_evaluator: TargetEvaluator,
    ) -> None:
        """Passive observer for stochastic/deterministic events to track event-exact retention."""
        nodes = self._target_nodes(state, target_evaluator)
        m_y = float(np.mean(state.y[nodes, 0]))
        m_x = float(np.mean(state.x[nodes, 0]))
        s_0 = self._initial_sign if self._initial_sign is not None else 1.0

        align_y = s_0 * m_y
        align_x = s_0 * m_x
        if align_y < self._event_exact_min_my:
            self._event_exact_min_my = align_y
        if align_x < self._event_exact_min_mx:
            self._event_exact_min_mx = align_x

        thresh = self.config.retention_threshold
        exc = max(0.0, thresh - align_y, thresh - align_x)
        if exc > self._event_exact_max_exc:
            self._event_exact_max_exc = exc

        if (align_y < thresh - 1e-12) or (align_x < thresh - 1e-12):
            if self._event_exact_retained:
                self._event_exact_retained = False
                self._event_exact_loss_time = float(current_time)

    def _capture_network_sample(
        self, state: State, ledger: BudgetLedger, t: float, problem: ControlProblem,
    ) -> None:
        """Keep bounded frames across intervention and post-release evolution.

        Scalar diagnostics retain their full cadence. Network matrices are much
        larger, so their separate deterministic schedule reserves early frames
        and one slot for the actual horizon. The schedule depends only on the
        declared experiment, making matched arms share timestamps.
        """
        cap = self.config.max_network_frames
        if not self.config.record_network_frames or cap <= 1:
            return
        if len(self.network_frames) >= cap - 1:
            return
        horizon = float(problem.horizon)
        release = min(float(problem.release_deadline), horizon)
        if t >= horizon - 1e-12:
            return  # finalize always captures the true final state

        early_slots = min(max(2, cap // 3), cap - 1)
        early_interval = max(self.config.cadence, release / max(early_slots - 1, 1))
        late_slots = max(2, cap - early_slots)
        late_interval = max(self.config.cadence, (horizon - release) / max(late_slots - 1, 1))
        previous = self.network_frames[-1].t if self.network_frames else 0.0
        if t <= release + 1e-12:
            due = t - previous >= early_interval - 1e-12
        else:
            due = t - max(previous, release) >= late_interval - 1e-12
        if due:
            self._record_network_frame(state, ledger, t=t)

    def advance_cadence_sampling(
        self,
        t_bound: float,
        state: State,
        ledger: BudgetLedger,
        target_evaluator: TargetEvaluator,
        problem: ControlProblem,
    ) -> None:
        """Sample diagnostic points at regular cadence up to t_bound."""
        cadence = self.config.cadence
        if cadence <= 0.0:
            return

        tol = 1e-12
        while self._next_record_time <= t_bound + tol:
            t_sample = min(self._next_record_time, problem.horizon)
            if self._last_sampled_time is None or abs(t_sample - self._last_sampled_time) > tol:
                self._record_scalar_point(state, ledger, target_evaluator, problem, t=t_sample)
                self._capture_network_sample(state, ledger, t_sample, problem)
                self._last_sampled_time = t_sample
            self._next_record_time += cadence

    def record_action(
        self,
        action: Action,
        decision_time: float,
        report_sample_time: Optional[float],
        effective_end: Optional[float],
        ledger: BudgetLedger,
        reason: str = "controller_actuation",
    ) -> None:
        """Record an intervention action or actuation schedule change."""
        if not self.config.record_action_history:
            return

        nodes = sorted(list(action.actuated_nodes()))
        field_entries = [
            {"node": int(e.node), "component": int(e.component), "field": float(e.field)}
            for e in action.entries
        ]
        requested_end = action.intended_end_time
        clamped = False
        if requested_end is not None and effective_end is not None:
            if effective_end < requested_end - 1e-9:
                clamped = True

        event_type = "actuation"
        if action.release:
            event_type = "explicit_release"
        elif action.spending_rate <= 1e-15 and len(action.structural_entries) == 0:
            event_type = "zero_action"

        rec = ActionRecord(
            decision_time=float(decision_time),
            report_sample_time=float(report_sample_time) if report_sample_time is not None else None,
            actuated_nodes=nodes,
            field_entries=field_entries,
            requested_end=float(requested_end) if requested_end is not None else None,
            effective_end=float(effective_end) if effective_end is not None else None,
            budget_clamped=clamped,
            shutoff_or_release=event_type,
            reason=reason,
            spending_rate=float(action.spending_rate),
            cumulative_exposure=float(ledger.integrated_exposure),
        )
        self.action_history.append(rec)

    def record_shutoff(
        self,
        current_time: float,
        ledger: BudgetLedger,
        reason: str,
    ) -> None:
        """Record a field shutoff, release deadline, or budget exhaustion event."""
        if not self.config.record_action_history:
            return

        rec = ActionRecord(
            decision_time=float(current_time),
            report_sample_time=None,
            actuated_nodes=[],
            field_entries=[],
            requested_end=float(current_time),
            effective_end=float(current_time),
            budget_clamped=False,
            shutoff_or_release=reason,
            reason=reason,
            spending_rate=0.0,
            cumulative_exposure=float(ledger.integrated_exposure),
        )
        self.action_history.append(rec)

    def finalize(
        self,
        state: State,
        ledger: BudgetLedger,
        target_evaluator: TargetEvaluator,
        problem: ControlProblem,
    ) -> None:
        """Sample final horizon point and complete trajectory records."""
        # Ensure points up to horizon are sampled
        self.advance_cadence_sampling(problem.horizon, state, ledger, target_evaluator, problem)

        # Force record at exact horizon if not already recorded
        tol = 1e-12
        if not self.scalar_points or abs(self.scalar_points[-1].t - problem.horizon) > tol:
            self._record_scalar_point(state, ledger, target_evaluator, problem, t=problem.horizon)
            self._last_sampled_time = problem.horizon
        if self.config.record_network_frames and self.config.max_network_frames > 0:
            if not self.network_frames or abs(self.network_frames[-1].t - problem.horizon) > tol:
                if len(self.network_frames) >= self.config.max_network_frames:
                    self.network_frames.pop()
                self._record_network_frame(state, ledger, t=problem.horizon)

    def _record_scalar_point(
        self,
        state: State,
        ledger: BudgetLedger,
        target_evaluator: TargetEvaluator,
        problem: ControlProblem,
        t: float,
    ) -> None:
        """Create and store a single ScalarTrajectoryPoint with copied arrays."""
        # Private means
        nodes = self._target_nodes(state, target_evaluator)
        m_x_all = [float(np.mean(state.x[nodes, alpha])) for alpha in range(state.D)]
        m_y_all = [float(np.mean(state.y[nodes, alpha])) for alpha in range(state.D)]
        m_x0 = m_x_all[0] if m_x_all else 0.0
        m_y0 = m_y_all[0] if m_y_all else 0.0

        # Target evaluation
        target_dir = (
            float(target_evaluator.defn.s_star)
            if target_evaluator.defn.s_star is not None
            else 1.0
        )
        dist = target_evaluator.compute_terminal_target_violation(state)
        pred = target_evaluator.evaluate_target_condition(state)

        # Hold progress
        if target_evaluator.is_success:
            hold_prog = float(problem.holding_duration)
        elif target_evaluator.t_start_hold is not None:
            hold_prog = max(0.0, float(t - target_evaluator.t_start_hold))
        else:
            hold_prog = 0.0

        # Ledger state evaluated at t
        # If t == ledger.current_time, values are exact
        if abs(t - ledger.current_time) <= 1e-12:
            exp = float(ledger.integrated_exposure)
            rate = float(ledger.current_spending_rate)
            rel = bool(ledger.is_released)
            nodes_used = len(ledger.distinct_nodes_actuated)
        else:
            # Interpolate exposure along constant spending rate interval
            dt = t - ledger.current_time
            rate = float(ledger.current_spending_rate)
            exp = float(ledger.integrated_exposure + max(0.0, dt) * rate)
            rel = bool(ledger.is_released or (t >= ledger.release_deadline - 1e-12))
            nodes_used = len(ledger.distinct_nodes_actuated)

        pt = ScalarTrajectoryPoint(
            t=float(t),
            m_x=m_x0,
            m_y=m_y0,
            m_x_all=m_x_all,
            m_y_all=m_y_all,
            target_direction=target_dir,
            distance_to_target=float(dist),
            target_predicate=bool(pred),
            hold_progress=float(hold_prog),
            actual_release=rel,
            exposure=exp,
            spending_rate=rate,
            distinct_nodes_used=int(nodes_used),
        )
        self.scalar_points.append(pt)

    def _record_network_frame(
        self,
        state: State,
        ledger: BudgetLedger,
        t: float,
    ) -> None:
        """Capture deep-copied network frame."""
        ctrl_mask = np.zeros(state.N, dtype=bool)
        if ledger.current_field is not None:
            ctrl_mask = np.any(np.abs(ledger.current_field) > 1e-15, axis=1)

        inf_inputs = (
            ledger.current_field.copy()
            if ledger.current_field is not None
            else np.zeros((state.N, state.D), dtype=np.float64)
        )

        frame = NetworkFrame(
            t=float(t),
            x=state.x.copy(),
            y=state.y.copy(),
            W=state.W.copy(),
            controlled_mask=ctrl_mask,
            influence_inputs=inf_inputs,
        )
        self.network_frames.append(frame)

    def integrated_spending(self) -> float:
        """Calculate total integrated exposure spending from recorded trajectory points.

        Returns the terminal cumulative exposure recorded in the scalar trajectory,
        which exactly matches the ledger's analytical exposure expenditure.
        """
        if not self.scalar_points:
            return 0.0
        return float(self.scalar_points[-1].exposure - self.scalar_points[0].exposure)

    def integrate_spending_rate(self, method: str = "trapezoidal") -> float:
        """Calculate exposure via numerical integration of recorded spending rates.

        Supports 'trapezoidal', 'left' (Riemann sum), and 'right'.
        """
        if len(self.scalar_points) < 2:
            return 0.0

        pts = self.scalar_points
        total = 0.0
        for i in range(len(pts) - 1):
            dt = pts[i + 1].t - pts[i].t
            if dt > 1e-12:
                if method == "left":
                    total += pts[i].spending_rate * dt
                elif method == "right":
                    total += pts[i + 1].spending_rate * dt
                else:  # trapezoidal
                    total += 0.5 * (pts[i].spending_rate + pts[i + 1].spending_rate) * dt
        return float(total)


    def evaluate_retention(
        self,
        threshold: Optional[float] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> RetentionResult:
        """Evaluate original-state retention from recorded trajectory points."""
        thresh = threshold if threshold is not None else self.config.retention_threshold
        win = window if window is not None else self.config.retention_window
        return evaluate_original_state_retention(
            trajectory=self.scalar_points,
            threshold=thresh,
            window=win,
            mode="sampled",
            initial_m_y=self.scalar_points[0].m_y if self.scalar_points else None,
            initial_m_x=self.scalar_points[0].m_x if self.scalar_points else None,
        )

    def event_exact_retention(
        self,
        threshold: Optional[float] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> RetentionResult:
        """Return event-exact retention tracking evaluated at every event."""
        thresh = threshold if threshold is not None else self.config.retention_threshold
        t_start = window[0] if window is not None else 0.0
        t_end = window[1] if window is not None else (self.scalar_points[-1].t if self.scalar_points else 0.0)

        min_my = self._event_exact_min_my
        min_mx = self._event_exact_min_mx
        if np.isinf(min_my):
            min_my = 0.0
        if np.isinf(min_mx):
            min_mx = 0.0
        min_align = min(min_my, min_mx)

        return RetentionResult(
            is_retained=self._event_exact_retained,
            initial_sign=self._initial_sign if self._initial_sign is not None else 1.0,
            threshold=float(thresh),
            window_start=float(t_start),
            window_end=float(t_end),
            min_m_y_alignment=float(min_my),
            min_m_x_alignment=float(min_mx),
            min_alignment=float(min_align),
            max_excursion=float(self._event_exact_max_exc),
            loss_time=self._event_exact_loss_time,
            loss_duration=0.0,
            evaluation_mode="event_exact",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert all recorded rollout telemetry to a serializable dictionary."""
        return {
            "scalar_points": [p.to_dict() for p in self.scalar_points],
            "action_history": [a.to_dict() for a in self.action_history],
            "network_frames_count": len(self.network_frames),
            "integrated_spending": self.integrated_spending(),
            "retention_sampled": self.evaluate_retention().to_dict(),
            "retention_event_exact": self.event_exact_retention().to_dict(),
        }
