"""Target definitions, exact holding clocks, and persistence evaluation.

Implements:
1. Reversal target:
   - Eligibility: |m_y(0)| >= 0.5 and sign(m_y(0)) * m_x(0) >= 0.5.
   - Target sign: s_* = -sign(m_y(0)).
   - Direction knowledge contract (F3): s_* is the immutable evaluation
     target fixed from the TRUE initial public majority. A controller may
     infer the direction from restricted public reports; any mismatch with
     s_* is the controller's inference mistake and is reported/counted,
     never absorbed by changing this evaluation target.
   - Condition: s_* * m_x(t) >= 0.5 and s_* * m_y(t) >= 0.5 simultaneously.
   - Terminal target violation: d_G = max(0, 0.5 - s_* * m_x, 0.5 - s_* * m_y).
2. Decentralization target:
   - Eligibility: initial C_N >= 0.002, active fraction M/N >= 0.8, largest weak component fraction L_w/N >= 0.8.
   - Condition: C_N(t) <= 0.5 * C_N(0), M/N >= 0.8, L_w/N >= 0.8.
   - Terminal target violation: d_G = max(0, C_N/(0.5*C_N(0)) - 1, 1 - (M/N)/0.8, 1 - (L_w/N)/0.8).
3. Exact Holding & Persistence Logic:
   - Holding timer starts ONLY at or after release when target condition holds.
   - Violating event resets holding timer to None.
   - Re-entry after release starts a new timer.
   - Conservative tie convention: a violating event exactly at the deadline prevents completion.
   - Completion is certified at the first deadline reached with target continuously true.
   - Continues monitoring to horizon H to record subsequent relapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.analytics.communities import connected_components_analysis
from src.analytics.influence import calculate_influence_metrics
from src.control.contracts import RolloutResult, TargetType
from src.core.state import State


def compute_top_10_percent_set(W: np.ndarray) -> Set[int]:
    """Compute the top-10% received attention leader set with all ties at the cutoff."""
    N = W.shape[0]
    if N == 0:
        return set()
    row_sums = W.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(row_sums > 0, W / row_sums, 0.0)
    r = A.sum(axis=0)
    n_10 = max(1, int(round(0.10 * N)))
    sorted_r = np.sort(r)[::-1]
    cutoff = sorted_r[min(n_10 - 1, len(sorted_r) - 1)]
    return {int(j) for j in range(N) if r[j] >= cutoff - 1e-12}


@dataclass
class TargetDefinition:
    """Configuration and baseline metadata for a transformation endpoint."""
    target_type: TargetType
    holding_duration: float = 50.0  # Delta
    horizon: float = 270.0          # H
    release_deadline: float = 20.0  # T_c
    s_star: Optional[float] = None  # Reversal target sign
    threshold: float = 0.5          # Signed mean; binary support=(1+threshold)/2
    component: int = 0              # Opinion component
    focal_indices: Optional[Tuple[int, ...]] = None  # None means the full population.
    C_N_0: Optional[float] = None   # Initial normalized concentration
    L_0: Optional[Set[int]] = None  # Initial top-10% influence set for turnover
    initial_metrics: Dict[str, Any] = field(default_factory=dict)
    is_eligible: bool = True
    ineligibility_reason: Optional[str] = None
    is_already_target: bool = False


class TargetEvaluator:
    """Evaluates physical states against target predicates and tracks holding timers."""

    def __init__(
        self,
        target_definition: TargetDefinition,
        problem_id: str = "problem_default",
        policy_id: str = "policy_default",
        origin_id: str = "origin_default",
        future_id: int = 0,
    ) -> None:
        self.defn = target_definition
        self.problem_id = problem_id
        self.policy_id = policy_id
        self.origin_id = origin_id
        self.future_id = future_id

        self.t_start_hold: Optional[float] = None
        self.completion_time: Optional[float] = None
        self.is_success: bool = False
        self.relapsed: bool = False

        self.last_condition_value: bool = False
        self.is_released: bool = False
        self.history_events: List[Tuple[float, bool]] = []
        self.terminal_violation: float = 0.0

    @classmethod
    def create_for_state(
        cls,
        target_type: TargetType,
        initial_state: State,
        holding_duration: float = 50.0,
        horizon: float = 270.0,
        release_deadline: float = 20.0,
        problem_id: str = "problem_0",
        policy_id: str = "policy_0",
        origin_id: str = "origin_0",
        future_id: int = 0,
        target_params: Optional[Dict[str, Any]] = None,
    ) -> TargetEvaluator:
        """Inspect initial state and construct TargetDefinition and Evaluator.

        With eligibility_mode="declared", use the explicit task parameter
        declared_destination_sign equally available to every policy. This new
        versioned contract does not inspect the initial majority for direction.

        Legacy direction knowledge contract (audit finding F3, option (b)):
        for a REVERSAL target this method fixes the immutable evaluation
        target s_* from the TRUE initial public majority of the physical
        state (s_* = -sign(m_y(0)); for the focal variant, the focal mean).
        This is the ground truth of the task objective: the controller-side
        target sign in ``ObservableSelectorPolicy`` is instead a RESTRICTED
        INFERENCE from delayed/sampled/noisy public reports and may differ
        from s_*. Such a mismatch is an inference mistake of the restricted
        policy: it is exported (controller 'effective_settings' inferred sign
        vs this definition's s_star) and must be COUNTED as an error, never
        absorbed by re-deriving s_* from the controller's belief, never
        corrected by feeding the true target or private states to the policy,
        and never used to evaluate against a different, easier destination.
        The information budget (panel fraction, noise, delivery delay) is not
        changed by this contract.
        """
        N, D = initial_state.N, initial_state.D
        init_metrics: Dict[str, Any] = {}
        is_eligible = True
        reason = None
        s_star = None
        C_N_0 = None
        L_0 = None
        threshold = 0.5
        if target_params and "threshold" in target_params:
            threshold = target_params["threshold"]
        focal_indices = None
        if target_params and target_params.get("focal_indices") is not None:
            focal_indices = tuple(int(i) for i in target_params["focal_indices"])
            if (target_type != TargetType.REVERSAL or not focal_indices
                    or len(set(focal_indices)) != len(focal_indices)
                    or any(i < 0 or i >= N for i in focal_indices)):
                raise ValueError("focal_indices must be unique valid nodes for a reversal target")
        component = 0
        if target_params and "component" in target_params:
            component = int(target_params["component"])
        focal_x = initial_state.x[list(focal_indices), component] if focal_indices is not None else initial_state.x[:, component]
        focal_y = initial_state.y[list(focal_indices), component] if focal_indices is not None else initial_state.y[:, component]

        if target_type == TargetType.REVERSAL:
            # Component scalar checks
            m_y_0 = float(np.mean(focal_y))
            m_x_0 = float(np.mean(focal_x))
            init_metrics["m_y_0"] = m_y_0
            init_metrics["m_x_0"] = m_x_0

            eligibility_mode = target_params.get("eligibility_mode", "legacy_strong") if target_params else "legacy_strong"
            if eligibility_mode == "declared":
                # v1 task parameter: independent of all hidden initial states.
                s_star = target_params.get("declared_destination_sign")
                if s_star not in (-1.0, 1.0):
                    raise ValueError("declared_destination_sign must be -1 or +1")
                s_star = float(s_star)
                init_metrics["direction_source"] = "declared_task"
            elif eligibility_mode == "legacy_strong":
                # Eligibility: |m_y(0)| >= threshold and sign(m_y(0)) * m_x(0) >= threshold
                if abs(m_y_0) < threshold - 1e-12:
                    is_eligible = False
                    reason = f"Initial |m_y(0)| = {abs(m_y_0):.4f} < {threshold}"
                else:
                    sgn = 1.0 if m_y_0 > 0 else -1.0
                    if sgn * m_x_0 < threshold - 1e-12:
                        is_eligible = False
                        reason = f"Initial sign(m_y(0))*m_x(0) = {sgn * m_x_0:.4f} < {threshold}"
                    else:
                        s_star = -sgn
            elif eligibility_mode == "public_majority":
                # Eligibility: |m_y(0)| > 0 (no tie)
                if abs(m_y_0) < 1e-12:
                    is_eligible = False
                    reason = f"Initial |m_y(0)| = {abs(m_y_0):.4f} is a tie"
                else:
                    s_star = -1.0 if m_y_0 > 0 else 1.0
            else:
                is_eligible = False
                reason = f"Unknown eligibility mode: {eligibility_mode}"

        elif target_type == TargetType.DECENTRALIZATION:
            inf = calculate_influence_metrics(initial_state.W)
            cc = connected_components_analysis(initial_state.W)

            C_N_0 = inf["C_N"]
            f_active_0 = inf["f_active"]
            lw_frac_0 = cc["largest_wcc_fraction"]

            init_metrics["C_N_0"] = C_N_0
            init_metrics["f_active_0"] = f_active_0
            init_metrics["largest_wcc_fraction_0"] = lw_frac_0

            # Eligibility: C_N >= 0.002, M/N >= 0.8, L_w/N >= 0.8
            if np.isnan(C_N_0) or C_N_0 < 0.002:
                is_eligible = False
                reason = f"Initial C_N(0) = {C_N_0} < 0.002"
            elif f_active_0 < 0.8:
                is_eligible = False
                reason = f"Initial active fraction M/N = {f_active_0:.4f} < 0.8"
            elif lw_frac_0 < 0.8:
                is_eligible = False
                reason = f"Initial largest weak component L_w/N = {lw_frac_0:.4f} < 0.8"

        elif target_type == TargetType.TURNOVER:
            inf = calculate_influence_metrics(initial_state.W)
            cc = connected_components_analysis(initial_state.W)

            C_N_0 = inf["C_N"]
            f_active_0 = inf["f_active"]
            lw_frac_0 = cc["largest_wcc_fraction"]
            L_0 = compute_top_10_percent_set(initial_state.W)

            init_metrics["C_N_0"] = C_N_0
            init_metrics["f_active_0"] = f_active_0
            init_metrics["largest_wcc_fraction_0"] = lw_frac_0
            init_metrics["L_0_size"] = len(L_0)

            # Eligibility: C_N >= 0.002, M/N >= 0.8, L_w/N >= 0.8, and |L_0| <= 0.20 * N
            if np.isnan(C_N_0) or C_N_0 < 0.002:
                is_eligible = False
                reason = f"Initial C_N(0) = {C_N_0} < 0.002"
            elif f_active_0 < 0.8:
                is_eligible = False
                reason = f"Initial active fraction M/N = {f_active_0:.4f} < 0.8"
            elif lw_frac_0 < 0.8:
                is_eligible = False
                reason = f"Initial largest weak component L_w/N = {lw_frac_0:.4f} < 0.8"
            elif len(L_0) > int(np.floor(0.20 * N)):
                is_eligible = False
                reason = f"Ambiguous rank cutoff: top-10% set size {len(L_0)} > 20% of N={N}"

        defn = TargetDefinition(
            target_type=target_type,
            holding_duration=holding_duration,
            horizon=horizon,
            release_deadline=release_deadline,
            s_star=s_star,
            threshold=threshold,
            component=component,
            focal_indices=focal_indices,
            C_N_0=C_N_0,
            L_0=L_0,
            initial_metrics=init_metrics,
            is_eligible=is_eligible,
            ineligibility_reason=reason,
        )
        instance = cls(
            target_definition=defn,
            problem_id=problem_id,
            policy_id=policy_id,
            origin_id=origin_id,
            future_id=future_id,
        )
        if is_eligible and instance.evaluate_target_condition(initial_state):
            defn.is_already_target = True
            defn.is_eligible = False
            defn.ineligibility_reason = "already_target"
        return instance

    def evaluate_target_condition(self, state: State) -> bool:
        """Evaluate target predicate on complete physical state."""
        if self.defn.target_type == TargetType.REVERSAL:
            s_star = self.defn.s_star
            if s_star is None:
                return False
            nodes = list(self.defn.focal_indices) if self.defn.focal_indices is not None else slice(None)
            m_x = float(np.mean(state.x[nodes, self.defn.component]))
            m_y = float(np.mean(state.y[nodes, self.defn.component]))
            # Both s_* * m_x >= threshold and s_* * m_y >= threshold simultaneously
            return bool((s_star * m_x >= self.defn.threshold - 1e-12) and (s_star * m_y >= self.defn.threshold - 1e-12))

        elif self.defn.target_type == TargetType.DECENTRALIZATION:
            C_N_0 = self.defn.C_N_0
            if C_N_0 is None or np.isnan(C_N_0):
                return False
            inf = calculate_influence_metrics(state.W)
            C_N = inf["C_N"]
            f_active = inf["f_active"]
            if np.isnan(C_N) or f_active < 0.8 - 1e-12:
                return False
            cc = connected_components_analysis(state.W)
            lw_frac = cc["largest_wcc_fraction"]
            if lw_frac < 0.8 - 1e-12:
                return False
            return bool(C_N <= 0.5 * C_N_0 + 1e-12)

        elif self.defn.target_type == TargetType.TURNOVER:
            if self.defn.L_0 is None or len(self.defn.L_0) == 0 or self.defn.C_N_0 is None:
                return False
            inf = calculate_influence_metrics(state.W)
            C_N = inf["C_N"]
            f_active = inf["f_active"]
            if np.isnan(C_N) or f_active < 0.8 - 1e-12:
                return False
            cc = connected_components_analysis(state.W)
            lw_frac = cc["largest_wcc_fraction"]
            if lw_frac < 0.8 - 1e-12:
                return False
            if C_N < 0.5 * self.defn.C_N_0 - 1e-12:
                return False
            L_t = compute_top_10_percent_set(state.W)
            retained = len(self.defn.L_0 & L_t) / len(self.defn.L_0)
            return bool(retained <= 0.5 + 1e-12)

        return False

    def compute_terminal_target_violation(self, state: State) -> float:
        """Compute terminal violation surrogate d_G for plateau tie-breaking."""
        if self.defn.target_type == TargetType.REVERSAL:
            s_star = self.defn.s_star if self.defn.s_star is not None else 1.0
            nodes = list(self.defn.focal_indices) if self.defn.focal_indices is not None else slice(None)
            m_x = float(np.mean(state.x[nodes, self.defn.component]))
            m_y = float(np.mean(state.y[nodes, self.defn.component]))
            d_G = max(0.0, self.defn.threshold - s_star * m_x, self.defn.threshold - s_star * m_y)
            return float(d_G)

        elif self.defn.target_type == TargetType.DECENTRALIZATION:
            C_N_0 = self.defn.C_N_0
            if C_N_0 is None or np.isnan(C_N_0) or C_N_0 == 0.0:
                return 10.0
            inf = calculate_influence_metrics(state.W)
            C_N = inf["C_N"]
            f_active = inf["f_active"]
            if np.isnan(C_N):
                return 10.0
            cc = connected_components_analysis(state.W)
            lw_frac = cc["largest_wcc_fraction"]
            d_G = max(
                0.0,
                C_N / (0.5 * C_N_0) - 1.0,
                1.0 - (f_active / 0.8),
                1.0 - (lw_frac / 0.8),
            )
            return float(d_G)

        elif self.defn.target_type == TargetType.TURNOVER:
            if self.defn.L_0 is None or len(self.defn.L_0) == 0 or self.defn.C_N_0 is None:
                return 10.0
            inf = calculate_influence_metrics(state.W)
            C_N = inf["C_N"]
            f_active = inf["f_active"]
            if np.isnan(C_N):
                return 10.0
            cc = connected_components_analysis(state.W)
            lw_frac = cc["largest_wcc_fraction"]
            L_t = compute_top_10_percent_set(state.W)
            retained = len(self.defn.L_0 & L_t) / len(self.defn.L_0)
            d_G = max(
                0.0,
                (retained / 0.5) - 1.0,
                1.0 - (C_N / (0.5 * self.defn.C_N_0)),
                1.0 - (f_active / 0.8),
                1.0 - (lw_frac / 0.8),
            )
            return float(d_G)

        return 0.0

    def process_state_update(
        self,
        state: State,
        current_time: float,
        is_released: bool,
    ) -> None:
        """Process physical state change or timestamp checkpoint."""
        self.is_released = bool(is_released)
        t = float(current_time)
        cond = self.evaluate_target_condition(state)
        self.last_condition_value = cond
        # Store exact predicate transitions, not millions of identical values.
        # Holding still processes EVERY event below, including tied violations.
        if not self.history_events or cond != self.history_events[-1][1]:
            self.history_events.append((t, cond))

        # Check conservative tie completion before processing violating event at this timestamp
        if is_released:
            if not cond:
                # If hold completed prior to this violating event (strictly after deadline)
                if (
                    not self.is_success
                    and self.t_start_hold is not None
                    and t > self.t_start_hold + self.defn.holding_duration + 1e-12
                ):
                    self.is_success = True
                    self.completion_time = self.t_start_hold + self.defn.holding_duration

                # Violating event at timestamp t resets holding timer
                self.t_start_hold = None
                if self.is_success:
                    self.relapsed = True
            else:
                # Target condition holds
                if self.t_start_hold is None:
                    self.t_start_hold = t

                # Check if hold completed
                if self.t_start_hold is not None and t >= self.t_start_hold + self.defn.holding_duration - 1e-12:
                    if not self.is_success:
                        self.is_success = True
                        self.completion_time = self.t_start_hold + self.defn.holding_duration
        else:
            # Before release: holding timer NEVER starts
            self.t_start_hold = None

    def process_holding_check(self, current_time: float, is_released: bool) -> None:
        """Check holding timer at a scheduled holding check/deadline timestamp."""
        self.is_released = bool(is_released)
        t = float(current_time)
        if not is_released or not self.last_condition_value:
            self.t_start_hold = None
            return

        if self.t_start_hold is not None:
            if t >= self.t_start_hold + self.defn.holding_duration - 1e-12:
                if not self.is_success:
                    self.is_success = True
                    self.completion_time = self.t_start_hold + self.defn.holding_duration

    def finalize(
        self,
        state: State,
        current_time: float,
        expenditure: float,
        distinct_nodes_used: int,
        event_counts: Optional[Dict[str, int]] = None,
        summaries: Optional[Dict[str, Any]] = None,
    ) -> RolloutResult:
        """Finalize evaluation at horizon H and build RolloutResult."""
        t = float(current_time)
        self.terminal_violation = self.compute_terminal_target_violation(state)

        # Check final holding status at horizon
        if self.is_released and self.last_condition_value and self.t_start_hold is not None:
            if t >= self.t_start_hold + self.defn.holding_duration - 1e-12 and not self.is_success:
                self.is_success = True
                self.completion_time = self.t_start_hold + self.defn.holding_duration

        if self.is_success and self.completion_time is not None:
            restricted_time = min(self.completion_time, self.defn.horizon)
        else:
            restricted_time = self.defn.horizon

        return RolloutResult(
            problem_id=self.problem_id,
            policy_id=self.policy_id,
            origin_id=self.origin_id,
            future_id=self.future_id,
            is_valid=True,
            is_eligible=self.defn.is_eligible,
            success=self.is_success,
            completion_time=self.completion_time,
            restricted_time=restricted_time,
            expenditure=float(expenditure),
            distinct_nodes_used=int(distinct_nodes_used),
            holding_start_time=self.t_start_hold,
            relapsed=self.relapsed,
            terminal_target_violation=self.terminal_violation,
            event_counts=event_counts or {},
            summaries={**(summaries or {}), "horizon": self.defn.horizon,
                       "certified_once": bool(self.is_success),
                       "sustained_success": bool(self.is_success and not self.relapsed)},
        )


def evaluate_holding_trace(
    trace: List[Tuple[float, bool]],
    release_time: float,
    holding_duration: float,
    horizon: float,
) -> Optional[float]:
    """Reference trace evaluator implementing the exact conservative event-tie holding convention.

    Directly matches hold_completion in validate_control_plan.py.
    """
    events = dict(trace)
    times = sorted(set(events.keys()) | {release_time, horizon})
    truth = False
    start: Optional[float] = None

    for t in times:
        if t > horizon:
            break
        if start is not None and start + holding_duration < t:
            return start + holding_duration
        truth = events.get(t, truth)
        if t >= release_time:
            if not truth:
                start = None
            elif start is None:
                start = t
            if start is not None and t >= start + holding_duration:
                return start + holding_duration
    return None
