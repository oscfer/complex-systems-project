"""Strict budget and actuator ledger for optimal transformation control.

Enforces:
1. Exposure cap B: integrated sum of absolute fields over time.
2. Distinct-node cap k: cumulative count of distinct nodes actuated.
3. Amplitude cap u_max: maximum permitted instantaneous field magnitude.
4. Mandatory release deadline T_c: permanent shutoff of interventions.
5. Accessible roster membership: interventions strictly restricted to designated nodes.
6. Irreversible release semantics: declaring release permanently disables actuation.
7. Exact piecewise-constant budget exhaustion calculation without silent overspending.
8. Structural modification budget B_struct and per-step limits.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple
import numpy as np

from src.control.contracts import Action


class BudgetLedger:
    """Strict accounting ledger for intervention resources."""

    def __init__(
        self,
        N: int,
        D: int = 1,
        k: int = 1,
        exposure_cap: float = 10.0,
        amplitude_cap: float = 4.0,
        release_deadline: float = 20.0,
        accessible_roster: Optional[Set[int]] = None,
        relative_tolerance: float = 1e-10,
        structural_budget: Optional[float] = None,
        max_structural_per_step: Optional[int] = None,
    ) -> None:
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}")
        if k < 1 or k > N:
            raise ValueError(f"k must be in [1, N], got {k} (N={N})")
        if exposure_cap < 0.0:
            raise ValueError(f"exposure_cap cannot be negative, got {exposure_cap}")
        if amplitude_cap <= 0.0:
            raise ValueError(f"amplitude_cap must be > 0, got {amplitude_cap}")
        if release_deadline < 0.0:
            raise ValueError(f"release_deadline cannot be negative, got {release_deadline}")
        if structural_budget is not None and structural_budget < 0.0:
            raise ValueError(f"structural_budget cannot be negative, got {structural_budget}")
        if max_structural_per_step is not None and max_structural_per_step < 1:
            raise ValueError(f"max_structural_per_step must be >= 1, got {max_structural_per_step}")

        self.N = N
        self.D = D
        self.k = k
        self.exposure_cap = float(exposure_cap)
        self.amplitude_cap = float(amplitude_cap)
        self.release_deadline = float(release_deadline)
        self.accessible_roster = set(accessible_roster) if accessible_roster is not None else None
        self.tol = float(relative_tolerance) * max(1.0, self.exposure_cap)

        self.structural_budget = float(structural_budget) if structural_budget is not None else None
        self.max_structural_per_step = int(max_structural_per_step) if max_structural_per_step is not None else None
        self.integrated_structural_modifications: int = 0
        self.is_structural_exhausted: bool = False

        self.current_time: float = 0.0
        self.integrated_exposure: float = 0.0
        self.distinct_nodes_actuated: Set[int] = set()
        self.is_released: bool = False
        self.release_time: Optional[float] = None
        self.is_exhausted: bool = False

        self.current_field: np.ndarray = np.zeros((N, D), dtype=np.float64)
        self.active_action: Optional[Action] = None
        self.active_action_end_time: Optional[float] = None

        if self.exposure_cap <= self.tol:
            self.is_exhausted = True
            self._declare_release(0.0)

        if self.structural_budget is not None and self.structural_budget <= self.tol:
            self.is_structural_exhausted = True

    @property
    def remaining_exposure(self) -> float:
        """Remaining unspent exposure capacity."""
        return max(0.0, self.exposure_cap - self.integrated_exposure)

    @property
    def remaining_structural_budget(self) -> float:
        """Remaining unspent structural modification capacity."""
        if self.structural_budget is None:
            return float("inf")
        return max(0.0, self.structural_budget - float(self.integrated_structural_modifications))

    @property
    def remaining_distinct_nodes(self) -> int:
        """Remaining unallocated distinct node slots."""
        return max(0, self.k - len(self.distinct_nodes_actuated))

    @property
    def current_spending_rate(self) -> float:
        """Instantaneous exposure spending rate c_u = sum_{i, alpha} |u_{i, alpha}|."""
        return float(np.sum(np.abs(self.current_field)))

    def copy(self) -> BudgetLedger:
        """Create an exact, isolated copy of this BudgetLedger state."""
        new_ledger = BudgetLedger(
            N=self.N,
            D=self.D,
            k=self.k,
            exposure_cap=self.exposure_cap,
            amplitude_cap=self.amplitude_cap,
            release_deadline=self.release_deadline,
            accessible_roster=set(self.accessible_roster) if self.accessible_roster is not None else None,
            relative_tolerance=self.tol / max(1.0, self.exposure_cap),
            structural_budget=self.structural_budget,
            max_structural_per_step=self.max_structural_per_step,
        )
        new_ledger.current_time = self.current_time
        new_ledger.integrated_exposure = self.integrated_exposure
        new_ledger.integrated_structural_modifications = self.integrated_structural_modifications
        new_ledger.distinct_nodes_actuated = set(self.distinct_nodes_actuated)
        new_ledger.is_released = self.is_released
        new_ledger.release_time = self.release_time
        new_ledger.is_exhausted = self.is_exhausted
        new_ledger.is_structural_exhausted = self.is_structural_exhausted
        new_ledger.current_field = self.current_field.copy()
        new_ledger.active_action = self.active_action
        new_ledger.active_action_end_time = self.active_action_end_time
        return new_ledger

    def validate_action(self, action: Action, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """Validate proposed action against ledger constraints without applying it."""
        t = self.current_time if current_time is None else float(current_time)
        if t < self.current_time - 1e-12:
            return False, f"Cannot validate action backwards in time: current={self.current_time}, requested={t}"

        # 1. Release check
        if self.is_released:
            if (
                action.spending_rate > 1e-15
                or len(action.structural_entries) > 0
                or (not action.release and len(action.entries) > 0)
            ):
                return False, "Cannot actuate after control release has been declared (release is irreversible)"

        # 2. Deadline check
        if t >= self.release_deadline and (action.spending_rate > 1e-15 or len(action.structural_entries) > 0):
            return False, f"Cannot apply nonzero action at or after release deadline {self.release_deadline}"

        # 3. Budget exhaustion check
        if self.remaining_exposure <= self.tol and action.spending_rate > 1e-15:
            return False, "Exposure budget is fully exhausted"

        if action.structural_entries:
            if self.structural_budget is not None and action.structural_spending > self.remaining_structural_budget + self.tol:
                return False, (
                    f"Action requires {action.structural_spending} structural modifications, "
                    f"which exceeds remaining structural budget {self.remaining_structural_budget}"
                )
            if self.max_structural_per_step is not None and action.structural_spending > self.max_structural_per_step:
                return False, (
                    f"Action structural spending {action.structural_spending} exceeds "
                    f"per-step limit {self.max_structural_per_step}"
                )

        # 4. Action structure and amplitude cap
        try:
            action.validate(
                N=self.N,
                D=self.D,
                amplitude_cap=self.amplitude_cap,
                accessible_roster=self.accessible_roster,
            )
        except ValueError as err:
            return False, str(err)

        # 5. Distinct node limit k
        struct_nodes = {se.source for se in action.structural_entries}
        new_nodes = (action.actuated_nodes() | struct_nodes) - self.distinct_nodes_actuated
        if len(self.distinct_nodes_actuated) + len(new_nodes) > self.k:
            return False, (
                f"Action actuates {len(new_nodes)} new nodes, which exceeds distinct-node cap "
                f"k={self.k} (already actuated {len(self.distinct_nodes_actuated)})"
            )

        # 6. Duration sanity
        if action.intended_end_time is not None:
            if action.intended_end_time <= t and action.spending_rate > 1e-15:
                return False, f"Zero-length or past nonzero pulse rejected (end={action.intended_end_time} <= now={t})"

        return True, "Valid"

    def apply_action(self, action: Action, current_time: Optional[float] = None) -> float:
        """Validate and apply action, replacing active field vector.

        Returns:
            effective_end_time: The timestamp when this action will shut off or expire,
            accounting for scheduled duration, mandatory deadline T_c, or early budget exhaustion.
        """
        t = self.current_time if current_time is None else float(current_time)
        if t < self.current_time - 1e-12:
            raise ValueError(f"Cannot apply action backwards in time: current={self.current_time}, requested={t}")
        if t > self.current_time:
            self.advance_to(t)

        is_valid, msg = self.validate_action(action, current_time=t)
        if not is_valid:
            raise ValueError(f"Invalid action: {msg}")

        # Check explicit release flag
        if action.release:
            self._declare_release(t)
            return t

        # Mark actuated nodes (both field targets and structural sources)
        for node in action.actuated_nodes():
            self.distinct_nodes_actuated.add(node)
        for se in action.structural_entries:
            self.distinct_nodes_actuated.add(se.source)

        # Apply structural spending
        if action.structural_entries:
            self.integrated_structural_modifications += action.structural_spending
            if self.structural_budget is not None and self.remaining_structural_budget <= self.tol:
                self.is_structural_exhausted = True

        # If zero field action, shut off current field
        c_u = action.spending_rate
        if c_u <= 1e-15:
            self.current_field.fill(0.0)
            self.active_action = action
            self.active_action_end_time = action.intended_end_time
            return action.intended_end_time if action.intended_end_time is not None else self.release_deadline

        # Install new field vector (replaces previous field vector)
        self.current_field = action.to_array(self.N, self.D)
        self.active_action = action

        # Compute effective shutoff time based on duration, deadline, and remaining budget
        rem_budget = self.remaining_exposure
        time_to_exhaustion = rem_budget / c_u
        exhaustion_time = t + time_to_exhaustion

        nominal_end = action.intended_end_time if action.intended_end_time is not None else self.release_deadline
        effective_end = min(nominal_end, self.release_deadline, exhaustion_time)

        self.active_action_end_time = effective_end
        return effective_end

    def advance_to(self, target_time: float) -> None:
        """Advance ledger clock to target_time, integrating exposure strictly over active intervals."""
        t_target = float(target_time)
        if t_target < self.current_time - 1e-12:
            raise ValueError(f"Time cannot move backwards: current={self.current_time}, target={t_target}")

        dt = t_target - self.current_time
        if dt <= 1e-15:
            return

        # Determine the earliest timestamp when current field shuts off or expires
        field_end = self.release_deadline
        if self.active_action_end_time is not None:
            field_end = min(field_end, self.active_action_end_time)

        c_u = self.current_spending_rate
        if c_u > 1e-15:
            exhaustion_time = self.current_time + self.remaining_exposure / c_u
            field_end = min(field_end, exhaustion_time)

            # Integrate exposure only over the active field duration [current_time, min(t_target, field_end)]
            t_active_end = min(t_target, field_end)
            dt_active = max(0.0, t_active_end - self.current_time)
            if dt_active > 0.0:
                delta_exposure = c_u * dt_active
                rem = self.remaining_exposure
                if delta_exposure > rem + self.tol:
                    raise RuntimeError(
                        f"Exposure violation: spending {delta_exposure} exceeds remaining {rem} "
                        f"by more than tolerance {self.tol}"
                    )
                actual_delta = min(delta_exposure, rem)
                self.integrated_exposure += actual_delta

        # Check exhaustion release declaration
        if self.remaining_exposure <= self.tol:
            self.is_exhausted = True
            self._declare_release(min(t_target, field_end))

        # Check action expiration
        if self.active_action_end_time is not None and t_target >= self.active_action_end_time - 1e-12:
            self.current_field.fill(0.0)
            self.active_action = None
            self.active_action_end_time = None

        # Check deadline shutoff
        if t_target >= self.release_deadline - 1e-12 and not self.is_released:
            self._declare_release(self.release_deadline)

        self.current_time = t_target

    def _declare_release(self, timestamp: float) -> None:
        """Permanently declare release status."""
        if not self.is_released:
            self.is_released = True
            self.release_time = min(float(timestamp), self.release_deadline)
        self.current_field.fill(0.0)
        self.active_action = None
        self.active_action_end_time = None

    def release_control(self, current_time: Optional[float] = None) -> None:
        """External call to declare immediate irreversible control release."""
        t = self.current_time if current_time is None else float(current_time)
        if t > self.current_time:
            self.advance_to(t)
        self._declare_release(t)
