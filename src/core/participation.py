"""Participation dynamics and institutional threshold-holding tracking (Stages 2 & 3).

Provides:
1. Binary collective action participation states n_i in {0, 1}.
2. Opposition incentive q_i = (1 - I * x_{i, 1}) / 2 and local peer exposure rho_i = (A @ n)_i.
3. Coordination payoff difference Delta u_i = b * q_i + g * rho_i - c.
4. Logistic participation redraw: P(n_i = 1) = 1 / (1 + exp(-Delta u_i / T_n)) without factor of 2.
5. Exact potential function Phi(n) for symmetric attention matrices where Delta Phi = Delta u_k.
6. Institutional threshold fraction tracking M_n >= theta with uninterrupted holding duration t_hold,
   monitored via preemption-ready deterministic deadline events and monotonic generation counters.

Simulation Stage Mapping:
    - Stage 2: Extension campaigns (collective action games, institutional threshold holding theta, t_hold).
    - Stage 3: Adaptive learning interaction (coordination payoffs influencing Q-value dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class ParticipationParams:
    """Parameters for collective action participation game and institutional transitions."""

    I: int = 1                  # Incumbent orientation on component 1, in {-1, +1}
    b: float = 1.0              # Private grievance / opposition incentive weight >= 0
    g: float = 0.0              # Social coordination benefit from peers' participation >= 0
    c: float = 0.5              # Participation cost >= 0
    T_n: float = 0.1            # Participation decision noise temperature > 0
    tau_n: float = 1.0          # Participation attempt timescale > 0
    theta: float = 0.5          # Institutional transition threshold fraction in (0, 1]
    t_hold: float = 1.0         # Required uninterrupted holding duration > 0


def compute_participation_potential(
    n: np.ndarray,
    q: np.ndarray,
    A: np.ndarray,
    b: float,
    g: float,
    c: float,
) -> float:
    """Compute exact potential Phi(n) on symmetric attention matrix A (A == A.T):

    Phi(n) = sum_i (b * q_i - c) * n_i + (g / 2) * sum_{i, j} A_{ij} * n_i * n_j.

    Satisfies Phi(n with n_k=1) - Phi(n with n_k=0) == Delta u_k exactly.

    Args:
        n: Binary participation vector of shape (N,), dtype uint8 or float.
        q: Opposition incentive vector of shape (N,).
        A: Symmetric attention matrix of shape (N, N).
        b: Grievance incentive weight.
        g: Social coordination weight.
        c: Participation cost.

    Returns:
        Potential value float.
    """
    n_float = n.astype(np.float64)
    linear_term = np.sum((b * q - c) * n_float)
    interaction_term = 0.5 * g * float(np.dot(n_float, np.dot(A, n_float)))
    return float(linear_term + interaction_term)


class ParticipationTracker:
    """Tracks participation states n and monitors institutional threshold-holding transitions.

    Attributes:
        N: Population size.
        params: ParticipationParams instance.
        n: Binary participation state array of shape (N,), dtype uint8.
        holding_active: Whether participation fraction M_n >= theta currently holds.
        t_start_hold: Timestamp when M_n first reached or exceeded theta.
        t_deadline: Scheduled transition deadline (t_start_hold + t_hold).
        deadline_generation: Monotonic generation ID for lazy invalidation of preempted deadlines.
        transition_occurred: Whether institutional transition has completed.
        t_transition: Timestamp of recorded institutional transition, or None.
    """

    def __init__(
        self,
        N: int,
        params: Optional[ParticipationParams] = None,
        initial_n: Optional[np.ndarray] = None,
    ) -> None:
        if N < 1:
            raise ValueError(f"Population size N must be >= 1, got {N}")
        self.N = N
        self.params = params if params is not None else ParticipationParams()

        if initial_n is not None:
            if initial_n.shape != (N,):
                raise ValueError(f"initial_n shape mismatch: expected ({N},), got {initial_n.shape}")
            if np.any(~np.isin(initial_n, [0, 1])):
                raise ValueError("initial_n values must be in {0, 1}")
            self.n = np.array(initial_n, dtype=np.uint8)
        else:
            self.n = np.zeros(N, dtype=np.uint8)

        self.holding_active: bool = False
        self.t_start_hold: Optional[float] = None
        self.t_deadline: Optional[float] = None
        self.deadline_generation: int = 0
        self.transition_occurred: bool = False
        self.t_transition: Optional[float] = None

    def compute_opposition(self, x1: np.ndarray) -> np.ndarray:
        """Compute private opposition q_i = (1 - I * x_{i, 1}) / 2 in [0, 1].

        Args:
            x1: First belief component array of shape (N,).

        Returns:
            Opposition values of shape (N,), dtype float64.
        """
        return 0.5 * (1.0 - float(self.params.I) * x1)

    def compute_exposure(self, A: np.ndarray) -> np.ndarray:
        """Compute local participation exposure rho_i = sum_j A_{ij} n_j = (A @ n)_i.

        Args:
            A: Row-normalized attention matrix of shape (N, N).

        Returns:
            Exposure values of shape (N,), dtype float64 in [0.0, 1.0].
        """
        return np.dot(A, self.n.astype(np.float64))

    def compute_payoff_diff(self, q_i: float, rho_i: float) -> float:
        """Compute local coordination payoff difference:

        Delta u_i = b * q_i + g * rho_i - c.
        """
        return float(self.params.b * q_i + self.params.g * rho_i - self.params.c)

    def draw_action(self, delta_u: float, rng: np.random.Generator) -> int:
        """Sample binary participation redraw from logistic distribution:

        P(n_i = 1 | Delta u_i) = 1 / (1 + exp(-Delta u_i / T_n)).
        Notice: exponent has NO factor of 2.

        Args:
            delta_u: Payoff difference.
            rng: NumPy random Generator.

        Returns:
            0 or 1.
        """
        z = delta_u / self.params.T_n
        # Overflow protection
        if z >= 350.0:
            p1 = 1.0
        elif z <= -350.0:
            p1 = 0.0
        else:
            p1 = 1.0 / (1.0 + np.exp(-z))
        return 1 if rng.random() < p1 else 0

    def get_participation_fraction(self) -> float:
        """Return current participation fraction M_n(t) = (1/N) sum_i n_i."""
        return float(np.mean(self.n))

    def check_threshold(self, t: float, scheduler: Optional[Any] = None) -> None:
        """Update institutional threshold holding state at time t.

        If M_n >= theta and holding is not active:
            Activates holding interval, schedules deadline at t + t_hold.
        If M_n < theta and holding was active:
            Resets holding interval and increments generation to invalidate pending event.
        """
        if self.transition_occurred:
            return

        m_n = self.get_participation_fraction()
        if m_n >= self.params.theta:
            if not self.holding_active:
                self.holding_active = True
                self.t_start_hold = t
                self.t_deadline = t + self.params.t_hold
                self.deadline_generation += 1
                if scheduler is not None:
                    scheduler.schedule_deterministic_event(
                        time=self.t_deadline,
                        priority=2,  # PRIORITY_HOLDING
                        event_type="holding_deadline",
                        payload={"generation": self.deadline_generation},
                    )
        else:
            if self.holding_active:
                self.holding_active = False
                self.t_start_hold = None
                self.t_deadline = None
                self.deadline_generation += 1  # Invalidate pending deadline in scheduler heap

    def update_agent(
        self,
        i: int,
        x1_val: float,
        A_row: np.ndarray,
        t: float,
        rng: np.random.Generator,
        scheduler: Optional[Any] = None,
    ) -> Tuple[bool, int, int]:
        """Redraw participation action for agent i.

        Args:
            i: Agent index.
            x1_val: Value of private component 1: x[i, 0].
            A_row: Attention vector A[i, :] of shape (N,).
            t: Current simulation time.
            rng: NumPy random Generator.
            scheduler: Optional PoissonScheduler instance for preemption registration.

        Returns:
            Tuple of (changed: bool, old_action: int, new_action: int).
        """
        q_i = 0.5 * (1.0 - float(self.params.I) * x1_val)
        rho_i = float(np.dot(A_row, self.n))
        delta_u = self.compute_payoff_diff(q_i, rho_i)
        old_val = int(self.n[i])
        new_val = self.draw_action(delta_u, rng)

        changed = old_val != new_val
        if changed:
            self.n[i] = new_val
            self.check_threshold(t, scheduler)

        return changed, old_val, new_val

    def handle_deadline_event(self, t: float, payload: Dict[str, Any]) -> bool:
        """Execute institutional transition when holding deadline event fires.

        Validates that the holding interval remained uninterrupted.

        Returns:
            True if institutional transition occurred, False if invalidated.
        """
        gen = payload.get("generation", -1)
        if (
            self.holding_active
            and gen == self.deadline_generation
            and not self.transition_occurred
            and self.get_participation_fraction() >= self.params.theta
            and self.t_deadline is not None
            and abs(t - self.t_deadline) <= 1e-6
        ):
            self.transition_occurred = True
            self.t_transition = t
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable dictionary for checkpointing."""
        return {
            "n": self.n.copy(),
            "holding_active": bool(self.holding_active),
            "t_start_hold": self.t_start_hold,
            "t_deadline": self.t_deadline,
            "deadline_generation": int(self.deadline_generation),
            "transition_occurred": bool(self.transition_occurred),
            "t_transition": self.t_transition,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore state from checkpoint dictionary."""
        self.n = np.array(data["n"], dtype=np.uint8)
        self.holding_active = bool(data["holding_active"])
        self.t_start_hold = float(data["t_start_hold"]) if data.get("t_start_hold") is not None else None
        self.t_deadline = float(data["t_deadline"]) if data.get("t_deadline") is not None else None
        self.deadline_generation = int(data.get("deadline_generation", 0))
        self.transition_occurred = bool(data.get("transition_occurred", False))
        self.t_transition = float(data["t_transition"]) if data.get("t_transition") is not None else None
