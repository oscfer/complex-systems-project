"""State estimation and simulated rollout forecasting for Model Predictive Control.

Implements:
1. StateEstimator: Reconstructs state (beliefs x, y, memory W) from restricted/noisy Observations
   under strict Layer 2 compliance without accessing private states x_i or true memory W.
2. SimulatedRolloutForecaster: Executes simulated network rollouts over a finite prediction
   horizon H to forecast future network states under candidate intervention trajectories.
3. Target distance evaluation: Computes exact discrepancy metrics across reversal, decentralization,
   and turnover control objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.control.advanced.structural import StructuralActuator
from src.control.contracts import Action, ControlProblem, Observation, TargetType
from src.control.ledger import BudgetLedger
from src.core.fields import compute_single_public_field
from src.core.heat_baths import sample_binary_glauber_scalar, sample_continuous_inverse_cdf_scalar
from src.core.memory import MemoryManager
from src.core.scheduler import PRIORITY_RESTORE, PRIORITY_SNAPSHOT, PoissonScheduler
from src.core.state import State, StateMode


@dataclass
class ForecastResult:
    """Outcome of a simulated candidate trajectory rollout."""
    trajectory_index: int
    forecasted_states: List[State]
    target_distances: List[float]
    total_cost: float
    exposure_spent: float
    structural_spent: int
    terminal_violation: float
    is_valid: bool = True
    error_message: Optional[str] = None


class StateEstimator:
    """Estimates unobserved private beliefs and memory from noisy, delayed Observations.

    Strict Architectural Compliance:
    - Never reads or accesses `Observation.private_states` or `Observation.true_memory`
      unless observation is explicitly flagged as oracle.
    - Imputes unobserved panel nodes from observed public statistics.
    - Estimates private beliefs from public beliefs using equilibrium conditional expectations.
    - Reconstructs interaction memory matrix from revealed interaction ties.
    """

    def __init__(self, default_state_mode: StateMode = StateMode.BINARY) -> None:
        self.default_state_mode = default_state_mode

    def estimate_state(
        self,
        observation: Observation,
        problem: ControlProblem,
        state_mode: Optional[StateMode] = None,
        J: float = 1.0,
        T: float = 0.3,
        reciprocal: bool = False,
        row_caps: Optional[np.ndarray] = None,
    ) -> State:
        """Construct an estimated State object from an Observation."""
        N = problem.N
        D = problem.D
        mode = state_mode if state_mode is not None else self.default_state_mode

        # If Oracle mode with full ground truth available, use it directly
        if observation.is_oracle and observation.private_states is not None and observation.true_memory is not None:
            return State(
                N=N,
                D=D,
                state_mode=mode,
                x=observation.private_states.copy(),
                y=observation.public_states.copy(),
                W=observation.true_memory.copy(),
                n=np.zeros(N, dtype=np.uint8),
                Q=np.zeros((N, D, 2), dtype=np.float64),
                t=observation.time,
                opportunity_groups=observation.group_labels.copy() if observation.group_labels is not None else None,
                row_caps=row_caps.copy() if row_caps is not None else None,
            )

        # 1. Estimate public states y
        y_hat = np.zeros((N, D), dtype=np.float64)
        for alpha in range(D):
            # Compute global panel mean for imputation fallback
            obs_vals = observation.public_states[observation.panel_mask, alpha]
            valid_vals = obs_vals[np.isfinite(obs_vals)]
            if len(valid_vals) > 0:
                global_mean = float(np.mean(valid_vals))
            elif problem.target_params and "s_star" in problem.target_params:
                # In non-oracle mode under 0% panel visibility, prior unobserved state opposes target
                global_mean = -float(problem.target_params["s_star"])
            else:
                global_mean = 0.0

            # Compute group means if group labels are available
            group_means: Dict[int, float] = {}
            if observation.group_labels is not None:
                for g in np.unique(observation.group_labels):
                    g_mask = observation.panel_mask & (observation.group_labels == g)
                    g_vals = observation.public_states[g_mask, alpha]
                    g_valid = g_vals[np.isfinite(g_vals)]
                    if len(g_valid) > 0:
                        group_means[int(g)] = float(np.mean(g_valid))

            for i in range(N):
                if observation.panel_mask[i] and np.isfinite(observation.public_states[i, alpha]):
                    y_hat[i, alpha] = observation.public_states[i, alpha]
                else:
                    if observation.group_labels is not None and int(observation.group_labels[i]) in group_means:
                        y_hat[i, alpha] = group_means[int(observation.group_labels[i])]
                    else:
                        y_hat[i, alpha] = global_mean

        # Clip beliefs to bounded range
        y_hat = np.clip(y_hat, -1.0, 1.0)
        if mode == StateMode.BINARY:
            # Discrete thresholding for binary states
            y_hat = np.where(y_hat >= 0.0, 1.0, -1.0)

        # 2. Estimate private states x from y using Glauber equilibrium expectation
        x_hat = np.zeros((N, D), dtype=np.float64)
        if mode == StateMode.BINARY:
            # In binary mode, Glauber mean has same sign as public belief
            x_hat = np.where(y_hat >= 0.0, 1.0, -1.0)
        else:
            # In continuous mode, surrogate x matches public y
            x_hat = y_hat.copy()

        # 3. Estimate memory matrix W from revealed interaction ties
        W_hat = np.zeros((N, N), dtype=np.uint32)
        for (i, j), count in observation.observed_edges.items():
            if 0 <= i < N and 0 <= j < N and i != j:
                W_hat[i, j] = max(W_hat[i, j], int(count))

        # Enforce strictly zero diagonal and reciprocal symmetry if enabled
        np.fill_diagonal(W_hat, 0)
        if reciprocal:
            W_hat = np.maximum(W_hat, W_hat.T)

        if row_caps is not None:
            for i in range(N):
                c_i = int(row_caps[i])
                r_sum = int(np.sum(W_hat[i]))
                if r_sum > c_i:
                    to_rem = r_sum - c_i
                    for _ in range(to_rem):
                        nz = np.where(W_hat[i] > 0)[0]
                        if len(nz) == 0:
                            break
                        W_hat[i, nz[0]] -= 1

        return State(
            N=N,
            D=D,
            state_mode=mode,
            x=x_hat,
            y=y_hat,
            W=W_hat,
            n=np.zeros(N, dtype=np.uint8),
            Q=np.zeros((N, D, 2), dtype=np.float64),
            t=observation.time,
            opportunity_groups=observation.group_labels.copy() if observation.group_labels is not None else None,
            row_caps=row_caps.copy() if row_caps is not None else None,
        )


class SimulatedRolloutForecaster:
    """Executes simulated rollouts to forecast future network states under candidate actions."""

    def __init__(
        self,
        J: float = 1.0,
        T: float = 0.3,
        delta: float = 0.1,
        epsilon: float = 0.1,
        closure: float = 0.0,
        reciprocal: bool = False,
        discount_factor: float = 0.9,
        expenditure_weight: float = 0.05,
        structural_weight: float = 0.1,
    ) -> None:
        self.J = float(J)
        self.T = float(T)
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.closure = float(closure)
        self.reciprocal = bool(reciprocal)
        self.discount_factor = float(discount_factor)
        self.expenditure_weight = float(expenditure_weight)
        self.structural_weight = float(structural_weight)
        self.structural_actuator = StructuralActuator()

    def forecast_trajectory(
        self,
        initial_state: State,
        action_sequence: Sequence[Action],
        problem: ControlProblem,
        ledger: BudgetLedger,
        step_interval: float = 5.0,
        s_star: float = -1.0,
        target_threshold: float = 0.5,
        target_params: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        trajectory_index: int = 0,
    ) -> ForecastResult:
        """Run simulated rollout for a sequence of candidate actions over a finite horizon.

        Args:
            initial_state: Estimated starting state S_t.
            action_sequence: Sequence of candidate Actions [u_0, u_1, ..., u_{H-1}].
            problem: ControlProblem specification.
            ledger: Current BudgetLedger state.
            step_interval: Duration Delta t of each horizon stage.
            s_star: Target orientation direction for reversal.
            target_threshold: Target threshold theta.
            rng: Independent RNG stream for rollout stochasticity.
            trajectory_index: Index identifier of the candidate trajectory.

        Returns:
            ForecastResult detailing forecasted states, distances, and cumulative cost.
        """
        sim_rng = rng if rng is not None else np.random.default_rng(problem.search_seed + trajectory_index)
        state = initial_state.copy()
        N, D = state.N, state.D

        # Create isolated rollout ledger copy
        rollout_ledger = ledger.copy()

        # Initialize rollout MemoryManager
        W_init = state.W.copy()
        if self.reciprocal and not np.array_equal(W_init, W_init.T):
            W_init = np.maximum(W_init, W_init.T)

        if state.row_caps is not None:
            for i in range(N):
                cap_i = int(state.row_caps[i])
                row_sum = int(np.sum(W_init[i]))
                if row_sum > cap_i:
                    to_remove = row_sum - cap_i
                    for _ in range(to_remove):
                        nz = np.where(W_init[i] > 0)[0]
                        if len(nz) == 0:
                            break
                        p_nz = W_init[i, nz].astype(np.float64) / np.sum(W_init[i, nz])
                        idx_evict = nz[sim_rng.choice(len(nz), p=p_nz)]
                        W_init[i, idx_evict] -= 1

        memory_mgr = MemoryManager(
            N=N,
            reciprocal=self.reciprocal,
            epsilon=self.epsilon,
            chi=self.closure,
            initial_W=W_init,
            capacity=None if self.reciprocal else state.row_caps,
            opportunity_groups=state.opportunity_groups,
            rng=sim_rng,
        )

        # Initialize rollout PoissonScheduler
        tau_x_param = state.tau_x if state.tau_x is not None else 5.0
        nu_param = state.nu if state.nu is not None else 2.0
        scheduler = PoissonScheduler(
            N=N,
            D=D,
            tau_x=tau_x_param,
            tau_y=1.0,
            nu=nu_param,
            delta=self.delta,
            rng=sim_rng,
        )
        scheduler.time = ledger.current_time
        scheduler.set_total_memory(int(memory_mgr.total_memory))

        K_arr = state.K if state.K is not None else np.full(N, 0.5)
        g_arr = state.g if state.g is not None else np.ones(N)
        b_arr = state.b if state.b is not None else np.zeros((N, D))

        # Register jump callbacks
        def cb_private(i: int) -> None:
            rollout_ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            for alpha in range(D):
                f_x = float(K_arr[i] * state.y[i, alpha] + b_arr[i, alpha])
                kappa = f_x / self.T
                if state.state_mode == StateMode.BINARY:
                    state.x[i, alpha] = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
                else:
                    state.x[i, alpha] = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))

        def cb_public(i: int) -> None:
            rollout_ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            W_row = memory_mgr.W[i, :]
            for alpha in range(D):
                u_val = float(rollout_ledger.current_field[i, alpha])
                f_y = compute_single_public_field(
                    W_row=W_row,
                    y_col=state.y[:, alpha],
                    x_val=float(state.x[i, alpha]),
                    J=self.J,
                    K=float(K_arr[i]),
                    h_val=0.0,
                    g_val=float(g_arr[i]),
                    u_val=u_val,
                )
                kappa = f_y / self.T
                if state.state_mode == StateMode.BINARY:
                    state.y[i, alpha] = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
                else:
                    state.y[i, alpha] = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))

        def cb_encounter(initiator: int) -> None:
            rollout_ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            res = memory_mgr.sample_and_apply_encounter(i=initiator, y=state.y, rng=sim_rng)
            if not res.is_null:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W

        def cb_decay() -> None:
            rollout_ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            i_dec, _ = memory_mgr.sample_and_apply_decay(rng=sim_rng)
            if i_dec != -1:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W

        scheduler.register_callbacks(
            cb_private=cb_private,
            cb_public=cb_public,
            cb_encounter=cb_encounter,
            cb_decay=cb_decay,
        )

        forecasted_states: List[State] = []
        target_distances: List[float] = []
        cumulative_cost = 0.0
        start_exposure = rollout_ledger.integrated_exposure
        start_structural = rollout_ledger.integrated_structural_modifications

        # Roll out each action in the candidate sequence
        for step_idx, action in enumerate(action_sequence):
            stage_start_t = scheduler.time
            stage_target_t = min(stage_start_t + step_interval, problem.horizon)

            # Validate action under current rollout ledger
            is_valid, err_msg = rollout_ledger.validate_action(action, current_time=stage_start_t)
            if not is_valid:
                return ForecastResult(
                    trajectory_index=trajectory_index,
                    forecasted_states=forecasted_states,
                    target_distances=[float("inf")],
                    total_cost=float("inf"),
                    exposure_spent=0.0,
                    structural_spent=0,
                    terminal_violation=float("inf"),
                    is_valid=False,
                    error_message=err_msg,
                )

            # Apply field action to ledger first to validate and consume budget
            eff_end = rollout_ledger.apply_action(action, current_time=stage_start_t)

            # Apply structural operations only after ledger validates action
            if action.structural_entries:
                self.structural_actuator.apply_action(
                    action=action,
                    state=state,
                    memory_mgr=memory_mgr,
                    scheduler=scheduler,
                    ledger=rollout_ledger,
                )
            if eff_end is not None and stage_start_t < eff_end < stage_target_t:
                scheduler.schedule_deterministic_event(
                    time=eff_end,
                    priority=PRIORITY_RESTORE,
                    event_type="field_shutoff",
                    callback=lambda s, e: rollout_ledger.advance_to(s.time),
                )

            # Schedule deterministic stage boundary event so scheduler stops exactly at stage_target_t
            scheduler.schedule_deterministic_event(
                time=stage_target_t,
                priority=PRIORITY_SNAPSHOT,
                event_type="stage_boundary",
                callback=lambda s, e: rollout_ledger.advance_to(s.time),
            )

            # Advance jump scheduler until stage_target_t
            while scheduler.time < stage_target_t:
                advanced = scheduler.step()
                if not advanced or scheduler.time >= stage_target_t - 1e-12:
                    break

            # Sync time and state
            if rollout_ledger.current_time < stage_target_t:
                rollout_ledger.advance_to(stage_target_t)
            state.t = stage_target_t
            state.W = memory_mgr.W

            # Evaluate distance to target
            t_params = target_params if target_params is not None else problem.target_params
            dist = self.evaluate_target_distance(
                state=state,
                target_type=problem.target_type,
                s_star=s_star,
                threshold=target_threshold,
                target_params=t_params,
            )
            target_distances.append(dist)
            forecasted_states.append(state.copy())

            # Add discounted stage cost
            discount = self.discount_factor ** step_idx
            cumulative_cost += discount * dist

        exp_spent = rollout_ledger.integrated_exposure - start_exposure
        struct_spent = rollout_ledger.integrated_structural_modifications - start_structural

        # Add penalty for resource expenditure
        total_cost = (
            cumulative_cost
            + self.expenditure_weight * exp_spent
            + self.structural_weight * struct_spent
        )

        terminal_violation = target_distances[-1] if target_distances else 0.0

        return ForecastResult(
            trajectory_index=trajectory_index,
            forecasted_states=forecasted_states,
            target_distances=target_distances,
            total_cost=float(total_cost),
            exposure_spent=float(exp_spent),
            structural_spent=int(struct_spent),
            terminal_violation=float(terminal_violation),
            is_valid=True,
        )

    def evaluate_target_distance(
        self,
        state: State,
        target_type: TargetType,
        s_star: float = -1.0,
        threshold: float = 0.5,
        target_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate mathematical distance of state S to target set."""
        if target_type == TargetType.REVERSAL:
            # Mean violation: max(0, threshold - s_* * y_i)
            # When s_* * y_i >= threshold for all nodes, violation is 0
            violations = np.maximum(0.0, threshold - s_star * state.y[:, 0])
            return float(np.mean(violations))

        elif target_type == TargetType.DECENTRALIZATION:
            C_N_0 = float(target_params.get("C_N_0", 0.0)) if target_params else 0.0
            if C_N_0 > 0.0:
                from src.analytics.influence import calculate_influence_metrics
                inf = calculate_influence_metrics(state.W)
                C_N = float(inf.get("C_N", 0.0))
                target_C_N = 0.5 * C_N_0
                return float(max(0.0, (C_N - target_C_N) / max(1e-6, target_C_N)))

            # Attention centralization: variance of received attention r
            N = state.N
            if N <= 1:
                return 0.0
            row_sums = np.sum(state.W, axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                A = np.where(row_sums > 0, state.W / row_sums, 1.0 / (N - 1))
                np.fill_diagonal(A, 0.0)
            r = np.sum(A, axis=0)
            target_var = float(target_params.get("max_variance", 0.05)) if target_params else 0.05
            var_r = float(np.var(r))
            return max(0.0, var_r - target_var)

        elif target_type == TargetType.TURNOVER:
            # Check for elite network turnover objective
            L_0 = target_params.get("L_0") if target_params else None
            if L_0 is not None and len(L_0) > 0:
                L_0_set = set(L_0)
                from src.control.evaluator import compute_top_10_percent_set
                L_t = compute_top_10_percent_set(state.W)
                max_retained = float(target_params.get("max_retained_fraction", 0.5))
                retained_frac = len(L_0_set & L_t) / len(L_0_set)
                return float(max(0.0, retained_frac - max_retained))
            else:
                # Target turnover fraction
                req_turnover = float(target_params.get("turnover_fraction", 0.5)) if target_params else 0.5
                turned = np.mean(state.y[:, 0] * s_star >= threshold)
                return float(max(0.0, req_turnover - turned))

        return 0.0
