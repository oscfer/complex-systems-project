"""Model Predictive Control (MPC) for optimal network transformation.

Implements:
1. ModelPredictiveController: Finite-horizon receding-horizon controller evaluating candidate
   intervention trajectories via simulated network rollouts.
2. State estimation from noisy/delayed observations under strict Layer 2 compliance.
3. Candidate trajectory generator producing diverse thermodynamic field, structural, and hybrid interventions.
4. Optimal trajectory selection minimizing target state distance and respecting BudgetLedger limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.advanced.forecaster import ForecastResult, SimulatedRolloutForecaster, StateEstimator
from src.control.advanced.structural import (
    StructuralActuator,
    force_connection_action,
    rewire_tie_action,
    sever_tie_action,
)
from src.control.contracts import (
    Action,
    ActionEntry,
    ControlProblem,
    Observation,
    StructuralActionEntry,
    StructuralOp,
    TargetType,
)
from src.control.controllers import Controller, TIE_TOLERANCE
from src.control.ledger import BudgetLedger

from src.core.state import StateMode


@dataclass
class MPCConfig:
    """Configuration parameters for Model Predictive Control."""
    horizon_steps: int = 3
    step_interval: float = 5.0
    rollout_samples: int = 2
    discount_factor: float = 0.9
    expenditure_weight: float = 0.05
    structural_weight: float = 0.1
    nominal_amplitude: float = 1.0
    use_structural: bool = False
    reciprocal: bool = False
    max_candidates: int = 15
    seed: int = 42


@dataclass
class TrajectoryCandidate:
    """A sequence of candidate actions spanning the prediction horizon."""
    name: str
    actions: Tuple[Action, ...]
    forecast_results: List[ForecastResult] = field(default_factory=list)
    average_cost: float = float("inf")


class ModelPredictiveController(Controller):
    """Model Predictive Controller executing simulated rollouts over a finite forecast horizon.

    Architectural Compliance:
    - Operates STRICTLY on Layer 2 via Observation and BudgetLedger contracts.
    - NEVER accesses private states (x_i) or true memory (W) in non-oracle mode.
    - Evaluates multi-step intervention trajectories using internal physics rollouts.
    - Enforces BudgetLedger exposure cap B, distinct-node limit k, and structural budget.
    - Applies receding-horizon execution: executes first action, re-plans at next delivery.
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        component: int = 0,
    ) -> None:
        self.config = config if config is not None else MPCConfig()
        self.component = int(component)

        self.problem: Optional[ControlProblem] = None
        self.rng: Optional[np.random.Generator] = None
        self.state_estimator = StateEstimator()
        self.forecaster: Optional[SimulatedRolloutForecaster] = None

        self.s_star: Optional[float] = None
        self.target_threshold: float = 0.5
        self.last_optimal_candidate: Optional[TrajectoryCandidate] = None
        self.execution_history: List[Dict[str, Any]] = []

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize controller for a new episode."""
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)

        # Infer target threshold
        if problem.target_params and "threshold" in problem.target_params:
            self.target_threshold = float(problem.target_params["threshold"])
        else:
            self.target_threshold = 0.5

        # Infer target orientation s_*. Follows the declared direction-knowledge
        # contract: a missing (None) or tied (|mean| <= TIE_TOLERANCE) initial
        # public report is ambiguous, so inference is DEFERRED to act() rather
        # than fabricating a majority from an ambiguous observation.
        if problem.target_params and "s_star" in problem.target_params:
            self.s_star = float(problem.target_params["s_star"])
        elif problem.target_type in (TargetType.REVERSAL, TargetType.TURNOVER):
            obs_mean = initial_observation.observed_public_mean(self.component)
            if obs_mean is None or abs(obs_mean) <= TIE_TOLERANCE:
                self.s_star = None  # deferred; act() waits until a decisive report
            else:
                self.s_star = -1.0 if obs_mean >= 0.0 else 1.0
        else:
            self.s_star = 1.0

        # Infer target metric parameters if missing
        self.target_params = dict(problem.target_params) if problem.target_params else {}
        if problem.target_type == TargetType.TURNOVER and "L_0" not in self.target_params:
            from src.control.evaluator import compute_top_10_percent_set
            est_state = self.state_estimator.estimate_state(
                observation=initial_observation,
                problem=problem,
                state_mode=StateMode.BINARY,
                reciprocal=self.config.reciprocal,
            )
            self.target_params["L_0"] = list(compute_top_10_percent_set(est_state.W))
        if problem.target_type in (TargetType.TURNOVER, TargetType.DECENTRALIZATION) and "C_N_0" not in self.target_params:
            from src.analytics.influence import calculate_influence_metrics
            est_state = self.state_estimator.estimate_state(
                observation=initial_observation,
                problem=problem,
                state_mode=StateMode.BINARY,
                reciprocal=self.config.reciprocal,
            )
            inf = calculate_influence_metrics(est_state.W)
            if "C_N" in inf and not np.isnan(inf["C_N"]):
                self.target_params["C_N_0"] = float(inf["C_N"])

        # Initialize forecaster
        self.forecaster = SimulatedRolloutForecaster(
            J=1.0,
            T=0.3,
            delta=0.1,
            epsilon=0.1,
            closure=0.0,
            reciprocal=self.config.reciprocal,
            discount_factor=self.config.discount_factor,
            expenditure_weight=self.config.expenditure_weight,
            structural_weight=self.config.structural_weight,
        )

        self.last_optimal_candidate = None
        self.execution_history.clear()

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        """Compute optimal intervention action via simulated rollout forecast."""
        if self.problem is None:
            raise RuntimeError("Controller must be reset before act()")

        # Check termination conditions
        if ledger.is_released or ledger.is_exhausted:
            return Action(entries=(), release=True)

        if observation.time >= self.problem.release_deadline - 1e-12:
            return Action(entries=(), release=True)

        # Deferred direction inference (declared direction-knowledge contract):
        # if s_* was ambiguous at construction, retry with the current
        # observation; while it stays missing/tied, wait without consuming
        # actuation budget instead of fabricating a majority.
        if self.s_star is None:
            obs_mean = observation.observed_public_mean(self.component)
            if obs_mean is not None and abs(obs_mean) > TIE_TOLERANCE:
                self.s_star = -1.0 if obs_mean >= 0.0 else 1.0
            else:
                return Action(entries=())

        # Estimate current state from Observation under Layer 2 contract
        estimated_state = self.state_estimator.estimate_state(
            observation=observation,
            problem=self.problem,
            state_mode=StateMode.BINARY,
            reciprocal=self.config.reciprocal,
        )

        # Determine horizon steps remaining before release deadline
        time_to_deadline = self.problem.release_deadline - observation.time
        max_possible_steps = max(1, int(np.ceil(time_to_deadline / self.config.step_interval)))
        actual_horizon_steps = min(self.config.horizon_steps, max_possible_steps)

        # Generate candidate intervention trajectories
        candidates = self._generate_candidates(
            observation=observation,
            ledger=ledger,
            horizon_steps=actual_horizon_steps,
        )

        if not candidates:
            # Fallback zero action
            return Action(entries=(), intended_end_time=observation.time + self.config.step_interval)

        # Evaluate candidate trajectories via simulated rollouts
        best_candidate: Optional[TrajectoryCandidate] = None
        best_cost = float("inf")

        for cand_idx, candidate in enumerate(candidates):
            rollout_costs: List[float] = []
            for sample_idx in range(self.config.rollout_samples):
                sample_seed = self.config.seed + sample_idx * 1000
                sample_rng = np.random.default_rng(sample_seed)

                result = self.forecaster.forecast_trajectory(
                    initial_state=estimated_state,
                    action_sequence=candidate.actions,
                    problem=self.problem,
                    ledger=ledger,
                    step_interval=self.config.step_interval,
                    s_star=self.s_star if self.s_star is not None else -1.0,
                    target_threshold=self.target_threshold,
                    target_params=self.target_params,
                    rng=sample_rng,
                    trajectory_index=cand_idx,
                )
                if result.is_valid:
                    rollout_costs.append(result.total_cost)
                    candidate.forecast_results.append(result)
                else:
                    # Invalid action (e.g. violated budget during rollout)
                    rollout_costs.append(float("inf"))

            avg_cost = float(np.mean(rollout_costs)) if rollout_costs else float("inf")
            candidate.average_cost = avg_cost

            if avg_cost < best_cost:
                best_cost = avg_cost
                best_candidate = candidate

        # Select first action from best trajectory (receding horizon principle)
        if best_candidate is None or not np.isfinite(best_cost):
            # Fallback to zero action
            chosen_action = Action(entries=(), intended_end_time=observation.time + self.config.step_interval)
        else:
            chosen_action = best_candidate.actions[0]
            self.last_optimal_candidate = best_candidate

        # Verify action against ledger constraints
        is_valid, _ = ledger.validate_action(chosen_action, current_time=observation.time)
        if not is_valid:
            # If chosen action fails validation on ledger, fallback to safe zero action
            chosen_action = Action(entries=(), intended_end_time=observation.time + self.config.step_interval)

        # Record telemetry
        self.execution_history.append({
            "time": observation.time,
            "chosen_candidate": best_candidate.name if best_candidate else "fallback",
            "best_cost": best_cost,
            "spending_rate": chosen_action.spending_rate,
            "structural_spending": chosen_action.structural_spending,
        })

        return chosen_action

    def _generate_candidates(
        self,
        observation: Observation,
        ledger: BudgetLedger,
        horizon_steps: int,
    ) -> List[TrajectoryCandidate]:
        """Generate a diverse pool of candidate intervention sequences."""
        candidates: List[TrajectoryCandidate] = []
        t = observation.time
        dt = self.config.step_interval
        rem_B = ledger.remaining_exposure
        rem_k = ledger.remaining_distinct_nodes
        accessible = list(observation.accessible_roster)
        s_star = self.s_star if self.s_star is not None else -1.0
        u_max = self.problem.amplitude_cap if self.problem else 4.0

        # Candidate 0: Zero Control (Unactuated coasting)
        zero_actions = tuple(
            Action(entries=(), intended_end_time=t + (i + 1) * dt)
            for i in range(horizon_steps)
        )
        candidates.append(TrajectoryCandidate(name="zero_control", actions=zero_actions))

        if not accessible:
            return candidates

        # Determine candidate nodes for field interventions respecting distinct-node limit k
        already_actuated = [n for n in accessible if n in ledger.distinct_nodes_actuated]
        unactuated = [n for n in accessible if n not in ledger.distinct_nodes_actuated]

        opposing_unactuated: List[int] = []
        for n in unactuated:
            if observation.panel_mask[n]:
                y_val = observation.public_states[n, self.component]
                if np.isfinite(y_val) and s_star * y_val < self.target_threshold:
                    opposing_unactuated.append(n)
            else:
                opposing_unactuated.append(n)
        if not opposing_unactuated:
            opposing_unactuated = unactuated

        from src.control.optimizers.heuristics import compute_degree_scores, compute_influence_scores
        N_pop = observation.public_states.shape[0]
        deg_arr = compute_degree_scores(N=N_pop, observed_edges=observation.observed_edges, mode="total")
        inf_arr = compute_influence_scores(N=N_pop, observed_edges=observation.observed_edges, mode="received_attention")

        deg_ranked_unact = sorted(opposing_unactuated, key=lambda n: float(deg_arr[n]), reverse=True)
        inf_ranked_unact = sorted(opposing_unactuated, key=lambda n: float(inf_arr[n]), reverse=True)

        k_new = min(max(0, rem_k), len(opposing_unactuated))

        # Target nodes: already actuated nodes (0 marginal k cost) + up to k_new unactuated nodes
        deg_targets = list(already_actuated) + deg_ranked_unact[:k_new]
        inf_targets = list(already_actuated) + inf_ranked_unact[:k_new]

        step_share = 1.0 / max(1, horizon_steps)

        # Helper to construct bounded field action for a step of duration dt
        def _make_field_action(
            target_nodes: Sequence[int],
            nominal_amp: float,
            end_t: float,
            budget_share: float = 1.0,
        ) -> Action:
            if not target_nodes or rem_B <= 1e-12:
                return Action(entries=(), intended_end_time=end_t)
            dur = max(1e-6, dt)
            available_budget = rem_B * budget_share * (1.0 - 1e-9)
            max_amp_by_budget = available_budget / (len(target_nodes) * dur) if (len(target_nodes) * dur) > 0 else 0.0
            a_eff = min(nominal_amp, max_amp_by_budget, u_max)
            if a_eff <= 1e-15:
                return Action(entries=(), intended_end_time=end_t)
            entries = tuple(
                ActionEntry(node=n, component=self.component, field=float(s_star * a_eff))
                for n in target_nodes
            )
            return Action(entries=entries, intended_end_time=end_t)

        # Generate field candidates if exposure budget remains and targets exist
        if rem_B > 1e-12 and deg_targets:
            # Candidate 1: Degree-Heuristic Constant Pulse (budget shared across horizon)
            deg_seq = tuple(
                _make_field_action(deg_targets, self.config.nominal_amplitude, t + (i + 1) * dt, budget_share=step_share)
                for i in range(horizon_steps)
            )
            candidates.append(TrajectoryCandidate(name="degree_heuristic", actions=deg_seq))

            # Candidate 2: Influence-Heuristic Constant Pulse
            if inf_targets:
                inf_seq = tuple(
                    _make_field_action(inf_targets, self.config.nominal_amplitude, t + (i + 1) * dt, budget_share=step_share)
                    for i in range(horizon_steps)
                )
                candidates.append(TrajectoryCandidate(name="influence_heuristic", actions=inf_seq))

            # Candidate 3: Full Amplitude Pulse then Coast (all budget permitted in step 0)
            full_act = _make_field_action(deg_targets, u_max, t + dt, budget_share=1.0)
            pulse_coast_seq = [full_act] + [
                Action(entries=(), intended_end_time=t + (i + 1) * dt)
                for i in range(1, horizon_steps)
            ]
            candidates.append(TrajectoryCandidate(name="pulse_then_coast", actions=tuple(pulse_coast_seq)))

            # Candidate 4: Conservative Half-Amplitude Pulse
            half_amp = 0.5 * self.config.nominal_amplitude
            half_seq = tuple(
                _make_field_action(deg_targets, half_amp, t + (i + 1) * dt, budget_share=step_share)
                for i in range(horizon_steps)
            )
            candidates.append(TrajectoryCandidate(name="half_amplitude", actions=half_seq))

            # Candidate 5: Single Leading Target Focus (k=1 maximal exertion)
            lead_node = [deg_targets[0]]
            lead_seq = tuple(
                _make_field_action(lead_node, self.config.nominal_amplitude, t + (i + 1) * dt, budget_share=step_share)
                for i in range(horizon_steps)
            )
            candidates.append(TrajectoryCandidate(name="single_lead_focus", actions=lead_seq))

        # Structural candidates (if enabled or requested)
        rem_struct = ledger.remaining_structural_budget
        if np.isneginf(rem_struct) or np.isposinf(rem_struct) or rem_struct == float("inf"):
            max_step_struct = (
                self.problem.max_structural_per_step
                if self.problem and self.problem.max_structural_per_step is not None
                else 10
            )
            allowed_struct_this_step = max_step_struct
        else:
            max_step_struct = (
                self.problem.max_structural_per_step
                if self.problem and self.problem.max_structural_per_step is not None
                else int(rem_struct)
            )
            allowed_struct_this_step = min(int(rem_struct), max_step_struct)

        should_consider_structural = (
            self.config.use_structural
            or rem_struct < float("inf")
            or (self.problem is not None and self.problem.target_type in (TargetType.DECENTRALIZATION, TargetType.TURNOVER))
            or (self.problem is not None and self.problem.structural_budget is not None)
        )

        if should_consider_structural and allowed_struct_this_step >= 1 and len(accessible) >= 2:
            # Respect distinct node cap: if rem_k == 0, source node must be in distinct_nodes_actuated
            valid_sources = [n for n in accessible if (rem_k > 0 or n in ledger.distinct_nodes_actuated)]

            if valid_sources:
                # Structural candidate A: Sever ties between prominent nodes and others
                ranked_sources = sorted(valid_sources, key=lambda n: float(deg_arr[n]), reverse=True)
                sever_candidates: List[StructuralActionEntry] = []
                for src in ranked_sources:
                    for (s, tgt), count in observation.observed_edges.items():
                        if s == src and tgt != src and count > 0:
                            entry = StructuralActionEntry(source=src, target=tgt, operation=StructuralOp.SEVER, count=1)
                            if entry not in sever_candidates:
                                sever_candidates.append(entry)
                            if len(sever_candidates) >= min(2, allowed_struct_this_step):
                                break
                    if len(sever_candidates) >= min(2, allowed_struct_this_step):
                        break

                if self.config.reciprocal and len(sever_candidates) < min(2, allowed_struct_this_step):
                    for src in ranked_sources:
                        for (tgt, s), count in observation.observed_edges.items():
                            if s == src and tgt != src and count > 0:
                                entry = StructuralActionEntry(source=src, target=tgt, operation=StructuralOp.SEVER, count=1)
                                if entry not in sever_candidates:
                                    sever_candidates.append(entry)
                                if len(sever_candidates) >= min(2, allowed_struct_this_step):
                                    break
                        if len(sever_candidates) >= min(2, allowed_struct_this_step):
                            break

                if sever_candidates:
                    act_sever = Action.structural(sever_candidates, intended_end_time=t + dt)
                    sever_seq = [act_sever] + [
                        Action(entries=(), intended_end_time=t + (i + 1) * dt)
                        for i in range(1, horizon_steps)
                    ]
                    candidates.append(TrajectoryCandidate(name="structural_sever", actions=tuple(sever_seq)))

                # Structural candidate B: Force connection between accessible nodes
                src_node = valid_sources[0]
                target_candidates = [n for n in accessible if n != src_node]
                if target_candidates:
                    target_candidates_sorted = sorted(target_candidates, key=lambda n: float(deg_arr[n]))
                    tgt_node = target_candidates_sorted[0]
                    force_entry = StructuralActionEntry(
                        source=src_node,
                        target=tgt_node,
                        operation=StructuralOp.FORCE,
                        count=min(1, allowed_struct_this_step),
                    )
                    act_force = Action.structural([force_entry], intended_end_time=t + dt)
                    force_seq = [act_force] + [
                        Action(entries=(), intended_end_time=t + (i + 1) * dt)
                        for i in range(1, horizon_steps)
                    ]
                    candidates.append(TrajectoryCandidate(name="structural_force", actions=tuple(force_seq)))

                # Structural candidate C: Rewire tie
                if allowed_struct_this_step >= 2 and sever_candidates:
                    old_sev = sever_candidates[0]
                    rewire_targets = [n for n in accessible if n != old_sev.source and n != old_sev.target]
                    if rewire_targets:
                        rewire_entry = StructuralActionEntry(
                            source=old_sev.source,
                            target=old_sev.target,
                            operation=StructuralOp.REWIRE,
                            count=1,
                            rewire_target=rewire_targets[0],
                        )
                        act_rewire = Action.structural([rewire_entry], intended_end_time=t + dt)
                        rewire_seq = [act_rewire] + [
                            Action(entries=(), intended_end_time=t + (i + 1) * dt)
                            for i in range(1, horizon_steps)
                        ]
                        candidates.append(TrajectoryCandidate(name="structural_rewire", actions=tuple(rewire_seq)))

                # Hybrid candidate: Field pulse + Structural manipulation
                if deg_targets and rem_B > 1e-12:
                    # Prefer a source node already in deg_targets or distinct_nodes_actuated to avoid k violation
                    common_sources = [n for n in deg_targets if n in valid_sources]
                    src_node = common_sources[0] if common_sources else valid_sources[0]
                    field_new_nodes = {n for n in deg_targets if n not in ledger.distinct_nodes_actuated}
                    struct_new_nodes = {src_node} if src_node not in ledger.distinct_nodes_actuated else set()
                    if len(field_new_nodes | struct_new_nodes) <= rem_k:
                        target_candidates = [n for n in accessible if n != src_node]
                        if target_candidates:
                            tgt_node = target_candidates[0]
                            force_entry = StructuralActionEntry(
                                source=src_node,
                                target=tgt_node,
                                operation=StructuralOp.FORCE,
                                count=min(1, allowed_struct_this_step),
                            )
                            field_act = _make_field_action(deg_targets, self.config.nominal_amplitude, t + dt, budget_share=1.0)
                            if field_act.entries:
                                hybrid_act = Action(
                                    entries=field_act.entries,
                                    intended_end_time=t + dt,
                                    structural_entries=(force_entry,),
                                )
                                hybrid_seq = [hybrid_act] + [
                                    Action(entries=(), intended_end_time=t + (i + 1) * dt)
                                    for i in range(1, horizon_steps)
                                ]
                                candidates.append(TrajectoryCandidate(name="hybrid_field_structural", actions=tuple(hybrid_seq)))

        # Limit to max_candidates
        return candidates[:self.config.max_candidates]
