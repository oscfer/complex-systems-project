"""Controlled simulation runner integrating control contracts with realism physics.

Connects:
- State (beliefs x, y, memory W, heterogeneity traits K, tau_x, nu, g, b, row_caps)
- MemoryManager (encounter and decay updates with capacity caps and eligibility)
- PoissonScheduler (5-channel jump process with deterministic preemption)
- BudgetLedger (resource accounting and field actuation)
- ObservationModel (noisy, delayed sampled-public panel views)
- Controller (observable feedback / open-loop / zero policies)
- TargetEvaluator (exact event-tie persistence and holding clocks)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

from src.control.contracts import Action, ControlProblem, RolloutResult, TargetType
from src.control.controllers import Controller, ZeroController
from src.control.evaluator import TargetDefinition, TargetEvaluator
from src.control.ledger import BudgetLedger
from src.control.observation import ObservationModel, SampledPublicObservationModel
from src.control.recording import RolloutRecorder
from src.core.fields import compute_single_public_field
from src.core.heat_baths import sample_binary_glauber_scalar, sample_continuous_inverse_cdf_scalar
from src.core.memory import MemoryManager
from src.core.scheduler import (
    PRIORITY_HOLDING,
    PRIORITY_HORIZON,
    PRIORITY_INTERVENTION,
    PRIORITY_RESTORE,
    PRIORITY_SNAPSHOT,
    PoissonScheduler,
)
from src.core.state import State, StateMode


def run_controlled_simulation(
    initial_state: Optional[State] = None,
    problem: Optional[ControlProblem] = None,
    controller: Optional[Controller] = None,
    observation_model: Optional[ObservationModel] = None,
    target_evaluator: Optional[TargetEvaluator] = None,
    J: float = 1.0,
    T: float = 0.3,
    delta: float = 0.1,
    epsilon: float = 0.1,
    closure: float = 0.3,
    sample_interval: float = 5.0,
    reciprocal: bool = False,
    adaptive_memory: bool = True,
    between_eligibility: float = 0.05,
    within_eligibility: float = 1.0,
    a: float = 0.0,
    c: float = 1.0,
    tau_y: float = 1.0,
    sim_rng: Optional[np.random.Generator] = None,
    controller_rng: Optional[np.random.Generator] = None,
    recorder: Optional[RolloutRecorder] = None,
    checkpoint_at: Optional[float] = None,
    stop_at_checkpoint: bool = False,
    checkpoint_file: Optional[Union[str, Path]] = None,
    on_checkpoint: Optional[Callable[[Any], None]] = None,
    resume_checkpoint: Optional[Union[Any, Dict[str, Any], str, Path]] = None,
    record_target_transitions: bool = False,
) -> Tuple[RolloutResult, State, BudgetLedger]:
    """Execute a single closed-loop controlled simulation rollout.

    Defaults match the principal profile (prestige_global=0.0, matching=1.0,
    tau_public=1.0); pass values explicitly when the declared profile differs. `a` and `c` are forwarded to MemoryManager (encounter
    kernel prestige/matching biases), `tau_y` to the PoissonScheduler public channel.

    Supports mid-rollout checkpointing (checkpoint_at, stop_at_checkpoint, checkpoint_file)
    and exact bit-for-bit resumption from checkpoint (resume_checkpoint).

    Returns:
        (rollout_result, final_state, ledger)
    """
    from src.control.checkpointing import (
        ControlledRolloutCheckpoint,
        capture_rollout_checkpoint,
        load_controlled_checkpoint,
        reconstruct_engine_callbacks,
        restore_rollout_state,
        save_controlled_checkpoint,
    )

    physics = {
        "J": float(J),
        "T": float(T),
        "delta": float(delta),
        "epsilon": float(epsilon),
        "closure": float(closure),
        "sample_interval": float(sample_interval),
        "reciprocal": bool(reciprocal),
        "adaptive_memory": bool(adaptive_memory),
        "between_eligibility": float(between_eligibility),
        "within_eligibility": float(within_eligibility),
        "a": float(a),
        "c": float(c),
        "tau_y": float(tau_y),
    }

    if resume_checkpoint is not None:
        if isinstance(resume_checkpoint, (str, Path)):
            chk = load_controlled_checkpoint(resume_checkpoint, expected_problem=problem)
        elif isinstance(resume_checkpoint, dict):
            chk = ControlledRolloutCheckpoint.from_dict(resume_checkpoint)
        elif isinstance(resume_checkpoint, ControlledRolloutCheckpoint):
            chk = resume_checkpoint
        else:
            raise TypeError(f"Invalid resume_checkpoint type: {type(resume_checkpoint)}")

        (
            state,
            memory_mgr,
            scheduler,
            ledger,
            observation_model,
            controller,
            target_evaluator,
            recorder,
            sim_rng,
            controller_rng,
            pending_observations,
            scheduled_holding_deadlines,
        ) = restore_rollout_state(chk, problem=problem)

        if problem is None:
            # Reconstruct problem from checkpoint
            p_dict = chk.problem_dict
            problem = ControlProblem(
                problem_id=p_dict["problem_id"],
                target_type=TargetType(p_dict["target_type"]),
                target_params=p_dict.get("target_params", {}),
                N=int(p_dict["N"]),
                D=int(p_dict["D"]),
                k=int(p_dict["k"]),
                exposure_cap=float(p_dict["exposure_cap"]),
                amplitude_cap=float(p_dict["amplitude_cap"]),
                release_deadline=float(p_dict["release_deadline"]),
                holding_duration=float(p_dict["holding_duration"]),
                horizon=float(p_dict["horizon"]),
                accessible_roster=tuple(p_dict["accessible_roster"]) if p_dict.get("accessible_roster") else None,
                config_version=p_dict.get("config_version", "control-campaign-m4-1.0"),
            )

        reconstruct_engine_callbacks(
            scheduler=scheduler,
            state=state,
            memory_mgr=memory_mgr,
            ledger=ledger,
            target_evaluator=target_evaluator,
            observation_model=observation_model,
            controller=controller,
            problem=problem,
            physics=physics,
            sim_rng=sim_rng,
            pending_observations=pending_observations,
            scheduled_holding_deadlines=scheduled_holding_deadlines,
            recorder=recorder,
        )
    else:
        if initial_state is None:
            raise ValueError("initial_state must be provided when resume_checkpoint is None")
        if problem is None:
            raise ValueError("problem must be provided when resume_checkpoint is None")

        state = initial_state.copy()
        N, D = state.N, state.D

        if controller is None:
            controller = ZeroController()

        # Simulator RNG stream
        if sim_rng is None:
            sim_rng = np.random.default_rng(problem.search_seed)

        # Controller RNG stream (strictly independent of simulator RNG)
        if controller_rng is None:
            controller_rng = np.random.default_rng(problem.proposal_seed)

        # 1. Initialize Budget Ledger
        accessible_set = set(problem.accessible_roster) if problem.accessible_roster else None
        ledger = BudgetLedger(
            N=N,
            D=D,
            k=problem.k,
            exposure_cap=problem.exposure_cap,
            amplitude_cap=problem.amplitude_cap,
            release_deadline=problem.release_deadline,
            accessible_roster=accessible_set,
            structural_budget=problem.structural_budget,
            max_structural_per_step=problem.max_structural_per_step,
        )
        from src.control.advanced.structural import StructuralActuator
        structural_actuator = StructuralActuator(rng=sim_rng)

        # 2. Initialize Observation Model
        if observation_model is None:
            observation_model = SampledPublicObservationModel(
                N=N,
                panel_fraction=0.5,
                accessible_roster=problem.accessible_roster,
                group_labels=state.opportunity_groups,
                rng=controller_rng,
            )

        # 3. Initialize Target Evaluator
        if target_evaluator is None:
            target_evaluator = TargetEvaluator.create_for_state(
                target_type=problem.target_type,
                initial_state=state,
                holding_duration=problem.holding_duration,
                horizon=problem.horizon,
                release_deadline=problem.release_deadline,
                problem_id=problem.problem_id,
                target_params=problem.target_params,
            )

        # 4. Initialize Memory Manager (a/c forwarded explicitly; no silent defaults)
        memory_mgr = MemoryManager(
            N=N,
            reciprocal=reciprocal,
            epsilon=epsilon,
            a=a,
            c=c,
            chi=closure,
            initial_W=state.W,
            capacity=state.row_caps,
            opportunity_groups=state.opportunity_groups,
            C_matrix=state.opportunity_matrix,
            between_eligibility=between_eligibility,
            within_eligibility=within_eligibility,
            rng=sim_rng,
        )

        # 5. Initialize Poisson Jump Scheduler
        tau_x_param = state.tau_x if state.tau_x is not None else 5.0
        nu_param = state.nu if state.nu is not None else 2.0
        if not adaptive_memory:
            nu_param = 0.0
        scheduler_delta = delta if adaptive_memory else 0.0
        scheduler = PoissonScheduler(
            N=N,
            D=D,
            tau_x=tau_x_param,
            tau_y=tau_y,
            nu=nu_param,
            delta=scheduler_delta,
            rng=sim_rng,
        )
        scheduler.set_total_memory(int(memory_mgr.total_memory))

        # Heterogeneity traits
        K_arr = state.K if state.K is not None else np.full(N, 0.5)
        g_arr = state.g if state.g is not None else np.ones(N)
        b_arr = state.b if state.b is not None else np.zeros((N, D))

        scheduled_holding_deadlines: set[float] = set()

        def check_and_schedule_holding_deadline(current_t: float) -> None:
            if (
                ledger.is_released
                and target_evaluator.t_start_hold is not None
                and not target_evaluator.is_success
            ):
                t_comp = target_evaluator.t_start_hold + problem.holding_duration
                if (
                    t_comp <= problem.horizon
                    and t_comp not in scheduled_holding_deadlines
                    and t_comp > current_t - 1e-12
                ):
                    scheduled_holding_deadlines.add(t_comp)
                    scheduler.schedule_deterministic_event(
                        time=t_comp,
                        priority=PRIORITY_HOLDING,
                        event_type="holding_deadline",
                        callback=lambda s, e: (
                            target_evaluator.process_holding_check(s.time, ledger.is_released),
                            check_and_schedule_holding_deadline(s.time),
                        ),
                    )

        # Callbacks for stochastic jump channels
        B_mat: Optional[np.ndarray] = None
        if D == 3:
            try:
                from src.core.couplings import get_coupling_matrix
                B_mat = get_coupling_matrix("B+", D=3)
            except Exception:
                B_mat = None
        L_int = 1.0

        def cb_private(node_idx: int) -> None:
            ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            i = node_idx
            # Spec: an event changes exactly ONE component; channel rate
            # D * sum_i 1/tau_x,i then yields per-component rate 1/tau_x,i.
            alpha = int(sim_rng.integers(0, D))
            # F^x_{i, alpha} = K_i * y_{i, alpha} + L * sum_{beta != alpha} B_{alpha, beta} x_{i, beta} + b_{i, alpha}
            f_x = float(K_arr[i] * state.y[i, alpha] + b_arr[i, alpha])
            if B_mat is not None and D > 1:
                f_x += L_int * float(np.dot(B_mat[alpha], state.x[i, :]))
            kappa = f_x / T
            if state.state_mode == StateMode.BINARY:
                new_x = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
            else:
                new_x = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))
            state.x[i, alpha] = new_x
            target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(scheduler.time)

        def cb_public(node_idx: int) -> None:
            ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            i = node_idx
            W_row = memory_mgr.W[i, :]
            alpha = int(sim_rng.integers(0, D))
            u_val = float(ledger.current_field[i, alpha])
            f_y = compute_single_public_field(
                W_row=W_row,
                y_col=state.y[:, alpha],
                x_val=float(state.x[i, alpha]),
                J=J,
                K=float(K_arr[i]),
                h_val=0.0,
                g_val=float(g_arr[i]),
                u_val=u_val,
            )
            kappa = f_y / T
            if state.state_mode == StateMode.BINARY:
                new_y = float(sample_binary_glauber_scalar(kappa, rng=sim_rng))
            else:
                new_y = float(sample_continuous_inverse_cdf_scalar(kappa, sim_rng.random()))
            state.y[i, alpha] = new_y
            target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(scheduler.time)

        def cb_encounter(initiator_idx: int) -> None:
            ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            res = memory_mgr.sample_and_apply_encounter(
                i=initiator_idx,
                y=state.y,
                rng=sim_rng,
            )
            if not res.is_null:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W
                target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
                if recorder is not None:
                    recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
                check_and_schedule_holding_deadline(scheduler.time)

        def cb_decay() -> None:
            ledger.advance_to(scheduler.time)
            state.t = scheduler.time
            i_dec, j_dec = memory_mgr.sample_and_apply_decay(rng=sim_rng)
            if i_dec != -1:
                scheduler.set_total_memory(int(memory_mgr.total_memory))
                state.W = memory_mgr.W
                target_evaluator.process_state_update(state, scheduler.time, ledger.is_released)
                if recorder is not None:
                    recorder.on_event_state(state, scheduler.time, ledger, target_evaluator)
                check_and_schedule_holding_deadline(scheduler.time)

        scheduler.register_callbacks(
            cb_private=cb_private,
            cb_public=cb_public,
            cb_encounter=cb_encounter,
            cb_decay=cb_decay,
        )

        # 6. Reset controller with initial state observation and handle time-zero actuation
        initial_obs = observation_model.sample_observation(
            state=state,
            current_time=0.0,
            remaining_exposure=ledger.remaining_exposure,
            remaining_distinct_nodes=ledger.remaining_distinct_nodes,
            current_spending_rate=0.0,
        )
        controller.reset(problem, initial_obs, rng=controller_rng)

        # Time-zero actuation (e.g. ZeroController immediate release or zero delivery delay)
        if isinstance(controller, ZeroController):
            initial_act = controller.act(initial_obs, ledger)
            eff_end = ledger.apply_action(initial_act, current_time=0.0)
            if recorder is not None:
                recorder.record_action(
                    action=initial_act,
                    decision_time=0.0,
                    report_sample_time=0.0,
                    effective_end=eff_end,
                    ledger=ledger,
                    reason="zero_controller_time_zero",
                )
            if initial_act.structural_entries:
                structural_actuator.apply_action(
                    action=initial_act,
                    state=state,
                    memory_mgr=memory_mgr,
                    scheduler=scheduler,
                    ledger=ledger,
                )
        elif initial_obs.delivery_time <= 0.0:
            initial_act = controller.act(initial_obs, ledger)
            eff_end = ledger.apply_action(initial_act, current_time=0.0)
            if recorder is not None:
                recorder.record_action(
                    action=initial_act,
                    decision_time=0.0,
                    report_sample_time=initial_obs.sample_time if hasattr(initial_obs, "sample_time") else 0.0,
                    effective_end=eff_end,
                    ledger=ledger,
                    reason="time_zero_delivery",
                )
            if initial_act.structural_entries:
                structural_actuator.apply_action(
                    action=initial_act,
                    state=state,
                    memory_mgr=memory_mgr,
                    scheduler=scheduler,
                    ledger=ledger,
                )
            if eff_end is not None and eff_end > 0.0 and eff_end <= problem.horizon:
                def _shutoff_t0(s: PoissonScheduler, e: Any) -> None:
                    ledger.advance_to(s.time)
                    if recorder is not None:
                        recorder.record_shutoff(s.time, ledger, reason="field_shutoff")
                    target_evaluator.process_state_update(state, s.time, ledger.is_released)
                    if recorder is not None:
                        recorder.on_event_state(state, s.time, ledger, target_evaluator)
                    check_and_schedule_holding_deadline(s.time)

                scheduler.schedule_deterministic_event(
                    time=eff_end,
                    priority=PRIORITY_RESTORE,
                    event_type="field_shutoff",
                    callback=_shutoff_t0,
                )

        # 7. Schedule Observation Snapshots and Deliveries
        # If initial observation was already delivered at t=0.0, periodic snapshots begin at sample_interval
        start_snap = sample_interval if initial_obs.delivery_time <= 0.0 else 0.0
        snapshot_times = np.arange(start_snap, problem.release_deadline, sample_interval)

        # We store captured observations for delivery
        pending_observations: Dict[float, Any] = {}

        def make_snapshot_callback(t_snap: float):
            def _cb(sched: PoissonScheduler, evt: Any = None) -> None:
                ledger.advance_to(sched.time)
                obs = observation_model.sample_observation(
                    state=state,
                    current_time=sched.time,
                    remaining_exposure=ledger.remaining_exposure,
                    remaining_distinct_nodes=ledger.remaining_distinct_nodes,
                    current_spending_rate=ledger.current_spending_rate,
                )
                delivery_t = obs.delivery_time
                pending_observations[delivery_t] = obs

                # Schedule delivery event if within horizon
                if delivery_t <= problem.horizon:
                    sched.schedule_deterministic_event(
                        time=delivery_t,
                        priority=PRIORITY_INTERVENTION,
                        event_type="observation_delivery",
                        payload={"delivery_time": delivery_t},
                        callback=make_delivery_callback(delivery_t),
                    )
            return _cb

        def make_delivery_callback(t_del: float):
            def _cb(sched: PoissonScheduler, evt: Any) -> None:
                ledger.advance_to(sched.time)
                obs = pending_observations.pop(t_del, None)
                if obs is not None and not ledger.is_released and sched.time < problem.release_deadline:
                    action = controller.act(obs, ledger)
                    eff_end = ledger.apply_action(action, current_time=sched.time)
                    if recorder is not None:
                        recorder.record_action(
                            action=action,
                            decision_time=sched.time,
                            report_sample_time=obs.sample_time if hasattr(obs, "sample_time") else sched.time,
                            effective_end=eff_end,
                            ledger=ledger,
                            reason="observation_delivery",
                        )
                    if action.structural_entries:
                        structural_actuator.apply_action(
                            action=action,
                            state=state,
                            memory_mgr=memory_mgr,
                            scheduler=sched,
                            ledger=ledger,
                        )
                    if eff_end is not None and eff_end > sched.time and eff_end <= problem.horizon:
                        def _shutoff_delivery(s: PoissonScheduler, e: Any) -> None:
                            ledger.advance_to(s.time)
                            if recorder is not None:
                                recorder.record_shutoff(s.time, ledger, reason="field_shutoff")
                            target_evaluator.process_state_update(state, s.time, ledger.is_released)
                            if recorder is not None:
                                recorder.on_event_state(state, s.time, ledger, target_evaluator)
                            check_and_schedule_holding_deadline(s.time)

                        sched.schedule_deterministic_event(
                            time=eff_end,
                            priority=PRIORITY_RESTORE,
                            event_type="field_shutoff",
                            callback=_shutoff_delivery,
                        )
                    target_evaluator.process_state_update(state, sched.time, ledger.is_released)
                    if recorder is not None:
                        recorder.on_event_state(state, sched.time, ledger, target_evaluator)
                    check_and_schedule_holding_deadline(sched.time)
            return _cb

        # Schedule initial snapshots
        for t_s in snapshot_times:
            scheduler.schedule_deterministic_event(
                time=float(t_s),
                priority=PRIORITY_SNAPSHOT,
                event_type="observation_snapshot",
                callback=make_snapshot_callback(float(t_s)),
            )

        def on_release_deadline(s: PoissonScheduler, e: Any) -> None:
            ledger.release_control(s.time)
            if recorder is not None:
                recorder.record_shutoff(s.time, ledger, reason="release_deadline")
            target_evaluator.process_state_update(state, s.time, ledger.is_released)
            if recorder is not None:
                recorder.on_event_state(state, s.time, ledger, target_evaluator)
            check_and_schedule_holding_deadline(s.time)

        # Schedule mandatory release deadline shutoff
        scheduler.schedule_deterministic_event(
            time=float(problem.release_deadline),
            priority=PRIORITY_RESTORE,
            event_type="release_deadline",
            callback=on_release_deadline,
        )

        # Schedule periodic holding check events (every 1.0 time unit after release)
        # as fallback to guarantee hold completion even if no stochastic events fire
        holding_check_times = np.arange(problem.release_deadline, problem.horizon + 1.0, 1.0)
        for t_h in holding_check_times:
            scheduler.schedule_deterministic_event(
                time=float(t_h),
                priority=PRIORITY_HOLDING,
                event_type="holding_check",
                callback=lambda s, e: (
                    target_evaluator.process_holding_check(s.time, ledger.is_released),
                    check_and_schedule_holding_deadline(s.time),
                ),
            )

        # Schedule horizon event
        scheduler.schedule_deterministic_event(
            time=float(problem.horizon),
            priority=PRIORITY_HORIZON,
            event_type="horizon",
            callback=lambda s, e: None,
        )

        # Evaluate initial state after time-zero action is installed
        target_evaluator.process_state_update(state, 0.0, ledger.is_released)
        check_and_schedule_holding_deadline(0.0)
        if recorder is not None:
            recorder.initialize(state=state, ledger=ledger, target_evaluator=target_evaluator, problem=problem)

    # Main event loop with exact mid-rollout checkpointing support
    target_stop_time = float(problem.horizon)
    checkpoint_pending = (checkpoint_at is not None and checkpoint_at < problem.horizon)
    captured_checkpoint = None

    while scheduler.time < target_stop_time:
        next_event_time = scheduler.peek_next_event_time()

        # Check if checkpoint should be taken before the next event
        if checkpoint_pending and checkpoint_at is not None:
            if next_event_time is None or next_event_time > checkpoint_at:
                scheduler.time = float(checkpoint_at)
                ledger.advance_to(checkpoint_at)
                state.t = float(checkpoint_at)
                if recorder is not None:
                    recorder.advance_cadence_sampling(checkpoint_at, state, ledger, target_evaluator, problem)

                captured_checkpoint = capture_rollout_checkpoint(
                    current_time=checkpoint_at,
                    problem=problem,
                    physics=physics,
                    state=state,
                    memory_mgr=memory_mgr,
                    scheduler=scheduler,
                    ledger=ledger,
                    observation_model=observation_model,
                    pending_observations=pending_observations,
                    controller=controller,
                    target_evaluator=target_evaluator,
                    sim_rng=sim_rng,
                    controller_rng=controller_rng,
                    scheduled_holding_deadlines=scheduled_holding_deadlines,
                    recorder=recorder,
                )
                if on_checkpoint is not None:
                    on_checkpoint(captured_checkpoint)
                if checkpoint_file is not None:
                    save_controlled_checkpoint(captured_checkpoint, checkpoint_file)

                checkpoint_pending = False
                if stop_at_checkpoint:
                    break

        if next_event_time is None or next_event_time > problem.horizon:
            scheduler.time = problem.horizon
            break

        if recorder is not None:
            t_target = min(next_event_time, problem.horizon)
            recorder.advance_cadence_sampling(t_target, state, ledger, target_evaluator, problem)

        advanced = scheduler.step()
        if not advanced:
            break

        # Check if checkpoint_at was reached exactly by the event that just stepped
        if checkpoint_pending and checkpoint_at is not None and scheduler.time >= checkpoint_at:
            captured_checkpoint = capture_rollout_checkpoint(
                current_time=scheduler.time,
                problem=problem,
                physics=physics,
                state=state,
                memory_mgr=memory_mgr,
                scheduler=scheduler,
                ledger=ledger,
                observation_model=observation_model,
                pending_observations=pending_observations,
                controller=controller,
                target_evaluator=target_evaluator,
                sim_rng=sim_rng,
                controller_rng=controller_rng,
                scheduled_holding_deadlines=scheduled_holding_deadlines,
                recorder=recorder,
            )
            if on_checkpoint is not None:
                on_checkpoint(captured_checkpoint)
            if checkpoint_file is not None:
                save_controlled_checkpoint(captured_checkpoint, checkpoint_file)

            checkpoint_pending = False
            if stop_at_checkpoint:
                break

    cur_end_time = scheduler.time if (stop_at_checkpoint and scheduler.time < problem.horizon) else problem.horizon
    ledger.advance_to(cur_end_time)
    state.t = cur_end_time
    state.W = memory_mgr.W

    if recorder is not None:
        recorder.finalize(state=state, ledger=ledger, target_evaluator=target_evaluator, problem=problem)

    result = target_evaluator.finalize(
        state=state,
        current_time=cur_end_time,
        expenditure=ledger.integrated_exposure,
        distinct_nodes_used=len(ledger.distinct_nodes_actuated),
        event_counts={
            "stochastic_events": scheduler.stats.total_stochastic_events,
            "private_updates": scheduler.stats.private_updates,
            "public_updates": scheduler.stats.public_updates,
            "encounters": scheduler.stats.encounters,
            "memory_decays": scheduler.stats.memory_decays,
            "deterministic_events": scheduler.stats.deterministic_events,
        },
    )
    result.structural_modifications = int(ledger.integrated_structural_modifications)
    if hasattr(controller, "fallback_records"):
        result.summaries["fallback_records"] = list(controller.fallback_records)
    if hasattr(controller, "feedback_opportunities_encountered"):
        result.summaries["feedback_opportunities"] = int(controller.feedback_opportunities_encountered)
    if hasattr(controller, "feedback_updates_performed"):
        result.summaries["feedback_updates"] = int(controller.feedback_updates_performed)
    if hasattr(controller, "requested_settings"):
        result.summaries["requested_settings"] = dict(controller.requested_settings)
    if hasattr(controller, "effective_settings"):
        result.summaries["effective_settings"] = dict(controller.effective_settings)
    if record_target_transitions:
        transitions = []
        for t, condition in target_evaluator.history_events:
            if not transitions or condition != transitions[-1][1]:
                transitions.append([float(t), bool(condition)])
        result.summaries["target_condition_transitions"] = transitions
        result.summaries["actual_release_time"] = ledger.release_time
    if recorder is not None:
        result.recorder = recorder
        result.summaries["recorder"] = recorder.to_dict()

    if captured_checkpoint is not None:
        result.checkpoint = captured_checkpoint
        result.summaries["checkpoint_time"] = captured_checkpoint.checkpoint_time
    result.is_checkpoint_interrupted = bool(stop_at_checkpoint and scheduler.time < problem.horizon)

    return result, state, ledger
