"""Execution adapter for the t200/t500 sequential experiment; old runners unchanged."""
from dataclasses import asdict
import time
import numpy as np
from src.control.contracts import ControlProblem,TargetType
from src.control.controllers import ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import SampledPublicObservationModel
from src.control.recording import RecorderConfig
from src.control.nucleation_recording import NucleationRecorder
from src.control.checkpointing import serialize_state
from src.control.optimizers.campaign import _origin_state,_role_seed
from src.control.optimizers.sequential_low_power import SequentialLowPowerController,seed_for


class SequentialRecorder(NucleationRecorder):
    def __init__(self,config,*,frame_interval=5.,J=1.,T=.3):
        super().__init__(config,{},J=J,T=T);self.frame_interval=frame_interval;self.action_contexts=[];self.state=None;self._pending_contexts=[]

    def initialize(self,state,ledger,target_evaluator,problem):
        self.state=state;super().initialize(state,ledger,target_evaluator,problem)
        # The engine records t=0 actions before initializing the recorder.
        # Resolve their diagnostics now, before any stochastic event occurs.
        for decision_time,nodes in self._pending_contexts:self._record_context(decision_time,nodes)
        self._pending_contexts.clear()

    def _capture_network_sample(self,state,ledger,t,problem):
        if (self.config.record_network_frames and 0<t<problem.horizon and
            abs(t/self.frame_interval-round(t/self.frame_interval))<1e-10 and len(self.network_frames)<self.config.max_network_frames-1):
            self._record_network_frame(state,ledger,t=t)

    def on_event_state(self,state,current_time,ledger,target_evaluator):
        super().on_event_state(state,current_time,ledger,target_evaluator)
        # Cadence sampling precedes events at the same timestamp. Replace that
        # frame after each tied event so pulse endpoints display the final field.
        if self.network_frames and abs(self.network_frames[-1].t-current_time)<1e-10:
            self.network_frames.pop()
            self._record_network_frame(state,ledger,t=current_time)

    def record_action(self,**kwargs):
        super().record_action(**kwargs)
        context=(float(kwargs['decision_time']),sorted(kwargs['action'].actuated_nodes()))
        if self.state is None:self._pending_contexts.append(context)
        else:self._record_context(*context)

    def _record_context(self,decision_time,nodes):
        W=self.state.W.astype(float);tot=W.sum(1)[:,None];A=np.divide(W,tot,out=np.zeros_like(W),where=tot>0);power=A.sum(0)
        self.action_contexts.append(dict(time=decision_time,true_received_attention=power.tolist(),
            private_blue=int((self.state.x[:,0]<0).sum()),public_blue=int((self.state.y[:,0]<0).sum()),
            nodes=nodes))

    def to_dict(self):
        d=super().to_dict();d['sequential_context']=dict(action_contexts=self.action_contexts,
            meaning='Retrospective true influence and states at decisions; never exposed to the controller.');return d


def observation_model(profile,seed,spec):
    complete=spec['visibility']=='complete_public'
    return SampledPublicObservationModel(N=profile.N,accessible_roster=tuple(range(profile.N)),
        panel_fraction=1. if complete else profile.panel_fraction,
        binary_flip_prob=0. if complete else profile.binary_flip_prob,
        edge_reveal_prob=1. if complete else profile.edge_reveal_prob,
        delivery_delay=profile.delivery_delay_tau_public,rng=np.random.default_rng(_role_seed(seed,'observation')))


def run_unit(profile,plan,base,seed,future,spec,*,initial_state=None,recorder=None):
    state=_origin_state(profile,seed) if initial_state is None else initial_state.copy()
    problem=ControlProblem(problem_id=f'{base}_{spec["name"]}_{future}',target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1.,'threshold':.5},
        N=profile.N,D=1,k=profile.N,exposure_cap=spec['budget'],amplitude_cap=spec['amplitude'],
        accessible_roster=tuple(range(profile.N)),release_deadline=plan['release_deadline'],holding_duration=plan['hold'],horizon=plan['horizon'],config_version=plan['version'])
    if spec['kind']=='zero':controller=ZeroController()
    elif spec['kind']=='response':
        from src.control.optimizers.response_low_power import ResponseLowPowerController
        controller=ResponseLowPowerController(spec)
    else:controller=SequentialLowPowerController(spec)
    if recorder is None:
        config=RecorderConfig(cadence=plan['scalar_cadence'],record_network_frames=future in plan['frame_futures'],
            max_network_frames=plan['max_frames'],retention_window=(plan['release_deadline'],plan['horizon']))
        recorder=SequentialRecorder(config,frame_interval=plan['frame_interval'],J=profile.J,T=profile.temperature)
    seed_fn=seed_for
    if plan['version']=='response-low-power-v1':
        from src.control.optimizers.response_low_power import seed_for as seed_fn
    sim_seed=seed_fn(base,'future',future);control_seed=seed_fn(base,'controller',spec.get('controller_seed_group',spec['name']))
    evaluator=TargetEvaluator.create_for_state(TargetType.REVERSAL,state,plan['hold'],plan['horizon'],plan['release_deadline'],target_params=problem.target_params)
    begin=time.monotonic()
    result,final,ledger=run_controlled_simulation(initial_state=state,problem=problem,controller=controller,
        observation_model=observation_model(profile,seed,spec),recorder=recorder,target_evaluator=evaluator,
        record_target_transitions=True,adaptive_memory=spec['memory']=='adaptive',
        sim_rng=np.random.default_rng(sim_seed),controller_rng=np.random.default_rng(control_seed),
        J=profile.J,T=profile.temperature,delta=profile.delta,epsilon=profile.exploration,closure=profile.closure,
        a=profile.prestige_global,c=profile.matching,tau_y=profile.tau_public,sample_interval=profile.sample_interval_tau_public)
    return dict(origin_id=base,policy=spec['name'],policy_spec=spec,recipient_nodes=sorted(ledger.distinct_nodes_actuated),
        memory_mode=spec['memory'],visibility=spec['visibility'],is_valid=bool(result.is_valid),is_eligible=bool(result.is_eligible),
        success=bool(result.success),relapsed=bool(result.relapsed),success_through_horizon=bool(result.success and not result.relapsed),
        completion_time=result.completion_time,expenditure=result.expenditure,distinct_nodes_used=result.distinct_nodes_used,
        actual_release_time=ledger.release_time,controller_log=getattr(controller,'log',[]),target_definition=asdict(evaluator.defn),
        problem=asdict(problem),event_counts=dict(result.event_counts),summaries=dict(result.summaries),final_state=serialize_state(final),
        network_frames=[f.to_dict() for f in result.recorder.network_frames],
        elapsed_seconds_this_invocation=time.monotonic()-begin,seeds=dict(simulation=sim_seed,controller=control_seed,origin=seed))
