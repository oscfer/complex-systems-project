"""Explicit prior-based forecasts for initial recipient selection.

Uses the real simulator/evaluator on *estimated* worlds. This is not a calibrated
posterior: missing edges use a uniform completion prior, private opinions use a
declared public/private agreement prior, and missing public opinions use a
smoothed panel frequency. These assumptions require predictive validation.
Only the initial t=0 report is supported; this is not a feedback state filter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import numpy as np

from src.control.contracts import ControlProblem, Observation, TargetType
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import CompletePublicObservationModel
from src.control.optimizers.search import PolicyCandidate
from src.core.state import State, StateMode


@dataclass(frozen=True)
class ForecastAssumptions:
    # Explicit forecast parameters, not inferred properties of the real origin.
    row_memory: int = 40
    public_error_probability: float = 0.05
    private_public_agreement: float = 0.8
    K: float = 0.5
    tau_private: float = 5.0
    encounter_rate: float = 2.0
    J: float = 1.0
    temperature: float = 0.3
    decay: float = 0.1
    exploration: float = 0.1
    closure: float = 0.3
    between_eligibility: float = 0.05
    within_eligibility: float = 1.0

    def __post_init__(self):
        if not isinstance(self.row_memory, int) or self.row_memory < 0:
            raise ValueError('row_memory must be a nonnegative integer')
        if any(not np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Forecast parameters must be finite')
        if not 0 <= self.public_error_probability < 0.5 or not 0 <= self.private_public_agreement <= 1:
            raise ValueError('Invalid opinion priors')
        if self.tau_private <= 0 or self.temperature <= 0 or min(self.K,self.J,self.encounter_rate,self.decay) < 0:
            raise ValueError('Invalid forecast dynamics')
        if any(not 0 <= v <= 1 for v in (self.exploration,self.closure,self.between_eligibility,self.within_eligibility)):
            raise ValueError('Invalid forecast probabilities')


class ObservedInitialForecasts:
    """Callable evaluator for search_observed_subsets, with no real-state input.

    Same seed means the same sampled world and simulation stream for each group.
    Explicit supplied direction is a controller/task input; never derive it here
    from the guessed world's majority. Runtime budgets count every forecast as a
    simulation. Independent forecast validation does not replace real evaluation.
    """
    def __init__(self, observation: Observation, problem: ControlProblem, *,
                 target_sign: float, assumptions: ForecastAssumptions):
        if observation.is_oracle or observation.private_states is not None or observation.true_memory is not None:
            raise ValueError('Hidden state is not permitted in restricted forecasts')
        if observation.sample_time != 0 or observation.time < 0:
            raise ValueError('Only an initial t=0 report is supported')
        if problem.target_type != TargetType.REVERSAL or problem.D != 1 or target_sign not in (-1,1):
            raise ValueError('Only a declared binary one-topic direction is supported')
        if observation.public_states.shape != (problem.N,1) or observation.panel_mask.shape != (problem.N,):
            raise ValueError('Observation dimensions differ from the task')
        if not 0 <= observation.time < problem.release_deadline:
            raise ValueError('No treatment time remains')
        if set(observation.accessible_roster) != set(problem.accessible_roster):
            raise ValueError('Observation and task permissions differ')
        self.observation=copy.deepcopy(observation)
        self.assumptions=assumptions
        self.target_sign=float(target_sign)
        # Forecast target stays fixed across guessed worlds, including blue ones.
        self.problem=replace(problem, target_params={**problem.target_params,
            'eligibility_mode':'declared','declared_destination_sign':target_sign}, model_hash='')

    def estimate_world(self, seed: int) -> State:
        o=self.observation; p=self.problem; a=self.assumptions
        rng=np.random.default_rng(np.random.SeedSequence([seed,0]))
        observed=o.panel_mask & np.isfinite(o.public_states[:,0])
        reported=o.public_states[observed,0]
        if np.any(~np.isin(reported,[-1,1])):
            raise ValueError('Forecast requires binary public reports')
        # Laplace smoothing prevents a small unanimous panel implying certainty.
        q=(np.count_nonzero(reported < 0)+1)/(len(reported)+2)
        y=np.where(rng.random(p.N)<q,-1.,1.)
        y[observed]=reported
        flips=rng.random(p.N)<a.public_error_probability
        y[observed & flips]*=-1
        x=y*np.where(rng.random(p.N)<a.private_public_agreement,1.,-1.)
        W=np.zeros((p.N,p.N),dtype=np.uint32)
        for (i,j),count in o.observed_edges.items():
            if not (0<=i<p.N and 0<=j<p.N) or count<0 or int(count)!=count:
                raise ValueError('Invalid revealed interaction count')
            if i!=j:
                W[i,j]=count
        # Revealed counts remain exact. A missing tie is unknown, not zero.
        for i in range(p.N):
            residual=max(0,a.row_memory-int(W[i].sum()))
            possible=[j for j in range(p.N) if j!=i and (i,j) not in o.observed_edges]
            if residual and possible:
                W[i,possible]+=rng.multinomial(residual,np.full(len(possible),1/len(possible))).astype(np.uint32)
        caps=np.maximum(W.sum(axis=1),a.row_memory).astype(np.int64)
        return State(N=p.N,D=1,state_mode=StateMode.BINARY,x=x[:,None],y=y[:,None],W=W,
            n=np.zeros(p.N,dtype=np.uint8),Q=np.zeros((p.N,1,2)),t=0.,
            K=np.full(p.N,a.K),g=np.ones(p.N),b=np.zeros((p.N,1)),
            tau_x=np.full(p.N,a.tau_private),nu=np.full(p.N,a.encounter_rate),row_caps=caps,
            opportunity_groups=None if o.group_labels is None else o.group_labels.copy())

    def __call__(self, candidate: PolicyCandidate, seeds: tuple[int,...]):
        p=self.problem; a=self.assumptions; o=self.observation
        nodes=candidate.target_nodes
        if len(nodes)>p.k or not set(nodes).issubset(p.accessible_roster):
            raise ValueError('Candidate exceeds permissions')
        if candidate.component!=0 or any(sign!=self.target_sign for _,sign in candidate.signs):
            raise ValueError('Candidate direction/component differs from the task')
        duration=min(candidate.duration,p.release_deadline-o.time)
        if candidate.nominal_amplitude>p.amplitude_cap or candidate.nominal_amplitude*len(nodes)*duration>p.exposure_cap+1e-9:
            raise ValueError('Candidate would be clipped; compare feasible schedules explicitly')
        # OpenLoopPulseController reserves its stated duration when clipping.
        if candidate.duration>p.release_deadline-o.time+1e-9:
            raise ValueError('Candidate duration must fit the available treatment window')
        results=[]
        for seed in seeds:
            world=self.estimate_world(seed)
            evaluator=TargetEvaluator.create_for_state(target_type=p.target_type,initial_state=world,
                holding_duration=p.holding_duration,horizon=p.horizon,release_deadline=p.release_deadline,
                problem_id=p.problem_id,policy_id=candidate.candidate_id,origin_id='estimated_world',
                future_id=seed,target_params=p.target_params)
            result,_,_=run_controlled_simulation(initial_state=world,problem=p,
                controller=candidate.to_controller(),target_evaluator=evaluator,
                observation_model=CompletePublicObservationModel(p.N,accessible_roster=p.accessible_roster,delivery_delay=o.time),
                sim_rng=np.random.default_rng(np.random.SeedSequence([seed,1])),
                controller_rng=np.random.default_rng(np.random.SeedSequence([seed,2])),
                J=a.J,T=a.temperature,delta=a.decay,epsilon=a.exploration,closure=a.closure,
                between_eligibility=a.between_eligibility,within_eligibility=a.within_eligibility,
                adaptive_memory=True,record_target_transitions=True)
            result.summaries['forecast_assumptions']=asdict(a)
            result.summaries['evidence_kind']='forecast_only'
            results.append(result)
        return results
