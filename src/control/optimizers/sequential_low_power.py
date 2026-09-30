"""Prospective intermittent low-power control; only delivered public information."""
from __future__ import annotations
import hashlib
from collections import deque
import numpy as np
from src.control.contracts import Action,ActionEntry
from src.control.controllers import Controller


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,('sequential-low-power-v1',*parts))).encode()).digest()[:4],'big')


def policy_spec(name,**updates):
    base=dict(name=name,kind='sequential',ranking='dynamic_low',slots=1,pulse=5.,pause=5.,
        low_quantile=.5,early_stop=True,visibility='restricted',memory='adaptive',budget=80.,amplitude=2.)
    base.update(updates);return base


def pilot_plan():
    families=[policy_spec('dynamic_low1'),policy_spec('dynamic_low2',slots=2),
        policy_spec('initial_low1',ranking='initial_low'),policy_spec('random1',ranking='random'),
        policy_spec('fallen_low1',ranking='fallen_low'),
        policy_spec('simultaneous_low4',kind='simultaneous',ranking='dynamic_low',slots=4),
        policy_spec('simultaneous_attention4',kind='simultaneous',ranking='attention',slots=4),
        policy_spec('simultaneous_coverage4',kind='simultaneous',ranking='coverage',slots=4)]
    policies={f'{s["name"]}_B{b}':dict(s,budget=float(b)) for b in (40,80,152) for s in families}
    extras=[policy_spec('no_pause',pause=0.),policy_spec('long_pulse',pulse=10.),
        policy_spec('strict_low',low_quantile=.25),policy_spec('no_early_stop',early_stop=False),
        policy_spec('fallen_only',ranking='fallen_only'),policy_spec('frozen_low',memory='frozen'),
        policy_spec('complete_low',visibility='complete_public')]
    policies.update({f'{s["name"]}_B80':s for s in extras})
    policies.update(zero=policy_spec('zero',kind='zero',budget=0.),
        zero_frozen=policy_spec('zero_frozen',kind='zero',budget=0.,memory='frozen'),
        zero_complete=policy_spec('zero_complete',kind='zero',budget=0.,visibility='complete_public'))
    return dict(version='sequential-low-power-v1',protocol_revision='h500-followup300',N=40,horizon=500.,release_deadline=200.,hold=100.,target_support=.75,
        origins=[f'sequential_{i:02d}' for i in range(8)],origin_seeds={f'sequential_{i:02d}':seed_for('origin',j) for i,j in enumerate([0,8,2,3,4,5,6,7])},
        initial_red_majority_required=True,origin_candidate_indices=[0,8,2,3,4,5,6,7],
        origin_selection='Strict initial red majority in both layers, before any trajectory. Candidate1 excluded (21/40 blue in each layer); next candidate8 replaces it (11/40 private,15/40 public blue). No outcome filtering.',
        timing_seed=seed_for('timing'),policies=policies,roles_per_block=len(policies),budgets=[40,80,152],
        maximum_futures=4,allocation_tiers=[[8,4],[8,3],[8,2],[6,2],[4,2]],timing_units=3,
        planning_seconds=21600.,session_seconds=28800.,export_reserve_seconds=1200.,timing_safety_factor=1.35,
        scalar_cadence=5.,frame_interval=5.,max_frames=101,frame_futures=[0],persistence_offsets=[100,200,300],
        bootstrap_draws=10000,bootstrap_seed=314159,
        primary=dict(a='dynamic_low1_B80',b='initial_low1_B80',metric='persistent'),
        information='Main arms: fixed half-node public panel, incomplete ties, delayed noisy reports. Complete-public diagnostic separately labeled. Private state never an input.',
        low_power_rule='Lowest smoothed positive reported received-attention scores within current bottom half (ties included), among observed non-confirmed-blue nodes. Three-report window; current positive evidence required. Unknown is not low.',
        former_power_rule='Previously top quartile, now low on two fresh reports and at least25% below that recorded high score. Decline refers to power, never a color change.',
        stop_rule='Two reports with at least80% public blue in observed panel, assessed after a treatment-free pause; optional diagnostic disables this. Final release is irrevocable.',
        cost_rule='Compare persistent success at matched budget, then actual integrated absolute field, distinct recipients, concurrency, time and power. Failures never disappear from cost summaries.',
        interpretation='Development hypotheses; no optimality/reliability claim. No outcome-selected allocation, retuning, extra work to fill hours, or nominal local runs.')


def choose_allocation(plan,timings,remaining):
    if len(timings)!=3 or any(not np.isfinite(x) or x<=0 for x in timings):raise ValueError('Three positive full-unit timings required')
    cost=max(timings)*plan['timing_safety_factor'];budget=min(plan['planning_seconds'],remaining-plan['export_reserve_seconds'])
    for n,f in plan['allocation_tiers']:
        ceiling=n*f*plan['roles_per_block']
        if cost*ceiling<=budget:return dict(origins=plan['origins'][:n],futures=f,units=ceiling,estimated_seconds=cost*ceiling,planning_budget_seconds=budget,seconds_per_unit=cost)
    return None


class SequentialLowPowerController(Controller):
    """Pulses replace fields; pauses never declare irreversible release.

    A currently powerful recipient loses eligibility at the next report; the
    policy does not secretly select on private alignment or true influence.
    """
    def __init__(self,spec):self.spec=dict(spec)

    def reset(self,problem,initial_observation,rng=None):
        self.problem=problem;self.rng=rng or np.random.default_rng(0)
        self.history=deque(maxlen=3);self.initial_scores=None;self.high_scores={};self.low_streak={};self.blue_streak={}
        self.last_sample=-np.inf;self.active=[];self.pulse_end=0.;self.next_start=0.;self.last_end=None
        self.public_stop_streak=0;self.log=[];self.started=False;self.fixed_nodes=[];self.fixed_end=None
        self.tie={int(n):i for i,n in enumerate(self.rng.permutation(problem.accessible_roster))}

    def _observe(self,obs):
        if obs.is_oracle or obs.private_states is not None or obs.true_memory is not None:raise ValueError('Restricted controller rejects hidden/oracle payloads')
        sample=float(obs.sample_time if obs.sample_time is not None else obs.time)
        if sample>obs.time+1e-10 or (obs.delivery_time is not None and obs.delivery_time>obs.time+1e-10):raise ValueError('Undelivered/future observation')
        if sample<=self.last_sample:return None
        self.last_sample=sample;N=obs.public_states.shape[0];W=np.zeros((N,N))
        for (i,j),v in obs.observed_edges.items():
            if not 0<=i<N or not 0<=j<N or not np.isfinite(v) or v<0:raise ValueError('Invalid reported edge')
            if i!=j:W[i,j]=v
        totals=W.sum(1);A=np.divide(W,totals[:,None],out=np.zeros_like(W),where=totals[:,None]>0);incoming=A.sum(0)
        present={int(i):float(incoming[i]) for i in obs.accessible_roster if obs.is_node_observed(i) and incoming[i]>0 and np.isfinite(obs.public_states[i,0])}
        self.history.append(present)
        scores={i:float(np.mean([h[i] for h in self.history if i in h])) for i in present}
        if self.initial_scores is None and scores:self.initial_scores=dict(scores)
        low_cut=float(np.quantile(list(scores.values()),self.spec['low_quantile'])) if scores else None
        high_cut=float(np.quantile(list(scores.values()),.75)) if scores else None
        low={i for i,v in scores.items() if v<=low_cut+1e-12} if scores else set()
        for i in set(self.low_streak)|set(scores):self.low_streak[i]=self.low_streak.get(i,0)+1 if i in low else 0
        for i,v in scores.items():
            if v>=high_cut-1e-12:self.high_scores[i]=max(v,self.high_scores.get(i,0.))
        observed=[i for i in obs.accessible_roster if obs.is_node_observed(i) and np.isfinite(obs.public_states[i,0])]
        for i in list(self.blue_streak):
            if i not in observed:self.blue_streak[i]=0
        for i in observed:self.blue_streak[i]=self.blue_streak.get(i,0)+1 if obs.public_states[i,0]<0 else 0
        fraction=float(np.mean([obs.public_states[i,0]<0 for i in observed])) if observed else None
        self.public_stop_streak=self.public_stop_streak+1 if fraction is not None and fraction>=.8 else 0
        fallen={i for i in low if self.low_streak[i]>=2 and i in self.high_scores and scores[i]<=.75*self.high_scores[i]}
        return dict(sample=sample,scores=scores,low=low,fallen=fallen,public_blue=fraction,A=A,
            observed=observed,currently_red={i for i in observed if obs.public_states[i,0]>0})

    def _rank(self,view,require_red=True):
        scores=view['scores'];mode=self.spec['ranking'];available=set(scores)
        if mode in ('dynamic_low','fallen_low','fallen_only'):available &= view['low']
        elif mode=='initial_low':
            initial=self.initial_scores or {};cut=np.quantile(list(initial.values()),self.spec['low_quantile']) if initial else -1
            available &= {i for i,v in initial.items() if v<=cut+1e-12}
        # Fresh red observation required to recruit; a single noisy blue does not
        # end an in-progress pulse until alignment is confirmed on two reports.
        if require_red:available &= view['currently_red']
        if mode=='fallen_only':available &= view['fallen']
        if mode=='random':return [int(x) for x in self.rng.permutation(sorted(available))]
        if mode=='coverage':
            out=[];covered=np.zeros(len(view['A']))
            while available:
                i=max(available,key=lambda j:(float(np.maximum(view['A'][:,j]-covered,0).sum()),-self.tie[j]));out.append(i);available.remove(i);covered=np.maximum(covered,view['A'][:,i])
            return out
        key_scores=self.initial_scores if mode=='initial_low' else scores
        return sorted(available,key=lambda i:((0 if i in view['fallen'] else 1) if mode=='fallen_low' else 0,
            -scores[i] if mode=='attention' else key_scores[i],self.tie[i]))

    def _record(self,t,view,ledger,nodes,reason,end=None,release=False):
        scores=view['scores'] if view else {};values=list(scores.values())
        self.log.append(dict(time=float(t),sample_time=view['sample'] if view else None,reason=reason,nodes=list(nodes),
            intended_end=end,release=release,spent_before=float(ledger.integrated_exposure),public_blue_estimate=view['public_blue'] if view else None,
            scores={str(i):v for i,v in scores.items()},low_nodes=sorted(view['low']) if view else [],fallen_nodes=sorted(view['fallen']) if view else [],
            ranks={str(i):float(sum(v<=scores[i]+1e-12 for v in values)/len(values)) for i in nodes if i in scores},
            high_power_history={str(i):v for i,v in self.high_scores.items()}))
        return Action(tuple(ActionEntry(i,0,-self.spec['amplitude']) for i in sorted(nodes)),intended_end_time=end if end is not None else t,release=release)

    def act(self,observation,ledger):
        t=float(observation.time);v=self._observe(observation)
        if ledger.is_released or t>=self.problem.release_deadline or ledger.remaining_exposure<=ledger.tol:
            return self._record(t,v,ledger,[],'final_release',release=True)
        if v is None:return self._record(t,None,ledger,self.active if t<self.pulse_end else [],'duplicate_report',self.pulse_end if t<self.pulse_end else t)
        if self.spec['kind']=='simultaneous':
            if self.started:
                if t>=self.fixed_end:return self._record(t,v,ledger,[],'simultaneous_complete',release=True)
                return self._record(t,v,ledger,self.fixed_nodes,'simultaneous_continue',self.fixed_end)
            ranked=self._rank(v,require_red=False)
            # Missing candidates is reported as no action; never replace unknown
            # influence by zero or silently widen the low-power set.
            if len(ranked)<self.spec['slots']:return self._record(t,v,ledger,[],'insufficient_initial_candidates',release=True)
            self.fixed_nodes=ranked[:self.spec['slots']];self.fixed_end=min(t+self.spec['budget']/(self.spec['amplitude']*len(self.fixed_nodes)),self.problem.release_deadline);self.started=True
            return self._record(t,v,ledger,self.fixed_nodes,'simultaneous_start',self.fixed_end)
        # Preserve an unexpired pulse without extending its original end.
        if t<self.pulse_end-1e-10:
            eligible=set(v['scores'])
            if self.spec['ranking'] in ('dynamic_low','fallen_low','fallen_only'):eligible &= v['low']
            self.active=[i for i in self.active if i in eligible and self.blue_streak.get(i,0)<2]
            if self.active:return self._record(t,v,ledger,self.active,'pulse_continue',self.pulse_end)
            self.last_end=t;self.pulse_end=t;self.next_start=t+self.spec['pause']
        if self.spec['early_stop'] and t>=(self.last_end if self.last_end is not None else 0.)+max(5.,self.spec['pause'])-1e-10 and self.public_stop_streak>=2:
            return self._record(t,v,ledger,[],'public_stop_confirmed',release=True)
        if t<self.next_start-1e-10:return self._record(t,v,ledger,[],'observe_pause')
        ranked=self._rank(v);nodes=ranked[:self.spec['slots']]
        if not nodes:return self._record(t,v,ledger,[],'no_eligible_red_nodes')
        self.active=nodes;self.pulse_end=min(t+self.spec['pulse'],self.problem.release_deadline,t+ledger.remaining_exposure/(self.spec['amplitude']*len(nodes)))
        self.last_end=self.pulse_end;self.next_start=self.pulse_end+self.spec['pause'];self.started=True
        return self._record(t,v,ledger,nodes,'pulse_start',self.pulse_end)
