"""Prospective response-guided, single-recipient control using public reports only."""
from __future__ import annotations
import hashlib
import numpy as np
from src.control.optimizers.sequential_low_power import SequentialLowPowerController, policy_spec


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, ('response-low-power-v1', *parts))).encode()).digest()[:4], 'big')


def response_spec(name, **updates):
    spec = policy_spec(name, kind='response', ranking='hybrid', pause=10., early_stop=True)
    spec.update(controller_seed_group='matched-response',wait_for_growth=True, restart=True, episode_budget=20., seed_rule='connected', core_rule='response')
    spec.update(updates)
    return spec


def final_plan():
    # Seeds are frozen by initial-state-only preflight, never trajectory outcomes.
    candidates = INITIAL_CANDIDATES
    families = [response_spec('response'), response_spec('weakest', ranking='weakest', wait_for_growth=False),
                response_spec('random_low', ranking='random_low', wait_for_growth=False),
                response_spec('reinforce', ranking='reinforce'), response_spec('expand', ranking='expand')]
    policies = {f'{s["name"]}_B{b}':dict(s, budget=float(b)) for b in (40, 80, 152) for s in families}
    extras = [response_spec('topology', ranking='topology', wait_for_growth=False),
              response_spec('no_growth_wait', wait_for_growth=False),
              response_spec('pulse10', pulse=10.), response_spec('pulse15', pulse=15.),
              response_spec('bottom_quarter', low_quantile=.25),
              response_spec('fallen_first', fallen_priority=True),
              response_spec('no_restart', restart=False), response_spec('weakest_seed', seed_rule='weakest'),
              response_spec('all_blue_core', core_rule='all_blue'),
              response_spec('frozen_response', memory='frozen'),
              response_spec('complete_response', visibility='complete_public'),
              response_spec('frozen_weakest', ranking='weakest', wait_for_growth=False, memory='frozen'),
              response_spec('complete_weakest', ranking='weakest', wait_for_growth=False, visibility='complete_public')]
    policies.update({f'{s["name"]}_B80':s for s in extras})
    policies.update(zero=response_spec('zero', kind='zero', budget=0.),
                    zero_frozen=response_spec('zero_frozen', kind='zero', budget=0., memory='frozen'),
                    zero_complete=response_spec('zero_complete', kind='zero', budget=0., visibility='complete_public'))
    contrasts = []
    for budget in (40,80,152):
        contrasts.extend(dict(a=f'response_B{budget}', b=f'{other}_B{budget}', hypothesis=other)
                         for other in ('weakest','random_low','reinforce','expand'))
        contrasts.append(dict(a=f'response_B{budget}', b='zero', hypothesis='net benefit over no treatment'))
    contrasts.extend(dict(a=f'{s["name"]}_B80', b='weakest_B80' if s['ranking']=='weakest' else 'response_B80', hypothesis=s['name']) for s in extras)
    contrasts.extend([dict(a='frozen_response_B80',b='frozen_weakest_B80',hypothesis='response advantage under frozen attention'),
                      dict(a='complete_response_B80',b='complete_weakest_B80',hypothesis='response advantage with complete public reports')])
    contrasts.extend([dict(a='response_B152',b='response_B80',hypothesis='larger budget'),
                      dict(a='response_B80',b='response_B40',hypothesis='larger budget')])
    return dict(version='response-low-power-v1', protocol_revision='final-prospective-v1',
        title='Response-guided weak-node targeting — final comparison', bundle_prefix='response_low_power_v1',
        N=40, horizon=500., release_deadline=200., hold=100., target_support=.75,
        origins=[f'response_{i:02d}' for i in range(8)],
        origin_seeds={f'response_{i:02d}':seed_for('origin',j) for i,j in enumerate(candidates)},
        origin_candidate_indices=candidates, initial_red_majority_required=True,
        origin_selection='First eight fresh candidate seeds with strict initial red majorities in both layers; initial-state-only screening.',
        timing_seed=seed_for('timing'), timing_origin='response_timing',
        timing_roles=['zero','response_B152','complete_response_B80'],
        policies=policies, roles_per_block=len(policies), budgets=[40,80,152],
        maximum_futures=4, allocation_tiers=[[8,4],[8,3],[8,2],[6,2],[4,2]], timing_units=3,
        planning_seconds=21600.,session_seconds=28800.,export_reserve_seconds=1200.,timing_safety_factor=1.35,
        scalar_cadence=5.,frame_interval=5.,max_frames=101,frame_futures=[0],persistence_offsets=[100,200,300],
        bootstrap_draws=10000,bootstrap_seed=271828,
        primary=dict(a='response_B80',b='weakest_B80',metric='persistent'), contrasts=contrasts,
        interactions=[dict(name='memory_by_response',a='frozen_response_B80',b='frozen_weakest_B80',c='response_B80',d='weakest_B80'),
                      dict(name='information_by_response',a='complete_response_B80',b='complete_weakest_B80',c='response_B80',d='weakest_B80')],
        plot_families=['response','weakest','random_low','reinforce','expand'],
        secondary_majority=dict(minimum_blue_nodes=21, hold=100., horizon=500., note='Predeclared secondary, never substituted for the 75% primary endpoint.'),
        low_power_rule='Current bottom half of positive, observed, three-report-smoothed received-attention estimates; quarter diagnostic separate. Fresh red report to recruit; ties included; missing is not weak.',
        information='Fixed half-node public panel, incomplete noisy ties/states, cadence5 delay1. No private states, true memory, evaluator outcomes or future draws enter decisions.',
        interpretation='All arms single recipient and currently weak. Diagnostic complete-public/frozen arms are mechanism tests. No global optimum, automatic retuning, or follow-up run.')


INITIAL_CANDIDATES = [0, 1, 2, 3, 4, 5, 7, 8]


class ResponseLowPowerController(SequentialLowPowerController):
    """Use observed response to choose the next weak recipient, never an oracle.

    An episode is a seed attempt, with a public baseline and observed response
    group. That group is an association after treatment, not a causal estimate.
    """
    def reset(self,problem,initial_observation,rng=None):
        super().reset(problem,initial_observation,rng)
        self.episode=0; self.episode_start_spend=0.; self.episode_nodes=set(); self.baseline_blue=set()
        self.core=set(); self.off_streak={}; self.off_reports=0; self.stall_reports=0
        self.attempted_seeds=set(); self.seed_node=None; self.last_view=None; self.candidates={}
        self.episode_reason='not_started'; self.growth=0

    def _feedback(self,v):
        # Delayed samples taken while a pulse was active cannot establish survival.
        if self.last_end is None or v['sample'] < self.last_end-1e-10:return
        self.off_reports += 1
        blue=set(v['observed'])-v['currently_red']
        for i in set(self.off_streak)|set(v['observed']):
            self.off_streak[i]=self.off_streak.get(i,0)+1 if i in blue else 0
        stable={i for i,n in self.off_streak.items() if n>=2}
        previous=set(self.core)
        if self.off_reports<2:
            self.core &= blue
            self.growth=0
            return
        if self.spec['core_rule']=='all_blue':
            self.core=stable
        else:
            self.core &= blue
            anchors=self.episode_nodes|self.core
            pool=(stable-self.baseline_blue)| (stable & self.episode_nodes)
            # Expansion follows observed listening to a treated/response node.
            for _ in range(len(pool)+1):
                added={i for i in pool if i in self.episode_nodes or any(v['A'][i,j]>0 for j in anchors)}
                if added<=anchors:break
                anchors |= added
            self.core=pool & anchors
        self.growth=len(self.core-previous)
        self.stall_reports=0 if self.growth else self.stall_reports+1

    def _new_episode(self,v,ledger):
        self.episode+=1; self.episode_start_spend=float(ledger.integrated_exposure)
        self.episode_nodes=set();self.core=set();self.off_streak={};self.off_reports=0;self.stall_reports=0
        self.baseline_blue=set(v['observed'])-v['currently_red'];self.seed_node=None
        self.episode_reason='seed' if self.episode==1 else 'restart_after_no_response'

    def _rank_response(self,v,ledger):
        eligible=v['low'] & v['currently_red'];mode=self.spec['ranking'];A=v['A'];self.candidates={}
        if not eligible:return []
        if self.episode==0:self._new_episode(v,ledger)
        if mode=='weakest':return sorted(eligible,key=lambda i:(v['scores'][i],self.tie[i]))
        if mode=='random_low':return [int(i) for i in self.rng.permutation(sorted(eligible))]
        if (self.episode_nodes and not self.core and self.off_reports>=2 and
                ledger.integrated_exposure-self.episode_start_spend>=self.spec['episode_budget']-1e-10):
            if not self.spec['restart']:
                self.episode_reason='unproductive_seed_wait';return []
            self._new_episode(v,ledger)
        if self.core and mode!='topology':
            linked={i for i in eligible if sum(A[i,j]+A[j,i] for j in self.core)>0}
            eligible=linked
            if not eligible:self.episode_reason='no_weak_neighbor';return []
        elif self.seed_node is not None and self.seed_node in eligible and mode!='topology':
            # One bounded retry reinforces a seed that has not held its change.
            eligible={self.seed_node}
        elif mode!='topology':
            untried=eligible-self.attempted_seeds
            if not untried:self.episode_reason='no_untried_weak_seed';return []
            eligible=untried
        for i in eligible:
            # A[j,i] means j listens to i. Reinforce existing blue members;
            # recruit nodes that listen to them; reach remaining observed red nodes.
            reinforce=sum(A[j,i] for j in self.core)
            receptivity=sum(A[i,j] for j in self.core)
            reach=sum(A[j,i] for j in v['currently_red'] if j!=i and j not in self.core)
            useful={'reinforce':reinforce+receptivity, 'expand':receptivity+reach,
                    'hybrid':reinforce+receptivity+reach,'topology':reach}.get(mode)
            if useful is None:raise ValueError('Unknown response ranking')
            if not self.core:useful=reach  # Same connected seed across response ablations.
            self.candidates[str(i)]=dict(reinforce=float(reinforce),receptivity=float(receptivity),reach=float(reach),score=float(useful))
        def key(i):
            fallen=0 if self.spec.get('fallen_priority') and i in v['fallen'] else 1
            score=self.candidates[str(i)]['score']
            if not self.core and self.spec['seed_rule']=='weakest' and mode!='topology':score=0.
            return (fallen,-score,v['scores'][i],self.tie[i])
        return sorted(eligible,key=key)

    def _record(self,t,view,ledger,nodes,reason,end=None,release=False):
        action=super()._record(t,view,ledger,nodes,reason,end,release)
        self.log[-1].update(episode=self.episode,episode_reason=self.episode_reason,
            seed_node=self.seed_node,response_core=sorted(self.core),episode_nodes=sorted(self.episode_nodes),
            off_reports=self.off_reports,stall_reports=self.stall_reports,observed_growth=self.growth,
            candidate_components=dict(self.candidates),baseline_blue=sorted(self.baseline_blue),
            currently_red=sorted(view['currently_red']) if view else [],
            episode_exposure=float(ledger.integrated_exposure)-self.episode_start_spend)
        return action

    def act(self,observation,ledger):
        t=float(observation.time);v=self._observe(observation)
        if ledger.is_released or t>=self.problem.release_deadline or ledger.remaining_exposure<=ledger.tol:
            return self._record(t,v,ledger,[],'final_release',release=True)
        if v is None:
            return self._record(t,None,ledger,self.active if t<self.pulse_end else [],'duplicate_report',self.pulse_end if t<self.pulse_end else t)
        self.last_view=v;self.candidates={}
        if t<self.pulse_end-1e-10:
            self.active=[i for i in self.active if i in v['low']]
            if self.active:return self._record(t,v,ledger,self.active,'pulse_continue',self.pulse_end)
            self.last_end=t;self.pulse_end=t;self.next_start=t+self.spec['pause']
            self.off_streak={};self.off_reports=0;self.stall_reports=0
            return self._record(t,v,ledger,[],'recipient_no_longer_weak')
        self._feedback(v)
        if (self.spec['early_stop'] and self.off_reports>=2 and self.public_stop_streak>=2):
            return self._record(t,v,ledger,[],'public_stop_confirmed',release=True)
        if t<self.next_start-1e-10:return self._record(t,v,ledger,[],'observe_pause')
        if self.spec['wait_for_growth'] and self.core and self.stall_reports<2:
            return self._record(t,v,ledger,[],'wait_for_spontaneous_spread')
        ranked=self._rank_response(v,ledger)
        if not ranked:return self._record(t,v,ledger,[],'no_eligible_response_target')
        node=ranked[0]
        if self.seed_node is None:
            self.seed_node=node;self.attempted_seeds.add(node)
        self.episode_nodes.add(node);self.active=[node]
        self.pulse_end=min(t+self.spec['pulse'],self.problem.release_deadline,t+ledger.remaining_exposure/self.spec['amplitude'])
        self.last_end=self.pulse_end;self.next_start=self.pulse_end+self.spec['pause'];self.started=True
        self.off_streak={};self.off_reports=0;self.stall_reports=0
        return self._record(t,v,ledger,[node],'pulse_start',self.pulse_end)
