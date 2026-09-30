"""Observation-constrained proposals for offline simulator-assisted discovery.

Candidate membership obeys public information and observed-power ceilings.
Choosing among candidates uses actual development rollouts, so this is NOT a
restricted online controller or independent evaluation. Held-out futures are
separate from the search, and final new-network validation remains required.
"""
import hashlib
import itertools
import numpy as np
from dataclasses import replace
from src.control.optimizers.observed_subsets import _proposal_orders

SIZES=(2,4,6,10)
POWER_QUANTILES=(0.25,0.5,1.0)


def discovery_plan():
    return {'version':'sparse-discovery-v1','sizes':list(SIZES),'power_quantiles':list(POWER_QUANTILES),
            'origins':[f'development_{i:02d}' for i in range(4)],
            'initial_candidates_per_stratum':6,'exchange_candidates_per_stratum':4,
            'search_futures':3,'audit_futures':8,'representatives_per_stratum':1,
            'search_ceiling':1440,'audit_ceiling':512,'zero_ceiling':44,'physical_ceiling':1996,
            'session_seconds':28800,'amplitude':2.,'duration':19.,'release':20.,'hold':100.,'horizon':320.,
            'target_support':.75,'selection_rule':'Within each size/power stratum: more nonrelapsing holds, then lower mean failure-penalized completion time; lower initial reported power breaks ties.',
            'missing_data_rule':'Positive reported incoming scores only; unknown excluded at every power ceiling. Full-roster random comparators remain separate.',
            'scope':'Offline simulator-assisted development search; audit futures untouched until groups frozen; four networks, not independent futures-as-networks.',
            'unavailable_rule':'Skip impossible strata without relaxing count or power limit.'}


def stream_seed(origin,phase,future):
    return int.from_bytes(hashlib.sha256(f'sparse-discovery-v1:{origin}:{phase}:{future}'.encode()).digest()[:8],'big')


def strata(observation):
    roster,incoming,_,_=_proposal_orders(observation,np.random.default_rng(0))
    positive=[n for n in roster if incoming[n]>0]
    result=[]
    for q in POWER_QUANTILES:
        ceiling=float(np.quantile(incoming[positive],q)) if positive else None
        eligible=tuple(n for n in positive if incoming[n]<=ceiling) if positive else ()
        for size in SIZES:
            result.append({'id':f'n{size}_q{q:g}','size':size,'quantile':q,'ceiling':ceiling,
                           'eligible':eligible,'available':len(eligible)>=size})
    return result,incoming


def initial_groups(observation,stratum,seed,limit=6):
    if not stratum['available']:return []
    rng=np.random.default_rng(seed)
    restricted=replace(observation,accessible_roster=tuple(stratum['eligible']))
    orders=_proposal_orders(restricted,rng)[3]
    size=stratum['size'];groups=[]
    def add(nodes):
        group=tuple(sorted(int(n) for n in nodes))
        if group not in groups:groups.append(group)
    for _,order in orders:add(order[:size])
    for _ in range(100):
        if len(groups)>=limit:break
        add(rng.choice(stratum['eligible'],size=size,replace=False))
    return groups[:limit]


def exchange_groups(best,stratum,seen,seed,limit=4):
    neighbors=sorted({tuple(sorted((set(best)-{old})|{new}))
                     for old in best for new in stratum['eligible'] if new not in best}-set(seen))
    rng=np.random.default_rng(seed)
    return [neighbors[int(i)] for i in rng.permutation(len(neighbors))[:limit]]


def group_score(nodes,results,incoming):
    if not results or any(not r['is_valid'] or not r['is_eligible'] for r in results):
        raise ValueError('Invalid or missing development result; do not score as a scientific failure')
    success=[bool(r['success_through_horizon']) for r in results]
    times=[r['completion_time'] if ok else 320. for r,ok in zip(results,success)]
    return (-sum(success),float(np.mean(times)),float(max(incoming[list(nodes)])),
            float(sum(incoming[list(nodes)])),tuple(nodes))
