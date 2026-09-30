"""Frozen, observation-only nucleation hypotheses. No simulator-based tuning."""
from __future__ import annotations
import hashlib
import numpy as np
from src.control.optimizers.observed_subsets import _proposal_orders


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, ('nucleation-pilot-v1', *parts))).encode()).digest()[:4], 'big')


def pilot_plan():
    return dict(version='nucleation-pilot-v2', N=40, sizes=[4,10], audit_futures=6,
        origins=[f'nucleation_{i:02d}' for i in range(8)],
        origin_seeds={f'nucleation_{i:02d}':seed_for('origin',i) for i in range(8)},
        timing_seed=seed_for('separate_timing_origin'),
        selectors=['random','attention','audience_coverage','nucleus','nucleus_bridge','random_core','random_bridge'],
        frozen_roles=['zero','nucleus4','nucleus10','nucleus_bridge4','nucleus_bridge10'],
        amplitude=2., duration=19., release=20., hold=100., horizon=320., target_support=.75,
        destination=-1, access='full_roster', core_extra_nodes=4, bridge_counts={'4':1,'10':2},
        local_support=.75, local_hold=20., session_seconds=28800, planning_seconds=21600,
        timing_safety_factor=1.35, export_reserve_seconds=1200, timing_units=3,
        allocation_tiers=[[8,6],[8,4],[6,4],[4,4]], roles_per_origin_future=20,
        role_ceiling=960, physical_ceiling=963, search_simulations=0,
        primary='nucleus_bridge4 minus audience_coverage4: mean per-network failure-penalized completion time; lower is better',
        secondary=['nucleus_bridge10 versus audience_coverage10', 'bridge versus nucleus at each count',
            'nucleus versus random recipients in the same core', 'chosen versus random outside bridges with the same retained core seeds',
            'adaptive versus frozen zero, nucleus and nucleus_bridge; memory-by-bridge interaction',
            'exact first joint target entry, local onset/hold, core versus outside spread, initial majority, recipient traits and influence'],
        failure_penalty=320., bootstrap_seed=271828, bootstrap_draws=10000,
        information='One noisy partial public t0 report delivered at t1. No hidden states, true W, future data, or rollout search in selection.',
        recording='Scalar cadence1; exact changed-spin events and core predicate transitions on all units; network frames every2 units on future0.',
        interpretation='Exploratory mechanism study, not reliability confirmation. Fixed timing-only allocation; stop when complete, never fill spare hours with extra searches.',
        decision='Advance bridge hypothesis only if primary mean time difference is negative, persistent-rate difference is nonnegative, and bridge4 persists on at least two networks; otherwise retain negative/mixed evidence. No automatic retuning or further runs.')


def choose_allocation(plan,timing_seconds,remaining_seconds):
    """Timing outcomes are not an input. Preserve networks before extra futures."""
    if len(timing_seconds)!=3 or any(not np.isfinite(v) or v<=0 for v in timing_seconds):
        raise ValueError('Three positive full-unit timings are required')
    per_unit=max(timing_seconds)*plan['timing_safety_factor']
    budget=min(plan['planning_seconds'],remaining_seconds-plan['export_reserve_seconds'])
    for networks,futures in plan['allocation_tiers']:
        ceiling=networks*futures*plan['roles_per_origin_future']
        if ceiling*per_unit<=budget:
            return dict(origins=plan['origins'][:networks],audit_futures=futures,
                audit_ceiling=ceiling,conservative_seconds_per_unit=per_unit,
                estimated_seconds=ceiling*per_unit,planning_budget_seconds=budget,
                rule='First fitting predeclared tier; max timing x1.35; at most6h planned audit work; no outcome inputs.')
    return None


def select_groups(observation, plan, seed):
    """A[i,j] means listener i attends to j. Missing rows never become weak nodes.

    Core candidates: deterministic greedy mutual-connectivity growth from each
    known node, size k+4; choose highest mean internal reported attention.
    Nucleus seeds: greedy marginal maximum coverage of listeners inside core.
    Bridges: outside core, maximize receiving attention from selected core seeds
    times outgoing influence to outsiders; later bridges may receive from earlier ones. Ties use one fixed seeded order.
    Reported-row normalization is a partial-data proxy, not true attention.
    """
    roster,incoming,_,orders=_proposal_orders(observation,np.random.default_rng(seed))
    orders=dict(orders);N=observation.public_states.shape[0]
    W=np.zeros((N,N))
    for (i,j),v in observation.observed_edges.items():
        if i!=j:W[i,j]=v
    totals=W.sum(axis=1)
    A=np.divide(W,totals[:,None],out=np.zeros_like(W),where=totals[:,None]>0)
    ties={int(n):i for i,n in enumerate(orders['random'])}
    known=[i for i in roster if totals[i]>0 and incoming[i]>0]
    groups={'zero':dict(nodes=[],available=True,memory='adaptive')};cores={}
    for k in plan['sizes']:
        for name in ('random','attention','audience_coverage'):
            groups[f'{name}{k}']=dict(nodes=sorted(map(int,orders[name][:k])),available=len(roster)>=k,memory='adaptive')
        size=k+plan['core_extra_nodes'];nb=plan['bridge_counts'][str(k)]
        reason='insufficient nodes with both reported outgoing and incoming edges'
        if len(known)<size+nb:
            for name in ('nucleus','nucleus_bridge','random_core','random_bridge'):
                groups[f'{name}{k}']=dict(nodes=[],available=False,memory='adaptive',reason=reason)
            continue
        candidates=[]
        for root in known:
            core=[root]
            while len(core)<size:
                j=max((j for j in known if j not in core),key=lambda j:(float(A[j,core].sum()+A[core,j].sum()),-ties[j]))
                core.append(j)
            core=tuple(sorted(core));cohesion=float(A[np.ix_(core,core)].sum()/size)
            candidates.append((cohesion,tuple(-ties[j] for j in core),core))
        core=list(max(candidates)[2]);cores[f'core{k}']=core
        coverage=np.zeros(len(core));seeds=[]
        while len(seeds)<k:
            j=max((j for j in core if j not in seeds),key=lambda j:(float(np.maximum(A[core,j]-coverage,0).sum()),-ties[j]))
            seeds.append(j);coverage=np.maximum(coverage,A[core,j])
        mixed=seeds[:k-nb];outside=[i for i in range(N) if i not in core];bridge_scores=[]
        reached=np.zeros(len(outside))
        for _ in range(nb):
            scores={j:float(A[j,mixed].sum()*np.maximum(A[outside,j]-reached,0).sum()) for j in known if j not in core and j not in mixed}
            j=max(scores,key=lambda j:(scores[j],-ties[j]));bridge_scores.append(dict(node=int(j),score=scores[j]));mixed.append(j);reached=np.maximum(reached,A[outside,j])
        common=dict(available=True,memory='adaptive',core=core,
            observed_core_internal=float(A[np.ix_(core,core)].sum()/size),known_nodes=len(known),
            reported_rows=int((totals>0).sum()),information='partial reported graph only')
        groups[f'nucleus{k}']=dict(common,nodes=sorted(seeds))
        groups[f'nucleus_bridge{k}']=dict(common,nodes=sorted(mixed),bridge_scores=bridge_scores,
            zero_bridge_score=any(r['score']==0 for r in bridge_scores))
        # Membership ablations share the selected core, retained seeds, size and dose.
        rng=np.random.default_rng(seed_for(seed,'composition_controls',k))
        random_core=sorted(int(j) for j in rng.choice(core,k,replace=False))
        outside_known=[j for j in known if j not in core]
        random_outside=sorted(int(j) for j in rng.choice(outside_known,nb,replace=False))
        retained=seeds[:k-nb]
        groups[f'random_core{k}']=dict(common,nodes=random_core,control_for=f'nucleus{k}')
        groups[f'random_bridge{k}']=dict(common,nodes=sorted(retained+random_outside),
            retained_core_seeds=sorted(retained),outside_nodes=random_outside,control_for=f'nucleus_bridge{k}')
        groups[f'nucleus_bridge{k}']['retained_core_seeds']=sorted(retained)
        groups[f'nucleus_bridge{k}']['outside_nodes']=sorted(set(mixed)-set(core))
    for role in plan['frozen_roles']:
        groups[role+'_frozen']=dict(groups[role],memory='frozen')
    return groups,cores
