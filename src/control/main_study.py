"""Shared, bounded N40/N500 study; physics remains in the controlled engine.

Timing origins never enter inference. Allocation uses elapsed times only and is
frozen before study origins. Selection uses separate search futures; no audit
result is used to recruit, select, or extend an allocation.
"""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import platform
import queue
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.optimizers.campaign import (_atomic_json, _sha, _origin_state,
    _case_problem, reconstruct_t0_observation, run_actual_unit)
from src.control.optimizers.observed_subsets import _proposal_orders
from src.control.optimizers.sparse_discovery import initial_groups, exchange_groups, group_score
from src.control.optimizers.group_characteristics import state_group, observed_group
from src.control.transformation_study import source_identity, _runtime_versions, load_transformation_contract


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, ('main-study-v1', *parts))).encode()).digest()[:8], 'big')


def study_plan(tier):
    if tier not in ('light', 'extended'):
        raise ValueError('Choose light or extended')
    N = 40 if tier == 'light' else 500
    return dict(version='main-study-v1', tier=tier, N=N, sizes=[4, 10] if N == 40 else [4, 10, 50, 125],
        session_seconds=28800, amplitude=2., duration=19., release=20., hold=100., horizon=320.,
        target_support=.75, destination=-1, memory='adaptive', access='full_roster',
        origin_seeds={f'{tier}_{i:02d}':seed_for(tier, 'origin', i) for i in range(8)},
        timing_seed=seed_for(tier, 'timing'), timing_safety_factor=2.,
        search_sizes=[4, 10], power_quantiles=[.5, 1.], initial_candidates=6,
        exchange_candidates=4, search_futures=3, max_network_frames=24 if N == 40 else 12,
        core_selectors=['random', 'attention', 'audience_coverage', 'low_observed'],
        primary='audience_coverage10 minus random10 (persistent success, paired by origin)',
        scaling_comparisons=['fixed 4 and 10 recipients', '10% and 25% population fractions'],
        bootstrap_draws=10000, bootstrap_seed=271828,
        information='Restricted public proposals; optional offline full-simulator ranking; full-state features retrospective.',
        incomplete_rule='Do not interpret interrupted or missing units as failures; no primary inference until complete.')


def allocation_cost(plan, origins, futures, search):
    # A ceiling: unavailable strata and duplicate memberships can reduce work.
    roles = 1 + len(plan['sizes']) * len(plan['core_selectors'])
    strata = len(plan['search_sizes']) * len(plan['power_quantiles']) if search else 0
    return origins * ((roles + strata) * futures + strata *
        (plan['initial_candidates'] + plan['exchange_candidates']) * plan['search_futures'])


def choose_allocation(plan, timing_seconds, remaining_seconds):
    """Cost-only gate. Four networks x two futures is the minimum ensemble."""
    if not timing_seconds or any(not np.isfinite(x) or x <= 0 for x in timing_seconds):
        raise ValueError('Positive measured durations required')
    conservative = max(timing_seconds) * plan['timing_safety_factor']
    if plan['tier'] == 'light':
        # Fresh light runs reproduce a substantial ensemble, not a single demo.
        choices = [(8,8,True), (8,4,True), (8,4,False), (4,4,False), (4,2,False)]
    else:
        # More independent networks take precedence over extra futures/search.
        choices = [(8,4,True), (8,4,False), (8,2,False), (4,2,False)]
    for n, f, search in choices:
        ceiling = allocation_cost(plan, n, f, search)
        if ceiling * conservative <= remaining_seconds * .90:
            return dict(origins=list(plan['origin_seeds'])[:n], audit_futures=f, search_enabled=search,
                physical_ceiling=ceiling, conservative_seconds_per_unit=conservative,
                estimated_seconds=ceiling*conservative,
                rationale='Largest predeclared feasible tier using timing only; 10% reserve for orchestration/export.')
    return None


def make_specs(observation, plan):
    _, incoming, _, _ = _proposal_orders(observation, np.random.default_rng(0))
    positive = np.flatnonzero(incoming > 0)
    specs = []
    for k in plan['search_sizes']:
        for q in plan['power_quantiles']:
            ceiling = float(np.quantile(incoming[positive], q)) if len(positive) else None
            eligible = tuple(int(n) for n in positive if incoming[n] <= ceiling)
            specs.append(dict(id=f'n{k}_q{q:g}', size=k, quantile=q, ceiling=ceiling,
                              eligible=eligible, available=len(eligible)>=k))
    return specs, incoming


def make_benchmarks(observation, sizes, seed):
    _, incoming, _, orders = _proposal_orders(observation, np.random.default_rng(seed))
    orders = dict(orders)
    positive = incoming[incoming > 0]
    ceiling = float(np.quantile(positive, .5)) if len(positive) else None
    low = [int(n) for n in orders['low_observed_attention'] if incoming[n]>0 and incoming[n]<=ceiling]
    result = {'zero':dict(nodes=[], available=True)}
    for size in sizes:
        for name in ('random','attention','audience_coverage'):
            result[f'{name}{size}'] = dict(nodes=sorted(int(n) for n in orders[name][:size]), available=True)
        result[f'low_observed{size}'] = dict(nodes=sorted(low[:size]) if len(low)>=size else [],
            available=len(low)>=size, observed_ceiling=ceiling)
    return result


def _identity(plan):
    return dict(source_sha256=source_identity()['source_sha256'], runtime=dict(python=platform.python_version(),
        packages=_runtime_versions()), plan=plan,
        profile=LIGHT_TIERED.model_copy(update={'N':plan['N']}).model_dump(),
        contract=load_transformation_contract())


def _run(out_dir, plan, resume_bundle=None, progress_queue=None):
    identity=json.loads(json.dumps(_identity(plan)))
    profile=LIGHT_TIERED.model_copy(update={'N':plan['N']})
    if resume_bundle:
        bundle=Path(resume_bundle);m=json.loads((bundle/'manifest.json').read_text())
        if m['identity']!=identity: raise ValueError('Resume requires identical source/runtime/plan')
        for info in m['actual'].values():
            if _sha(bundle/info['path'])!=info['sha256']: raise ValueError('Saved unit checksum mismatch')
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        bundle=Path(out_dir)/f'main_study_{plan["tier"]}_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True,exist_ok=False)
        m=dict(schema_version='main-study-v1', identity=identity, status='incomplete', actual={},
               audit_roles={}, groups={}, origin_info={}, allocation=None, selection_frozen=False,
               timing_durations=[], started_utc=stamp)
    def save(): _atomic_json(bundle/'manifest.json',m)
    save()
    if progress_queue is not None: progress_queue.put(('bundle',str(bundle)))
    print('Durable bundle:',bundle,flush=True)
    started=time.monotonic()

    def evaluate(base, origin_seed, nodes, phase, future):
        nodes=tuple(sorted(nodes));token=hashlib.sha256(str(nodes).encode()).hexdigest()[:16]
        key=f'{base}__{phase}{future:02d}__{token}'
        if key not in m['actual']:
            keep_frames=(phase=='audit' and future==0)
            u=run_actual_unit(profile,origin_seed,f'{base}_{phase}{future:02d}',identity['contract'],
                'adaptive','broad_diagnostic',nodes,plan['amplitude'] if nodes else 0.,
                plan['duration'] if nodes else 0.,-1,
                simulation_seed=seed_for(base,phase,future),
                record_network_frames=keep_frames,max_network_frames=plan['max_network_frames'])
            if not u['is_valid'] or not u['is_eligible']:
                # Preserve invalid evidence but never count it as physical failure.
                _atomic_json(bundle/'invalid'/f'{key}.json',u)
                raise ValueError(f'Invalid or ineligible unit: {key}')
            path=Path('units')/f'{key}.json';_atomic_json(bundle/path,u)
            m['actual'][key]=dict(path=str(path),sha256=_sha(bundle/path),kind=phase,
                origin_id=f'{base}_{phase}{future:02d}',base_origin=base,seed=int(origin_seed),future=future,
                recipient_nodes=list(nodes),success=u['success'],relapsed=u['relapsed'],
                success_through_horizon=u['success_through_horizon'],
                elapsed_seconds=u['elapsed_seconds_this_invocation'])
            save();print(f'{base} {phase} future {future}: {len(nodes)} recipients; {len(m["actual"])} units saved',flush=True)
        return key,json.loads((bundle/m['actual'][key]['path']).read_text())

    if m['allocation'] is None:
        # Timing results are retained but never enter scientific selection/inference.
        base=f'{plan["tier"]}_timing';s=plan['timing_seed']
        state=_origin_state(profile,s)
        problem=_case_problem(profile,s,'broad_diagnostic',-1,base)
        obs=reconstruct_t0_observation(profile,state,s,problem.accessible_roster)
        benches=make_benchmarks(obs,plan['sizes'],seed_for(base,'bench'))
        durations=[]
        for role in ('zero','audience_coverage4',f'audience_coverage{max(plan["sizes"])}'):
            _,u=evaluate(base,s,benches[role]['nodes'],'timing',0)
            durations.append(u['elapsed_seconds_this_invocation'])
        m['timing_durations']=durations
        m['allocation']=choose_allocation(plan,durations,plan['session_seconds']-(time.monotonic()-started))
        if m['allocation'] is None:
            m['status']='timing_only_insufficient_budget';save();return bundle
        m['allocation_frozen_before_study']=True;save()
    allocation=m['allocation']
    # Freeze each group before auditing any network; descriptors never enter score.
    for base in allocation['origins']:
        s=plan['origin_seeds'][base]
        if base in m['groups']: continue
        state=_origin_state(profile,s);problem=_case_problem(profile,s,'broad_diagnostic',-1,base)
        obs=reconstruct_t0_observation(profile,state,s,problem.accessible_roster)
        roles=make_benchmarks(obs,plan['sizes'],seed_for(base,'bench'))
        if allocation['search_enabled']:
            specs,incoming=make_specs(obs,plan)
            for spec in specs:
                if not spec['available']:
                    roles[spec['id']]=dict(nodes=[],available=False,spec=spec);continue
                initial=initial_groups(obs,spec,seed_for(base,spec['id'],'initial'),plan['initial_candidates'])
                scored={}
                def score(nodes):
                    nodes=tuple(nodes)
                    outcomes=[evaluate(base,s,nodes,'search',f)[1] for f in range(plan['search_futures'])]
                    scored[nodes]=group_score(nodes,outcomes,incoming)
                for nodes in initial:score(nodes)
                best=min(scored,key=scored.get)
                for nodes in exchange_groups(best,spec,initial,seed_for(base,spec['id'],'exchange'),plan['exchange_candidates']):score(nodes)
                best=min(scored,key=scored.get)
                roles[spec['id']]=dict(nodes=list(best),available=True,spec=spec,
                    candidates=[dict(nodes=list(n),score=list(v)) for n,v in scored.items()])
        for info in roles.values():
            if info['available'] and info['nodes']:
                info['full_network_initial']=state_group(state,info['nodes'])
                info['observed_initial']=observed_group(obs,info['nodes'])
        m['origin_info'][base]=dict(N=plan['N'],private_blue=float(np.mean(state.x<0)),public_blue=float(np.mean(state.y<0)))
        m['groups'][base]=roles;save()
    m['selection_frozen']=True;save()
    # Future-major ordering gives broad coverage if the hard session ends early.
    for future in range(allocation['audit_futures']):
        for base in allocation['origins']:
            for role,info in m['groups'][base].items():
                if not info['available']:continue
                key,_=evaluate(base,plan['origin_seeds'][base],info['nodes'],'audit',future)
                m['audit_roles'][f'{base}__{role}__future{future}']=key;save()
    m['status']='complete';save();return bundle


def _worker(out_dir,plan,resume_bundle,events):
    try: events.put(('done',str(_run(out_dir,plan,resume_bundle,events))))
    except BaseException:
        import traceback
        events.put(('error',traceback.format_exc()))


def run_study(out_dir, tier, *, resume_bundle=None):
    """Cloud only; hard deadline also stops an active simulation."""
    import sys
    if 'google.colab' not in sys.modules:
        raise RuntimeError('Nominal studies are cloud-only. Use saved-data analysis locally.')
    plan=study_plan(tier);ctx=mp.get_context('spawn');events=ctx.Queue()
    worker=ctx.Process(target=_worker,args=(str(out_dir),plan,resume_bundle,events))
    start=time.monotonic();worker.start();bundle=Path(resume_bundle) if resume_bundle else None
    done=False;error=None
    try:
        while time.monotonic()-start<plan['session_seconds']:
            try:kind,value=events.get(timeout=min(.25,max(.001,plan['session_seconds']-(time.monotonic()-start))))
            except queue.Empty:
                if not worker.is_alive():break
                continue
            if kind=='bundle':bundle=Path(value)
            elif kind=='done':bundle=Path(value);done=True;break
            elif kind=='error':error=value;break
        if worker.is_alive():
            worker.terminate();worker.join(timeout=1)
            if worker.is_alive():worker.kill();worker.join(timeout=1)
        if bundle and not done:
            m=json.loads((bundle/'manifest.json').read_text())
            m['status']='incomplete_worker_error' if error else ('incomplete_hard_time_limit' if time.monotonic()-start>=plan['session_seconds'] else 'incomplete_worker_exit')
            if error:m['error']=error
            _atomic_json(bundle/'manifest.json',m)
        if error:raise RuntimeError(f'{error}\nSaved bundle: {bundle}')
        if not bundle:raise RuntimeError('Worker exited before a manifest was saved')
        return bundle
    finally:
        if worker.is_alive():worker.terminate();worker.join(timeout=1)
        events.close()
