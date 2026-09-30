"""Bounded Colab pilot, durable units, frozen observation-only memberships."""
from __future__ import annotations
import hashlib,json,multiprocessing as mp,platform,queue,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.optimizers.campaign import (_atomic_json,_sha,_origin_state,_case_problem,reconstruct_t0_observation,observation_identity,run_actual_unit)
from src.control.optimizers.nucleation_pilot import pilot_plan,seed_for,select_groups,choose_allocation
from src.control.optimizers.group_characteristics import state_group,observed_group
from src.control.transformation_study import source_identity,_runtime_versions,load_transformation_contract
from src.control.checkpointing import serialize_state
from src.control.recording import RecorderConfig
from src.control.nucleation_recording import NucleationRecorder


def identity_for(plan):
    return json.loads(json.dumps(dict(source=source_identity(),runtime=dict(python=platform.python_version(),packages=_runtime_versions()),
        plan=plan,profile=LIGHT_TIERED.model_copy(update={'N':plan['N']}).model_dump(),contract=load_transformation_contract())))


def physical_key(base,future,memory,nodes):
    token=hashlib.sha256(json.dumps(sorted(nodes)).encode()).hexdigest()[:16]
    return f'{base}__audit{future:02d}__{memory}__{token}'


def resolved_queue(plan,groups):
    roles={};units={}
    for f in range(plan['audit_futures']):
        for base in plan['origins']:
            for role,g in groups[base].items():
                if not g['available']:continue
                key=physical_key(base,f,g['memory'],g['nodes'])
                roles[f'{base}__{role}__future{f}']=key
                units[key]=dict(base=base,future=f,memory=g['memory'],nodes=g['nodes'])
    return roles,units


def _run(out_dir,plan,resume_bundle=None,events=None,budget_seconds=None):
    started=time.monotonic();budget_seconds=plan['session_seconds'] if budget_seconds is None else budget_seconds
    identity=identity_for(plan);profile=LIGHT_TIERED.model_copy(update={'N':plan['N']})
    if resume_bundle:
        bundle=Path(resume_bundle);m=json.loads((bundle/'manifest.json').read_text())
        if m['identity']!=identity:raise ValueError('Resume requires identical source, runtime, profile and frozen plan')
        for info in list(m['actual'].values())+list(m['initial_states'].values()):
            if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Saved checksum mismatch')
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');bundle=Path(out_dir)/f'nucleation_pilot_v2_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True,exist_ok=False)
        m=dict(schema_version='nucleation-pilot-v2',identity=identity,status='incomplete',actual={},audit_roles={},groups={},cores={},initial_states={},origin_info={},selection_frozen=False,
            allocation=None,allocation_decided=False,timing_durations=[],planned_roles={},resolved_units={},report_status='pending')
    def save():_atomic_json(bundle/'manifest.json',m)
    save()
    if events is not None:events.put(('bundle',str(bundle)))
    print('Durable overnight bundle:',bundle,flush=True)

    def prepare(base,seed):
        state=_origin_state(profile,seed);problem=_case_problem(profile,seed,'broad_diagnostic',-1,base)
        obs=reconstruct_t0_observation(profile,state,seed,problem.accessible_roster)
        groups,cores=select_groups(obs,plan,seed_for(base,'selector'))
        path=Path('origins')/f'{base}.json';_atomic_json(bundle/path,serialize_state(state))
        m['initial_states'][base]=dict(path=str(path),sha256=_sha(bundle/path))
        return state,obs,groups,cores

    def evaluate(base,future,memory,nodes,phase,seed,cores):
        key=physical_key(base,future,memory,nodes)
        if key not in m['actual']:
            begin=time.monotonic()
            config=RecorderConfig(cadence=1.,record_network_frames=(future==0),max_network_frames=161,retention_window=(20.,320.))
            recorder=NucleationRecorder(config,cores,J=profile.J,T=profile.temperature)
            u=run_actual_unit(profile,seed,f'{base}_{phase}{future:02d}',identity['contract'],memory,
                'broad_diagnostic',tuple(nodes),plan['amplitude'] if nodes else 0.,plan['duration'] if nodes else 0.,-1,
                simulation_seed=seed_for(base,phase,future),recorder=recorder)
            if not(u['is_valid'] and u['is_eligible']):
                _atomic_json(bundle/'invalid'/f'{key}.json',u);raise ValueError(f'Invalid unit retained: {key}')
            path=Path('units')/f'{key}.json';_atomic_json(bundle/path,u)
            m['actual'][key]=dict(path=str(path),sha256=_sha(bundle/path),kind=phase,base_origin=base,future=future,
                origin_id=u['origin_id'],recipient_nodes=list(nodes),memory=memory,
                elapsed_seconds=u['elapsed_seconds_this_invocation'],elapsed_full_unit_seconds=time.monotonic()-begin)
            save();print(f'{phase}: {key}; {len(m["actual"])} physical units saved',flush=True)
        return key,m['actual'][key]

    if not m['allocation_decided']:
        base='nucleation_timing';seed=plan['timing_seed']
        _,_,groups,cores=prepare(base,seed)
        timings=[]
        # Outcomes are saved for provenance, but only elapsed costs reach the gate.
        for nodes,memory in [([], 'adaptive'),(groups['audience_coverage10']['nodes'],'adaptive'),
                             (groups['audience_coverage10']['nodes'],'frozen')]:
            _,info=evaluate(base,0,memory,nodes,'timing',seed,cores)
            timings.append(info['elapsed_full_unit_seconds'])
        m['timing_durations']=timings
        m['allocation']=choose_allocation(plan,timings,budget_seconds-(time.monotonic()-started))
        m['allocation_decided']=True;save()
    if m['allocation'] is None:
        m['status']='timing_only_insufficient_budget';save();return bundle
    allocation=m['allocation'];active_plan=dict(plan,origins=allocation['origins'],audit_futures=allocation['audit_futures'])
    for base in allocation['origins']:
        if base in m['groups']:continue
        state,obs,groups,cores=prepare(base,plan['origin_seeds'][base])
        for g in groups.values():
            if g['available'] and g['nodes']:
                g['full_network_initial']=state_group(state,g['nodes']);g['observed_initial']=observed_group(obs,g['nodes'])
        m['origin_info'][base]=dict(N=plan['N'],observation_sha256=observation_identity(obs),private_blue=float(np.mean(state.x<0)),public_blue=float(np.mean(state.y<0)))
        m['groups'][base]=groups;m['cores'][base]=cores;save()
    roles,units=resolved_queue(active_plan,m['groups'])
    if m['selection_frozen'] and (m['planned_roles']!=roles or m['resolved_units']!=units):raise ValueError('Frozen queue mismatch')
    m['planned_roles']=roles;m['resolved_units']=units;m['selection_frozen']=True
    m['resolved_physical_units']=len(units);m['resolved_role_rows']=len(roles);save()
    print('Frozen allocation:',allocation,'resolved audit units:',len(units),'role rows:',len(roles),flush=True)
    # Complete matched origin/future blocks; no outcome-dependent priority or extension.
    for key,job in units.items():
        base=job['base'];future=job['future']
        saved,_=evaluate(base,future,job['memory'],job['nodes'],'audit',plan['origin_seeds'][base],m['cores'][base])
        assert saved==key
        for role,unit in roles.items():
            if unit==key:m['audit_roles'][role]=key
        save()
    m['status']='complete';save();return bundle


def _worker(out_dir,plan,resume_bundle,events,budget_seconds):
    try:
        bundle=_run(out_dir,plan,resume_bundle,events,budget_seconds)
        events.put(('compute_done',str(bundle)))
        from src.control.nucleation_analysis import build_saved_report
        build_saved_report(bundle)
        events.put(('done',str(bundle)))
    except BaseException:
        import traceback
        events.put(('error',traceback.format_exc()))


def run_pilot(out_dir,*,resume_bundle=None,plan_only=False,wall_time_limit=None):
    plan=pilot_plan()
    if plan_only:return plan
    import sys
    if 'google.colab' not in sys.modules:raise RuntimeError('Nominal pilot is Colab-only; local work is implementation verification or saved-data analysis.')
    limit=plan['session_seconds'] if wall_time_limit is None else min(plan['session_seconds'],float(wall_time_limit))
    if not np.isfinite(limit) or limit<=0:raise ValueError('No notebook time remains')
    ctx=mp.get_context('spawn');events=ctx.Queue();worker=ctx.Process(target=_worker,args=(str(out_dir),plan,resume_bundle,events,limit))
    start=time.monotonic();worker.start();bundle=Path(resume_bundle) if resume_bundle else None;done=False;compute_done=False;error=None
    try:
        while time.monotonic()-start<limit:
            try:kind,value=events.get(timeout=min(.25,max(.001,limit-(time.monotonic()-start))))
            except queue.Empty:
                if not worker.is_alive():break
                continue
            if kind=='bundle':bundle=Path(value)
            elif kind=='compute_done':bundle=Path(value);compute_done=True
            elif kind=='done':bundle=Path(value);compute_done=True;done=True;break
            elif kind=='error':error=value;break
        if worker.is_alive():
            worker.terminate();worker.join(timeout=1)
            if worker.is_alive():worker.kill();worker.join(timeout=1)
        if bundle:
            m=json.loads((bundle/'manifest.json').read_text())
            if not done:
                reason='worker_error' if error else ('hard_time_limit' if time.monotonic()-start>=limit else 'worker_exit')
                if not compute_done and m.get('status') not in ('complete','timing_only_insufficient_budget'):
                    m['status']='incomplete_'+reason
                m['report_status']='incomplete_'+reason
                if error:m['error']=error
            else:m['report_status']='complete'
            _atomic_json(bundle/'manifest.json',m)
        if not bundle:raise RuntimeError(error or 'Worker exited before durable manifest')
        if error:print('Worker stopped with an error; retain/resume this exact bundle:',bundle,'\n',error,flush=True)
        return bundle
    finally:
        if worker.is_alive():worker.terminate();worker.join(timeout=1)
        events.close()
