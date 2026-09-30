"""Durable sequential pilot: full-horizon timing, frozen queue, eight-hour supervisor."""
from __future__ import annotations
import json,multiprocessing as mp,platform,queue,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.checkpointing import serialize_state
from src.control.optimizers.campaign import _origin_state,_atomic_json,_sha,observation_identity
from src.control.transformation_study import source_identity,_runtime_versions
from src.control.optimizers.sequential_low_power import pilot_plan,choose_allocation
from src.control.sequential_runner import run_unit,observation_model


def identity_for(plan):
    profile=LIGHT_TIERED.model_copy(update={'N':plan['N'],'horizon':plan['horizon'],'release_deadline':plan['release_deadline'],'holding_duration':plan['hold']})
    return json.loads(json.dumps(dict(plan=plan,profile=profile.model_dump(),source=source_identity(),runtime=dict(python=platform.python_version(),packages=_runtime_versions()))))


def key_for(base,future,role):return f'{base}__{role}__future{future}'


def resolved_queue(plan,allocation):
    return {key_for(o,f,r):dict(origin=o,future=f,role=r) for f in range(allocation['futures']) for o in allocation['origins'] for r in plan['policies']}


def _run(out_dir,plan,resume_bundle=None,events=None,budget_seconds=None):
    start=time.monotonic();identity=identity_for(plan);profile=LIGHT_TIERED.model_copy(update=identity['profile'])
    if resume_bundle:
        bundle=Path(resume_bundle);m=json.loads((bundle/'manifest.json').read_text())
        if m['identity']!=identity:raise ValueError('Resume requires identical source, runtime, profile and plan')
        for info in list(m['actual'].values())+list(m['initial_states'].values()):
            if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Saved checksum mismatch')
    else:
        bundle=Path(out_dir)/f'{plan.get("bundle_prefix","sequential_low_power_v1")}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}';bundle.mkdir(parents=True)
        m=dict(schema_version=plan['version'],identity=identity,status='incomplete',report_status='pending',actual={},initial_states={},origin_info={},
            allocation=None,allocation_decided=False,selection_frozen=False,queue={},audit_roles={},timing_durations=[])
    def save():_atomic_json(bundle/'manifest.json',m)
    save()
    if events is not None:events.put(('bundle',str(bundle)))
    print('Durable bundle:',bundle,flush=True)
    def prepare(base,seed):
        if base in m['initial_states']:return
        state=_origin_state(profile,seed)
        if base in plan['origins'] and plan.get('initial_red_majority_required') and not (np.mean(state.x>0)>.5 and np.mean(state.y>0)>.5):
            raise ValueError('Audit origin violates required initial red majority in both layers')
        p=Path('origins')/f'{base}.json';_atomic_json(bundle/p,serialize_state(state))
        m['initial_states'][base]=dict(path=str(p),sha256=_sha(bundle/p))
        hashes={}
        for visibility in ['restricted','complete_public']:
            om=observation_model(profile,seed,dict(visibility=visibility));om.sample_observation(state,0.);obs=om.sample_observation(state,0.);hashes[visibility]=observation_identity(obs)
        m['origin_info'][base]=dict(private_blue=float(np.mean(state.x<0)),public_blue=float(np.mean(state.y<0)),observation_sha256=hashes)
        save()
    def evaluate(base,seed,f,role,phase):
        key=key_for(base,f,role)
        if key not in m['actual']:
            begin=time.monotonic();u=run_unit(profile,plan,base,seed,f,plan['policies'][role])
            if not(u['is_valid'] and u['is_eligible']):
                _atomic_json(bundle/'invalid'/f'{key}.json',u);raise ValueError(f'Invalid unit retained: {key}')
            p=Path('units')/f'{key}.json';_atomic_json(bundle/p,u)
            m['actual'][key]=dict(path=str(p),sha256=_sha(bundle/p),kind=phase,origin=base,future=f,role=role,elapsed_full_unit_seconds=time.monotonic()-begin)
            save();print(phase,key,'saved',len(m['actual']),flush=True)
        return key,m['actual'][key]
    if not m['allocation_decided']:
        base=plan.get('timing_origin','sequential_timing');seed=plan['timing_seed'];prepare(base,seed)
        timing_roles=plan.get('timing_roles',['zero','dynamic_low2_B152','complete_low_B80']);durations=[]
        for role in timing_roles:
            _,info=evaluate(base,seed,0,role,'timing');durations.append(info['elapsed_full_unit_seconds'])
        m['timing_durations']=durations;m['allocation']=choose_allocation(plan,durations,(budget_seconds or plan['session_seconds'])-(time.monotonic()-start));m['allocation_decided']=True;save()
    if m['allocation'] is None:m['status']='timing_only_insufficient_budget';save();return bundle
    for base in m['allocation']['origins']:prepare(base,plan['origin_seeds'][base])
    jobs=resolved_queue(plan,m['allocation'])
    if m['selection_frozen'] and jobs!=m['queue']:raise ValueError('Frozen queue mismatch')
    m['queue']=jobs;m['selection_frozen']=True;save()
    print('Frozen allocation:',m['allocation'],flush=True)
    for key,j in jobs.items():
        saved,_=evaluate(j['origin'],plan['origin_seeds'][j['origin']],j['future'],j['role'],'audit')
        assert key==saved;m['audit_roles'][key]=key;save()
    m['status']='complete';save();return bundle


def _worker(out_dir,plan,resume_bundle,events,budget_seconds):
    try:
        bundle=_run(out_dir,plan,resume_bundle,events,budget_seconds);events.put(('compute_done',str(bundle)))
        from src.control.sequential_analysis import build_saved_report
        build_saved_report(bundle);events.put(('done',str(bundle)))
    except BaseException:
        import traceback
        events.put(('error',traceback.format_exc()))


def run_pilot(out_dir,*,resume_bundle=None,plan_only=False,wall_time_limit=None,plan=None):
    plan=pilot_plan() if plan is None else plan
    if plan_only:return plan
    import sys
    if 'google.colab' not in sys.modules:raise RuntimeError('Nominal sequential pilot is Colab-only; local work is implementation verification or saved-data analysis.')
    limit=plan['session_seconds'] if wall_time_limit is None else min(plan['session_seconds'],float(wall_time_limit))
    if not np.isfinite(limit) or limit<=0:raise ValueError('No notebook time remains')
    ctx=mp.get_context('spawn');events=ctx.Queue();worker=ctx.Process(target=_worker,args=(str(out_dir),plan,resume_bundle,events,limit))
    start=time.monotonic();bundle=Path(resume_bundle) if resume_bundle else None;done=False;error=None;worker.start()
    try:
        while time.monotonic()-start<limit:
            try:kind,value=events.get(timeout=min(.25,max(.001,limit-(time.monotonic()-start))))
            except queue.Empty:
                if not worker.is_alive():break
                continue
            if kind in ('bundle','compute_done'):bundle=Path(value)
            elif kind=='done':bundle=Path(value);done=True;break
            elif kind=='error':error=value;break
        if worker.is_alive():
            worker.terminate();worker.join(timeout=1)
            if worker.is_alive():worker.kill();worker.join(timeout=1)
        if bundle:
            m=json.loads((bundle/'manifest.json').read_text())
            if not done:
                reason='worker_error' if error else ('hard_time_limit' if time.monotonic()-start>=limit else 'worker_exit')
                if m['status'] not in ('complete','timing_only_insufficient_budget'):m['status']='incomplete_'+reason
                m['report_status']='incomplete_'+reason
                if error:m['error']=error
            else:m['report_status']='complete'
            _atomic_json(bundle/'manifest.json',m)
        else:raise RuntimeError(error or 'Worker exited before durable manifest')
        if error:print('Worker error; retain this bundle for exact resume:',bundle,'\n',error,flush=True)
        return bundle
    finally:
        if worker.is_alive():worker.terminate();worker.join(timeout=1)
        events.close()
