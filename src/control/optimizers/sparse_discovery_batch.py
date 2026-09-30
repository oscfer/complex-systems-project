"""Durable offline search with fixed count/power strata and untouched audit futures."""
import json
import platform
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.optimizers.campaign import (_atomic_json,_sha,_origin_state,_case_problem,
    reconstruct_t0_observation,run_actual_unit,_write_study_bundle)
from src.control.optimizers.sparse_discovery import (discovery_plan,strata,initial_groups,
    exchange_groups,group_score,stream_seed)
from src.control.transformation_study import source_identity,_runtime_versions,load_transformation_contract


def _plan(validation):
    if validation:
        from src.control.optimizers.validation_batch import validation_plan
        return validation_plan()
    return discovery_plan()


class SessionFinished(Exception):pass


def _run_sparse_discovery(out_dir,*,resume_bundle=None,plan_only=False,progress_queue=None,validation=False):
    plan=_plan(validation)
    if plan_only:return plan
    profile=LIGHT_TIERED;contract=load_transformation_contract()
    origins=plan.get('origin_seeds') or {r['origin_id']:r['seed'] for r in contract['development']}
    identity=json.loads(json.dumps({'source_sha256':source_identity()['source_sha256'],
        'runtime':{'python':platform.python_version(),'packages':_runtime_versions()},
        'profile':profile.model_dump(),'contract':contract,'plan':plan}))
    if resume_bundle is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        prefix='sparse_validation_v1' if validation else 'sparse_discovery_v1'
        bundle=Path(out_dir)/f'{prefix}_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True,exist_ok=False)
        manifest={'identity':identity,'status':'incomplete','actual':{},'strata':{},'audit_roles':{},'actual_units_completed':0}
        _atomic_json(bundle/'manifest.json',manifest)
    else:
        bundle=Path(resume_bundle);manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest['identity']!=identity:raise ValueError('Resume identity changed')
        for info in manifest['actual'].values():
            if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Saved trajectory checksum mismatch')
    if progress_queue is not None:progress_queue.put(('bundle',str(bundle)))
    print('Bundle:',bundle,flush=True)
    started=time.monotonic()
    def save():_atomic_json(bundle/'manifest.json',manifest)
    def evaluate(base,nodes,phase,future):
        nodes=tuple(sorted(int(n) for n in nodes));origin=f'{base}_{phase}{future:02d}'
        key=f"{origin}__broad_diagnostic__{'_'.join(map(str,nodes)) or 'zero'}"
        if key not in manifest['actual']:
            if time.monotonic()-started>=plan['session_seconds']:raise SessionFinished()
            u=run_actual_unit(profile,origins[base],origin,contract,'adaptive','broad_diagnostic',nodes,
                plan['amplitude'] if nodes else 0.,plan['duration'] if nodes else 0.,-1,
                simulation_seed=stream_seed(base,phase,future))
            path=Path('units')/(key+'.json');_atomic_json(bundle/path,u)
            manifest['actual'][key]={'path':str(path),'sha256':_sha(bundle/path),'kind':phase,
                'origin_id':origin,'base_origin':base,'future':future,'seed':origins[base],
                'package_id':'broad_diagnostic','recipient_nodes':list(nodes),
                'success':u['success'],'relapsed':u['relapsed'],'success_through_horizon':u['success_through_horizon']}
            manifest['actual_units_completed']+=1;save()
            print(base,phase,future,'nodes',len(nodes),'completed',manifest['actual_units_completed'],flush=True)
        return key,json.loads((bundle/manifest['actual'][key]['path']).read_text())
    try:
        # Freeze every representative before any audit future is evaluated.
        for base in plan['origins']:
            state=_origin_state(profile,origins[base]);problem=_case_problem(profile,origins[base],'broad_diagnostic',-1,base)
            obs=reconstruct_t0_observation(profile,state,origins[base],problem.accessible_roster)
            specs,incoming=strata(obs)
            for spec in specs:
                if spec['size'] not in plan['sizes'] or spec['quantile'] not in plan['power_quantiles']:continue
                label=f"{base}__{spec['id']}"
                if label in manifest['strata']:continue
                if not spec['available']:
                    manifest['strata'][label]={'spec':spec,'available':False};save();continue
                groups=initial_groups(obs,spec,stream_seed(base,spec['id'],0),plan['initial_candidates_per_stratum'])
                scored={}
                def score(nodes):
                    outcomes=[evaluate(base,nodes,'search',f)[1] for f in range(plan['search_futures'])]
                    scored[nodes]=group_score(nodes,outcomes,incoming)
                for nodes in groups:score(nodes)
                best=min(scored,key=scored.get)
                for nodes in exchange_groups(best,spec,groups,stream_seed(base,spec['id'],1),plan['exchange_candidates_per_stratum']):score(nodes)
                best=min(scored,key=scored.get)
                manifest['strata'][label]={'spec':spec,'available':True,'selected':list(best),
                    'candidates':[{'nodes':list(n),'score':list(v)} for n,v in scored.items()]};save()
            for f in range(plan['search_futures']):evaluate(base,(),'search',f)
            if validation:
                from src.control.optimizers.observed_subsets import _proposal_orders
                rng=np.random.default_rng(stream_seed(base,'benchmark_membership',0))
                orders=dict(_proposal_orders(obs,rng)[3])
                manifest.setdefault('benchmark_groups',{})[base]={
                    f'{name}{size}': sorted(int(n) for n in orders[name][:size])
                    for name in plan['benchmark_strategies'] for size in plan['sizes']}
                save()
        manifest['selection_frozen']=True;save()
        for base in plan['origins']:
            selected=[(label,tuple(r['selected'])) for label,r in manifest['strata'].items()
                      if label.startswith(base+'__') and r['available']]
            # Outcome-independent full-roster random benchmarks at every size.
            if validation:
                selected.extend((f'{base}__{role}',tuple(nodes))
                    for role,nodes in manifest['benchmark_groups'][base].items())
            else:
                rng=np.random.default_rng(stream_seed(base,'benchmark_membership',0))
                order=rng.permutation(40)
                selected.extend((f'{base}__random{size}',tuple(sorted(order[:size]))) for size in plan['sizes'])
            selected.append((base+'__zero',()))
            for f in range(plan['audit_futures']):
                for role,nodes in selected:
                    key,u=evaluate(base,nodes,'audit',f)
                    manifest['audit_roles'][f'{role}__future{f}']=key;save()
    except SessionFinished:
        manifest['status']='incomplete_time_ceiling';save();return bundle
    manifest['status']='complete';save()
    # Only untouched audit trajectories appear in the scientific comparison report.
    audit=dict(manifest);audit['actual']={k:v for k,v in manifest['actual'].items() if v['kind']=='audit'}
    study=_write_study_bundle(bundle,audit,profile,contract,'adaptive')
    sm=json.loads((study/'manifest.json').read_text());sm['label']=plan.get('study_label','development_offline_search_heldout_futures')
    sm.setdefault('config',{})['independent_origins']=False
    if validation:sm['config']['stage']='validation'
    _atomic_json(study/'manifest.json',sm)
    rows=[]
    for role,key in manifest['audit_roles'].items():
        u=json.loads((bundle/manifest['actual'][key]['path']).read_text())
        rows.append({'role':role,'physical_unit':key,'nodes':manifest['actual'][key]['recipient_nodes'],
            'valid':u['is_valid'],'eligible':u['is_eligible'],'completion_time':u['completion_time'],
            'hold_completed':u['success'],'relapsed':u['relapsed'],
            'no_relapse_after_certification':u['success_through_horizon'],'exposure':u['expenditure']})
    _atomic_json(bundle/'audit_comparisons.json',rows)
    from src.control.optimizers.group_characteristics import export_characteristics
    export_characteristics(bundle)
    return bundle


def _supervised_worker(out_dir,resume_bundle,queue,validation=False):
    try:
        result=_run_sparse_discovery(out_dir,resume_bundle=resume_bundle,progress_queue=queue,validation=validation)
        queue.put(('done',str(result)))
    except BaseException:
        import traceback
        queue.put(('error',traceback.format_exc()))


def run_sparse_discovery(out_dir,*,resume_bundle=None,plan_only=False,validation=False):
    """Wall-clock supervisor: kill the worker at8h, including an active rollout.

    Completed atomic units remain usable. Rendering is deliberately outside the
    timed compute call. A resumed invocation receives another explicitly started
    session; the wrapper never launches one automatically.
    """
    plan=_plan(validation)
    if plan_only:return plan
    import multiprocessing as mp
    import queue as queue_module
    context=mp.get_context('spawn')
    queue=context.Queue()
    args=(str(out_dir),resume_bundle,queue)
    if validation:args=(*args,True)
    worker=context.Process(target=_supervised_worker,args=args)
    started=time.monotonic();worker.start()
    bundle=Path(resume_bundle) if resume_bundle is not None else None
    done=False;error=None
    try:
        while time.monotonic()-started<plan['session_seconds']:
            remaining=plan['session_seconds']-(time.monotonic()-started)
            try:kind,value=queue.get(timeout=min(.25,max(.001,remaining)))
            except queue_module.Empty:
                if not worker.is_alive():break
                continue
            if kind=='bundle':bundle=Path(value)
            elif kind=='done':bundle=Path(value);done=True;break
            elif kind=='error':error=value;break
        if worker.is_alive():
            worker.terminate();worker.join(timeout=1)
            if worker.is_alive():worker.kill();worker.join(timeout=1)
        if error:raise RuntimeError(error)
        if bundle is None:raise RuntimeError('Worker stopped before creating a bundle; inspect the notebook output')
        if not done:
            path=bundle/'manifest.json';manifest=json.loads(path.read_text())
            manifest['status']='incomplete_hard_time_limit' if time.monotonic()-started>=plan['session_seconds'] else 'incomplete_worker_exit'
            manifest['supervisor_elapsed_seconds']=time.monotonic()-started
            _atomic_json(path,manifest)
        return bundle
    finally:
        if worker.is_alive():worker.terminate();worker.join(timeout=1)
        queue.close()


def summarize_partial(bundle):
    """Use saved completed units without running or selecting any new groups."""
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    rows=[]
    for key,info in manifest['actual'].items():
        path=bundle/info['path']
        if _sha(path)!=info['sha256']:raise ValueError('Partial unit checksum mismatch')
        u=json.loads(path.read_text())
        rows.append({'physical_unit':key,'origin':info['base_origin'],'phase':info['kind'],
            'future':info['future'],'nodes':info['recipient_nodes'],'valid':u['is_valid'],
            'eligible':u['is_eligible'],'completion_time':u['completion_time'],
            'no_relapse_after_certification':u['success_through_horizon'],'exposure':u['expenditure']})
    _atomic_json(bundle/'partial_results.json',{'status':manifest['status'],
        'selection_frozen':manifest.get('selection_frozen',False),
        'completed_physical_units':len(rows),'rows':rows,
        'interpretation':'Completed trajectories only. Search and audit phases remain separate; incomplete allocation is not a completed comparison.'})
    return rows
