"""Pre-final fixed-group repeated-future experiment on development origins.

No surrogate fitting or outcome-dependent selection. Strategies and seeds are
frozen before execution. Completed physical trajectories are reused on resume;
interrupted trajectories repeat. Source/runtime/profile identity must match.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid
import numpy as np

from src.config.tiered_profiles import LIGHT_TIERED
from src.control.optimizers.campaign import (
    _atomic_json, _sha, _origin_state, _case_problem, reconstruct_t0_observation,
    run_actual_unit, _write_study_bundle)
from src.control.optimizers.observed_subsets import _proposal_orders
from src.control.transformation_study import source_identity, _runtime_versions, load_transformation_contract

from src.control.optimizers.hypothesis_plan import hypothesis_plan, future_seed, restricted_low_coverage


def _parent_records(parent_bundle):
    parent_bundle=Path(parent_bundle)
    m=json.loads((parent_bundle/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('Batch1 must be complete')
    result={}
    for key,info in m['actual'].items():
        if _sha(parent_bundle/info['path'])!=info['sha256']:raise ValueError('Batch1 unit integrity failure')
        for role in info.get('roles',[info['kind']]):
            result[(info['origin_id'],role)]=info
    return m,result


def run_hypothesis_batch(out_dir, parent_bundle, *, resume_bundle=None, session_seconds=28800, plan_only=False):
    plan=hypothesis_plan()
    if plan_only:return plan
    if not np.isfinite(session_seconds) or session_seconds<=0:raise ValueError('Invalid session budget')
    parent,records=_parent_records(parent_bundle)
    parent_hash=_sha(Path(parent_bundle)/'manifest.json')
    profile=LIGHT_TIERED
    contract=load_transformation_contract()
    identity={'source_sha256':source_identity()['source_sha256'], 'plan':plan,
              'runtime':{'python':platform.python_version(),'packages':_runtime_versions()},
              'profile':profile.model_dump(),'contract':contract,'parent_manifest_sha256':parent_hash}
    identity=json.loads(json.dumps(identity))
    if resume_bundle is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        bundle=Path(out_dir)/f'hypotheses_v1_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True,exist_ok=False)
        manifest={'schema_version':1,'identity':identity,'status':'incomplete',
                  'actual':{},'comparisons':{},'unavailable':{},'actual_units_completed':0}
        _atomic_json(bundle/'manifest.json',manifest)
    else:
        bundle=Path(resume_bundle);manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest['identity']!=identity:raise ValueError('Validation resume identity mismatch')
        for info in manifest['actual'].values():
            if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Saved unit integrity failure')
    print('Hypothesis batch bundle:',bundle,flush=True)
    start=time.monotonic()
    origin_seeds=parent['identity']['origins']
    for base_origin in plan['development_origins']:
      seed=origin_seeds[base_origin]
      state=_origin_state(profile,seed)
      # Verify exact recreated initial state against the saved Batch1 origin.
      from src.control.checkpointing import serialize_state
      from src.execution.runner import NumpyEncoder
      saved_path=Path(parent_bundle)/'study/origins'/f'{base_origin}.json'
      if saved_path.exists():
          recreated=json.loads(json.dumps(serialize_state(state),cls=NumpyEncoder))
          if recreated!=json.loads(saved_path.read_text()):
              raise ValueError('Initial state differs from Batch1; use its runtime before repeating futures')
      problem=_case_problem(profile,seed,'broad_diagnostic',-1,base_origin)
      observation=reconstruct_t0_observation(profile,state,seed,problem.accessible_roster)
      low=restricted_low_coverage(observation,4,seed)
      for future in range(plan['futures_per_origin']):
        origin=f'{base_origin}_future{future}'
        queue=[]
        for name,role,size in [('random4','selector_random_4',4),('coverage4','selector_audience_coverage_4',4),
                               ('random10','selector_random_10',10),('coverage10','selector_audience_coverage_10',10)]:
            queue.append((name,size,tuple(records[(base_origin,role)]['recipient_nodes'])))
        if low['available']:queue.append(('restricted_low_coverage4',4,tuple(low['nodes'])))
        else:manifest['unavailable'][origin]=low
        queue.append(('zero',0,()))
        for name,size,nodes in queue:
            role=f'{name}_{size}'
            key=f"{origin}__broad_diagnostic__{'_'.join(map(str,nodes)) or 'zero'}"
            comparison=f'{origin}__{role}'
            if key not in manifest['actual']:
                if time.monotonic()-start>=session_seconds:
                    manifest['status']='incomplete_time_ceiling';_atomic_json(bundle/'manifest.json',manifest)
                    return bundle
                unit=run_actual_unit(profile,seed,origin,contract,'adaptive','broad_diagnostic',nodes,
                                     2. if size else 0.,19. if size else 0.,-1,
                                     simulation_seed=future_seed(base_origin,future))
                path=Path('units')/(key+'.json');_atomic_json(bundle/path,unit)
                manifest['actual'][key]={'path':str(path),'sha256':_sha(bundle/path),'kind':role,'roles':[],
                    'recipient_nodes':list(nodes),'origin_id':origin,'package_id':'broad_diagnostic','seed':seed,'base_origin':base_origin,'future':future,
                    'success':unit['success'],'relapsed':unit['relapsed'],
                    'success_through_horizon':unit['success_through_horizon']}
                manifest['actual_units_completed']+=1
            info=manifest['actual'][key]
            if role not in info['roles']:info['roles'].append(role)
            manifest['comparisons'][comparison]=key
            manifest['status']='incomplete';_atomic_json(bundle/'manifest.json',manifest)
            print(origin,role,'saved',manifest['actual_units_completed'],'physical trajectories',flush=True)
    manifest['status']='complete';_atomic_json(bundle/'manifest.json',manifest)
    study=_write_study_bundle(bundle,manifest,profile,contract,'adaptive')
    sm=json.loads((study/'manifest.json').read_text())
    sm['label']='development_repeated_futures';sm['config']['stage']='development'
    _atomic_json(study/'manifest.json',sm)
    # Role table retains aliased strategies and validity, without counting aliases as replicates.
    rows=[]
    for comparison,key in manifest['comparisons'].items():
        info=manifest['actual'][key];u=json.loads((bundle/info['path']).read_text())
        rows.append({'comparison':comparison,'physical_unit':key,'nodes':info['recipient_nodes'],
                     'base_origin':info['base_origin'],'future':info['future'],
                     'role':comparison.split('__')[-1],
                     'valid':u['is_valid'],'eligible':u['is_eligible'],'exposure':u['expenditure'],
                     'completion_time':u['completion_time'],'relapsed':u['relapsed'],
                     'no_relapse_after_certification':u['success_through_horizon']})
    _atomic_json(bundle/'comparison_table.json',rows)
    groups={}
    for row in rows:
        key=(row['base_origin'],row['role'])
        groups.setdefault(key,[]).append(row)
    summaries=[]
    for (origin,role),group in sorted(groups.items()):
        summaries.append({'origin':origin,'role':role,'futures':len(group),
            'valid':sum(r['valid'] and r['eligible'] for r in group),
            'hold_completions':sum(r['completion_time'] is not None for r in group),
            'no_relapse_successes':sum(r['no_relapse_after_certification'] for r in group),
            'mean_restricted_completion_time':float(np.mean([
                r['completion_time'] if r['no_relapse_after_certification'] else 320. for r in group]))})
    _atomic_json(bundle/'per_origin_summary.json',summaries)
    return bundle
