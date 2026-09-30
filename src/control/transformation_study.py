"""Focused, resumable matched comparisons over the shared control engine.

No optimizer, response grid, animation or extended campaign runs implicitly.
Normal runs implement the v1 scientific contract; smoke runs are explicitly
shortened software rehearsals. Each completed unit is immutable and hashed.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import uuid

import numpy as np

from src.config.tiered_profiles import LIGHT_TIERED
from src.config.scenarios import tiered_origin_provider
from src.config.transformation import build_transformation_origin, contract_digest, load_transformation_contract, resolve_resource_package
from src.control.checkpointing import load_controlled_checkpoint, serialize_state
from src.control.contracts import ControlProblem, RolloutResult, TargetType
from src.control.controllers import ObservableSelectorPolicy, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import SampledPublicObservationModel
from src.control.recording import RecorderConfig, RolloutRecorder
from src.analytics.influence import calculate_influence_metrics
from src.analytics.communities import connected_components_analysis
from src.execution.runner import NumpyEncoder, _json_safe

ROOT = Path(__file__).resolve().parents[2]


def _runtime_versions() -> dict:
    versions = {}
    for package in ('numpy', 'pydantic', 'matplotlib', 'pandas', 'pillow', 'pyyaml', 'tqdm'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), cls=NumpyEncoder, sort_keys=True, indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload+'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(seed: int, role: str) -> int:
    return int.from_bytes(hashlib.sha256(f'transformation-v1:{seed}:{role}'.encode()).digest()[:8], 'big')


def source_identity() -> dict:
    paths = (sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'research/scenarios').rglob('*.json'))
             + [ROOT/'requirements.txt'])
    hashes = {str(p.relative_to(ROOT)): _sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        head = None
    return {'git_head':head,'source_sha256':digest,'file_sha256':hashes}


def make_policy(name: str):
    if name == 'zero':
        return ZeroController()
    optional = load_transformation_contract().get('optional_policies', {})
    if name in optional:
        return ObservableSelectorPolicy(**optional[name])
    if name not in load_transformation_contract()['policies']:
        raise ValueError(f'Unknown policy: {name}')
    return ObservableSelectorPolicy(
        selector='random' if name.startswith('random_') else 'reported_attention',
        nominal_amplitude=4. if name.endswith('_4') else 1., duration=20.,
        is_feedback=name=='attention_shutoff_1', initial_recruitment_fraction=1.,
        roster_selection=True,
    )


def _structure(state) -> dict:
    inf = calculate_influence_metrics(state.W)
    cc = connected_components_analysis(state.W)
    return {'C_N':float(inf['C_N']),'f_active':float(inf['f_active']),
            'largest_wcc_fraction':float(cc['largest_wcc_fraction'])}


def run_transformation_study(
    out_dir: str | Path, *, smoke: bool = False, n: int = 40,
    origin_ids: tuple[str,...] | None = None,
    policy_packages: tuple[tuple[str,str],...] | None = None,
    target_kind: str = 'state', memory_mode: str = 'adaptive',
    initial_conditions: str = 'synthetic_v1', notebook_code_digest: str | None = None,
    stage: str = 'development', resume_bundle: str | Path | None = None,
    checkpoint_interval: float = 20., progress: bool = True,
    max_units: int | None = None,
    include_zero_baselines: bool = True, plan_only: bool = False,
    stop_after_checkpoints: int | None = None,
) -> Path | dict:
    """Run an explicitly selected queue; resume requires exact configuration/source.

    max_units is a per-invocation execution limit, not a smaller scientific plan.
    It allows clean interruption/recovery tests and bounded cloud batches.
    stop_after_checkpoints returns after a durable within-unit checkpoint, for
    explicit restart rehearsals. It does not alter the scientific configuration.
    Selection requires a frozen explicit list of at most two active comparisons;
    their paired zero is included. No confirmation allocation is defined.
    """
    # plan_only resolves the real queue without creating files or simulating.
    if stop_after_checkpoints is not None and (type(stop_after_checkpoints) is not int or stop_after_checkpoints < 1):
        raise ValueError('stop_after_checkpoints must be a positive integer')
    c = load_transformation_contract()
    if stage == "selection" and not include_zero_baselines:
        raise ValueError("Selection requires matched zero baselines")
    if target_kind not in ('state', 'structural'):
        raise ValueError('target_kind must be state or structural; joint certification is not implemented')
    if memory_mode not in ('adaptive', 'frozen'):
        raise ValueError('memory_mode must be adaptive or frozen')
    target_type = TargetType.REVERSAL if target_kind == 'state' else TargetType.DECENTRALIZATION
    if initial_conditions not in ('synthetic_v1', 'tiered_reference'):
        raise ValueError('Unknown initial-condition family')
    if stage not in ('development','selection'):
        raise ValueError('Only frozen development/selection pools exist')
    if not np.isfinite(checkpoint_interval) or checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be finite and positive')
    if max_units is not None and max_units < 1:
        raise ValueError('max_units must be positive')
    if not smoke and n != 40:
        raise ValueError('v1 initial light development is N=40; freeze the extended allocation first')
    if smoke:
        n = 8
    profile = LIGHT_TIERED.model_copy(update={'N':n})
    origins = c[stage]
    if origin_ids is not None:
        if not origin_ids or len(set(origin_ids)) != len(origin_ids) or not set(origin_ids)<= {o['origin_id'] for o in origins}:
            raise ValueError('origin_ids must be unique members of the frozen pool')
        origins = [o for o in origins if o['origin_id'] in origin_ids]
    elif smoke:
        origins = origins[:1]
    if policy_packages is None:
        if stage == 'selection':
            raise ValueError('Selection needs frozen policy_packages chosen using development only')
        packages = c['packages'][:1] if smoke else c['packages']
        pairs = [(p,pack['id']) for pack in packages for p in c['policies']]
    else:
        pairs = list(policy_packages)
        if not pairs or len(set(pairs))!=len(pairs):
            raise ValueError('policy_packages must be nonempty and unique')
        if stage=='selection' and len([p for p,_ in pairs if p!='zero'])>2:
            raise ValueError('Selection permits at most two frozen active comparisons')
        for _,pack in list(pairs):
            if include_zero_baselines and ('zero',pack) not in pairs:
                pairs.insert(0,('zero',pack))
    for policy,package in pairs:
        make_policy(policy)
        resolve_resource_package(package,n)
    horizon,release,hold = (6.,2.,2.) if smoke else (c['horizon'],c['release_deadline'],c['holding_duration'])
    queue = [{'key':f"{o['origin_id']}__{pack}__{p}",'origin':{**o,'cell_id':o['cell_id'] if initial_conditions=='synthetic_v1' else None},'policy':p,'package':pack}
             for o in origins for p,pack in pairs]
    physical_count = sum(q['policy'] != 'zero' for q in queue) + len({
        q['origin']['origin_id'] for q in queue if q['policy'] == 'zero'})
    if plan_only:
        return {'queue': queue, 'comparisons': len(queue),
                'physical_continuations': physical_count, 'memory_mode': memory_mode,
                'initial_conditions': initial_conditions, 'N': n}
    identity = source_identity()
    config = {'runtime':{'python':platform.python_version(),'architecture':platform.machine(),'packages':_runtime_versions()},
              'contract_sha256':contract_digest(),'source_sha256':identity['source_sha256'],
              'target_kind':target_kind,'memory_mode':memory_mode,'stage':stage,'smoke':smoke,'N':n,'initial_conditions':initial_conditions,
              'notebook_code_digest':notebook_code_digest,'horizon':horizon,'release_deadline':release,
              'holding_duration':hold,'checkpoint_interval':checkpoint_interval,
              'profile':profile.model_dump(),'queue':queue}
    # Canonicalize tuples/numpy scalars exactly as the persisted JSON will.
    config = json.loads(json.dumps(config, cls=NumpyEncoder))
    if resume_bundle is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        bundle=Path(out_dir)/f'transformation_v1_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True,exist_ok=False)
        manifest={'schema_version':1,'status':'running','config':config,'contract':c,'source':identity,
                  'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'packages':_runtime_versions()},
                  'label':'SOFTWARE_SMOKE_NOT_SCIENTIFIC_EVIDENCE' if smoke else 'development_not_confirmation',
                  'units':{},'origins':{}}
        _atomic_json(bundle/'manifest.json',manifest)
    else:
        bundle=Path(resume_bundle)
        manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest['config']!=config:
            raise ValueError('Resume configuration/source mismatch; start a new result series')
        for info in list(manifest['units'].values())+list(manifest['origins'].values()):
            if _sha(bundle/info['path'])!=info['sha256']:
                raise ValueError(f"Saved artifact checksum mismatch: {info['path']}")
    if progress:
        print(f'Bundle: {bundle.resolve()} | {len(queue)} comparisons | memory={memory_mode}', flush=True)
    completed_this_call=0
    checkpoints_this_call=0
    for unit in queue:
        key=unit['key']
        if key in manifest['units']:
            continue
        origin=unit['origin']; seed=origin['seed']
        if unit['policy'] == 'zero':
            # Zero ignores the roster and spends nothing. With independent
            # observation RNG and identical physics/seed/target, packages do
            # not alter its physical trajectory. Reuse evidence explicitly.
            canonical = next((q['key'] for q in queue
                if q['origin']['origin_id'] == origin['origin_id'] and q['policy'] == 'zero'
                and q['key'] in manifest['units'] and 'reused_from' not in manifest['units'][q['key']]), None)
            if canonical is not None:
                path=Path('units')/(key+'.json')
                _atomic_json(bundle/path, {'reference_unit':canonical, 'unit':unit,
                    'comparison_geometry':resolve_resource_package(unit['package'],n)})
                manifest['units'][key]={'path':str(path),'sha256':_sha(bundle/path),'reused_from':canonical}
                manifest['physical_continuations_completed']=sum('reused_from' not in u for u in manifest['units'].values())
                manifest['status']='complete' if len(manifest['units'])==len(queue) else 'incomplete'
                _atomic_json(bundle/'manifest.json',manifest)
                if progress:
                    print(f'{key}: reused exact zero trajectory {canonical}',flush=True)
                continue
        state=(build_transformation_origin(profile,seed,origin['cell_id'])
               if initial_conditions=='synthetic_v1' else tiered_origin_provider(profile,seed))
        origin_path=Path('origins')/(origin['origin_id']+'.json')
        if origin['origin_id'] not in manifest['origins']:
            _atomic_json(bundle/origin_path,serialize_state(state))
            manifest['origins'][origin['origin_id']]={'path':str(origin_path),'sha256':_sha(bundle/origin_path)}
            _atomic_json(bundle/'manifest.json',manifest)
        geometry=resolve_resource_package(unit['package'],n)
        roster_rng=np.random.default_rng(_seed(seed,'roster'))
        roster=tuple(sorted(int(i) for i in roster_rng.permutation(n)[:geometry['accessible_count']]))
        problem=ControlProblem(problem_id=key, target_type=target_type,
            target_params={'eligibility_mode':'declared','declared_destination_sign':c['destination_sign'],
                           'threshold':c['state_mean_threshold']},N=n,k=geometry['k'],
            exposure_cap=geometry['exposure_cap'],amplitude_cap=geometry['amplitude_cap'],
            accessible_roster=roster,release_deadline=release,holding_duration=hold,horizon=horizon,
            config_version=c['contract_version'])
        evaluator=TargetEvaluator.create_for_state(target_type,state,hold,horizon,release,
            problem_id=key,policy_id=unit['policy'],origin_id=origin['origin_id'],target_params=problem.target_params)
        obs=SampledPublicObservationModel(N=n,accessible_roster=roster,
            panel_fraction=profile.panel_fraction,binary_flip_prob=profile.binary_flip_prob,
            edge_reveal_prob=profile.edge_reveal_prob,delivery_delay=profile.delivery_delay_tau_public,
            rng=np.random.default_rng(_seed(seed,'observation')))
        recorder=RolloutRecorder(RecorderConfig(cadence=1. if smoke else 5.,record_network_frames=True,
            max_network_frames=100,retention_window=(release,horizon)))
        checkpoint=bundle/'checkpoints'/(key+'.json')
        current=load_controlled_checkpoint(checkpoint,expected_problem=problem).checkpoint_time if checkpoint.exists() else 0.
        started=time.monotonic()
        while current<horizon:
            cut=min(current+checkpoint_interval,horizon)
            result,final,ledger=run_controlled_simulation(
                record_target_transitions=True, adaptive_memory=memory_mode == "adaptive",
                initial_state=state,problem=problem,controller=make_policy(unit['policy']),
                observation_model=obs,target_evaluator=evaluator,recorder=recorder,
                sim_rng=np.random.default_rng(_seed(seed,'simulation')),
                controller_rng=np.random.default_rng(_seed(seed,'controller')),
                J=profile.J,T=profile.temperature,delta=profile.delta,epsilon=profile.exploration,
                closure=profile.closure,a=profile.prestige_global,c=profile.matching,tau_y=profile.tau_public,
                sample_interval=profile.sample_interval_tau_public,
                checkpoint_at=cut if cut<horizon else None,stop_at_checkpoint=cut<horizon,
                checkpoint_file=checkpoint if cut<horizon else None,
                resume_checkpoint=checkpoint if checkpoint.exists() else None)
            current=float(final.t)
            if current < horizon:
                checkpoints_this_call += 1
                if stop_after_checkpoints is not None and checkpoints_this_call >= stop_after_checkpoints:
                    manifest['status'] = 'incomplete'
                    _atomic_json(bundle/'manifest.json',manifest)
                    if progress:
                        print(f'Durable checkpoint saved at t={current:g}; resume bundle {bundle}',flush=True)
                    return bundle
            if progress:
                print(f'{key}: t={current:g}/{horizon:g}',flush=True)
        output={f.name:getattr(result,f.name) for f in fields(RolloutResult) if f.name!='recorder'}
        output.update({'unit':unit,'target_definition':asdict(evaluator.defn),'problem':asdict(problem),'initial_conditions':initial_conditions,'geometry':geometry,'accessible_roster':roster,
            'seeds':{role:_seed(seed,role) for role in ('roster','simulation','observation','controller')},
            'initial_structure':_structure(state),'final_structure':_structure(final),
            'final_state':serialize_state(final),
            'network_frames':[f.to_dict() for f in result.recorder.network_frames],
            'elapsed_seconds_this_invocation':time.monotonic()-started,
            'success_through_horizon':bool(result.success and not result.relapsed)})
        path=Path('units')/(key+'.json')
        _atomic_json(bundle/path,output)
        manifest['units'][key]={'path':str(path),'sha256':_sha(bundle/path)}
        completed_this_call+=1
        manifest['physical_continuations_completed']=sum('reused_from' not in u for u in manifest['units'].values())
        manifest['status']='complete' if len(manifest['units'])==len(queue) else 'incomplete'
        _atomic_json(bundle/'manifest.json',manifest)
        if checkpoint.exists():
            checkpoint.unlink()
        if max_units is not None and completed_this_call>=max_units:
            break
    return bundle
