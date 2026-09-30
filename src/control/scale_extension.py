"""Prospective N40/N400 extension. Shared physics; frozen, outcome-blind design."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, platform, subprocess, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.optimizers.campaign import (_origin_state, reconstruct_t0_observation,
    observation_identity, _role_seed, _atomic_json, _sha)
from src.control.optimizers.observed_subsets import _proposal_orders
from src.control.checkpointing import serialize_state
from src.control.contracts import ControlProblem, TargetType
from src.control.controllers import OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import SampledPublicObservationModel
from src.control.recording import RolloutRecorder, RecorderConfig
from src.execution.runner import _json_safe, NumpyEncoder

VERSION='scale-extension-20261008-v1'
ROOT=Path(__file__).resolve().parents[2]

def seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,(VERSION,*parts))).encode()).digest()[:8],'big')

def plan():
    return dict(version=VERSION,sizes=[40,400],release=20.,hold=100.,horizon=320.,
        amplitude=2.,duration=19.,tiers=[[8,2],[6,2],[4,2],[4,1]],
        screening_limit=1000,session_seconds=28800,planning_seconds=21600,
        export_reserve=1200,safety_factor=1.5,bootstrap_seed=1082026,bootstrap_draws=10000,
        primary='Within each N: adaptive coverage10 minus adaptive random10 in persistent success',
        endpoint='ceil(0.75*N) in EACH layer continuously for 100 after common release20, with no later exit through320',
        scaling='largest-remainder tier/role/jurisdiction allocation; unchanged stance priors/traits/affinities; 20 initial tokens and40 cap per node; dense opportunity pool, bounded realized attention capacity',
        inference='Independent origins at each N; futures nested within origins; sizes NOT paired. Exploratory cluster bootstrap; no reliability or empirical prediction claim.',
        roles=['zero','random4','coverage4','random10','coverage10','coverage25','coverage10_equal_total'],
        memory=['adaptive','frozen'],
        analysis='All treated-versus-zero effects; coverage10-random10 primary; coverage25-coverage10 dose; coverage10-coverage4 count/fraction; coverage10_equal_total-coverage4 matched total; adaptive/frozen difference of differences; independent-size contrasts.',
        no_extension='First fitting timing-only tier, no spare-time expansion, no outcome-driven policy selection',
        recording='Exact changed-spin times and predicates all units; full initial/final state; sparse W plus both layers in65 frames all future0 units; exact ledger; seeds enable full event replay')

def profile(N): return LIGHT_TIERED.model_copy(update={'N':N})

def make_groups(N, obs, origin_seed):
    orders=dict(_proposal_orders(obs,np.random.default_rng(seed('selector',N,origin_seed)))[3])
    k10=math.ceil(.1*N);k25=math.ceil(.25*N)
    specs={'zero':([],0.),'random4':(orders['random'][:4],2.),
        'coverage4':(orders['audience_coverage'][:4],2.),
        'random10':(orders['random'][:k10],2.),'coverage10':(orders['audience_coverage'][:k10],2.),
        'coverage25':(orders['audience_coverage'][:k25],2.),
        'coverage10_equal_total':(orders['audience_coverage'][:k10],8./k10)}
    return {f'{role}_{memory}':dict(nodes=sorted(map(int,nodes)),amplitude=amp,memory=memory)
        for memory in ('adaptive','frozen') for role,(nodes,amp) in specs.items()}

def physical_signature(g): return json.dumps(g,sort_keys=True,separators=(',',':'))

def prepare(N, phase, count):
    accepted=[];screen=[]
    for candidate in range(plan()['screening_limit']):
        s=seed('origin',N,phase,candidate);state=_origin_state(profile(N),s)
        nx=int((state.x<0).sum());ny=int((state.y<0).sum())
        eligible=nx<N/2 and ny<N/2
        screen.append(dict(candidate=candidate,seed=s,private_blue=nx,public_blue=ny,eligible=eligible))
        if not eligible:continue
        obs=reconstruct_t0_observation(profile(N),state,s,tuple(range(N)))
        groups=make_groups(N,obs,s)
        accepted.append(dict(id=f'{phase}_N{N}_{len(accepted):02d}',N=N,seed=s,
            candidate=candidate,initial=serialize_state(state),groups=groups,
            observation=dict(panel=obs.panel_mask.tolist(),public=_json_safe(obs.public_states.tolist()),
                edges=[[int(i),int(j),int(v)] for (i,j),v in sorted(obs.observed_edges.items())],
                sha256=observation_identity(obs))))
        if len(accepted)==count:break
    if len(accepted)!=count:raise RuntimeError('Prespecified initial-state screening exhausted')
    return accepted,screen

class ExactRecorder(RolloutRecorder):
    """No RNG access. Compact exact spin changes; inherited event-exact target log."""
    def initialize(self,state,ledger,target_evaluator,problem):
        self.flips=[];self._x=state.x.copy();self._y=state.y.copy()
        super().initialize(state,ledger,target_evaluator,problem)
    def on_event_state(self,state,current_time,ledger,target_evaluator):
        super().on_event_state(state,current_time,ledger,target_evaluator)
        for layer,old,new in ((0,self._x,state.x),(1,self._y,state.y)):
            for i,a in np.argwhere(old!=new):
                self.flips.append([float(current_time),layer,int(i),int(old[i,a]),int(new[i,a])])
            old[:]=new
    def to_dict(self):
        d=super().to_dict();d['spin_changes']=self.flips;return d

def sparse_frame(frame):
    w=np.asarray(frame.W);i,j=np.nonzero(w)
    return dict(t=float(frame.t),x=frame.x[:,0].astype(int).tolist(),y=frame.y[:,0].astype(int).tolist(),
        edges=np.column_stack((i,j,w[i,j])).astype(int).tolist(),
        controlled=np.flatnonzero(frame.controlled_mask).tolist())

def simulate(origin,g,future,contract,phase='audit',frames=True):
    N=origin['N'];p=profile(N);s=origin['seed'];state=_origin_state(p,s)
    nodes=g['nodes'];amplitude=g['amplitude'];release=contract['release'];duration=contract['duration']
    problem=ControlProblem(problem_id=origin['id'],target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1,'threshold':.5},
        N=N,D=1,k=N,exposure_cap=max(1.,len(nodes)*amplitude*duration),amplitude_cap=2.,
        accessible_roster=tuple(range(N)),release_deadline=release,holding_duration=contract['hold'],
        horizon=contract['horizon'],config_version=VERSION)
    ev=TargetEvaluator.create_for_state(TargetType.REVERSAL,state,contract['hold'],contract['horizon'],release,
        target_params=problem.target_params)
    controller=OpenLoopPulseController(nodes,amplitude,duration,{i:-1. for i in nodes}) if nodes else ZeroController()
    obs=SampledPublicObservationModel(N=N,accessible_roster=tuple(range(N)),panel_fraction=p.panel_fraction,
        binary_flip_prob=p.binary_flip_prob,edge_reveal_prob=p.edge_reveal_prob,
        delivery_delay=p.delivery_delay_tau_public,rng=np.random.default_rng(_role_seed(s,'observation')))
    rec=ExactRecorder(RecorderConfig(cadence=min(1.,contract['horizon']/10),record_network_frames=frames,
        max_network_frames=65,retention_window=(release,contract['horizon'])))
    simseed=seed('future',N,origin['id'],phase,future)
    start=time.monotonic()
    result,final,ledger=run_controlled_simulation(record_target_transitions=True,
        adaptive_memory=g['memory']=='adaptive',initial_state=state,problem=problem,controller=controller,
        observation_model=obs,target_evaluator=ev,recorder=rec,sim_rng=np.random.default_rng(simseed),
        controller_rng=np.random.default_rng(_role_seed(s,'controller')),J=p.J,T=p.temperature,
        delta=p.delta,epsilon=p.exploration,closure=p.closure,a=p.prestige_global,c=p.matching,
        tau_y=p.tau_public,sample_interval=p.sample_interval_tau_public)
    if not result.is_valid or not result.is_eligible:raise RuntimeError('Invalid continuation')
    expected=len(nodes)*amplitude*duration
    if not math.isclose(result.expenditure,expected,rel_tol=1e-8,abs_tol=1e-7):
        raise AssertionError(f'Exposure {result.expenditure} != declared {expected}')
    output=dict(origin=origin['id'],N=N,origin_seed=s,simulation_seed=simseed,future=future,phase=phase,
        group=g,profile=p.model_dump(),contract=contract,elapsed=time.monotonic()-start,
        exposure=result.expenditure,event_counts=result.event_counts,
        engine_success=bool(result.success and not result.relapsed),engine_completion=result.completion_time,
        summaries=result.summaries,final_state=serialize_state(final),frames=[sparse_frame(f) for f in rec.network_frames])
    output['endpoint']=reconstruct_endpoint(origin['initial'],output)
    # Treated arms release at the declared endpoint; zero uses the common20 comparison window.
    if nodes and (output['endpoint']['persistent']!=output['engine_success']):
        raise AssertionError('Independent event replay disagrees with shared evaluator')
    return output

def reconstruct_endpoint(initial,u):
    c=u['contract'];N=u['N'];release=c['release'];H=c['horizon'];hold=c['hold'];threshold=math.ceil(.75*N)
    x=np.array(initial['x'],dtype=int).reshape(-1);y=np.array(initial['y'],dtype=int).reshape(-1)
    counts=[int((x<0).sum()),int((y<0).sum())];points=[[0.,*counts]];last=0.
    for t,layer,node,before,after in u['summaries']['recorder']['spin_changes']:
        if t<last or not 0<=node<N:raise AssertionError('Malformed spin trace')
        arr=x if layer==0 else y
        if arr[node]!=before:raise AssertionError('Spin replay before-state mismatch')
        arr[node]=after;counts[layer]+=int(after<0)-int(before<0)
        points.append([float(t),*counts]);last=t
    if not np.array_equal(x,np.asarray(u['final_state']['x']).reshape(-1)) or not np.array_equal(y,np.asarray(u['final_state']['y']).reshape(-1)):
        raise AssertionError('Spin replay final state mismatch')
    # Coalesce identical timestamps, then evaluate right-continuous intervals exactly.
    timeline=[]
    for row in points:
        if timeline and timeline[-1][0]==row[0]:timeline[-1]=row
        else:timeline.append(row)
    segments=[]
    for i,(t,a,b) in enumerate(timeline):
        end=timeline[i+1][0] if i+1<len(timeline) else H
        if (end>release or t>=release) and t<=H:segments.append((max(t,release),min(end,H),a,b))
    onset=None;cert=None;relapsed=False;below=0.;longest=0.;excursion=0.;mins=[N,N]
    postmins=[N,N];postbelow=0.
    for index,(lo,hi,a,b) in enumerate(segments):
        if hi<lo:continue
        mins=[min(mins[0],a),min(mins[1],b)];ok=a>=threshold and b>=threshold
        if ok:
            if onset is None:onset=lo
            boundary=onset+hold
            after=segments[index+1][2:] if index+1<len(segments) else (a,b)
            # A violating event exactly at the hold boundary resets before certification.
            if cert is None and (boundary<hi or (boundary==hi and min(after)>=threshold)):
                cert=boundary
            excursion=0.
        else:
            onset=None;below+=hi-lo;excursion+=hi-lo;longest=max(longest,excursion)
            if cert is not None and hi>=cert:relapsed=True
        if cert is not None and hi>=cert:
            postmins=[min(postmins[0],a),min(postmins[1],b)]
            if not ok:postbelow+=max(0.,hi-max(lo,cert))
    sensitivity={}
    for frac in (.6,.7,.75):
        cutoff=math.ceil(frac*N)
        sensitivity[str(frac)]=None if cert is None else bool(postmins[0]>=cutoff and postmins[1]>=cutoff)
    return dict(threshold=threshold,certification=cert,persistent=bool(cert is not None and not relapsed),
        relapsed=relapsed,min_private=mins[0],min_public=mins[1],below75_duration=below,
        longest_below75=longest,postcert_min_private=None if cert is None else postmins[0],
        postcert_min_public=None if cert is None else postmins[1],postcert_below75_duration=postbelow,
        hysteresis_sensitivity=sensitivity,final_private=int((x<0).sum()),final_public=int((y<0).sum()),
        penalized_time=cert if cert is not None and not relapsed else H)

def digest_data(obj):
    return hashlib.sha256(json.dumps(_json_safe(obj),cls=NumpyEncoder,sort_keys=True,allow_nan=False).encode()).hexdigest()

def write_gzip(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',encoding='utf8') as f:json.dump(_json_safe(data),f,cls=NumpyEncoder,separators=(',',':'),allow_nan=False)
    os.replace(temp,path)

def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)

def identity():
    files={str(p.relative_to(ROOT)):_sha(p) for folder in ('src','research/scenarios/tiered')
        for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}
    import importlib.metadata
    packages={name:importlib.metadata.version(name) for name in ('numpy','pydantic','pydantic_core','pandas','matplotlib','PyYAML','pillow','tqdm','ipywidgets')}
    return dict(files=files,python=platform.python_version(),packages=packages)

def allocate(timings,remaining,contract):
    cost={N:max(v for n,v in timings if n==N)*contract['safety_factor'] for N in contract['sizes']}
    budget=min(contract['planning_seconds'],remaining-contract['export_reserve'])
    # Upper physical counts per origin/future: N40 deduplicates to8, N400 to14.
    for origins,futures in contract['tiers']:
        estimate=origins*futures*(8*cost[40]+14*cost[400])
        if estimate<=budget:return dict(origins=origins,futures=futures,estimated_seconds=estimate,
            budget_seconds=budget,seconds_per_unit=cost)
    return None

def science(bundle,deadline):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text());c=m['plan']
    if m['identity']!=identity():raise ValueError('Exact source/runtime required for resume')
    if m.get('software_only'):
        if c['horizon']>4:raise ValueError('Software rehearsal must remain tiny')
    elif c!=plan():raise ValueError('Frozen nominal plan changed')
    for info in m['units'].values():
        if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Unit checksum mismatch')
    def save():_atomic_json(bundle/'manifest.json',m)
    if not m.get('preflight',{}).get('passed'):
        from src.control.scale_extension_preflight import preflight
        m['preflight']=preflight();save()
    def unit(origin,g,f,phase,key):
        if key in m['units']:return m['units'][key]
        if time.monotonic()>deadline:raise TimeoutError('Science deadline')
        start=time.monotonic();u=simulate(origin,g,f,c,phase,frames=f==0)
        path=f'units/{key}.json.gz';write_gzip(bundle/path,u)
        info=dict(path=path,sha256=_sha(bundle/path),elapsed=time.monotonic()-start,N=origin['N'],
            phase=phase,origin=origin['id'],future=f,group=g)
        m['units'][key]=info;save();print(key,'saved',round(info['elapsed'],2),'seconds',flush=True)
        return info
    if not m.get('prepared'):
        origins={};screen={}
        for N in c['sizes']:
            for phase,count in [('timing',1),('audit',8)]:
                rows,screening=prepare(N,phase,count);screen[f'{N}_{phase}']=screening
                for row in rows:origins[row['id']]=row
        write_gzip(bundle/'origins.json.gz',dict(origins=origins,screening=screen))
        m['origins_sha256']=_sha(bundle/'origins.json.gz');m['prepared']=True;save()
    if _sha(bundle/'origins.json.gz')!=m['origins_sha256']:raise ValueError('Origin checksum mismatch')
    origins=read_gzip(bundle/'origins.json.gz')['origins']
    if not m.get('allocation_decided'):
        timings=[]
        for N in c['sizes']:
            origin=origins[f'timing_N{N}_00']
            for role in ('zero_adaptive','coverage25_adaptive','coverage25_frozen'):
                info=unit(origin,origin['groups'][role],0,'timing',f'timing_N{N}_{role}')
                timings.append([N,info['elapsed']])
            # Exact fresh execution with identical source, state and stream. No extra replicate.
            role='zero_adaptive';a=read_gzip(bundle/m['units'][f'timing_N{N}_{role}']['path'])
            info=unit(origin,origin['groups'][role],0,'timing',f'replay_N{N}')
            b=read_gzip(bundle/info['path'])
            def comparable(u):return {k:v for k,v in u.items() if k!='elapsed'}
            if digest_data(comparable(a))!=digest_data(comparable(b)):raise AssertionError('Fresh deterministic replay mismatch')
            m.setdefault('replay',{})[str(N)]=dict(passed=True,original=f'timing_N{N}_{role}',repeat=f'replay_N{N}')
            timings.append([N,info['elapsed']]);save()
        m['timings']=timings
        # deadline supplied to science already excludes export reserve.
        m['allocation']=allocate(timings,deadline-time.monotonic()+c['export_reserve'],c)
        m['allocation_decided']=True;save()
    if m['allocation'] is None:m['science_status']='timing_only_insufficient_budget';save();return
    a=m['allocation'];queue=[];roles={}
    for f in range(a['futures']):
        for i in range(a['origins']):
            for N in c['sizes']:
                origin=origins[f'audit_N{N}_{i:02d}'];physical={}
                for role,g in origin['groups'].items():
                    sig=physical_signature(g)
                    key=f'{origin["id"]}_f{f}_{hashlib.sha256(sig.encode()).hexdigest()[:12]}'
                    roles[f'{origin["id"]}__f{f}__{role}']=key
                    if sig not in physical:queue.append(dict(key=key,origin=origin['id'],group=g,future=f));physical[sig]=key
    if m.get('queue') is not None and (m['queue']!=queue or m['roles']!=roles):raise ValueError('Frozen queue changed')
    m['queue']=queue;m['roles']=roles;save()
    for job in queue:unit(origins[job['origin']],job['group'],job['future'],'audit',job['key'])
    m['science_status']='complete';save()

def supervise(bundle,hard_deadline):
    """Parent enforces deadlines even inside a slow event, serializer or exporter."""
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    science_deadline=hard_deadline-m['plan']['export_reserve']
    failures=[]
    for mode,stop in [('science',science_deadline),('export',hard_deadline-2.)]:
        remaining=stop-time.monotonic()
        if remaining<=0:
            m=json.loads((bundle/'manifest.json').read_text())
            if m.get(mode+'_status')=='pending':m[mode+'_status']='incomplete_deadline'
            _atomic_json(bundle/'manifest.json',m)
            continue
        command=[sys.executable,'-m','src.control.scale_extension',mode,str(bundle),str(science_deadline)]
        p=subprocess.Popen(command,cwd=ROOT)
        timed_out=False
        print('Launching',mode,'worker',flush=True)
        try:p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out=True
            p.kill();p.wait()
            print(mode,'stopped at enforced deadline',flush=True)
        m=json.loads((bundle/'manifest.json').read_text())
        m[mode+'_process_returncode']=p.returncode
        if p.returncode and not timed_out:
            m[mode+'_status']='failed'
            failures.append(f'{mode} worker exited {p.returncode}; inspect {bundle / (mode + ".log")}')
        elif mode=='science' and m['science_status']=='pending':m['science_status']='incomplete'
        _atomic_json(bundle/'manifest.json',m)
    if failures:raise RuntimeError('; '.join(failures))
    return str(bundle)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['science','export']);parser.add_argument('bundle');parser.add_argument('deadline',type=float)
    args=parser.parse_args()
    log=(Path(args.bundle)/(args.mode+'.log')).open('a',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,text):self.stream.write(text);log.write(text)
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    if args.mode=='science':science(args.bundle,args.deadline)
    else:
        from src.control.scale_extension_analysis import export
        export(Path(args.bundle))
