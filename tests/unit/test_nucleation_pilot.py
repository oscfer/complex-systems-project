"""No nominal pilot physics. One N4/H2 diagnostic equivalence test only."""
import json,copy,sys,time
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from src.control.optimizers.nucleation_pilot import pilot_plan,select_groups,choose_allocation
from src.control import nucleation_batch as batch
from src.control import nucleation_analysis as analysis
from src.control.contracts import Observation,ControlProblem,TargetType
from src.control.recording import RecorderConfig,RolloutRecorder
from src.control.nucleation_recording import NucleationRecorder
from src.control.evaluator import evaluate_holding_trace


def observed():
    W=np.full((40,40),.05);W[:20,:20]=2;W[20:,20:]=2;np.fill_diagonal(W,0)
    return Observation(1.,np.ones((40,1)),np.ones(40,bool),{(i,j):float(W[i,j]) for i,j in zip(*np.nonzero(W))},tuple(range(40)))


def test_selection_information_and_membership():
    p=pilot_plan();o=observed();g,c=select_groups(o,p,42)
    assert len(g)==20 and len(c)==2
    for k in p['sizes']:
        core=set(c[f'core{k}']);n=set(g[f'nucleus{k}']['nodes']);b=set(g[f'nucleus_bridge{k}']['nodes'])
        assert len(n)==len(b)==k and n<=core
        assert len(b-core)==p['bridge_counts'][str(k)]
        assert len(core)==k+4
        assert len(g[f'random_core{k}']['nodes'])==k and set(g[f'random_core{k}']['nodes'])<=core
        random_bridge=g[f'random_bridge{k}'];chosen=g[f'nucleus_bridge{k}']
        assert random_bridge['retained_core_seeds']==chosen['retained_core_seeds']
        assert len(random_bridge['outside_nodes'])==len(chosen['outside_nodes'])
        assert not set(random_bridge['outside_nodes'])&core
        assert g[f'nucleus{k}_frozen']['nodes']==g[f'nucleus{k}']['nodes']
        assert g[f'nucleus_bridge{k}_frozen']['nodes']==g[f'nucleus_bridge{k}']['nodes']
    assert select_groups(o,p,42)==(g,c)
    o=replace(o,private_states=np.ones((40,1)))
    with pytest.raises(ValueError,match='restricted'):select_groups(o,p,42)
    o=replace(observed(),observed_edges={(0,1):1})
    g,c=select_groups(o,p,42);assert not g['nucleus4']['available'] and not c


def test_queue_reuses_roles_not_memory_or_futures(tmp_path):
    p=pilot_plan();g,_=select_groups(observed(),p,42);g['attention4']=dict(g['nucleus4'])
    roles,units=batch.resolved_queue(p,{b:g for b in p['origins']})
    assert len(roles)==960 and len(units)<960
    b=p['origins'][0]
    assert roles[f'{b}__attention4__future0']==roles[f'{b}__nucleus4__future0']
    assert roles[f'{b}__zero__future0']!=roles[f'{b}__zero_frozen__future0']
    assert roles[f'{b}__zero__future0']!=roles[f'{b}__zero__future1']
    assert batch.run_pilot(tmp_path,plan_only=True)==p
    with pytest.raises(RuntimeError,match='Colab-only'):batch.run_pilot(tmp_path)


def test_exact_trace_conservative_boundary():
    assert evaluate_holding_trace([(0,False),(21,True),(41,False)],20,20,320) is None
    assert evaluate_holding_trace([(0,False),(21,True),(41.1,False)],20,20,320)==41


def test_passive_recorder_same_physics_and_reconstructable_flips():
    from src.core.state import State,StateMode
    from src.control.engine import run_controlled_simulation
    from src.control.controllers import ZeroController
    from src.control.observation import SampledPublicObservationModel
    state=State(N=4,D=1,x=np.ones((4,1)),y=np.ones((4,1)),W=np.ones((4,4),dtype=int)-np.eye(4,dtype=int),
        n=np.zeros(4),Q=np.zeros((4,1,2)),state_mode=StateMode.BINARY,K=np.full(4,.3),b=np.full((4,1),-.15),g=np.ones(4),tau_x=np.ones(4),nu=np.ones(4))
    problem=ControlProblem(problem_id='tiny',target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1.,'threshold':.5},
        N=4,D=1,k=4,exposure_cap=20.,amplitude_cap=4.,accessible_roster=tuple(range(4)),
        release_deadline=.2,holding_duration=.2,horizon=2.)
    def run(rec):
        return run_controlled_simulation(initial_state=state.copy(),problem=problem,controller=ZeroController(),
            observation_model=SampledPublicObservationModel(N=4,accessible_roster=tuple(range(4)),rng=np.random.default_rng(2)),
            recorder=rec,sim_rng=np.random.default_rng(33),controller_rng=np.random.default_rng(4),T=1.,
            record_target_transitions=True)
    cfg=RecorderConfig(cadence=.1,record_network_frames=True,max_network_frames=10)
    baseline=run(RolloutRecorder(cfg));rec=NucleationRecorder(cfg,{'core4':[0,1,2,3]},T=1.);diagnostic=run(rec)
    for key in ['x','y','W']:assert np.array_equal(getattr(baseline[1],key),getattr(diagnostic[1],key))
    assert baseline[0].event_counts==diagnostic[0].event_counts
    assert baseline[0].summaries['target_condition_transitions']==diagnostic[0].summaries['target_condition_transitions']
    replay={'x':state.x.copy(),'y':state.y.copy()}
    assert rec.flips
    for e in rec.flips:
        i=e['node'];assert replay[e['layer']][i,0]==e['before'];replay[e['layer']][i,0]=e['after']
        assert abs(e['field']-e['social']-e['consistency']-e['bias']-e['external'])<1e-10
    for key in replay:assert np.array_equal(replay[key],getattr(diagnostic[1],key))


@pytest.fixture
def fake(monkeypatch,tmp_path):
    p=pilot_plan();p['origins']=['tiny0','tiny1'];p['origin_seeds']={'tiny0':123,'tiny1':456};p['audit_futures']=2;p['allocation_tiers']=[[2,2]]
    monkeypatch.setattr(batch,'identity_for',lambda plan:dict(plan=plan,profile={'N':40},contract={}))
    state=SimpleNamespace(W=np.ones((40,40))-np.eye(40),x=np.ones((40,1)),y=np.ones((40,1)),opportunity_groups=np.zeros(40))
    monkeypatch.setattr(batch,'_origin_state',lambda *args:state)
    monkeypatch.setattr(batch,'serialize_state',lambda state:dict(W=state.W.tolist(),x=state.x.tolist(),y=state.y.tolist(),K=[.3]*40,g=[1.]*40,b=[[-.15]]*40,tau_x=[1.]*40))
    monkeypatch.setattr(batch,'_case_problem',lambda *args:SimpleNamespace(accessible_roster=tuple(range(40))))
    monkeypatch.setattr(batch,'reconstruct_t0_observation',lambda *args:observed())
    calls=[];interrupt={'once':True}
    def unit(profile,seed,origin,contract,memory,package,nodes,amp,duration,sign,**kwargs):
        manifest=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
        if '_audit' in origin:assert manifest['selection_frozen'] and len(manifest['groups'])==2
        else:assert not manifest['selection_frozen'] and not manifest['groups']
        if interrupt['once'] and len(calls)==3:interrupt['once']=False;raise RuntimeError('interrupt')
        calls.append((origin,memory,tuple(nodes),kwargs['simulation_seed']))
        return dict(origin_id=origin,recipient_nodes=list(nodes),memory_mode=memory,is_valid=True,is_eligible=True,
            success=bool(nodes),relapsed=False,success_through_horizon=bool(nodes),completion_time=120. if nodes else None,
            expenditure=38*len(nodes),elapsed_seconds_this_invocation=.01,network_frames=[dict(t=t,x=(state.x if t==0 or not nodes else -state.x).tolist(),y=(state.y if t==0 or not nodes else -state.y).tolist(),W=state.W.tolist(),controlled_mask=[False]*40) for t in [0.,320.]] if origin.endswith('00') else [],
            summaries={'target_condition_transitions':[[0,False],[20,bool(nodes)]], 'recorder':{'nucleation':{'flips':[],
                'core_transitions':{'core4':[[0,False],[20,bool(nodes)]]}}}})
    monkeypatch.setattr(batch,'run_actual_unit',unit)
    return p,calls


def test_resume_identity_censoring_and_analysis(fake,tmp_path):
    p,calls=fake
    with pytest.raises(RuntimeError,match='interrupt'):batch._run(tmp_path,p)
    b=next(tmp_path.iterdir());partial=analysis.analyze_pilot(b)
    assert not partial['complete'] and partial['primary']['effect'] is None
    batch._run(tmp_path,p,b);n=len(calls);batch._run(tmp_path,p,b);assert len(calls)==n==len(set(calls))
    raw=(b/'manifest.json').read_bytes();d=analysis.analyze_pilot(b)
    assert d['complete'] and d['primary']['networks']==2
    assert (b/'manifest.json').read_bytes()==raw
    assert set(analysis.memory_view(d,'frozen')['roles'].memory)=={'frozen'}
    assert 'zero' in set(analysis.memory_view(d,'frozen')['roles'].role)
    changed=copy.deepcopy(p);changed['amplitude']=4
    with pytest.raises(ValueError,match='identical'):batch._run(tmp_path,changed,b)
    m=json.loads(raw);u=next(iter(m['actual'].values()));(b/u['path']).write_text('{}')
    with pytest.raises(ValueError,match='Checksum'):analysis.analyze_pilot(b)


def test_hard_deadline(monkeypatch,tmp_path):
    import multiprocessing as mp
    if 'fork' not in mp.get_all_start_methods():pytest.skip('fork needed for bounded sleeping worker')
    context=mp.get_context('fork');monkeypatch.setattr(batch.mp,'get_context',lambda method:context)
    monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace())
    monkeypatch.setattr(batch,'pilot_plan',lambda:dict(session_seconds=.25))
    def sleeper(out,plan,resume,events,budget_seconds):
        b=tmp_path/'partial';b.mkdir();(b/'manifest.json').write_text('{"status":"incomplete"}')
        events.put(('bundle',str(b)));time.sleep(10)
    monkeypatch.setattr(batch,'_worker',sleeper)
    start=time.monotonic();b=batch.run_pilot(tmp_path)
    assert time.monotonic()-start<3
    assert json.loads((b/'manifest.json').read_text())['status']=='incomplete_hard_time_limit'




def test_timing_allocation_uses_cost_and_preserves_networks():
    p=pilot_plan()
    for duration,expected in [(10,(8,6)),(20,(8,4)),(30,(6,4)),(49,(4,4))]:
        a=choose_allocation(p,[duration]*3,28000)
        assert (len(a['origins']),a['audit_futures'])==expected
        assert a['estimated_seconds']<=21600
    assert choose_allocation(p,[60]*3,28000) is None
    assert choose_allocation(p,[10]*3,1201) is None
    assert len(choose_allocation(p,[10]*3,6000)['origins'])==4
    with pytest.raises(ValueError):choose_allocation(p,[1,0,1],28000)
    with pytest.raises(ValueError):choose_allocation(p,[1,1],28000)


def test_network_interaction_and_available_scope(fake,tmp_path):
    p,calls=fake
    try:batch._run(tmp_path,p)
    except RuntimeError:pass
    b=next(tmp_path.iterdir());batch._run(tmp_path,p,b);d=analysis.analyze_pilot(b)
    assert len(d['recipient_characteristics'])>0
    assert all(r['effect']==0 and r['networks']==2 for r in d['memory_interactions'])
    # Deliberately asymmetric network/future effects; cluster means, not role counts.
    f=d['roles'];f.loc[(f.origin=='tiny0')&(f.role=='nucleus_bridge4'),'penalized_time']=100.
    f.loc[(f.origin=='tiny1')&(f.role=='nucleus_bridge4_frozen'),'penalized_time']=160.
    r=next(r for r in analysis.memory_interactions(d) if r['count']==4 and r['metric']=='penalized_time')
    assert r['origin_differences']=={'tiny0':-20.,'tiny1':-40.} and r['effect']==-30.
    c=analysis._contrast(f,'nucleus_bridge4','nucleus4','penalized_time',['tiny0'],p)
    assert c['effect']==-20. and c['networks']==1 and c['interval'] is None
    m=json.loads((b/'manifest.json').read_text())
    assert len([v for v in m['actual'].values() if v['kind']=='timing'])==3
    assert m['allocation_decided'] and m['allocation']['origins']==['tiny0','tiny1']


def test_deadline_includes_export_and_preserves_completed_science(monkeypatch,tmp_path):
    import multiprocessing as mp
    if 'fork' not in mp.get_all_start_methods():pytest.skip('fork needed for bounded sleeping worker')
    context=mp.get_context('fork');monkeypatch.setattr(batch.mp,'get_context',lambda method:context)
    monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace())
    monkeypatch.setattr(batch,'pilot_plan',lambda:dict(session_seconds=28800))
    def sleeper(out,plan,resume,events,budget_seconds):
        b=tmp_path/'complete';b.mkdir();(b/'manifest.json').write_text('{"status":"complete"}')
        events.put(('bundle',str(b)));events.put(('compute_done',str(b)));time.sleep(10)
    monkeypatch.setattr(batch,'_worker',sleeper)
    start=time.monotonic();b=batch.run_pilot(tmp_path,wall_time_limit=.25)
    assert time.monotonic()-start<3
    m=json.loads((b/'manifest.json').read_text())
    assert m['status']=='complete' and m['report_status']=='incomplete_hard_time_limit'


def test_timing_only_exports_without_audit_or_false_success(fake,tmp_path):
    p,calls=fake;p['planning_seconds']=0
    b=batch._run(tmp_path,p);assert len(calls)==3
    m=json.loads((b/'manifest.json').read_text())
    assert m['status']=='timing_only_insufficient_budget' and not m['audit_roles']
    report=analysis.build_saved_report(b)
    assert not report['complete'] and report['physical_audit_units']==0
    assert report['timing_units']==3 and not report['players']
    assert report['primary']['effect'] is None
