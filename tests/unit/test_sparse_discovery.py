import numpy as np
from src.control.contracts import Observation
from src.control.optimizers.sparse_discovery import strata,initial_groups,exchange_groups,group_score,discovery_plan,stream_seed


def test_power_bounds_and_small_groups():
    obs=Observation(1,np.ones((40,1)),np.ones(40,bool),{(39,j):j+1 for j in range(30)},tuple(range(40)))
    ss,power=strata(obs)
    for s in ss:
        initial=initial_groups(obs,s,1)
        if not s['available']:assert not initial;continue
        extra=exchange_groups(initial[0],s,initial,2)
        assert not set(initial)&set(extra)
        for nodes in initial+extra:
            assert len(nodes)==s['size'] and len(set(nodes))==s['size']
            assert all(0<power[n]<=s['ceiling'] for n in nodes)
    assert any(s['size']==2 and s['available'] for s in ss)


def test_persistence_before_speed_and_separate_audit():
    base={'is_valid':True,'is_eligible':True}
    good={**base,'success_through_horizon':True,'completion_time':200}
    fail={**base,'success_through_horizon':False,'completion_time':120}
    assert group_score((0,),[good],np.ones(2))<group_score((1,),[fail],np.ones(2))
    assert stream_seed('development_00','search',0)!=stream_seed('development_00','audit',0)
    p=discovery_plan();assert p['physical_ceiling']==p['search_ceiling']+p['audit_ceiling']+p['zero_ceiling']


def test_runner_interruption_and_audit_separation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from src.control.optimizers import sparse_discovery_batch as b
    plan=discovery_plan();plan.update(origins=['development_00'],search_futures=2,audit_futures=2,
        sizes=[2],initial_candidates_per_stratum=2,exchange_candidates_per_stratum=1)
    monkeypatch.setattr(b,'discovery_plan',lambda:plan)
    monkeypatch.setattr(b,'_origin_state',lambda *a:None)
    monkeypatch.setattr(b,'_case_problem',lambda *a:SimpleNamespace(accessible_roster=tuple(range(40))))
    obs=Observation(1,np.ones((40,1)),np.ones(40,bool),{(39,j):j+1 for j in range(30)},tuple(range(40)))
    monkeypatch.setattr(b,'reconstruct_t0_observation',lambda *a:obs)
    spec={'id':'n2_q0.5','size':2,'quantile':.5,'ceiling':5.,'eligible':tuple(range(5)),'available':True}
    monkeypatch.setattr(b,'strata',lambda o:([spec],np.arange(1,41)))
    calls=[];fail=[True]
    def run(profile,seed,origin,contract,memory,package,nodes,amp,duration,sign,simulation_seed=None):
        if len(calls)==1 and fail[0]:raise RuntimeError('interrupt')
        if '_audit' in origin:
            m=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
            assert m['selection_frozen'] and m['strata']['development_00__n2_q0.5']['selected']
        calls.append((origin,nodes,simulation_seed))
        return {'success':False,'relapsed':False,'success_through_horizon':False,
            'is_valid':True,'is_eligible':True,'completion_time':None,'expenditure':len(nodes)*amp*duration}
    monkeypatch.setattr(b,'run_actual_unit',run)
    def write(bundle,*a):
        p=bundle/'study';p.mkdir(exist_ok=True);(p/'manifest.json').write_text('{}');return p
    monkeypatch.setattr(b,'_write_study_bundle',write)
    monkeypatch.setattr('src.control.optimizers.group_characteristics.export_characteristics',lambda p:[])
    import pytest
    with pytest.raises(RuntimeError,match='interrupt'):b._run_sparse_discovery(tmp_path)
    bundle=next(tmp_path.iterdir());fail[0]=False
    b._run_sparse_discovery(tmp_path,resume_bundle=bundle)
    n=len(calls)
    b._run_sparse_discovery(tmp_path,resume_bundle=bundle)
    assert len(calls)==n==len(set(calls))
    assert {s for o,_,s in calls if '_search' in o}.isdisjoint({s for o,_,s in calls if '_audit' in o})
    m=json.loads((bundle/'manifest.json').read_text());assert m['status']=='complete'


def _slow_supervised_worker(out_dir,resume_bundle,queue):
    import json,time
    from pathlib import Path
    p=Path(out_dir)/'interrupted';p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'actual':{},'status':'incomplete'}))
    queue.put(('bundle',str(p)))
    time.sleep(30)


def test_hard_supervisor_terminates_active_work(tmp_path,monkeypatch):
    import json,time
    from src.control.optimizers import sparse_discovery_batch as b
    monkeypatch.setattr(b,'_supervised_worker',_slow_supervised_worker)
    monkeypatch.setattr(b,'discovery_plan',lambda:{'session_seconds':2})
    start=time.monotonic();bundle=b.run_sparse_discovery(tmp_path)
    assert time.monotonic()-start<5
    assert json.loads((bundle/'manifest.json').read_text())['status']=='incomplete_hard_time_limit'
    assert b.summarize_partial(bundle)==[]
