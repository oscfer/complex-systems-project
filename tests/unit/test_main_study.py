"""Bounded stand-in checks only: never launch a nominal physical simulation."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.control import main_study as study
from src.control import study_analysis as analysis
from src.control.contracts import Observation


def test_scaling_allocation_and_local_guard(tmp_path):
    p=study.study_plan('extended')
    assert p['N']==500 and p['sizes']==[4,10,50,125]
    assert p['horizon']==320 and p['hold']==100 and p['release']==20
    assert study.choose_allocation(p,[1,2,3],28000)['origins']==list(p['origin_seeds'])
    assert study.choose_allocation(p,[1000,1200,1300],28000) is None
    with pytest.raises(RuntimeError,match='cloud-only'):study.run_study(tmp_path,'extended')


def test_unknown_is_not_weak():
    obs=Observation(1.,np.ones((40,1)),np.ones(40,bool),{(39,j):j+1 for j in range(12)},tuple(range(40)))
    groups=study.make_benchmarks(obs,[4,10],10)
    assert groups['low_observed4']['available']
    assert not groups['low_observed10']['available']
    assert all(n<6 for n in groups['low_observed4']['nodes'])


@pytest.fixture
def fake_study(monkeypatch,tmp_path):
    p=study.study_plan('light');p['origin_seeds']={'test00':123,'test01':456};p['initial_candidates']=1;p['exchange_candidates']=0;p['search_futures']=1
    monkeypatch.setattr(study,'_identity',lambda plan:dict(plan=plan,profile={'N':40},contract={}))
    W=np.ones((40,40));np.fill_diagonal(W,0)
    state=SimpleNamespace(W=W,x=np.ones((40,1)),y=np.ones((40,1)),opportunity_groups=np.zeros(40))
    obs=Observation(1.,np.ones((40,1)),np.ones(40,bool),{(39,j):j+1 for j in range(30)},tuple(range(40)))
    monkeypatch.setattr(study,'_origin_state',lambda *a:state)
    monkeypatch.setattr(study,'_case_problem',lambda *a:SimpleNamespace(accessible_roster=tuple(range(40))))
    monkeypatch.setattr(study,'reconstruct_t0_observation',lambda *a:obs)
    monkeypatch.setattr(study,'choose_allocation',lambda *a:dict(origins=list(p['origin_seeds']),audit_futures=2,search_enabled=True))
    calls=[];switch={'interrupt':True}
    def fake(profile,seed,origin,contract,memory,package,nodes,amp,duration,sign,**kwargs):
        if '_audit' in origin:
            manifest=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
            assert manifest['selection_frozen'] and len(manifest['groups'])==2
            if switch['interrupt']:
                switch['interrupt']=False;raise RuntimeError('simulated interruption')
        calls.append((origin,tuple(nodes),kwargs['simulation_seed']))
        return dict(recipient_nodes=list(nodes),is_valid=True,is_eligible=True,success=bool(nodes),
                    relapsed=False,success_through_horizon=bool(nodes),completion_time=120 if nodes else None,
                    expenditure=38*len(nodes),elapsed_seconds_this_invocation=.001,
                    target_definition={'initial_metrics':{'m_x_0':1,'m_y_0':1}},network_frames=[])
    monkeypatch.setattr(study,'run_actual_unit',fake)
    return p,calls


def test_freeze_resume_and_clustered_readonly_analysis(fake_study,tmp_path):
    p,calls=fake_study
    with pytest.raises(RuntimeError,match='interruption'):study._run(tmp_path,p)
    bundle=next(tmp_path.iterdir())
    study._run(tmp_path,p,bundle);n=len(calls);study._run(tmp_path,p,bundle)
    assert len(calls)==n==len(set(calls))
    manifest_before=(bundle/'manifest.json').read_bytes()
    data=analysis.analyze_bundle(bundle)
    assert data['complete'] and data['primary']['networks']==2
    assert data['physical_audit_units']<len(data['roles']) # role reuse is not replication
    assert (bundle/'manifest.json').read_bytes()==manifest_before
    with pytest.raises(ValueError,match='immutable'):analysis.export_analysis(data,bundle/'derived')
    m=json.loads(manifest_before);key=next(iter(m['audit_roles']));del m['audit_roles'][key]
    (bundle/'manifest.json').write_text(json.dumps(m))
    partial=analysis.analyze_bundle(bundle)
    assert not partial['complete'] and partial['primary']['effect'] is None
    info=m['actual'][next(iter(m['audit_roles'].values()))];(bundle/info['path']).write_text('{}')
    with pytest.raises(ValueError,match='Checksum'):analysis.analyze_bundle(bundle)


def test_hard_deadline_retains_manifest(monkeypatch,tmp_path):
    import multiprocessing as mp
    import sys,time
    if 'fork' not in mp.get_all_start_methods():pytest.skip('Bounded supervisor check requires fork')
    context=mp.get_context('fork')
    monkeypatch.setattr(study.mp,'get_context',lambda method:context)
    monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace())
    original=study.study_plan
    monkeypatch.setattr(study,'study_plan',lambda tier:{**original(tier),'session_seconds':.25})
    def sleeping(out,plan,resume,events):
        b=tmp_path/'partial';b.mkdir();(b/'manifest.json').write_text('{"status":"incomplete","actual":{}}')
        events.put(('bundle',str(b)));time.sleep(10)
    monkeypatch.setattr(study,'_worker',sleeping)
    start=time.monotonic();bundle=study.run_study(tmp_path,'extended')
    assert time.monotonic()-start<3
    assert json.loads((bundle/'manifest.json').read_text())['status']=='incomplete_hard_time_limit'
