"""Orchestration checks only: never initialize a reserved network locally."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.control.contracts import Observation
from src.control.optimizers import validation_batch as v
from src.control.optimizers import sparse_discovery_batch as b
from src.control.transformation_study import load_transformation_contract


def test_frozen_plan_and_uninspected_seeds():
    p=v.validation_plan();seeds=set(p['origin_seeds'].values())
    contract=load_transformation_contract()
    assert len(seeds)==8
    assert not seeds & {r['seed'] for pool in ('development','selection') for r in contract[pool]}
    assert p['search_ceiling']==8*2*2*10*3
    assert p['audit_ceiling']==8*(4+6)*8
    assert p['zero_ceiling']==8*(3+8)
    assert p['physical_ceiling']==sum(p[k] for k in ('search_ceiling','audit_ceiling','zero_ceiling'))==1688
    assert p['session_seconds']==28800
    assert v.run_validation_batch('/unused',plan_only=True)==p


def test_final_search_freeze_resume_integrity(tmp_path,monkeypatch):
    p=v.validation_plan();p.update(origins=['validation_00'],origin_seeds={'validation_00':123},
        search_futures=2,audit_futures=2,initial_candidates_per_stratum=2,exchange_candidates_per_stratum=1)
    monkeypatch.setattr(v,'validation_plan',lambda:p)
    monkeypatch.setattr(b,'_origin_state',lambda *a:None)
    monkeypatch.setattr(b,'_case_problem',lambda *a:SimpleNamespace(accessible_roster=tuple(range(40))))
    obs=Observation(1.,np.ones((40,1)),np.ones(40,bool),{(39,j):j+1 for j in range(30)},tuple(range(40)))
    monkeypatch.setattr(b,'reconstruct_t0_observation',lambda *a:obs)
    # One selectable stratum plus one unavailable; other planned strata omitted
    # to keep this a bounded stand-in orchestration check.
    spec={'id':'n4_q0.5','size':4,'quantile':.5,'ceiling':5.,'eligible':tuple(range(5)),'available':True}
    missing={**spec,'id':'n10_q0.5','size':10,'available':False}
    monkeypatch.setattr(b,'strata',lambda o:([spec,missing],np.arange(1,41)))
    calls=[];fail=[True];frozen=[]
    def unit(profile,seed,origin,contract,memory,package,nodes,amp,duration,sign,simulation_seed=None):
        if len(calls)==1 and fail[0]:raise RuntimeError('interrupt')
        if '_audit' in origin:
            m=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
            assert m['selection_frozen'] and len(m['benchmark_groups']['validation_00'])==6
            assert not m['strata']['validation_00__n10_q0.5']['available']
            frozen.append(m['strata']['validation_00__n4_q0.5']['selected'])
        calls.append((origin,nodes,simulation_seed))
        return dict(success=False,relapsed=False,success_through_horizon=False,is_valid=True,
            is_eligible=True,completion_time=None,expenditure=len(nodes)*amp*duration)
    monkeypatch.setattr(b,'run_actual_unit',unit)
    def write(bundle,manifest,*a):
        assert all(x['kind']=='audit' for x in manifest['actual'].values())
        q=bundle/'study';q.mkdir(exist_ok=True);(q/'manifest.json').write_text('{"config":{}}');return q
    monkeypatch.setattr(b,'_write_study_bundle',write)
    monkeypatch.setattr('src.control.optimizers.group_characteristics.export_characteristics',lambda p:[])
    with pytest.raises(RuntimeError,match='interrupt'):b._run_sparse_discovery(tmp_path,validation=True)
    bundle=next(tmp_path.iterdir());fail[0]=False
    b._run_sparse_discovery(tmp_path,validation=True,resume_bundle=bundle)
    n=len(calls);b._run_sparse_discovery(tmp_path,validation=True,resume_bundle=bundle)
    assert len(calls)==n==len(set(calls))
    assert frozen and all(g==frozen[0] for g in frozen)
    assert {s for o,_,s in calls if '_search' in o}.isdisjoint({s for o,_,s in calls if '_audit' in o})
    m=json.loads((bundle/'manifest.json').read_text());assert m['status']=='complete'
    sm=json.loads((bundle/'study/manifest.json').read_text())
    assert not sm['config']['independent_origins'] and sm['config']['stage']=='validation'
    assert len(m['audit_roles'])==2*(1+6+1)
    info=next(iter(m['actual'].values()));(bundle/info['path']).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):b._run_sparse_discovery(tmp_path,validation=True,resume_bundle=bundle)
