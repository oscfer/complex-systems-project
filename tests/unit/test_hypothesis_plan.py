import numpy as np
from src.control.contracts import Observation
from src.control.optimizers.hypothesis_plan import restricted_low_coverage,future_seed,hypothesis_plan


def test_low_score_constraint_and_unavailable():
    obs=Observation(1,np.ones((8,1)),np.ones(8,bool),{(7,j):j+1 for j in range(6)},tuple(range(8)))
    r=restricted_low_coverage(obs,2,0)
    assert r['available'] and set(r['nodes'])<=set(range(3))
    assert r['cutoff']==3.5
    assert not restricted_low_coverage(obs,4,0)['available']
    assert 6 not in r['nodes'] and 7 not in r['nodes']


def test_future_schedule_distinct_and_bounded():
    p=hypothesis_plan()
    seeds=[future_seed(o,f) for o in p['development_origins'] for f in range(p['futures_per_origin'])]
    assert len(set(seeds))==160 and p['physical_ceiling']==4*40*6


def test_runner_keeps_groups_fixed_across_futures(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from src.control.optimizers import hypothesis_batch as b
    from types import SimpleNamespace
    plan=hypothesis_plan();plan['futures_per_origin']=3;plan['physical_ceiling']=72
    monkeypatch.setattr(b,'hypothesis_plan',lambda:plan)
    origins=plan['development_origins']
    roles={'selector_random_4':[0,1,2,3],'selector_audience_coverage_4':[4,5,6,7],
           'selector_random_10':list(range(10)),'selector_audience_coverage_10':list(range(10,20))}
    records={(o,r):{'recipient_nodes':nodes} for o in origins for r,nodes in roles.items()}
    parent=tmp_path/'parent';parent.mkdir();(parent/'manifest.json').write_text('{}')
    monkeypatch.setattr(b,'_parent_records',lambda p:({'identity':{'origins':dict(zip(origins,range(4)))}},records))
    monkeypatch.setattr(b,'_origin_state',lambda *a:None)
    monkeypatch.setattr(b,'_case_problem',lambda *a:SimpleNamespace(accessible_roster=tuple(range(40))))
    monkeypatch.setattr(b,'reconstruct_t0_observation',lambda *a:None)
    monkeypatch.setattr(b,'restricted_low_coverage',lambda *a:{'available':True,'nodes':[20,21,22,23]})
    calls=[]
    def run(profile,seed,origin,contract,memory,package,nodes,amp,duration,sign,simulation_seed=None):
        calls.append((seed,origin,tuple(nodes),simulation_seed))
        return dict(success=False,relapsed=False,success_through_horizon=False,is_valid=True,is_eligible=True,expenditure=len(nodes)*amp*duration,completion_time=None)
    monkeypatch.setattr(b,'run_actual_unit',run)
    def write(bundle,*a):
        p=bundle/'study';p.mkdir(exist_ok=True);(p/'manifest.json').write_text('{"config":{}}');return p
    monkeypatch.setattr(b,'_write_study_bundle',write)
    out=b.run_hypothesis_batch(tmp_path/'output',parent)
    assert len(calls)==72
    for seed in range(4):
        c=[r for r in calls if r[0]==seed]
        assert len({r[2] for r in c})==6 and len({r[3] for r in c})==3
        for nodes in {r[2] for r in c}:assert sum(r[2]==nodes for r in c)==3
    b.run_hypothesis_batch(tmp_path/'output',parent,resume_bundle=out)
    assert len(calls)==72
