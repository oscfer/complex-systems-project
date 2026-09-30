"""Bounded software checks; shortened horizons are never scientific results."""
import copy,time
import numpy as np
from src.control.scale_extension import prepare,simulate,plan,digest_data,reconstruct_endpoint,physical_signature

def preflight():
    started=time.monotonic();checks=[]
    for N in (40,400):
        origin=prepare(N,'preflight-only',1)[0][0];st=origin['initial'];W=np.asarray(st['W'])
        assert (W.sum(1)==20).all() and (np.diag(W)==0).all() and set(st['row_caps'])=={40}
        assert (np.asarray(st['x'])<0).sum()<N/2 and (np.asarray(st['y'])<0).sum()<N/2
        groups=origin['groups'];assert len(set(physical_signature(g) for g in groups.values()))==(8 if N==40 else 14)
        assert len(groups['coverage10_adaptive']['nodes'])==N//10
        assert len(groups['coverage25_adaptive']['nodes'])==N//4
        assert len(groups['coverage10_equal_total_adaptive']['nodes'])*groups['coverage10_equal_total_adaptive']['amplitude']==8.
        c=plan();c.update(release=2.,duration=1.,hold=1.,horizon=4.)
        g=groups['coverage10_adaptive']
        a=simulate(origin,g,0,c,'preflight',True);b=simulate(origin,g,0,c,'preflight',False)
        assert a['final_state']==b['final_state'] and a['endpoint']==b['endpoint']
        assert a['summaries']['recorder']['spin_changes']==b['summaries']['recorder']['spin_changes']
        frozen=simulate(origin,groups['coverage25_frozen'],0,c,'preflight',True)
        assert frozen['final_state']['W']==st['W']
        checks.append(dict(N=N,checks=['initial screening','memory invariants','fraction counts','dose scaling',
            'physical deduplication','fresh deterministic replay','passive recording','exact endpoint replay','frozen memory'],passed=True))
    initial={'x':[[-1]]*30+[[1]]*10,'y':[[-1]]*40}
    u=dict(N=40,contract=plan(),summaries={'recorder':{'spin_changes':[[150.,0,0,-1,1],[150.49,0,0,1,-1]]}},final_state=copy.deepcopy(initial))
    r=reconstruct_endpoint(initial,u)
    assert r['certification']==120 and not r['persistent'] and r['hysteresis_sensitivity']['0.6']
    assert abs(r['postcert_below75_duration']-.49)<1e-9
    return dict(passed=True,software_only=True,seconds=time.monotonic()-started,checks=checks,
        endpoint_fixture='30/40 to29/40 for0.49 units fails primary;60% exit diagnostic remains true')
