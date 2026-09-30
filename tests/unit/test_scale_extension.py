import copy,json,time
from pathlib import Path
import numpy as np
import pytest
from src.control import scale_extension as s
from src.control.scale_extension_analysis import export

@pytest.fixture(scope='module')
def origins():return {N:s.prepare(N,'software-test',1)[0][0] for N in (40,400)}

def small():
 p=s.plan();p.update(release=2.,duration=1.,hold=1.,horizon=4.);return p

@pytest.mark.parametrize('N',[40,400])
def test_initial_scaling_and_selection(origins,N):
 o=origins[N];st=o['initial'];W=np.array(st['W']);g=o['groups']
 assert W.shape==(N,N) and (W.sum(1)==20).all() and set(st['row_caps'])=={40}
 assert np.count_nonzero(np.diag(W))==0
 assert (np.asarray(st['x'])<0).sum()<N/2 and (np.asarray(st['y'])<0).sum()<N/2
 assert len(g['coverage10_adaptive']['nodes'])==N//10
 assert len(g['coverage25_adaptive']['nodes'])==N//4
 assert len(g['coverage4_adaptive']['nodes'])==4
 assert len(g['coverage10_equal_total_adaptive']['nodes'])*g['coverage10_equal_total_adaptive']['amplitude']*19==152
 assert len(set(s.physical_signature(v) for v in g.values()))==(8 if N==40 else 14)
 assert set(g['coverage10_adaptive']['nodes'])==set(g['coverage10_frozen']['nodes'])

@pytest.mark.parametrize('N',[40,400])
def test_passive_recording_deterministic_and_export(origins,N,tmp_path):
 o=origins[N];c=small();g=o['groups']['coverage10_adaptive']
 a=s.simulate(o,g,0,c,'test',True);b=s.simulate(o,g,0,c,'test',True)
 for u in (a,b):u.pop('elapsed')
 assert s.digest_data(a)==s.digest_data(b)
 without=s.simulate(o,g,0,c,'test',False)
 assert without['final_state']==a['final_state'] and without['endpoint']==a['endpoint']
 bundle=tmp_path;key=f'N{N}';s.write_gzip(bundle/'origins.json.gz',dict(origins={o['id']:o},screening={}))
 s.write_gzip(bundle/'units'/f'{key}.json.gz',a)
 info=dict(path=f'units/{key}.json.gz',sha256=s._sha(bundle/'units'/f'{key}.json.gz'),phase='audit',elapsed=1.)
 m=dict(prepared=True,origins_sha256=s._sha(bundle/'origins.json.gz'),plan=c,units={key:info},
   science_status='incomplete',roles={f'{o["id"]}__f0__coverage10_adaptive':key})
 s._atomic_json(bundle/'manifest.json',m);export(bundle)
 result=json.loads((bundle/'analysis/analysis.json').read_text())
 assert not result['complete'] and result['contrasts']==[] and result['players']==1
 player=(bundle/'analysis'/f'{key}.html').read_text()
 assert 'https://' not in player and 'http://' not in player
 assert f'"N":{N}' in player and 'modular-player-v1' in player
 import re
 payload=json.loads(re.search(r'<script id="player-data" type="application/json">(.*?)</script>',player,re.S)[1])
 assert payload['style']=='study' and not payload['matched_zero_present']
 assert payload['comparisons'][0]['zero']==payload['comparisons'][0]['arm']

@pytest.mark.parametrize('fraction',[.75,.7,.6])
def test_strict_threshold_and_hysteresis(fraction):
 N=40;initial={'x':[[-1]]*30+[[1]]*10,'y':[[-1]]*40}
 # First hold at120; private briefly dips29/40 for0.49 after certification.
 final=copy.deepcopy(initial)
 u=dict(N=N,contract=s.plan(),summaries={'recorder':{'spin_changes':[[150.,0,0,-1,1],[150.49,0,0,1,-1]]}},final_state=final)
 r=s.reconstruct_endpoint(initial,u)
 assert r['certification']==120 and not r['persistent'] and r['relapsed']
 assert r['postcert_min_private']==29 and abs(r['postcert_below75_duration']-.49)<1e-9
 assert r['hysteresis_sensitivity'][str(fraction)]==(fraction<.75)

def test_late_no_time_to_certify_and_terminal_exit():
 N=40;init={'x':[[1]]*40,'y':[[-1]]*40};end={'x':[[-1]]*30+[[1]]*10,'y':[[-1]]*40}
 u=dict(N=N,contract=s.plan(),summaries={'recorder':{'spin_changes':[[250.,0,i,1,-1] for i in range(30)]}},final_state=end)
 assert s.reconstruct_endpoint(init,u)['certification'] is None
 init={'x':[[-1]]*30+[[1]]*10,'y':[[-1]]*40};end=copy.deepcopy(init);end['x'][0]=[1]
 u.update(summaries={'recorder':{'spin_changes':[[320.,0,0,-1,1]]}},final_state=end)
 assert not s.reconstruct_endpoint(init,u)['persistent']

def test_allocation_only_timings():
 p=s.plan();assert s.allocate([(40,10),(400,100)],27000,p)['origins']==4
 assert s.allocate([(40,10000),(400,10000)],27000,p) is None
 assert s.allocate([(40,10),(400,10)],27000,p)['origins']==8

def test_seed_namespaces():
 vals=[s.seed('future',N,f'origin{i}',phase,f) for N in (40,400) for i in range(8) for phase in ('audit','timing') for f in range(2)]
 assert len(vals)==len(set(vals))

def test_supervisor_forcibly_stops_uncooperative_process(tmp_path,monkeypatch):
 import os,sys
 fake=tmp_path/'fake';module=fake/'src/control/scale_extension.py';module.parent.mkdir(parents=True)
 (fake/'src/__init__.py').write_text('');(fake/'src/control/__init__.py').write_text('')
 module.write_text('import time,os\nfrom pathlib import Path\nPath("child_pid").write_text(str(os.getpid()))\ntime.sleep(60)\n')
 bundle=tmp_path/'bundle';bundle.mkdir();s._atomic_json(bundle/'manifest.json',dict(plan={'export_reserve':3},science_status='pending'))
 monkeypatch.setattr(s,'ROOT',fake);monkeypatch.setenv('PYTHONPATH',str(fake))
 start=time.monotonic();s.supervise(bundle,start+5.)
 assert time.monotonic()-start<5.
 manifest=json.loads((bundle/'manifest.json').read_text())
 assert manifest['science_process_returncode']<0 and manifest['science_status']=='incomplete'
 pid=int((fake/'child_pid').read_text())
 with pytest.raises(ProcessLookupError):os.kill(pid,0)

def test_violation_exactly_at_holding_boundary_is_not_certification():
 initial={'x':[[-1]]*30+[[1]]*10,'y':[[-1]]*40};final=copy.deepcopy(initial);final['x'][0]=[1]
 u=dict(N=40,contract=s.plan(),summaries={'recorder':{'spin_changes':[[120.,0,0,-1,1]]}},final_state=final)
 assert s.reconstruct_endpoint(initial,u)['certification'] is None

@pytest.mark.parametrize('N',[40,400])
def test_topology_layout_is_fixed_by_initial_memory(origins,N):
 from src.control.scale_extension_analysis import fixed_layout
 a=fixed_layout(origins[N]['initial']);b=fixed_layout(copy.deepcopy(origins[N]['initial']))
 assert a==b
 xy=np.asarray(a);assert xy.shape==(N,2) and np.isfinite(xy).all()
 assert np.abs(xy).max()<=1.10000001
 assert len(np.unique(xy,axis=0))==N
