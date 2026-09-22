import gzip,hashlib,json,re
from pathlib import Path
import numpy as np
import pytest
from src.visualization.animation_player import decode_arm, render_player_html
from src.visualization.results_archive import build_results_archive

def bundle(root,name,N):
    b=root/name;(b/'units').mkdir(parents=True)
    manifest={'status':'complete','units':{}}
    for role in ['zero','attention_sustained_4']:
        frames=[dict(t=t,W=np.ones((N,N),dtype=int).tolist(),x=[-1]*N,y=[1]*N,controlled_mask=[role!='zero']+[False]*(N-1)) for t in [0,20,320]]
        for f in frames:
            for i in range(N):f['W'][i][i]=0
        u=dict(final_state={'N':N},target_definition=dict(horizon=320,release_deadline=20,holding_duration=100,threshold=.5),expenditure=0 if role=='zero' else 152,is_valid=True,is_eligible=True,success=False,success_through_horizon=False,relapsed=False,completion_time=None,network_frames=frames,unit={'package':'original'})
        p=b/'units'/(role+'.json');p.write_text(json.dumps(u));key='network__original__'+role
        manifest['units'][key]={'path':str(p.relative_to(b)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (b/'manifest.json').write_text(json.dumps(manifest));return b

def test_one_archive_preserves_mixed_sizes_and_all_frames(tmp_path):
    bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    bundle(tmp_path,'transformation_v1_20260927_010000_bbbb',12)
    result=build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False)
    payload=json.loads(re.search(r'<script id="player-data" type="application/json">(.*?)</script>',result['page'].read_text(),re.S)[1])
    assert {c['N'] for c in payload['comparisons']}=={4,12}
    assert len(payload['comparisons'])==2 and len(payload['records'])==4
    for c in payload['comparisons']:
        a=payload['arms'][c['arm']]
        import base64
        record=json.loads(gzip.decompress(base64.b64decode(a['gzip'])))
        decoded=decode_arm(record,c['N'])
        assert len(decoded)==3 and decoded[-1]['t']==320
        assert decoded[-1]['W'].shape==(c['N'],c['N'])
        assert c['matched'] and c['policy_label'].startswith('High observed attention · field strength 4')
    assert result['inventory']['retained_frames']==12
    assert build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False)['inventory']['player_sha256']==result['inventory']['player_sha256']

def test_checksum_failure_is_not_silently_omitted(tmp_path):
    b=bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    (b/'units/zero.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum mismatch'):build_results_archive(tmp_path,tmp_path/'viewer.html')

def test_renderer_hides_node_numbers_and_scales_rings():
    page=render_player_html({},title='test')
    assert 'fillText(String(i),' not in page
    assert 'strokeText(String(i),' not in page
    assert 'Math.max(8, rr + 5)' not in page
    assert 'ringRadius = rr *' in page
    assert 'filter-study' in page and 'search-results' in page


def payload(result):
    return json.loads(re.search(r'<script id="player-data" type="application/json">(.*?)</script>',result['page'].read_text(),re.S)[1])


def rewrite_unit(b, role, edit):
    manifest=json.loads((b/'manifest.json').read_text())
    key='network__original__'+role
    p=b/manifest['units'][key]['path']
    unit=json.loads(p.read_text());edit(unit);p.write_text(json.dumps(unit))
    manifest['units'][key]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (b/'manifest.json').write_text(json.dumps(manifest))


def test_scalar_future_is_retained_without_inventing_animation(tmp_path):
    b=bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    manifest=json.loads((b/'manifest.json').read_text())
    key='network__original__attention_sustained_4'
    original=json.loads((b/manifest['units'][key]['path']).read_text())
    original['network_frames']=[]
    p=b/'units/second_future.json';p.write_text(json.dumps(original))
    manifest['units']['second_future']={'path':'units/second_future.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'future':1,'base_origin':'network','roles':['attention_sustained_4']}
    (b/'manifest.json').write_text(json.dumps(manifest))
    data=payload(build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False))
    assert len(data['records'])==3 and len(data['comparisons'])==1
    scalar,=[r for r in data['records'] if r['future']==1]
    assert scalar['frames']==0 and scalar['valid'] and not scalar['persistent']


def test_reference_alias_reuses_physical_arm_but_retains_package(tmp_path):
    b=bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    manifest=json.loads((b/'manifest.json').read_text())
    for role in ['zero','attention_sustained_4']:
        p=b/'units'/('wider_'+role+'.json')
        p.write_text(json.dumps({'reference_unit':'network__original__'+role,'unit':{'package':'wider'}}))
        manifest['units']['network__wider__'+role]={'path':str(p.relative_to(b)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (b/'manifest.json').write_text(json.dumps(manifest))
    data=payload(build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False))
    assert len(data['records'])==4 and len(data['comparisons'])==2 and len(data['arms'])==2
    assert all(c['matched'] for c in data['comparisons'])
    assert {r['package'] for r in data['records']}=={'original','wider'}
    assert len({r['physical_sha256'] for r in data['records']})==2


def test_unsynchronized_control_is_explicitly_unpaired(tmp_path):
    b=bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    rewrite_unit(b,'attention_sustained_4',lambda u:u['network_frames'][1].update(t=21))
    result=build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False)
    data=payload(result)
    assert not data['comparisons'][0]['matched']
    assert data['comparisons'][0]['arm']==data['comparisons'][0]['zero']
    assert 'Unsynchronized' in result['inventory']['issues'][0]['reason']


def test_colab_compact_loader_reads_the_whole_file_in_chunks(tmp_path,monkeypatch):
    import base64,sys,types
    from unittest.mock import patch
    from src.visualization.results_archive import show_results_archive
    file=tmp_path/'viewer.html';raw=b'<html>'+b'x'*(600*1024)+b'</html>';file.write_bytes(raw)
    callbacks={}
    output=types.SimpleNamespace(register_callback=lambda name,fn:callbacks.update({name:fn}))
    colab=types.ModuleType('google.colab');colab.output=output
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    with patch('IPython.display.display') as display:
        assert show_results_archive(file)==file.resolve()
    callback=callbacks['modular3.results_chunk']
    combined=b''.join(base64.b64decode(callback(offset).data['chunk']) for offset in [0,512*1024,1024*1024])
    assert combined==raw
    from IPython.core.interactiveshell import InteractiveShell
    assert 'application/json' in InteractiveShell.instance().display_formatter.format(callback(0))[0]
    html=display.call_args.args[0].data
    assert len(html)<3000 and 'allowfullscreen' in html
    assert 'Save the complete offline viewer' in html


def test_completed_software_rehearsal_is_excluded_but_valid_failure_stays(tmp_path):
    bundle(tmp_path,'transformation_v1_20260926_010000_aaaa',4)
    rehearsal=bundle(tmp_path,'transformation_v1_20260927_010000_bbbb',4)
    m=json.loads((rehearsal/'manifest.json').read_text());m['config']={'smoke':True}
    (rehearsal/'manifest.json').write_text(json.dumps(m))
    result=build_results_archive(tmp_path,tmp_path/'viewer.html',progress=False)
    data=payload(result)
    assert len(data['studies'])==1 and len(data['comparisons'])==1
    assert not data['comparisons'][0]['outcome']['persistent']
    assert result['inventory']['excluded'][0]['bundle']==rehearsal.name
