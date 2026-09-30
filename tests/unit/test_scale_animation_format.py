"""Scale returns use the same navigable player as prior campaigns."""
import json, re
import numpy as np
import pytest
from src.control import scale_extension as s
from src.control import scale_extension_analysis as a
from src.visualization.animation_player import decode_arm
from src.visualization.network_layout import _seeded_spring_layout

def payload(path):
    return json.loads(re.search(r'<script id="player-data" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])

@pytest.fixture
def saved_bundle(tmp_path):
    origins = {}; units = {}; roles = {}; queue = []
    for N in (40, 400):
        for oi in range(2):
            oid = f'audit_N{N}_{oi:02d}'
            W = np.zeros((N, N), dtype=int)
            for i in range(N): W[i, (i + 1) % N] = 20
            initial = dict(N=N, W=W.tolist(), x=[-1] * N, y=[1] * N, row_caps=[40] * N)
            origins[oid] = dict(id=oid, initial=initial)
            for memory in ('adaptive', 'frozen'):
                for role in ('zero', 'coverage4', 'coverage10'):
                    physical = 'coverage4' if N == 40 and role == 'coverage10' else role
                    key = f'{oid}_{memory}_{physical}'
                    roles[f'{oid}__f0__{role}_{memory}'] = key
                    if key in units: continue
                    nodes = [] if role == 'zero' else list(range(4))
                    edges = [[i, (i+1)%N, 20] for i in range(N)]
                    frames = [dict(t=t, x=initial['x'], y=initial['y'], edges=edges,
                                   controlled=nodes if t == 1 else []) for t in (0., 1., 20., 320.)]
                    u = dict(N=N, origin=oid, future=0, frames=frames, group=dict(nodes=nodes, memory=memory),
                             contract=s.plan(), endpoint=dict(persistent=False, relapsed=False, certification=None, threshold=int(.75*N)),
                             summaries={'recorder': {'action_history': [] if not nodes else [dict(spending_rate=8, actuated_nodes=nodes, decision_time=1.)]}})
                    path = tmp_path / 'units' / (key + '.json.gz'); s.write_gzip(path, u)
                    units[key] = dict(path=str(path.relative_to(tmp_path)), sha256=s._sha(path), phase='audit')
                    queue.append(dict(key=key))
    s.write_gzip(tmp_path/'origins.json.gz', dict(origins=origins))
    m = dict(plan=s.plan(), roles=roles, units=units, queue=queue, origins_sha256=s._sha(tmp_path/'origins.json.gz'),
             science_status='complete', export_status='complete', exports={})
    s._atomic_json(tmp_path/'manifest.json', m)
    return tmp_path

def test_ensemble_navigation_matches_previous_campaigns(saved_bundle):
    result = a.export_animations(saved_bundle)
    assert len(result['players']) == 4
    for info in result['players'].values():
        p = payload(saved_bundle/info['path'])
        assert p['style'] == 'study'
        assert len(p['comparisons']) == 4 and len({c['origin'] for c in p['comparisons']}) == 2
        assert not p['dropped'] and not p['skipped']
        for c in p['comparisons']:
            assert c['zero'] != c['arm']
            assert c['layouts'][0] == c['layouts'][1]
            assert len(p['arms'][c['arm']]['t']) == 4
            assert p['arms'][c['zero']]['active'] == [[], [], [], []]
        assert len(p['arms']) == (4 if p['N'] == 40 else 6)
        for arm in p['arms']:
            rebuilt = decode_arm(arm, p['N'])
            assert [f['t'] for f in rebuilt] == [0., 1., 20., 320.]
            assert all(np.all(f['W'].sum(1) == 20) for f in rebuilt)
        assert 'https://' not in (saved_bundle/info['path']).read_text()

def test_layout_matches_previous_saved_campaigns():
    W = np.array([[0, 2, 1], [3, 0, 4], [1, 2, 0]])
    assert np.array_equal(a.fixed_layout({'W': W.tolist()}), _seeded_spring_layout(W, iterations=50))

def test_repair_is_render_only_and_tracks_hashes(saved_bundle, monkeypatch):
    (saved_bundle/'analysis').mkdir()
    report = saved_bundle/'analysis/report.html'; report.write_text('<h1>Saved report</h1>')
    csv = saved_bundle/'analysis/outcomes.csv'; csv.write_text('saved numerical results\n')
    before = {p: p.read_bytes() for p in saved_bundle.glob('units/*')}
    monkeypatch.setattr(s, 'simulate', lambda *args, **kw: pytest.fail('Visualization must not run science'))
    a.rebuild_animations(saved_bundle)
    assert csv.read_text() == 'saved numerical results\n'
    assert all(p.read_bytes() == data for p, data in before.items())
    assert 'animations/index.html' in report.read_text()
    m = json.loads((saved_bundle/'manifest.json').read_text())
    assert m['science_status'] == 'complete'
    assert m['exports'] and all(s._sha(saved_bundle/p) == digest for p, digest in m['exports'].items())

def test_rejects_corrupt_and_unmatched_saved_arms(saved_bundle):
    m = json.loads((saved_bundle/'manifest.json').read_text())
    key = next(k for k in m['units'] if k.endswith('adaptive_coverage4'))
    path = saved_bundle/m['units'][key]['path']
    u = s.read_gzip(path); u['frames'][0]['x'][0] = 1; s.write_gzip(path,u)
    with pytest.raises(ValueError, match='hash'): a.export_animations(saved_bundle)
    m['units'][key]['sha256'] = s._sha(path); s._atomic_json(saved_bundle/'manifest.json',m)
    with pytest.raises(ValueError, match='initial state'): a.export_animations(saved_bundle)

def test_missing_control_is_reported_without_inventing_a_pair(saved_bundle):
    m=json.loads((saved_bundle/'manifest.json').read_text())
    del m['roles']['audit_N40_00__f0__zero_adaptive']
    s._atomic_json(saved_bundle/'manifest.json',m)
    result=a.export_animations(saved_bundle)
    assert result['players']['N40_adaptive']['comparisons']==2
    assert len(result['skipped'])==2



