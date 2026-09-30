"""The self-contained player: offline, exact, navigable, and honest about budgets.

The animation used to be matplotlib's jshtml player, which fetched Font Awesome
from a CDN and labelled its buttons with icons only — opened from disk with no
network, the controls rendered as blank space. These tests pin the replacement:
one page, no external resource, drawn from the saved numbers rather than from
baked images, and never claiming more than the budget allowed it to carry.
"""

import base64
import hashlib
import html
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.visualization import animation_player as ap
from src.visualization.animation_player import (SCHEMA, decode_arm, inline_player_fixups,
                                                payload_from_data, render_player_html)
from src.visualization.study_figures import interactive_comparisons

N = 12


def _frames(count, seed, n=N):
    """A saved-shaped arm: integer memory, binary states, one actuation per step."""
    rng = np.random.default_rng(seed)
    W = np.triu((rng.random((n, n)) < 0.35) * rng.integers(1, 5, (n, n)), 1)
    W = W + W.T
    out = []
    for k in range(count):
        if k:
            i, j = (int(x) for x in rng.integers(0, n, 2))
            if i != j:
                W[min(i, j), max(i, j)] = int(rng.integers(0, 4))
        mask = np.zeros(n, dtype=bool)
        mask[:min(4, n)] = True
        out.append(dict(t=float(10 * k), W=W.copy(),
                        x=rng.choice([-1.0, 1.0], n), y=rng.choice([-1.0, 1.0], n),
                        controlled_mask=mask))
    return out


def _scenes(specs, frames_by_key):
    """specs: [(origin, role, future, [arm keys])]; one shared layout per key."""
    layouts = {}

    def layout(fs):
        return np.asarray([[np.cos(i), np.sin(i)] for i in range(N)]) * 0.5

    def scenes_for(k):
        out = []
        for origin, role, future, keys in specs:
            pairs = [(('Matched zero' if key.endswith('z') else key), key, frames_by_key[key]) for key in keys]
            out.append(ap._scene(origin, role, future, list(range(4)), f'{origin} · {role}',
                                 ap._arms_at(pairs, indices=list(range(k)), layouts=layouts, layout=layout),
                                 outcome=dict(persistent=True, valid=True)))
        return out

    return scenes_for


def _embedded(html):
    body = html.split('<script id="player-data" type="application/json">', 1)[1]
    return json.loads(body.split('</script>', 1)[0])


class TestEncoding(unittest.TestCase):
    def test_arm_round_trip_is_exact(self):
        frames = _frames(9, seed=4)
        record = ap._encode_arm(frames, N)
        self.assertEqual(record['wmode'], 'b', 'integer memory should use packed records')
        back = decode_arm(record, N)
        self.assertEqual(len(back), len(frames))
        for before, after in zip(frames, back):
            np.testing.assert_array_equal(after['W'], np.asarray(before['W'], dtype=int))
            np.testing.assert_array_equal(after['x'], np.asarray(before['x'], dtype=float))
            np.testing.assert_array_equal(after['y'], np.asarray(before['y'], dtype=float))
            np.testing.assert_array_equal(after['controlled_mask'],
                                          np.asarray(before['controlled_mask'], bool).reshape(-1)[:N])
            self.assertEqual(after['t'], before['t'])

    def test_non_integral_memory_falls_back_instead_of_rounding(self):
        frames = _frames(4, seed=7)
        frames[2]['W'] = np.asarray(frames[2]['W'], dtype=float) + 0.25
        record = ap._encode_arm(frames, N)
        self.assertEqual(record['wmode'], 'j')
        np.testing.assert_allclose(decode_arm(record, N)[2]['W'], frames[2]['W'])

    def test_packed_records_beat_json_int_arrays(self):
        keys = np.array([0, 5, 17, 160, 1599])
        vals = np.array([3, -2, 1, 0, 4])
        packed = ap._pack_records(keys, vals, 2, 1)
        loose = json.dumps([keys.tolist(), (keys // N).tolist(), (keys % N).tolist(), vals.tolist()])
        self.assertLess(len(packed), len(loose))
        back_k, back_v = ap._unpack_records(packed, 2, 1)
        np.testing.assert_array_equal(back_k, keys)
        np.testing.assert_array_equal(back_v, vals)


    def test_index_width_never_asks_for_a_dtype_that_does_not_exist(self):
        for nodes in (2, 12, 40, 256, 500, 1000):
            width = ap._index_width(nodes)
            self.assertIn(width, (1, 2, 4), nodes)
            self.assertGreaterEqual((1 << (8 * width)) - 1, nodes * nodes - 1, nodes)
            np.dtype(f'<u{width}')  # raises if the width is not a real integer type

    def test_a_large_network_round_trips_through_the_wide_index(self):
        frames = _frames(5, seed=13, n=500)
        record = ap._encode_arm(frames, 500)
        self.assertEqual(record['ib'], 4)
        back = decode_arm(record, 500)
        for before, after in zip(frames, back):
            np.testing.assert_array_equal(after['W'], np.asarray(before['W'], dtype=int))


class TestPayload(unittest.TestCase):
    def test_payload_carries_the_saved_numbers_without_re_running_anything(self):
        frames = {'a': _frames(6, 1), 'az': _frames(6, 1), 'b': _frames(6, 2), 'bz': _frames(6, 2)}
        scenes_for = _scenes([('origin00', 'coverage10', 0, ['az', 'a']),
                              ('origin01', 'random10', 0, ['bz', 'b'])], frames)
        payload, report = ap.assemble_payload(scenes_for, n=N, style='study', meta={}, max_frames=6,
                                              min_frames=2, max_bytes=None)
        self.assertEqual(payload['schema'], SCHEMA)
        self.assertEqual(report['comparisons'], 2)
        self.assertEqual(report['dropped'], [])
        html = render_player_html(payload, title='t')
        for scene in _embedded(html)['comparisons']:
            self.assertEqual(scene['count'], 4)
            self.assertIn(scene['origin'], ('origin00', 'origin01'))

    def test_budget_coarsens_sampling_before_dropping_comparisons(self):
        keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        frames = {}
        for i, key in enumerate(keys):
            frames[key] = _frames(24, seed=i)
            frames[f'{key}z'] = [dict(f) for f in frames[key]]
        specs = [(f'origin{i:02d}', 'coverage10', 0, [f'{key}z', key])
                 for i, key in enumerate(['a', 'h', 'g', 'f', 'e', 'd', 'c', 'b'])]
        scenes_for = _scenes(specs, frames)
        full, _ = ap.assemble_payload(scenes_for, n=N, style='study', meta={}, max_frames=24,
                                      min_frames=24, max_bytes=None)
        roomy, roomy_report = ap.assemble_payload(scenes_for, n=N, style='study', meta={},
                                                  max_frames=24, min_frames=4,
                                                  max_bytes=len(json.dumps(full)) // 2)
        self.assertEqual(roomy_report['comparisons'], 8, 'sampling should give before comparisons go')
        self.assertEqual(roomy_report['dropped'], [])
        self.assertLess(roomy_report['frames_per_arm'], 24)

        tight, tight_report = ap.assemble_payload(scenes_for, n=N, style='study', meta={},
                                                  max_frames=24, min_frames=4, max_bytes=6000)
        self.assertLess(tight_report['comparisons'], 8)
        self.assertTrue(tight_report['dropped'])
        self.assertLessEqual(tight_report['bytes'], 6000)
        # Dropping follows the declared order, never a success ranking: what is
        # kept is a prefix of the declared origins and what goes is the tail.
        self.assertEqual([c['origin'] for c in tight['comparisons']], ['origin00'])
        self.assertEqual(sorted(name.split('·')[0] for name in tight['dropped']),
                         [f'origin{i:02d}' for i in range(1, 8)])

    def test_budget_too_small_for_one_comparison_is_an_error_not_a_silent_page(self):
        frames = {'a': _frames(8, 3), 'az': _frames(8, 3)}
        scenes_for = _scenes([('origin00', 'coverage10', 0, ['az', 'a'])], frames)
        with self.assertRaises(ValueError):
            ap.assemble_payload(scenes_for, n=N, style='study', meta={}, max_frames=8, min_frames=8,
                                max_bytes=200)

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            ap.assemble_payload(lambda k: [], n=N, style='fancy', meta={})


class TestEmittedPage(unittest.TestCase):
    def setUp(self):
        frames = {'a': _frames(6, 5), 'az': _frames(6, 5), 'b': _frames(6, 6), 'bz': _frames(6, 6)}
        scenes_for = _scenes([('origin00', 'coverage10', 0, ['az', 'a']),
                              ('origin00', 'random10', 0, ['bz', 'b'])], frames)
        self.payload, _ = ap.assemble_payload(scenes_for, n=N, style='study',
                                              meta=dict(horizon=320.0, skipped=[]),
                                              max_frames=6, min_frames=2, max_bytes=None)
        self.html = render_player_html(self.payload, title='Persistent transformation')

    def test_no_external_resource_of_any_kind(self):
        self.assertEqual(re.findall(r'https?://', self.html), [])
        self.assertNotIn('<link', self.html)
        self.assertNotIn('src="', self.html)
        self.assertNotIn("src='", self.html)
        self.assertNotIn('<iframe', self.html)
        self.assertNotIn('@import', self.html)

    def test_embedded_data_reproduces_the_encoded_frames(self):
        page = _embedded(self.html)
        self.assertEqual(page['schema'], SCHEMA)
        self.assertEqual(len(page['comparisons']), 2)
        first = page['arms'][page['comparisons'][0]['zero']]
        rebuilt = decode_arm(first, page['N'])
        self.assertTrue(all(np.asarray(f['W']).any() for f in rebuilt))

    def test_navigation_hooks_and_keyboard_bindings_are_present(self):
        for hook in ('id="s-first"', 'id="s-prev"', 'id="s-next"', 'id="s-last"',
                     'id="f-first"', 'id="f-prev"', 'id="f-play"', 'id="f-next"',
                     'id="f-last"', 'id="scrub"', 'id="speed"', 'id="loop"', 'id="list"'):
            self.assertIn(hook, self.html, hook)
        for key in ('ArrowRight', 'ArrowLeft', 'Home', 'End', "' '"):
            self.assertIn(key, self.html, key)
        self.assertIn('shiftKey', self.html)

    def test_omissions_are_stated_on_the_page(self):
        payload = dict(self.payload, dropped=['origin07·coverage10·future0'])
        html = render_player_html(payload, title='t')
        # The wording is assembled in the browser, so pin its parts and the element.
        self.assertIn("'Byte budget omitted ' + P.dropped.length + ' comparison(s)", html)
        self.assertIn('id="warn"', html)
        self.assertEqual(_embedded(html)['dropped'], ['origin07·coverage10·future0'])

    def test_skipped_comparisons_are_stated_on_the_page(self):
        payload = dict(self.payload, skipped=['origin00·random10·future0: no retained frames'])
        html = render_player_html(payload, title='t')
        self.assertIn("' saved comparison(s) omitted: '", html)
        self.assertIn('no retained frames', html)

    def test_caption_keeps_the_sampled_frames_caveat(self):
        self.assertIn('do not certify an uninterrupted hold', self.html)


class TestInlineFixups(unittest.TestCase):
    LEGACY = ('<html><head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/'
              'font-awesome/4.7.0/css/font-awesome.min.css"><link rel="stylesheet" href="x.css">'
              '</head><body><div class="anim-controls"><button>'
              '<i class="fa fa-play"></i></button></div></body></html>')

    def test_cdn_stylesheet_is_removed_and_glyphs_are_inlined(self):
        fixed = inline_player_fixups(self.LEGACY)
        self.assertEqual(re.findall(r'https?://', fixed), [])
        self.assertNotIn('<link', fixed)
        self.assertIn('<style>', fixed)
        for glyph in ('.fa-play::before', '.fa-pause::before', '.fa-step-backward::before',
                      '.fa-fast-forward::before'):
            self.assertIn(glyph, fixed)

    def test_the_rest_of_the_document_survives(self):
        fixed = inline_player_fixups(self.LEGACY)
        self.assertIn('anim-controls', fixed)
        self.assertIn('</body></html>', fixed)

    def test_a_page_with_no_external_stylesheet_is_left_alone(self):
        html = '<html><head></head><body><div class="anim-controls"></div></body></html>'
        self.assertTrue(inline_player_fixups(html).endswith(html))


def _diverged(base, role, salt):
    """Same initial frame, different trajectory after it."""
    rng = np.random.default_rng(9000 + salt * 17 + len(role))
    out = [dict(base[0])]
    for frame in base[1:]:
        W = frame['W'].copy()
        W[0, 1] = int(rng.integers(0, 4))
        x = rng.choice([-1.0, 1.0], len(frame['x']))
        out.append(dict(t=frame['t'], W=W, x=x, y=rng.choice([-1.0, 1.0], len(frame['y'])),
                        controlled_mask=frame['controlled_mask']))
    return out


def _synthetic_bundle(root, origins=('origin00', 'origin01'), roles=('coverage10', 'random10'), n=N):
    """A manifest and unit files shaped like a real bundle, so load_raw verifies them."""
    bundle = Path(root) / 'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    actual, audit_roles, rows = {}, {}, []
    for index, origin in enumerate(origins):
        # Roles of one origin are matched pairs: they share the initial network
        # and diverge only afterwards, which is what the player asserts of them.
        base = _frames(7, seed=1000 + index, n=n)
        for role in ('zero',) + roles:
            key = f'{origin}__{role}__future0'
            frames = [dict(f) for f in base] if role == 'zero' else _diverged(base, role, index)
            unit = dict(recipient_nodes=[0, 1, 2, 3], is_valid=True, is_eligible=True, success=True,
                        success_through_horizon=role != 'random10', relapsed=False, completion_time=42.0,
                        expenditure=152.0, target_definition=dict(initial_metrics=dict(m_x_0=0.2, m_y_0=0.3)),
                        network_frames=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                         for k, v in frame.items()} for frame in frames])
            raw = json.dumps(unit).encode()
            name = f'{key}.json'
            (bundle / name).write_bytes(raw)
            actual[key] = dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), kind='audit',
                               base_origin=origin, future=0, recipient_nodes=[0, 1, 2, 3])
            audit_roles[f'{origin}__{role}__future0'] = key
            rows.append(dict(role=role, origin=origin, future=0,
                             count=0 if role == 'zero' else 4,
                             nodes=[] if role == 'zero' else [0, 1, 2, 3],
                             physical_unit=key, persistent=unit['success_through_horizon'],
                             hold=True, relapsed=False, valid=True, completion=42.0))
    manifest = dict(schema_version='sparse-v1', status='complete',
                    identity=dict(plan=dict(horizon=320.0, release_deadline=20.0, profile=dict(N=n))),
                    actual=actual, audit_roles=audit_roles)
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    import pandas as pd
    data = dict(bundle=str(bundle), N=n, manifest=manifest, roles=pd.DataFrame(rows))
    return bundle, data


class TestNotebookEntryPoint(unittest.TestCase):
    def test_it_builds_a_page_and_reports_without_widgets(self):
        try:
            import IPython.display as ipd
        except ImportError:  # pragma: no cover - IPython is a notebook dependency
            self.skipTest('IPython unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            _, data = _synthetic_bundle(tmp)
            shown = []
            original, ipd.display = ipd.display, lambda *a, **k: shown.append(a[0] if a else None)
            try:
                result = interactive_comparisons(data, Path(tmp) / 'comparisons', inline_bytes=300_000)
            finally:
                ipd.display = original
            self.assertTrue(result['page'].exists())
            self.assertEqual(len(shown), 1, 'the player should be displayed once')
            self.assertIn("createElement('canvas')", html.unescape(shown[0].data))
            self.assertIn('sandbox="allow-scripts"', shown[0].data)
            self.assertEqual(re.findall(r'https?://', result['page'].read_text()), [])
            metadata = json.loads(result['metadata'].read_text())
            self.assertEqual(metadata['schema'], SCHEMA)
            self.assertEqual(metadata['comparisons'], len(result['shown']['comparisons']))

    def test_it_refuses_a_bundle_with_no_saved_frames(self):
        import pandas as pd
        data = dict(bundle='nowhere', N=N, manifest=dict(identity=dict(plan={})),
                    roles=pd.DataFrame([dict(role='coverage10', origin='o', future=0, count=0, nodes=[],
                                             physical_unit='u', persistent=False, hold=False, relapsed=False,
                                             valid=False, completion=None)]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                payload_from_data(data, max_bytes=None)


if __name__ == '__main__':
    unittest.main()

def test_exact_treatment_history_survives_frame_sampling():
    frames=_frames(3,seed=8)
    for frame in frames:frame['controlled_mask'][:]=False
    frames[0]['first_treated']={'5':1.0,'7':15.0}
    record=ap._encode_arm([frames[0],frames[2]],N)
    assert record['first_treated']=={'5':1.0,'7':15.0}
    assert record['active']==[[],[]]  # History never falsely becomes active treatment.
    assert record['t']==[0.,20.]
