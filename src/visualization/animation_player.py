"""Self-contained, offline navigation of saved trajectories; never simulates.

The saved bundles already hold the numbers that were plotted (`network_frames`:
`W`, `x`, `y`, `controlled_mask`, `t`). This module encodes those numbers once
and draws them in the browser on a canvas, instead of baking one PNG per frame
into the page. That keeps a whole bundle's comparisons in a single file that
opens from local disk with no network, and lets the file's own buttons walk
through every saved comparison without re-rendering anything.

Two drawing styles are reproduced, so published panel titles and stance
percentages do not move:

- ``study``: the matched-pair panels of
  :func:`src.visualization.study_figures.render_comparison`.
- ``control``: the arms panels of
  :func:`src.visualization.control_plots._draw_control_network`.

Node areas follow each style's own formula exactly, and one scale is shared
across every frame and arm, which is what the encodings claim. The reference
points-per-unit constant below only fixes how large a point is relative to the
panel; both encodings are explicitly non-proportional to attention, so absolute
size carries no claim while relative size does.
"""
from __future__ import annotations

import base64
import datetime as _datetime
import hashlib
import html as _html
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 'modular-player-v1'
# Reference drawing scale: the delivered figures mapped about 90 points onto one
# data unit. Only the ratio between node sizes carries meaning; see module doc.
REFERENCE_POINTS_PER_UNIT = 90.0
EXTENT = {'study': 1.3, 'control': 1.28}
# Measured on the N40 validation bundle: 24 sampled frames per arm is 20.2 MB,
# 16 is 15.2 MB, 10 is 10.7 MB, for all 616 saved comparisons.
DEFAULT_MAX_BYTES = 16_000_000
# The notebook embeds the player in its own output, which is saved into the
# notebook file. A slice this size keeps that at a few megabytes while the page
# written to disk keeps every comparison; the omitted count always shows on the
# page itself, so a partial inline view never looks complete.
INLINE_MAX_BYTES = 3_000_000
DEFAULT_MAX_FRAMES = 24
DEFAULT_MIN_FRAMES = 10

Scene = Dict[str, Any]
Layout = Callable[[Sequence[Dict[str, Any]]], np.ndarray]


class PayloadBudgetError(ValueError):
    """The requested byte budget cannot hold one comparison."""


def _coolwarm_lut(n: int = 256) -> List[List[int]]:
    """Sample the exact colormap matplotlib uses, so browser colours match."""
    import matplotlib
    rgba = np.asarray(matplotlib.colormaps['coolwarm'](np.linspace(0.0, 1.0, n)))
    return [[int(round(255 * c)) for c in row[:3]] for row in rgba]


def _pack_states(values: Any, n: int) -> Optional[str]:
    """Pack a binary state layer as base64 bits; None when the layer is not binary.

    Saved states take the values -1 and +1. Anything else falls back to a
    rounded float list, so no trajectory is silently reinterpreted as binary.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size != n or not np.all((v == 1.0) | (v == -1.0)):
        return None
    return base64.b64encode(np.packbits((v < 0.0).astype(np.uint8)).tobytes()).decode('ascii')


def _float_states(values: Any, n: int) -> List[float]:
    return [float(q) for q in np.asarray(values, dtype=float).reshape(-1)[:n]]


def _encode_states(values: Any, n: int) -> Tuple[Optional[str], Optional[List[float]]]:
    code = _pack_states(values, n)
    return code, None if code is not None else _float_states(values, n)


def _index_width(n: int) -> int:
    """Bytes per flat index. Only 1, 2 and 4 exist: there is no 3-byte integer."""
    needed = max(1, (int(n) * int(n) - 1).bit_length() + 7 >> 3)
    return 4 if needed > 2 else needed


def _value_width(values: np.ndarray) -> int:
    top = int(np.max(np.abs(values))) if values.size else 0
    for width in (1, 2, 4):
        if top <= (1 << (8 * width - 1)) - 1:
            return width
    return 8


def _pack_records(indices: np.ndarray, values: np.ndarray, ib: int, vb: int) -> str:
    """Little-endian (index, value) records as base64.

    The uploaded figures are drawn from integer memory matrices, so an entry
    needs a small index and a tiny signed value; JSON arrays of ints cost about
    seven characters per entry, these records cost three bytes. The index is
    unsigned — a flat index into an N by N matrix is never negative, and `ib`
    is chosen to hold the largest one.
    """
    spec = np.dtype([('i', f'<u{ib}'), ('v', f'<i{vb}')])
    block = np.empty(len(indices), dtype=spec)
    block['i'] = np.asarray(indices, dtype=np.int64)
    block['v'] = np.asarray(values, dtype=np.int64)
    return base64.b64encode(block.tobytes()).decode('ascii')


def _unpack_records(code: str, ib: int, vb: int) -> Tuple[np.ndarray, np.ndarray]:
    spec = np.dtype([('i', f'<u{ib}'), ('v', f'<i{vb}')])
    block = np.frombuffer(base64.b64decode(code), dtype=spec)
    return block['i'].astype(np.int64), block['v'].astype(np.int64)


def _encode_arm(frames: Sequence[Dict[str, Any]], n: int) -> Dict[str, Any]:
    """Encode one saved arm: states per frame, and the memory matrix as deltas.

    `W` is stored once in full and then as exact per-frame differences. Memory
    updates touch few entries at a time, which keeps a whole bundle in a single
    page instead of repeating the matrix for every frame. Integral matrices use
    packed byte records; a non-integral matrix falls back to JSON arrays so no
    saved value is rounded away.
    """
    t: List[float] = []
    bx: List[Optional[str]] = []
    by: List[Optional[str]] = []
    fx: List[Optional[List[float]]] = []
    fy: List[Optional[List[float]]] = []
    active: List[List[int]] = []
    base_keys: List[int] = []
    base_vals: List[int] = []
    delta_keys: List[np.ndarray] = []
    delta_vals: List[np.ndarray] = []
    integral = True
    previous: Dict[int, int] = {}
    for k, frame in enumerate(frames):
        t.append(float(frame['t']))
        cx, rx = _encode_states(frame['x'], n)
        cy, ry = _encode_states(frame['y'], n)
        bx.append(cx); fx.append(rx); by.append(cy); fy.append(ry)
        mask = np.asarray(frame.get('controlled_mask', np.zeros(n)), dtype=bool).reshape(-1)
        active.append([int(i) for i in np.nonzero(mask[:n])[0]])
        W = np.asarray(frame['W'], dtype=float)
        integral = integral and bool(np.all(W == np.rint(W)))
        rows, cols = np.nonzero(W)
        # Kept as floats so a non-integral matrix falls back to exact JSON
        # values instead of being quietly truncated.
        current = {int(i) * n + int(j): float(W[i, j]) for i, j in zip(rows.tolist(), cols.tolist())}
        if k == 0:
            base_keys.extend(current)
            base_vals.extend(current.values())
        else:
            changed: Dict[int, float] = {}
            for key, value in current.items():
                if previous.get(key) != value:
                    changed[key] = value
            for key in previous:
                if key not in current:
                    changed[key] = 0.0
            keys = sorted(changed)
            delta_keys.append(np.asarray(keys, dtype=np.int64))
            delta_vals.append(np.asarray([changed[key] for key in keys], dtype=np.int64 if integral
                                         else float))
        previous = current
    first_treated = dict(frames[0].get('first_treated', {})) if frames else {}
    if not first_treated:
        for when, nodes in zip(t, active):
            for node in nodes:first_treated.setdefault(str(node), when)
    record = dict(t=t, bx=bx, by=by, fx=fx, fy=fy, active=active, first_treated=first_treated, wmode='b' if integral else 'j',
                  wn=[int(x.size) for x in delta_keys])
    if not integral:
        record.update(wi=[int(k) // n for k in base_keys], wj=[int(k) % n for k in base_keys],
                      wv=base_vals,
                      di=[[int(k) // n for k in keys] for keys in delta_keys],
                      dj=[[int(k) % n for k in keys] for keys in delta_keys],
                      dv=[[float(v) for v in vals] for vals in delta_vals])
        return record
    everything = [np.asarray(base_vals, dtype=np.int64)] + [v for v in delta_vals if v.size]
    ib = _index_width(n)
    vb = _value_width(np.concatenate(everything))
    record.update(ib=ib, vb=vb,
                  wb=_pack_records(np.asarray(base_keys, dtype=np.int64),
                                   np.asarray(base_vals, dtype=np.int64), ib, vb),
                  wd=_pack_records(np.concatenate(delta_keys) if delta_keys else np.zeros(0, dtype=np.int64),
                                   np.concatenate(delta_vals) if delta_vals else np.zeros(0, dtype=np.int64),
                                   ib, vb))
    return record


def decode_arm(record: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Rebuild an encoded arm's matrices exactly. Used by tests and audits."""
    frames = len(record['t'])
    if record['wmode'] == 'j':
        base = list(zip(record['wi'], record['wj'], record['wv']))
        deltas = [list(zip(a, b, v)) for a, b, v in zip(record['di'], record['dj'], record['dv'])]
    else:
        ib, vb = int(record['ib']), int(record['vb'])
        keys, vals = _unpack_records(record['wb'], ib, vb)
        base = [(int(k) // n, int(k) % n, int(v)) for k, v in zip(keys, vals)]
        keys, vals = _unpack_records(record['wd'], ib, vb)
        at, deltas = 0, []
        for count in record['wn']:
            count = int(count)
            deltas.append([(int(k) // n, int(k) % n, int(v))
                           for k, v in zip(keys[at:at + count], vals[at:at + count])])
            at += count
    out: List[Dict[str, Any]] = []
    W = np.zeros((n, n), dtype=int if record['wmode'] == 'b' else float)
    for i, j, v in base:
        W[i, j] = v
    for k in range(frames):
        if k:
            for i, j, v in deltas[k - 1]:
                W[i, j] = v
        x = np.array([frame_state(record, 'bx', 'fx', k, i) for i in range(n)], dtype=float)
        y = np.array([frame_state(record, 'by', 'fy', k, i) for i in range(n)], dtype=float)
        out.append(dict(t=record['t'][k], W=W.copy(), x=x, y=y,
                        controlled_mask=np.zeros(n, dtype=bool)))
        for node in record['active'][k]:
            out[-1]['controlled_mask'][int(node)] = True
    return out


def frame_state(record: Dict[str, Any], packed_key: str, float_key: str, k: int, i: int) -> float:
    code = record[packed_key][k]
    if code:
        bits = base64.b64decode(code)
        return -1.0 if (bits[i >> 3] >> (7 - (i & 7))) & 1 else 1.0
    return float(record[float_key][k][i])


def _encode_layout(pos: np.ndarray) -> List[List[float]]:
    return [[round(float(a), 4), round(float(b), 4)] for a, b in np.asarray(pos, dtype=float)]


def encode_scenes(scenes: Sequence[Scene], n: int) -> Dict[str, Dict[str, Any]]:
    """Encode every distinct arm once. This is the expensive pass of the module."""
    encoded: Dict[str, Dict[str, Any]] = {}
    for scene in scenes:
        for arm in scene['arms']:
            if arm['key'] in encoded:
                continue
            layout = _encode_layout(arm['pos'])
            encoded[arm['key']] = dict(
                label=str(arm['label']), targeted=[int(x) for x in arm.get('targeted') or []],
                layout_id=json.dumps(layout, separators=(',', ':')), layout_pos=layout,
                **_encode_arm(arm['frames'], n))
    return encoded


def _materialise(scenes: Sequence[Scene], encoded: Dict[str, Dict[str, Any]], *, n: int, style: str,
                 meta: Dict[str, Any], dropped: Sequence[str]) -> Dict[str, Any]:
    """Deduplicate arms and layouts, so a shared trajectory is stored once."""
    arms: List[Dict[str, Any]] = []
    arm_ids: Dict[str, int] = {}
    layouts: List[List[List[float]]] = []
    layout_ids: Dict[str, int] = {}
    comparisons: List[Dict[str, Any]] = []
    for scene in scenes:
        ids, scene_layouts = [], []
        for arm in scene['arms']:
            key = arm['key']
            if key not in arm_ids:
                arm_ids[key] = len(arms)
                arms.append(encoded[key])
            ids.append(arm_ids[key])
            code = encoded[key]['layout_id']
            if code not in layout_ids:
                layout_ids[code] = len(layouts)
                layouts.append(encoded[key]['layout_pos'])
            scene_layouts.append(layout_ids[code])
        comparisons.append(dict(
            origin=str(scene['origin']), role=str(scene['role']), future=int(scene['future']),
            nodes=[int(x) for x in scene['nodes']], count=int(len(scene['nodes'])),
            label=str(scene['label']), zero=ids[0], arm=ids[1],
            layouts=scene_layouts, outcome=scene.get('outcome') or {},
        ))
    return dict(schema=SCHEMA, N=int(n), style=style, comparisons=comparisons, arms=arms,
                layouts=layouts, coolwarm=_coolwarm_lut(), dropped=list(dropped), **meta)


def _finite(value: Any) -> Optional[float]:
    """A JSON-safe number: undefined durations become null, never NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _serialise(payload: Dict[str, Any]) -> str:
    """JSON the browser can actually parse.

    Python writes bare `NaN`/`Infinity` by default and JavaScript's `JSON.parse`
    rejects both, which silently kills the whole page. A never-completed run has
    an undefined completion time, so this is the common case, not a corner one:
    non-finite numbers are normalised to `null` at the source, and
    `allow_nan=False` makes any future leak a build-time error instead.
    """
    return json.dumps(payload, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _payload_bytes(payload: Dict[str, Any]) -> int:
    return len(_serialise(payload).encode('utf-8'))


def assemble_payload(scenes_for: Callable[[int], List[Scene]], *, n: int, style: str, meta: Dict[str, Any],
                     max_frames: int = DEFAULT_MAX_FRAMES, min_frames: int = DEFAULT_MIN_FRAMES,
                     max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit a scene set to a byte budget. Frame sampling is reduced before anything is dropped.

    Comparisons are never re-ranked. Sampling is coarsened down to `min_frames`
    first, because every saved comparison being reachable matters more than the
    density of the sample; only then are whole comparisons dropped from the end
    of the declared order (the last origin and role as recorded, never the least
    successful one), and every drop is listed in the payload for the page to show.
    """
    if style not in EXTENT:
        raise ValueError(f'Unknown style: {style}')
    if min_frames < 2:
        raise ValueError('min_frames must keep at least two frames')
    if max_frames < 2:
        raise ValueError('max_frames must keep at least two frames')
    min_frames = min(min_frames, int(max_frames))
    frames, tried = int(max_frames), []
    scenes, payload, size = [], None, 0
    while True:
        tried.append(frames)
        scenes = scenes_for(frames)
        payload = _materialise(scenes, encode_scenes(scenes, n), n=n, style=style, meta=meta, dropped=[])
        size = _payload_bytes(payload)
        if max_bytes is None or size <= max_bytes or frames <= min_frames:
            break
        frames = max(min_frames, min(frames - 1, int(frames * max_bytes / float(size) * 0.97)))
    kept, dropped = list(scenes), []
    while max_bytes is not None and size > max_bytes and kept:
        cut = max(1, int(len(kept) - len(kept) * max_bytes / float(size) * 0.97))
        for scene in kept[-cut:]:
            dropped.append(f"{scene['origin']}·{scene['role']}·future{scene['future']}")
        kept = kept[:-cut]
        if not kept:
            raise PayloadBudgetError('Byte budget too small for even one comparison; raise max_bytes')
        payload = _materialise(kept, encode_scenes(kept, n), n=n, style=style, meta=meta, dropped=dropped)
        size = _payload_bytes(payload)
    report = dict(bytes=size, dropped=dropped, comparisons=len(payload['comparisons']),
                  stored_arms=len(payload['arms']), frames_per_arm=frames,
                  frame_counts_tried=tried, layouts=len(payload['layouts']))
    return payload, report


def _scene(origin: str, role: str, future: int, nodes: Sequence[int], label: str,
           arms: Sequence[Dict[str, Any]], outcome: Optional[Dict[str, Any]] = None) -> Scene:
    return dict(origin=str(origin), role=str(role), future=int(future), nodes=[int(x) for x in nodes],
                label=str(label), arms=list(arms), outcome=outcome or {})


def _arms_at(pairs: Sequence[Tuple[str, str, Sequence[Dict[str, Any]]]], *, indices: Sequence[int],
             layouts: Dict[str, np.ndarray], layout: Callable[[Sequence[Dict[str, Any]]], np.ndarray],
             targeted: Optional[Dict[str, List[int]]] = None) -> List[Dict[str, Any]]:
    out = []
    for label, key, frames in pairs:
        if key not in layouts:
            layouts[key] = layout(frames)
        sample_key = key + '::frames=' + ','.join(map(str, indices))
        out.append(dict(key=sample_key, label=label, frames=[frames[i] for i in indices],
                        pos=layouts[key], targeted=(targeted or {}).get(key, [])))
    return out


def payload_from_data(data: Dict[str, Any], *, style: str = 'study', max_frames: int = DEFAULT_MAX_FRAMES,
                      min_frames: int = DEFAULT_MIN_FRAMES, max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
                      include: Optional[Iterable[Tuple[str, str, int]]] = None,
                      prepared: Optional[Dict[str, Any]] = None,
                      progress: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ctx = prepared if prepared is not None else prepare_comparisons(data, include=include, progress=progress)
    payload, report = assemble_payload(ctx['scenes_for'], n=ctx['n'], style=style, meta=ctx['meta'],
                                       max_frames=max_frames, min_frames=min_frames, max_bytes=max_bytes)
    report['skipped'] = ctx['skipped']
    return payload, report


def prepare_comparisons(data: Dict[str, Any], *,
                        include: Optional[Iterable[Tuple[str, str, int]]] = None,
                        progress: bool = True) -> Dict[str, Any]:
    """Load and verify every saved matched pair once, and hand back the layout cache.

    Loading dominates the cost of building a page, so the notebook prepares
    once and then fits two byte budgets from the same verified frames: the full
    page for disk, a smaller slice for the notebook's own saved output.
    """
    """Every saved matched comparison in an analysed bundle, as one navigable page.

    Order is origin, then audit future, then intervention, so stepping forward
    moves between selectors at one origin/future before advancing — the
    comparison the study actually makes. Nothing here consults outcomes.

    Pass `prepared` from :func:`prepare_comparisons` to build a second budget
    from the same verified frames instead of reading every unit again.
    """
    from src.control.study_analysis import load_raw
    from src.visualization.control_plots import _select_animation_indices
    from src.visualization.network_layout import _seeded_spring_layout

    df = data['roles']
    if df is None or len(df) == 0:
        raise ValueError('No saved comparison rows to navigate')
    manifest = data['manifest']
    main_study = manifest.get('schema_version') == 'main-study-v1'
    records = df.to_dict('records')
    zeros = {(str(r['origin']), int(r['future'])): r for r in records if r['role'] == 'zero'}
    wanted = {tuple(x) for x in include} if include is not None else None
    cache: Dict[str, List[Dict[str, Any]]] = {}
    layouts: Dict[str, np.ndarray] = {}

    def frames_for(unit: str) -> List[Dict[str, Any]]:
        if unit not in cache:
            raw = load_raw(data['bundle'], unit)
            frames = [dict(f) for f in (raw.get('network_frames') or [])]
            first_treated = {}
            for action in raw.get('summaries', {}).get('recorder', {}).get('action_history', []):
                if action.get('spending_rate', 0) > 0:
                    for node in action['actuated_nodes']:
                        first_treated.setdefault(str(node), float(action['decision_time']))
            if frames and first_treated:frames[0]['first_treated'] = first_treated
            cache[unit] = frames
        return cache[unit]

    ordered = [r for r in records
               if r['role'] != 'zero' and (int(r['count']) > 0 or data.get('include_unactuated_policies',False))
               and (str(r['origin']), int(r['future'])) in zeros
               and not (main_study and int(r['future']) != 0)
               and (wanted is None or (str(r['origin']), str(r['role']), int(r['future'])) in wanted)]
    if not ordered:
        raise ValueError('No saved comparison with a matched zero and retained frames')
    ordered.sort(key=lambda r: (str(r['origin']), int(r['future']), str(r['role'])))

    prepared: List[Tuple[Dict[str, Any], Sequence[Tuple[str, str, List[Dict[str, Any]]]]]] = []
    skipped: List[str] = []
    for seen, row in enumerate(ordered):
        if progress and seen and seen % 60 == 0:
            print(f'  reading comparison {seen}/{len(ordered)} (trajectories loaded: {len(cache)})', flush=True)
        tag = f"{row['origin']}·{row['role']}·future{row['future']}"
        members = [('Matched zero', zeros[(str(row['origin']), int(row['future']))]), (str(row['role']), row)]
        loaded: List[Tuple[str, str, List[Dict[str, Any]]]] = []
        for label, member in members:
            frames = frames_for(member['physical_unit'])
            if not frames:
                loaded = []
                skipped.append(f'{tag}: no retained frames')
                break
            loaded.append((label, member['physical_unit'], frames))
        if not loaded:
            continue
        first, second = loaded[0][2], loaded[1][2]
        if (len(first) != len(second)
                or any(float(a['t']) != float(b['t']) for a, b in zip(first, second))
                or any(not np.array_equal(np.asarray(first[0][key]), np.asarray(second[0][key]))
                       for key in ('W', 'x', 'y'))):
            skipped.append(f'{tag}: arms are not a synchronized comparison')
            continue
        prepared.append((row, loaded))
    if not prepared:
        raise ValueError('No saved comparison with a matched zero and retained frames')
    n = int(np.asarray(prepared[0][1][0][2][0]['W']).shape[0])

    def scenes_for(k: int) -> List[Scene]:
        out = []
        for row, loaded in prepared:
            indices = _select_animation_indices(loaded[1][2], k)
            out.append(_scene(
                row['origin'], row['role'], row['future'], row['nodes'],
                f"{row['origin']} · {row['role']} · future {row['future']}",
                _arms_at(loaded, indices=indices, layouts=layouts,
                         layout=lambda fs: _seeded_spring_layout(np.asarray(fs[0]['W']), iterations=50)),
                outcome=dict(persistent=bool(row['persistent']), hold=bool(row['hold']),
                             relapsed=bool(row['relapsed']), valid=bool(row['valid']),
                             completion=_finite(row['completion']))))
        return out

    plan = manifest.get('identity', {}).get('plan', {})
    meta = dict(
        horizon=_finite(plan.get('horizon')) or 320.0,
        release_deadline=_finite(plan.get('release_deadline')) or 20.0,
        matched_zero_present=True, skipped=skipped,
        source_manifest_sha256=hashlib.sha256((Path(data['bundle']) / 'manifest.json').read_bytes()).hexdigest(),
        future_note=('Only audit future 0 retains network frames in this bundle' if main_study
                     else 'Every audit future in this bundle retains network frames'))
    return dict(scenes_for=scenes_for, n=n, meta=meta, skipped=skipped,
                comparisons=len(prepared), layouts=layouts)


def payload_from_units(units: Sequence[Dict[str, Any]], *, style: str = 'control',
                       origin_id: Optional[str] = None, max_frames: int = DEFAULT_MAX_FRAMES,
                       min_frames: int = DEFAULT_MIN_FRAMES, max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
                       meta: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Navigate each active arm against its same-origin, same-package zero."""
    from src.visualization.control_plots import _select_animation_indices
    from src.visualization.network_layout import _seeded_spring_layout

    usable = [u for u in units if u.get('network_frames')]
    if not usable:
        raise ValueError('No saved units to animate')
    available = {u['origin_id'] for u in usable if u['policy_id'] != 'zero'}
    if not available:
        raise ValueError('No saved active arm with retained frames')
    if origin_id is None:
        origin_id = sorted(available)[0]
    if origin_id not in available:
        raise ValueError(f'No saved frames for origin {origin_id}')
    chosen = [u for u in usable if u['origin_id'] == origin_id]
    zeros = [u for u in chosen if u['policy_id'] == 'zero']
    active = [u for u in chosen if u['policy_id'] != 'zero']
    if not active:
        raise ValueError('No saved active arm for this origin')
    zeros_by_package = {}
    for unit in zeros:
        package = unit['unit']['package']
        if package in zeros_by_package:
            raise ValueError(f'Multiple saved zero arms for {origin_id}/{package}')
        zeros_by_package[package] = unit
    targeted, layouts, pairs = {}, {}, []

    def loaded(unit):
        key = f"{unit['origin_id']}::{unit['unit']['package']}::{unit['policy_id']}"
        targeted[key] = [int(x) for x in
                         ((unit.get('summaries') or {}).get('effective_settings') or {}).get('targeted_nodes') or []]
        label = ('Matched zero' if unit['policy_id'] == 'zero'
                 else f"{unit['unit']['package']} / {unit['policy_id']}")
        return label, key, unit['network_frames']

    n = int(np.asarray(active[0]['network_frames'][0]['W']).shape[0])
    for unit in sorted(active, key=lambda u: (u['unit']['package'], u['policy_id'])):
        treatment = loaded(unit)
        zero = zeros_by_package.get(unit['unit']['package'])
        reference = loaded(zero) if zero is not None else treatment
        first, second = reference[2], treatment[2]
        if ([float(f['t']) for f in first] != [float(f['t']) for f in second]
                or any(not np.array_equal(first[0][key], second[0][key])
                       for key in ('W', 'x', 'y'))):
            raise ValueError('Matched arms must share time grids and initial state')
        if np.asarray(second[0]['W']).shape != (n, n):
            raise ValueError('Saved arms must share the network size')
        pairs.append((reference, treatment))

    def scenes_for(k: int) -> List[Scene]:
        scenes = []
        for reference, treatment in pairs:
            indices = _select_animation_indices(treatment[2], k)
            nodes = sorted({int(i) for f in treatment[2]
                            for i in np.flatnonzero(f.get('controlled_mask', []))})
            scenes.append(_scene(origin_id, treatment[0], 0, nodes,
                                 f'{origin_id} · {treatment[0]}',
                                 _arms_at([reference, treatment], indices=indices, layouts=layouts,
                                          layout=lambda fs: _seeded_spring_layout(
                                              np.asarray(fs[0]['W']), layout_seed=42),
                                          targeted=targeted)))
        return scenes

    base = dict(horizon=None, release_deadline=None, skipped=[])
    base.update(meta or {})
    base['matched_zero_present'] = all(a[1] != b[1] for a, b in pairs)
    return assemble_payload(scenes_for, n=n, style=style, meta=base, max_frames=max_frames,
                            min_frames=min_frames, max_bytes=max_bytes)


def render_player_html(payload: Dict[str, Any], *, title: str) -> str:
    """The one self-contained page: no stylesheet, script or font is fetched."""
    blob = _serialise(payload).replace('<', '\\u003c')
    head, tail = _PAGE.split('__DATA__', 1)
    return head.replace('__TITLE__', _html.escape(str(title))) + blob + tail


def write_player_html(payload: Dict[str, Any], path: Path, *, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_player_html(payload, title=title), encoding='utf-8')
    return path


def write_player_metadata(payload: Dict[str, Any], path: Path, *, source: Optional[str] = None,
                          report: Optional[Dict[str, Any]] = None) -> Path:
    """Record what the page contains, so a rebuild can be skipped and audited."""
    path = Path(path)
    meta = dict(schema=payload['schema'], style=payload['style'], N=payload['N'],
                comparisons=len(payload['comparisons']), stored_arms=len(payload['arms']),
                stored_layouts=len(payload['layouts']),
                player_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                written_utc=_datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec='seconds'),
                horizon=payload.get('horizon'), release_deadline=payload.get('release_deadline'),
                source_manifest_sha256=payload.get('source_manifest_sha256'),
                matched_zero_present=payload.get('matched_zero_present'),
                frame_selection='sampled from saved frames; no trajectory is rerun for visualization',
                report=report or {}, source=source)
    out = path.with_name('player_metadata.json')
    out.write_text(json.dumps(meta, indent=2) + '\n')
    return out


def inline_player_fixups(html_text: str) -> str:
    """Make a matplotlib ``to_jshtml`` page usable offline.

    The stock player loads Font Awesome from a CDN and puts no text in its
    buttons, so with no network the controls render as blank space. This removes
    the external stylesheet and supplies inline glyphs plus button sizing.
    """
    out = html_text
    marker = '<link rel="stylesheet"'
    while True:
        start = out.find(marker)
        if start == -1:
            break
        end = out.find('>', start)
        if end == -1:
            out = out[:start]
            break
        out = out[:start] + out[end + 1:]
    css = (
        '<style>'
        '.anim-controls .fa{font-style:normal;font-family:inherit;line-height:1}'
        '.anim-controls button{min-width:2.4em;padding:.3em .5em;cursor:pointer;'
        'background:#f4f5f7;border:1px solid #b9bdc7;border-radius:4px;color:#202735}'
        '.anim-controls button:hover{background:#e6e9ee}'
        '.anim-buttons{display:flex;gap:4px}'
        '.fa-minus::before{content:"\\2796"}'
        '.fa-plus::before{content:"\\2795"}'
        '.fa-step-backward::before{content:"\\25C0\\25C0"}'
        '.fa-fast-backward::before{content:"\\23EE"}'
        '.fa-play::before{content:"\\25B6"}'
        '.fa-pause::before{content:"\\23F8"}'
        '.fa-flip-horizontal::before{content:"\\25C0"}'
        '.fa-step-forward::before{content:"\\25B6\\25B6"}'
        '.fa-fast-forward::before{content:"\\23ED"}'
        '</style>'
    )
    return css + out


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#1d2430;--muted:#5c6675;--line:#c9cfd8;--bg:#fff;--panel:#f6f7f9;--accent:#2f6ea8;--selected:#fff;--warning:#a4562a;--edge:#46546a;--edge-alpha:.24;--tie-alpha:.16;--plot-private:#316b9b;--plot-public:#b54c30}
:root[data-theme="dark"]{color-scheme:dark;--ink:#e7ebf1;--muted:#b9c3d1;--line:#596575;--bg:#12161c;--panel:#1a2029;--accent:#77a9d8;--selected:#12161c;--warning:#efb083;--edge:#c4d3e5;--edge-alpha:.38;--tie-alpha:.25;--plot-private:#80b9ea;--plot-public:#f49a7e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:14px 16px 40px}
h1{font-size:17px;margin:0 0 2px}
.prov{color:var(--muted);font-size:12px;margin-bottom:12px}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:10px}
button,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.45;cursor:default}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0 12px}
.filters label{font-size:12px;color:var(--muted)}
.filters input{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);min-width:180px}
.filters[hidden],.archive-extra[hidden],.frames[hidden],.network-viewport[hidden]{display:none}
.archive-extra{margin:10px 0;font-size:12px;color:var(--muted)}
.table-scroll{max-height:360px;overflow:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{position:sticky;top:0;background:var(--panel)}
.empty{padding:20px;color:var(--muted)}
body.fullscreen-view .filters,body.fullscreen-view .archive-extra{display:none}
button.icon{min-width:44px;font-variant-numeric:tabular-nums}
.counter{font-variant-numeric:tabular-nums;color:var(--muted);margin-left:auto;max-width:100%;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns:236px minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:820px){.layout{grid-template-columns:1fr}}
aside{background:var(--panel);border:1px solid var(--line);border-radius:8px;max-height:74vh;overflow:auto;padding:6px}
aside[hidden]{display:none}
aside h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:8px 6px 4px}
aside button{display:block;width:100%;text-align:left;background:transparent;border:0;border-radius:4px;padding:4px 6px;font-size:12.5px}
aside button[aria-current=true]{background:var(--accent);color:var(--selected)}
.layout.list-hidden{grid-template-columns:1fr}
.stage{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.network-viewport{overflow:auto;max-width:100%}
body.fullscreen-view{overflow:hidden}
body.fullscreen-view .wrap{max-width:none;width:100%;height:100dvh;padding:8px;display:flex;flex-direction:column}
body.fullscreen-view h1,body.fullscreen-view .prov,body.fullscreen-view #list,
body.fullscreen-view #toggle,body.fullscreen-view .key,body.fullscreen-view #members,
body.fullscreen-view #static-views,body.fullscreen-view #caption,body.fullscreen-view .help{display:none}
body.fullscreen-view .bar{flex:0 0 auto;margin-bottom:6px}
body.fullscreen-view .layout{display:block;flex:1;min-height:0}
body.fullscreen-view .layout>div{height:100%;display:flex;flex-direction:column;min-height:0}
body.fullscreen-view .network-viewport{flex:1;min-height:0;overflow:hidden;touch-action:none}
body.fullscreen-view #stage{height:100%;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:minmax(0,1fr)}
body.fullscreen-view #stage .panel{min-height:0;display:flex;flex-direction:column;overflow:hidden}
body.fullscreen-view #stage .panel[hidden]{display:none}
body.fullscreen-view #stage canvas{flex:1 1 0;height:0;min-height:0;aspect-ratio:auto}
body.fullscreen-view #stage .panel h3,body.fullscreen-view #stage .panel p{flex:0 0 auto}
body.fullscreen-view #stage[data-view="treatment"],body.fullscreen-view #stage[data-view="zero"]{grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr)}
body.fullscreen-view #stage[data-view="treatment"] .panel:first-child,
body.fullscreen-view #stage[data-view="zero"] .panel:last-child{display:none}
body.fullscreen-view .frames{flex:0 0 auto;margin-top:6px}
body.fullscreen-view .network-viewport.zoomed{cursor:grab}
body.fullscreen-view .network-viewport.dragging{cursor:grabbing}
@media(max-width:700px){body.fullscreen-view #stage{grid-template-columns:minmax(0,1fr);grid-template-rows:repeat(2,minmax(0,1fr))}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 8px 4px}
.panel h3{margin:0 0 6px;font-size:13px;font-weight:600}
.panel p{margin:4px 2px 2px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
canvas{width:100%;aspect-ratio:1/1;display:block;border-radius:6px}
.frames{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:12px}
input[type=range]{flex:1 1 240px;min-width:160px;accent-color:var(--accent)}
.key{margin-top:12px;font-size:12px;color:var(--muted);display:grid;gap:3px}
.key b{color:var(--ink);font-weight:600}
.note,.warn,.help{margin-top:8px;font-size:12px;color:var(--muted)}
.warn{color:var(--warning)}
kbd{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 4px;font:inherit}
</style></head><body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="prov" id="h-prov"></div>
  <div id="filters" class="filters" hidden>
    <label>Study <select id="filter-study" aria-label="Study"></select></label>
    <label>Nodes <select id="filter-size" aria-label="Network size"></select></label>
    <label>Memory <select id="filter-memory" aria-label="Memory"><option value="">All memory settings</option><option value="adaptive">Adaptive</option><option value="frozen">Fixed</option></select></label>
    <label>Outcome <select id="filter-outcome" aria-label="Outcome"><option value="">All outcomes</option><option value="persistent">Persistent</option><option value="breach">Threshold breach</option><option value="no-hold">Hold not completed</option><option value="invalid">Invalid / ineligible</option></select></label>
    <input id="search-results" type="search" aria-label="Search saved results" placeholder="Search policy, network, or study">
    <button id="reset-filters">Reset</button>
  </div>
  <details id="label-guide" class="archive-extra" hidden><summary>Label key and reading guide</summary><div id="label-key"></div></details>
  <details id="study-results" class="archive-extra" hidden><summary>Saved outcomes · all futures, including those without animation frames</summary>
    <p>Each study keeps its own cohort and success definition. Futures are nested within networks; policy aliases may reuse physical trajectories. Counts are descriptive and are not pooled across studies. The Outcome filter affects the animation list; this table and its CSV keep all outcomes so denominators remain visible.</p>
    <div class="table-scroll"><table><thead><tr><th>Study</th><th>N</th><th>Policy</th><th>Memory / reports</th><th>Networks</th><th>Valid outcomes</th><th>Persistent</th><th>Animated</th></tr></thead><tbody id="results-table"></tbody></table></div>
    <button id="download-outcomes">Download saved outcomes CSV</button>
  </details>
  <div id="study-context" class="archive-extra" hidden></div>
  <div class="bar">
    <button id="s-first" class="icon" title="First simulation (Shift+Home)">&#9198;</button>
    <button id="s-prev" class="icon" title="Previous simulation (Shift+Left)">&#9664;&#9664;</button>
    <button id="s-next" class="icon" title="Next simulation (Shift+Right)">&#9654;&#9654;</button>
    <button id="s-last" class="icon" title="Last simulation (Shift+End)">&#9197;</button>
    <button id="toggle" aria-expanded="true" title="Show or hide the list of simulations">List</button>
    <button id="fullscreen-view" aria-pressed="false" title="Automatically fit the networks and playback controls to the full screen">Full screen</button>
    <label id="full-view-control" hidden>View <select id="full-view" aria-label="Full screen panels">
      <option value="both" selected>Both panels</option><option value="treatment">Treatment only</option>
      <option value="zero">Matched zero only</option></select></label>
    <span id="zoom-controls" hidden>
      <button id="zoom-out" aria-label="Zoom out">−</button>
      <button id="zoom-fit" title="Fit the whole network and reset pan">Fit</button>
      <button id="zoom-in" aria-label="Zoom in">+</button>
      <span id="zoom-level">100%</span>
    </span>
    <label>Theme <select id="theme" aria-label="Color theme"><option value="light" selected>Light</option><option value="dark">Dark</option></select></label>
    <label>Sort <select id="sort" aria-label="Sort simulations">
      <option value="original">Saved order</option>
      <option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option>
      <option value="role">Intervention A–Z</option>
      <option value="future">Future: lowest first</option>
      <option value="count-asc">Recipients: fewest first</option>
      <option value="count-desc">Recipients: most first</option>
      <option value="outcome">Outcome: persistent first</option>
    </select></label>
    <span class="counter" id="h-count"></span>
  </div>
  <div class="layout">
    <aside id="list"><div id="list-body"></div></aside>
    <div>
      <div class="network-viewport" id="network-viewport"><div class="stage" id="stage"></div></div>
      <div class="frames">
        <button id="f-first" class="icon" title="First frame (Home)">&#9198;</button>
        <button id="f-prev" class="icon" title="Previous frame (Left)">&#9664;</button>
        <button id="f-play" style="min-width:86px" title="Play or pause (Space)">&#9654; Play</button>
        <button id="f-next" class="icon" title="Next frame (Right)">&#9654;</button>
        <button id="f-last" class="icon" title="Last frame (End)">&#9197;</button>
        <input type="range" id="scrub" min="0" step="1" value="0" aria-label="Frame">
        <span id="h-time" style="min-width:170px;font-variant-numeric:tabular-nums"></span>
        <label>Speed <select id="speed">
          <option value="0.5">0.5x</option><option value="1" selected>1x</option>
          <option value="2">2x</option><option value="4">4x</option></select></label>
        <label><input type="checkbox" id="loop" checked> Loop</label>
      </div>
      <div class="key" id="key"></div>
      <div class="note" id="members"></div>
      <details id="static-views"><summary>Snapshots and support curves</summary>
        <div class="stage" id="snapshots"></div>
        <p class="note">Deadline snapshots use the nearest displayed frame to the configured release deadline, not the actual release time. Support curves use sampled frames; certificates come from event records.</p>
        <div class="stage" id="curves"></div>
      </details>
      <div class="note" id="caption"></div>
      <details id="provenance" class="archive-extra" hidden><summary>Source identifiers</summary><div id="source-identifiers"></div></details>
      <div class="warn" id="warn" hidden></div>
      <div class="help">Keyboard: <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> frame,
        <kbd>Shift</kbd>+<kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> simulation,
        <kbd>Space</kbd> play/pause, <kbd>Home</kbd>/<kbd>End</kbd> first/last.</div>
    </div>
  </div>
</div>
<script id="player-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
var P = JSON.parse(document.getElementById('player-data').textContent);
var S = P.style, N = P.N, LUT = P.coolwarm, EXT = S === 'study' ? 1.3 : 1.28;
var BASE_MS = 300, TWO_PI = 6.283185307179586;
var state = {index: 0, frame: 0, playing: false, timer: null};
var camera = {zoom: 1, x: 0, y: 0}, drag = null;
var staticKey = null;
var order = P.comparisons.map(function(c, i){ return i; });
var canvases = [], contexts = [], armStates = [], cachedScales = null, scaleKey = null;
var decodedArms = new Map(), loadingSequence = 0, ready = false;
function armRecord(i){return decodedArms.get(i) || P.arms[i];}
async function ensureArm(i){
  if (!P.arms[i].gzip || decodedArms.has(i)) return;
  var bytes=b64bytes(P.arms[i].gzip);
  if (typeof DecompressionStream==='undefined') throw new Error('This viewer needs a current Chrome, Firefox, Edge, or Safari browser.');
  var stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  decodedArms.set(i,JSON.parse(await new Response(stream).text()));
}

function b64bytes(b64){
  var bin = atob(b64), out = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
var bitCache = Object.create(null);
function bits(b64){
  if (!(b64 in bitCache)) bitCache[b64] = b64bytes(b64);
  return bitCache[b64];
}
function stateAt(packed, floats, i){
  if (packed) { var b = bits(packed); return ((b[i >> 3] >> (7 - (i & 7))) & 1) ? -1 : 1; }
  return floats[i];
}
function lutColor(v){
  var t = (Math.max(-1, Math.min(1, v)) + 1) / 2;
  return 'rgb(' + LUT[Math.min(255, Math.max(0, Math.round(t * 255)))].join(',') + ')';
}
/* Packed records: little-endian index then signed value, ib and vb bytes each. */
function records(code, ib, vb){
  var b = b64bytes(code), view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  var stride = ib + vb, count = Math.floor(b.byteLength / stride), pairs = new Array(count);
  for (var i = 0; i < count; i++) {
    var off = i * stride;
    var key = ib === 1 ? view.getUint8(off)
        : ib === 2 ? view.getUint16(off, true) : view.getUint32(off, true);
    var val = vb === 1 ? view.getInt8(off + ib)
        : vb === 2 ? view.getInt16(off + ib, true) : vb === 4 ? view.getInt32(off + ib, true) : Number(view.getBigInt64(off + ib, true));
    pairs[i] = [key, val];
  }
  return pairs;
}
/* W is rebuilt from one full frame plus exact per-frame differences. */
function armState(armIdx){
  var arm = armRecord(armIdx), map = new Map(), upto = -1;
  var base, deltas = [], offsets = [0], i, j;
  if (arm.wmode === 'b') {
    base = records(arm.wb, arm.ib, arm.vb);
    deltas = records(arm.wd, arm.ib, arm.vb);
    for (i = 0; i < arm.wn.length; i++) offsets.push(offsets[i] + arm.wn[i]);
  } else {
    base = [];
    for (i = 0; i < arm.wi.length; i++) base.push([arm.wi[i] * N + arm.wj[i], arm.wv[i]]);
    for (i = 0; i < arm.di.length; i++) {
      for (j = 0; j < arm.di[i].length; j++) deltas.push([arm.di[i][j] * N + arm.dj[i][j], arm.dv[i][j]]);
      offsets.push(deltas.length);
    }
  }
  function buildTo(k){
    if (upto > k) { map = new Map(); upto = -1; }
    if (upto === -1) {
      for (i = 0; i < base.length; i++) map.set(base[i][0], base[i][1]);
      upto = 0;
    }
    while (upto < k) {
      for (j = offsets[upto]; j < offsets[upto + 1]; j++) {
        if (deltas[j][1] === 0) map.delete(deltas[j][0]); else map.set(deltas[j][0], deltas[j][1]);
      }
      upto++;
    }
    return map;
  }
  return {buildTo: buildTo};
}
function receivedAttention(map){
  var r = new Float64Array(N), rowsum = new Float64Array(N), i, j;
  map.forEach(function(v, k){ i = Math.floor(k / N); rowsum[i] += v; });
  map.forEach(function(v, k){
    i = Math.floor(k / N); j = k - i * N;
    if (rowsum[i] > 0) r[j] += v / rowsum[i];
  });
  return r;
}
function areasStudy(r){
  var out = new Float64Array(N);
  for (var i = 0; i < N; i++) {
    var c = Math.pow(Math.max(r[i], 0) / 3, 1.4); c = c / (1 + c);
    out[i] = N > 100 ? 4 + 86 * c : 12 + 360 * c;
  }
  return out;
}
function areasControl(r, rmax){
  var out = new Float64Array(N);
  for (var i = 0; i < N; i++) out[i] = 75 + 300 * r[i] / rmax;
  return out;
}
function circle(ctx, x, y, rad, fill, stroke, lw){
  ctx.beginPath(); ctx.arc(x, y, Math.max(rad, 0.4), 0, TWO_PI);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.lineWidth = lw; ctx.strokeStyle = stroke; ctx.stroke(); }
}
function star(ctx, x, y, rad){
  ctx.beginPath();
  for (var i = 0; i < 10; i++) {
    var ang = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? rad * 0.45 : rad;
    var sx = x + rr * Math.cos(ang), sy = y + rr * Math.sin(ang);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.closePath(); ctx.fillStyle = '#111111'; ctx.fill();
  ctx.lineWidth = 1; ctx.strokeStyle = '#ffffff'; ctx.stroke();
}
function scene(){ return P.comparisons[state.index]; }
function frameCount(){ return armRecord(scene().zero).t.length; }

/* One scale shared by every frame and both arms, as the encodings state. */
function computeScales(){
  var sc = scene(), rmax = 0, tie = 0;
  armStates = [];
  [sc.zero, sc.arm].forEach(function(a){ if (!armStates[a]) armStates[a] = armState(a); });
  if (S === 'study') return {rmax:1, tie:1};
  [sc.zero, sc.arm].forEach(function(a){
    var st = armStates[a], arm = armRecord(a);
    for (var k = 0; k < arm.t.length; k++) {
      var map = st.buildTo(k), att = receivedAttention(map);
      for (var i = 0; i < N; i++) if (att[i] > rmax) rmax = att[i];
      if (S === 'control') {
        map.forEach(function(v, key){
          var i2 = Math.floor(key / N), j2 = key - i2 * N;
          if (i2 === j2 || (i2 > j2 && map.has(j2 * N + i2))) return;
          var w = v + (map.get(j2 * N + i2) || 0);
          if (w > tie) tie = w;
        });
      }
    }
  });
  return {rmax: Math.max(rmax, 1e-12), tie: Math.max(tie, 1e-12)};
}

function drawPanel(canvas, ctx, armIdx, layoutIdx, k){
  var cssW = canvas.clientWidth || 320, cssH = canvas.clientHeight || 320;
  var dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  var pos = P.layouts[layoutIdx], arm = armRecord(armIdx), map = armStates[armIdx].buildTo(k);
  var att = receivedAttention(map);
  var areas = S === 'study' ? areasStudy(att) : areasControl(att, cachedScales.rmax);
  var inspecting = document.body.classList.contains('fullscreen-view') && canvas.parentElement.parentElement.id === 'stage';
  var unit = Math.min(cssW, cssH), zoom = inspecting ? camera.zoom : 1;
  var s = unit / (EXT * 2) * zoom;
  var cx = cssW / 2, cy = cssH / 2;
  if (inspecting) {
    var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (var q = 0; q < pos.length; q++) {
      xmin = Math.min(xmin,pos[q][0]); xmax = Math.max(xmax,pos[q][0]);
      ymin = Math.min(ymin,pos[q][1]); ymax = Math.max(ymax,pos[q][1]);
    }
    /* Fit fixed positions, with a fixed maximum-node allowance and room for rings/labels.
       The camera therefore stays stationary as attention and treatment change. */
    var maxArea = S === 'study' ? (N > 100 ? 90 : 372) : 375;
    var nodePad = Math.sqrt(maxArea / Math.PI) / 90;
    s = Math.max(1,Math.min((cssW-64)/Math.max(xmax-xmin+2*nodePad,1e-12),
                           (cssH-64)/Math.max(ymax-ymin+2*nodePad,1e-12))) * zoom;
    cx -= (xmin+xmax) / 2 * s; cy += (ymin+ymax) / 2 * s;
    cx += camera.x * unit; cy += camera.y * unit;
  }
  var PXPT = s / 90;
  function px(i){ return cx + pos[i][0] * s; }
  function py(i){ return cy - pos[i][1] * s; }

  if (S === 'study') {
    var palette = getComputedStyle(document.documentElement);
    ctx.globalAlpha = Number(palette.getPropertyValue('--edge-alpha'));
    ctx.strokeStyle = palette.getPropertyValue('--edge').trim(); ctx.lineWidth = 0.75;
    ctx.beginPath();
    map.forEach(function(v, key){
      var i = Math.floor(key / N), j = key - i * N;
      ctx.moveTo(px(i), py(i)); ctx.lineTo(px(j), py(j));
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  } else {
    var groups = new Map();
    map.forEach(function(v, key){
      var i = Math.floor(key / N), j = key - i * N;
      if (i === j || (i > j && map.has(j * N + i))) return;
      var w = (v + (map.get(j * N + i) || 0)) / cachedScales.tie;
      var g = groups.get(w.toFixed(4));
      if (!g) { g = {w: w, seg: []}; groups.set(w.toFixed(4), g); }
      g.seg.push([i, j]);
    });
    groups.forEach(function(g){
      var palette = getComputedStyle(document.documentElement);
      ctx.globalAlpha = Number(palette.getPropertyValue('--tie-alpha')) + 0.38 * g.w;
      ctx.strokeStyle = palette.getPropertyValue('--edge').trim();
      ctx.lineWidth = (0.35 + 1.6 * g.w) * 4 / 3;
      ctx.beginPath();
      for (var i = 0; i < g.seg.length; i++) {
        ctx.moveTo(px(g.seg[i][0]), py(g.seg[i][0]));
        ctx.lineTo(px(g.seg[i][1]), py(g.seg[i][1]));
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  for (var i = 0; i < N; i++) {
    var rr = Math.sqrt(areas[i] / Math.PI) * PXPT;
    circle(ctx, px(i), py(i), rr, lutColor(stateAt(arm.by[k], arm.fy[k], i)),
           S === 'study' ? '#ffffff' : '#202735', S === 'study' ? 0.53 : 1.07);
    circle(ctx, px(i), py(i), Math.sqrt(areas[i] * (S === 'study' ? 0.42 : 0.28) / Math.PI) * PXPT,
           lutColor(stateAt(arm.bx[k], arm.fx[k], i)), '#ffffff', S === 'study' ? 0.53 : 0.8);
    var isActive = arm.active[k].indexOf(i) !== -1;
    var wasTreated = arm.first_treated && arm.first_treated[String(i)] != null
      && arm.first_treated[String(i)] <= arm.t[k];
    if (isActive || wasTreated) {
      var ringRadius = rr * 1.24;
      var ringWidth = Math.max(0.45, Math.min(isActive ? 1.8 : 1.1, rr * (isActive ? 0.18 : 0.12)));
      ctx.save();
      if (!isActive) {ctx.setLineDash([ringWidth * 2.5, ringWidth * 2.2]);ctx.globalAlpha=0.8;}
      circle(ctx, px(i), py(i), ringRadius, null, '#725400', ringWidth + 0.45);
      circle(ctx, px(i), py(i), ringRadius, null, '#efbb23', ringWidth);
      ctx.setLineDash([]);
      ctx.restore();
    }
  }
  for (var q = 0; q < arm.targeted.length; q++) {
    var node = arm.targeted[q];
    if (node < N && arm.active[k].indexOf(node) !== -1)
      star(ctx, px(node), py(node), Math.sqrt(90 / Math.PI) * PXPT);
  }
}

function stats(armIdx, k){
  var arm = armRecord(armIdx), priv = 0, pub = 0;
  for (var i = 0; i < N; i++) {
    var x = stateAt(arm.bx[k], arm.fx[k], i), y = stateAt(arm.by[k], arm.fy[k], i);
    priv += S === 'control' ? (1-x)/2 : Number(x < 0);
    pub += S === 'control' ? (1-y)/2 : Number(y < 0);
  }
  return {priv: priv / N, pub: pub / N, active: arm.active[k].length};
}
function pct(v){ return Math.round(v * 100) + '%'; }
function plural(count, word){ return count + ' ' + word + (count === 1 ? '' : 's'); }

function render(){
  if (!ready || !order.length) return;
  var sc = scene(), k = state.frame, key = state.index + '|' + S;
  if (scaleKey !== key) { cachedScales = computeScales(); scaleKey = key; }
  for (var p = 0; p < 2; p++) {
    var armIdx = p === 0 ? sc.zero : sc.arm;
    canvases[p].parentElement.hidden = p === 1 && sc.zero === sc.arm;
    drawPanel(canvases[p], contexts[p], armIdx, sc.layouts[p], k);
    var st = stats(armIdx, k);
    document.getElementById('sub-' + p).textContent =
      'Private ' + pct(st.priv) + ' · Public ' + pct(st.pub) + ' · Nodes receiving intervention: ' + st.active;
    document.getElementById('lab-' + p).textContent = p === 1 || sc.zero === sc.arm ? (sc.policy_label || sc.role) : 'No intervention · matched control';
  }
  if (staticKey !== state.index && !document.body.classList.contains('fullscreen-view')) { drawStatic(); staticKey = state.index; }
  document.getElementById('caption').textContent =
    'Sampled saved frames; nothing is re-simulated, and sampled frames do not certify an uninterrupted hold.'
    + (P.future_note ? ' ' + P.future_note + '.' : '')
    + (sc.zero === sc.arm ? ' No matched uncontrolled arm is available for this package; this is not a controlled comparison.' : '');
  var total = frameCount(), t = armRecord(sc.zero).t[k];
  var scrub = document.getElementById('scrub');
  scrub.max = String(total - 1); scrub.value = String(k);
  document.getElementById('h-time').textContent =
    't = ' + Number(t.toFixed(2)) + ((sc.horizon || P.horizon) ? ' / ' + (sc.horizon || P.horizon) : '') + '   frame ' + (k + 1) + ' / ' + total;
  document.getElementById('f-prev').disabled = document.getElementById('f-first').disabled = k === 0;
  document.getElementById('f-next').disabled = document.getElementById('f-last').disabled = k === total - 1;
  var bitsTxt = [sc.network_label || sc.origin, sc.policy_label || sc.role, 'Future ' + (sc.future+1)];
  if(sc.study)bitsTxt.unshift(sc.study_label,'N'+N,sc.memory==='frozen'?'Fixed memory':'Adaptive memory');
  if (sc.count) bitsTxt.push(sc.count + ' recipients');
  if (sc.outcome && sc.outcome.valid) bitsTxt.push(outcomeLabel(sc));
  var treatment = armRecord(sc.arm), activeNow = treatment.active[k];
  var treatedByNow = Object.keys(treatment.first_treated || {}).filter(function(i){return treatment.first_treated[i] <= treatment.t[k];}).map(Number).sort(function(a,b){return a-b;});
  document.getElementById('members').textContent = 'Field active: ' + activeNow.length + ' nodes · Previously or currently treated: ' + treatedByNow.length
    + ' · Solid gold: field on; dashed gold: field off.';
  document.getElementById('h-count').textContent =
    'Comparison ' + (order.indexOf(state.index) + 1) + ' / ' + order.length + ' — ' + bitsTxt.join(' · ');
  if (sc.study) {
    document.getElementById('study-context').textContent=sc.study_label+' · N='+N+' · '+(sc.memory==='frozen'?'Fixed memory':'Adaptive memory')+' · '+(sc.visibility==='complete'?'Complete public reports':'Partial public reports')
      +' · Success: '+Math.round(sc.threshold*100)+'% in both layers for '+sc.holding_duration+' uninterrupted units after release ≤'+sc.release_deadline+', sustained through '+sc.horizon+'.';
    document.getElementById('source-identifiers').textContent='Network: '+sc.origin+' · policy: '+sc.role+' · saved future index: '+sc.future+'.';
  }
  var buttons = document.querySelectorAll('#list-body button');
  for (var b = 0; b < buttons.length; b++)
    buttons[b].setAttribute('aria-current', String(Number(buttons[b].dataset.i) === state.index));
}

async function go(index){
  if (index == null || !order.length) return;
  var sequence=++loadingSequence;ready=false;
  document.getElementById('h-count').textContent='Loading saved comparison…';
  try {await Promise.all([ensureArm(P.comparisons[index].zero),ensureArm(P.comparisons[index].arm)]);}
  catch(error){document.getElementById('h-count').textContent=error.message;return;}
  if(sequence!==loadingSequence)return;
  state.index = Math.max(0, Math.min(P.comparisons.length - 1, index));
  var previousN=N,sc=scene();N=sc.N || P.N;S=sc.style || P.style;EXT=S==='study'?1.3:1.28;
  if(previousN!==N)setZoom(1);
  state.frame = 0;scaleKey=null;staticKey=null;bitCache=Object.create(null);
  for (var key of decodedArms.keys()) if(key!==sc.zero && key!==sc.arm) decodedArms.delete(key);
  ready=true;buildKey();
  render();
}
function goPosition(position){
  go(order[Math.max(0, Math.min(order.length - 1, position))]);
}
function stepBy(delta){ if (delta) { stop(); goPosition(order.indexOf(state.index) + delta); } }
function moveFrame(k){
  if (!ready || !order.length) return;
  var last = frameCount() - 1;
  state.frame = Math.max(0, Math.min(last, k));
  render();
  if (state.frame >= last && !document.getElementById('loop').checked) stop();
}
function speed(){ return Number(document.getElementById('speed').value) || 1; }
function tick(){
  if (!ready || !order.length) return;
  if (state.frame >= frameCount() - 1) {
    if (document.getElementById('loop').checked) moveFrame(0); else stop();
    return;
  }
  moveFrame(state.frame + 1);
}
function play(){
  if (!ready || !order.length) return;
  if (state.playing) return;
  if (state.frame >= frameCount() - 1) state.frame = 0;
  state.playing = true;
  document.getElementById('f-play').textContent = '❚❚ Pause';
  state.timer = window.setInterval(tick, BASE_MS / speed());
}
function stop(){
  if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
  state.playing = false;
  document.getElementById('f-play').textContent = '▶ Play';
}

function buildStage(){
  var stage = document.getElementById('stage');
  for (var p = 0; p < 2; p++) {
    var panel = document.createElement('div'); panel.className = 'panel';
    var h = document.createElement('h3'); h.id = 'lab-' + p;
    var c = document.createElement('canvas'); c.setAttribute('role', 'img');
    var sub = document.createElement('p'); sub.id = 'sub-' + p;
    panel.appendChild(h); panel.appendChild(c); panel.appendChild(sub);
    stage.appendChild(panel);
    canvases.push(c); contexts.push(c.getContext('2d'));
  }
}
function drawStatic(){
  var sc = scene(), box = document.getElementById('snapshots'), curves = document.getElementById('curves');
  box.replaceChildren(); curves.replaceChildren();
  if (!document.getElementById('static-views').open) return;
  var pending = [];
  var arms = sc.zero === sc.arm ? [sc.zero] : [sc.zero, sc.arm];
  arms.forEach(function(a, panel){
    var arm = armRecord(a), label = panel === 1 || sc.zero === sc.arm ? (sc.policy_label || sc.role) : 'No intervention', times = arm.t, release = sc.release_deadline == null ? P.release_deadline : sc.release_deadline;
    var near = 0;
    times.forEach(function(t,i){ if(Math.abs(t-release)<Math.abs(times[near]-release)) near=i; });
    var samples = [[0, 'Initial']];
    if (release != null) samples.push([near, 'Release deadline (nearest displayed)']);
    samples.push([times.length-1, 'Final']);
    samples.forEach(function(sample){
      var k = sample[0];
      var wrap=document.createElement('div'); wrap.className='panel';
      var h=document.createElement('h3'); h.textContent=label+' · '+sample[1]+' · t='+times[k];
      var canvas=document.createElement('canvas'); wrap.appendChild(h); wrap.appendChild(canvas); box.appendChild(wrap);
      pending.push(function(){ drawPanel(canvas,canvas.getContext('2d'),a,sc.layouts[panel],k); });
    });
    var wrap=document.createElement('div'); wrap.className='panel';
    var h=document.createElement('h3'); h.textContent=label+' · blue support (solid private; dashed public)';
    var canvas=document.createElement('canvas'); wrap.appendChild(h); wrap.appendChild(canvas); curves.appendChild(wrap);
    pending.push(function(){
    var w=canvas.clientWidth||320, height=w*.65, dpr=window.devicePixelRatio||1;
    canvas.style.aspectRatio='auto'; canvas.style.height=height+'px'; canvas.width=w*dpr; canvas.height=height*dpr;
    var ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
    var left=38,top=15,bottom=height-28,right=w-12, span=times[times.length-1]-times[0]||1;
    function xx(t){return left+(right-left)*(t-times[0])/span;} function yy(v){return bottom-(bottom-top)*v;}
    var palette=getComputedStyle(document.documentElement);
    ctx.font='11px sans-serif';ctx.fillStyle=palette.getPropertyValue('--muted').trim();ctx.strokeStyle=palette.getPropertyValue('--line').trim();ctx.lineWidth=1;
    [0,.5,.75,1].forEach(function(v){ctx.fillText(Math.round(v*100)+'%',0,yy(v)+3);ctx.beginPath();ctx.moveTo(left,yy(v));ctx.lineTo(right,yy(v));ctx.stroke();});
    ctx.fillText(String(times[0]),left,bottom+18);ctx.fillText(String(times[times.length-1]),right-25,bottom+18);
    ['priv','pub'].forEach(function(layer,l){ctx.beginPath();ctx.strokeStyle=palette.getPropertyValue(l?'--plot-public':'--plot-private').trim();ctx.lineWidth=2;ctx.setLineDash(l?[5,4]:[]);
      times.forEach(function(t,k){var v=stats(a,k)[layer];if(k===0)ctx.moveTo(xx(t),yy(v));else ctx.lineTo(xx(t),yy(v));});ctx.stroke();});ctx.setLineDash([]);
    });
  });
  pending.forEach(function(draw){ draw(); });
}
function sortList(){
  var mode = document.getElementById('sort').value;
  var collator = new Intl.Collator('en', {numeric:true, sensitivity:'base'});
  function name(a,b){ return collator.compare(a.network_label || a.origin,b.network_label || b.origin) || collator.compare(a.policy_label || a.role,b.policy_label || b.role) || a.future-b.future; }
  function outcome(c){ return c.outcome && c.outcome.valid ? (c.outcome.persistent ? 0 : 1) : 2; }
  order.sort(function(i,j){
    var a=P.comparisons[i], b=P.comparisons[j], result=0;
    if(mode==='original') return i-j;
    if(mode==='name-asc') result=name(a,b);
    else if(mode==='name-desc') result=-name(a,b);
    else if(mode==='role') result=collator.compare(a.policy_label || a.role,b.policy_label || b.role);
    else if(mode==='future') result=a.future-b.future;
    else if(mode==='count-asc') result=a.count-b.count;
    else if(mode==='count-desc') result=b.count-a.count;
    else if(mode==='outcome') result=outcome(a)-outcome(b);
    return result || name(a,b) || i-j;
  });
  buildList(); render();
  var selected = document.querySelector('#list-body button[aria-current="true"]');
  if (selected && !document.getElementById('list').hidden)
    selected.scrollIntoView({block:'nearest'});
}
function outcomeLabel(c){
  var o=c.outcome || {};
  if(!o.valid)return 'Invalid / ineligible';
  return o.persistent ? 'Persistent' : (o.relapsed ? 'Threshold breach' : 'Hold not completed');
}
function buildArchiveControls(){
  if(!P.studies)return;
  ['filters','label-guide','study-results','study-context','provenance'].forEach(function(id){document.getElementById(id).hidden=false;});
  var study=document.getElementById('filter-study'),size=document.getElementById('filter-size');
  function option(select,value,label){var o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o);}
  option(study,'','All studies');P.studies.forEach(function(s){option(study,s.id,s.label);});
  option(size,'','All sizes');Array.from(new Set(P.comparisons.map(function(c){return c.N;}))).sort(function(a,b){return a-b;}).forEach(function(n){option(size,String(n),'N = '+n);});
  study.value=P.studies[0].id;
  var guide=document.getElementById('label-key');
  (P.label_key || []).forEach(function(row){var p=document.createElement('p'),b=document.createElement('b');b.textContent=row[0]+': ';p.appendChild(b);p.appendChild(document.createTextNode(row[1]));guide.appendChild(p);});
  var notes=document.createElement('p');notes.textContent='All available saved frames are retained. Timing/replay and search-training runs are excluded from audit counts. Incomplete runs and duplicate study mirrors are not additional evidence. N500 has no completed results.';guide.appendChild(notes);
  var catalog=document.createElement('details'),catalogTitle=document.createElement('summary');catalogTitle.textContent='Included studies and why they are useful';catalog.appendChild(catalogTitle);
  P.studies.forEach(function(s){var p=document.createElement('p');p.textContent=s.label+': '+s.retained_reason;catalog.appendChild(p);});guide.appendChild(catalog);
  if(P.archive_notes && P.archive_notes.excluded.length){var excluded=document.createElement('p');excluded.textContent=P.archive_notes.excluded.length+' incomplete or software-only bundles are excluded. Valid negative results remain included; inclusion never depends on success.';guide.appendChild(excluded);}
  if(P.archive_notes && P.archive_notes.issues.length){var warning=document.createElement('p');warning.textContent=P.archive_notes.issues.length+' source issue(s) recorded in the companion inventory. Unpaired development trajectories are explicitly identified.';guide.appendChild(warning);}
  function criteria(){return {study:study.value,N:size.value,memory:document.getElementById('filter-memory').value,outcome:document.getElementById('filter-outcome').value,text:document.getElementById('search-results').value.toLowerCase().trim()};}
  function accepts(c,f,withOutcome){
    var label=c.study_label || (P.studies.find(function(s){return s.id===c.study;}) || {}).label || '';
    if(f.study && c.study!==f.study || f.N && String(c.N)!==f.N || f.memory && c.memory!==f.memory)return false;
    if(f.text && (label+' '+(c.network_label || c.origin)+' '+(c.policy_label || c.role)+' '+c.role).toLowerCase().indexOf(f.text)===-1)return false;
    if(withOutcome && f.outcome){var o=c.outcome || {};
      if(f.outcome==='persistent' && !o.persistent)return false;
      if(f.outcome==='breach' && !o.relapsed)return false;
      if(f.outcome==='no-hold' && (!o.valid || o.persistent || o.relapsed))return false;
      if(f.outcome==='invalid' && o.valid)return false;
    }return true;
  }
  function updateTable(f){
    var groups=new Map();
    (P.records || []).filter(function(r){return accepts(r,f,false);}).forEach(function(r){
      var key=[r.study,r.N,r.role,r.memory,r.visibility,r.package].join('|');
      if(!groups.has(key))groups.set(key,{r:r,networks:new Set(),valid:0,total:0,success:0,animated:0});
      var g=groups.get(key);g.networks.add(r.origin);g.total++;g.valid+=Number(r.valid);g.success+=Number(r.valid && r.persistent);g.animated+=Number(r.frames>0);
    });
    var body=document.getElementById('results-table');body.replaceChildren();
    groups.forEach(function(g){var tr=document.createElement('tr'),r=g.r,studyLabel=(P.studies.find(function(s){return s.id===r.study;}) || {}).label;
      [studyLabel,r.N,r.policy_label,(r.memory==='frozen'?'Fixed':'Adaptive')+' / '+(r.visibility==='complete'?'complete':'partial'),g.networks.size,g.valid+' / '+g.total,g.success+' / '+g.valid,g.animated+' / '+g.total].forEach(function(value){var td=document.createElement('td');td.textContent=value;tr.appendChild(td);});body.appendChild(tr);
    });
  }
  function applyFilters(){
    stop();var f=criteria();order=P.comparisons.map(function(c,i){return i;}).filter(function(i){return accepts(P.comparisons[i],f,true);});
    document.getElementById('network-viewport').hidden=!order.length;document.querySelector('.frames').hidden=!order.length;
    document.getElementById('static-views').hidden=!order.length;
    updateTable(f);buildList();
    if(order.length){sortList();go(order.indexOf(state.index)>=0 ? state.index : order[0]);}
    else{ready=false;++loadingSequence;document.getElementById('h-count').textContent='No saved comparisons match these filters.';document.getElementById('members').textContent='Try another study, outcome, or search.';}
  }
  ['filter-study','filter-size','filter-memory','filter-outcome'].forEach(function(id){document.getElementById(id).addEventListener('change',applyFilters);});
  var searchTimer;document.getElementById('search-results').addEventListener('input',function(){clearTimeout(searchTimer);searchTimer=setTimeout(applyFilters,120);});
  document.getElementById('reset-filters').addEventListener('click',function(){study.value=P.studies[0].id;size.value='';document.getElementById('filter-memory').value='';document.getElementById('filter-outcome').value='';document.getElementById('search-results').value='';applyFilters();});
  document.getElementById('download-outcomes').addEventListener('click',function(){
    var rows=(P.records || []).filter(function(r){return accepts(r,criteria(),false);});if(!rows.length)return;
    var keys=Object.keys(rows[0]);function cell(v){return '"'+String(v==null?'':v).replace(/"/g,'""')+'"';}
    var text=[keys.map(cell).join(',')].concat(rows.map(function(r){return keys.map(function(k){return cell(r[k]);}).join(',');})).join('\n');
    var url=URL.createObjectURL(new Blob([text],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='saved_outcomes.csv';a.click();setTimeout(function(){URL.revokeObjectURL(url);},1000);
  });
  applyFilters();
}
function buildList(){
  var body = document.getElementById('list-body');
  var grouped = document.getElementById('sort').value === 'original', previousOrigin = null;
  body.replaceChildren();
  order.forEach(function(i){
    var c = P.comparisons[i], b = document.createElement('button');
    var groupKey=[c.study,c.N,c.origin,c.memory,c.visibility].join('|');
    if (grouped && groupKey !== previousOrigin) {
      var head = document.createElement('h2'); head.textContent = (c.study_label ? c.study_label+' · N'+c.N+' · ' : '')+(c.network_label || c.origin)+(c.memory ? ' · '+(c.memory==='frozen'?'Fixed':'Adaptive') : ''); body.appendChild(head);
      previousOrigin = groupKey;
    }
    b.type = 'button'; b.dataset.i = String(i);
    var valid = c.outcome && c.outcome.valid;
    b.textContent = (grouped ? '' : (c.network_label || c.origin) + ' · ') + (c.policy_label || c.role)
      + ' · ' + c.count + ' recipients'+(c.N ? ' ('+Math.round(c.count/c.N*100)+'%)' : '')+' · Future '+(c.future+1)
      + ' · '+outcomeLabel(c);
    b.title = c.label;
    b.addEventListener('click', function(){ stop(); go(i); });
    body.appendChild(b);
  });
}
function buildKey(){
  var rows = S === 'study' ? [
    '<b>Center dot</b> private state; <b>outer disk</b> public state. Blue = -1, red = +1.',
    '<b>Size</b> current received attention, non-linear and bounded on one scale shared by every frame and both arms.',
    '<b>Gold ring</b> solid: field on; dashed: treated earlier, field off. Rings scale with nodes. Lines show directed memory support without arrowheads.',
  ] : [
    '<b>Outer disk</b> public stance; <b>center dot</b> private stance. Blue = -1, red = +1.',
    '<b>Node area</b> 75 + 300 r/r_max (monotonic, not proportional); r_max is shared across this pair and its sampled frames.',
    '<b>Support</b> mean (1 - stance)/2; for binary states this is the blue fraction.',
    '<b>Lines</b> W_ij + W_ji: remembered interaction volume, direction not shown.',
    '<b>Solid gold ring</b> field on; <b>dashed gold ring</b> previously treated, field off. <b>Star</b> additional targeted field while active.',
  ];
  rows.push('<b>Positions</b> fixed from initial memory; they are not geography.');
  document.getElementById('key').innerHTML = rows.map(function(r){ return '<div>' + r + '</div>'; }).join('');
  document.getElementById('caption').textContent =
    'Sampled saved frames; nothing is re-simulated, and sampled frames do not certify an uninterrupted hold.'
    + (P.future_note ? ' ' + P.future_note + '.' : '')
    + (P.matched_zero_present === false
       ? ' No matched uncontrolled arm is available in this bundle; this is not a controlled comparison.' : '');
  var notes = [];
  if (P.skipped && P.skipped.length)
    notes.push(P.skipped.length + ' saved comparison(s) omitted: ' + P.skipped.slice(0, 3).join('; ')
               + (P.skipped.length > 3 ? ' …' : ''));
  if (P.dropped && P.dropped.length)
    notes.push('Byte budget omitted ' + P.dropped.length + ' comparison(s) from the end of the declared order: '
               + P.dropped.slice(0, 3).join('; ') + (P.dropped.length > 3 ? ' …' : ''));
  if (notes.length) {
    var warn = document.getElementById('warn');
    warn.hidden = false; warn.textContent = notes.join(' ');
  }
}
function setFullscreenView(enabled){
  document.body.classList.toggle('fullscreen-view', enabled);
  var button = document.getElementById('fullscreen-view');
  button.setAttribute('aria-pressed', String(enabled));
  button.textContent = enabled ? 'Exit full screen' : 'Full screen';
  document.getElementById('full-view-control').hidden = !enabled;
  document.getElementById('zoom-controls').hidden = !enabled;
  document.getElementById('stage').dataset.view = document.getElementById('full-view').value;
  staticKey = null; render();
}
function setZoom(value){
  camera.zoom = Math.max(1, Math.min(6, value));
  if (camera.zoom === 1) { camera.x = 0; camera.y = 0; }
  document.getElementById('zoom-level').textContent = Math.round(camera.zoom * 100) + '%';
  document.getElementById('network-viewport').classList.toggle('zoomed', camera.zoom > 1);
  render();
}
function wire(){
  document.getElementById('fullscreen-view').addEventListener('click', async function(){
    if (document.body.classList.contains('fullscreen-view')) {
      if (document.fullscreenElement && document.exitFullscreen) await document.exitFullscreen();
      setFullscreenView(false);
    } else {
      setZoom(1); setFullscreenView(true);
      if (document.documentElement.requestFullscreen) {
        try { await document.documentElement.requestFullscreen(); } catch (_) { /* Fit this viewport if embedding disallows browser fullscreen. */ }
      }
      render();
    }
  });
  document.addEventListener('fullscreenchange', function(){
    setFullscreenView(Boolean(document.fullscreenElement));
  });
  document.getElementById('full-view').addEventListener('change', function(){
    document.getElementById('stage').dataset.view = this.value;
    setZoom(1);
  });
  document.getElementById('zoom-in').addEventListener('click', function(){ setZoom(camera.zoom * 1.25); });
  document.getElementById('zoom-out').addEventListener('click', function(){ setZoom(camera.zoom / 1.25); });
  document.getElementById('zoom-fit').addEventListener('click', function(){ setZoom(1); });
  var viewport = document.getElementById('network-viewport');
  viewport.addEventListener('wheel', function(event){
    if (!document.body.classList.contains('fullscreen-view')) return;
    event.preventDefault(); setZoom(camera.zoom * Math.exp(-event.deltaY * 0.0015));
  }, {passive:false});
  viewport.addEventListener('pointerdown', function(event){
    if (!document.body.classList.contains('fullscreen-view') || camera.zoom <= 1 || event.button !== 0) return;
    var canvas = event.target.closest('canvas'); if (!canvas) return;
    drag = {x:event.clientX,y:event.clientY,panX:camera.x,panY:camera.y,
      unit:Math.min(canvas.clientWidth,canvas.clientHeight)};
    viewport.setPointerCapture(event.pointerId); viewport.classList.add('dragging');
  });
  viewport.addEventListener('pointermove', function(event){
    if (!drag) return;
    camera.x = drag.panX + (event.clientX - drag.x) / drag.unit;
    camera.y = drag.panY + (event.clientY - drag.y) / drag.unit;
    render();
  });
  function endDrag(){ drag = null; viewport.classList.remove('dragging'); }
  viewport.addEventListener('pointerup', endDrag);
  viewport.addEventListener('pointercancel', endDrag);
  document.addEventListener('keydown', function(event){
    if (event.key === 'Escape' && !document.fullscreenElement && document.body.classList.contains('fullscreen-view')) setFullscreenView(false);
  });
  document.getElementById('sort').addEventListener('change', sortList);
  document.getElementById('theme').addEventListener('change', function(){
    document.documentElement.dataset.theme = this.value;
    staticKey = null; render();
  });
  document.getElementById('static-views').addEventListener('toggle', function(){
    if (this.open) { staticKey = null; render(); }
  });
  document.getElementById('s-first').addEventListener('click', function(){ stop(); goPosition(0); });
  document.getElementById('s-prev').addEventListener('click', function(){ stepBy(-1); });
  document.getElementById('s-next').addEventListener('click', function(){ stepBy(1); });
  document.getElementById('s-last').addEventListener('click', function(){ stop(); goPosition(order.length - 1); });
  document.getElementById('toggle').addEventListener('click', function(){
    var aside = document.getElementById('list'), open = aside.hidden;
    aside.hidden = !open; this.setAttribute('aria-expanded', String(open));
    document.querySelector('.layout').classList.toggle('list-hidden', !open);
    staticKey = null; render();
  });
  document.getElementById('f-first').addEventListener('click', function(){ moveFrame(0); });
  document.getElementById('f-prev').addEventListener('click', function(){ moveFrame(state.frame - 1); });
  document.getElementById('f-next').addEventListener('click', function(){ moveFrame(state.frame + 1); });
  document.getElementById('f-last').addEventListener('click', function(){ moveFrame(frameCount() - 1); });
  document.getElementById('f-play').addEventListener('click', function(){ state.playing ? stop() : play(); });
  document.getElementById('scrub').addEventListener('input', function(){ moveFrame(Number(this.value)); });
  document.getElementById('speed').addEventListener('change', function(){
    if (state.playing) { window.clearInterval(state.timer); state.timer = window.setInterval(tick, BASE_MS / speed()); }
  });
  window.addEventListener('keydown', function(ev){
    if (ev.target && /^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) return;
    if (ev.key === 'ArrowRight') { ev.preventDefault(); ev.shiftKey ? stepBy(1) : moveFrame(state.frame + 1); }
    else if (ev.key === 'ArrowLeft') { ev.preventDefault(); ev.shiftKey ? stepBy(-1) : moveFrame(state.frame - 1); }
    else if (ev.key === ' ') { ev.preventDefault(); state.playing ? stop() : play(); }
    else if (ev.key === 'Home') { ev.preventDefault(); ev.shiftKey ? goPosition(0) : moveFrame(0); }
    else if (ev.key === 'End') { ev.preventDefault(); ev.shiftKey ? goPosition(order.length - 1) : moveFrame(frameCount() - 1); }
  });
  window.addEventListener('resize', function(){ staticKey = null; render(); });
}

buildStage(); buildList(); buildKey(); wire(); buildArchiveControls();
document.getElementById('h-prov').textContent =
  (P.studies ? plural(P.studies.length,'study')+' · ' : 'N = ' + N + ' · ') + plural(P.comparisons.length, 'saved comparison') + ' · ' + plural(P.arms.length, 'stored trajectory')
  + ' (shared arms are stored once)'
  + (P.source_manifest_sha256 ? ' · source manifest ' + P.source_manifest_sha256.slice(0, 12) : '');
if(!P.studies)go(order[0]);
})();
</script>
</body></html>
"""
