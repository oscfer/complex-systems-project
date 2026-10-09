"""Interactive animation of saved focused-study frames; never simulates."""
from pathlib import Path
import hashlib
import json

from src.visualization.transformation_report import _load_with_baselines
from src.visualization.animation_player import (payload_from_units,
                                                write_player_html)


def render_transformation_animation(bundle_dir, *, origin_id=None, max_frames=40,
                                    output_dir=None, baseline_bundle=None, source_review=None):
    """Show each saved active arm with its package-matched zero for an origin.

    Default origin is the first declared origin with saved results, never selected
    by success. Frames are synchronized exactly; no interpolation invents states.
    """
    if max_frames < 4:
        raise ValueError('At least four frames are required')
    bundle = Path(bundle_dir)
    manifest, units = _load_with_baselines(bundle, baseline_bundle, source_review=source_review)
    if not units:
        raise ValueError('No saved units to animate')
    available = {u['origin_id'] for u in units
                 if u['policy_id'] != 'zero' and u.get('network_frames')}
    if not available:
        raise ValueError('No saved active arm with retained frames')
    explicit_origin = origin_id is not None
    if origin_id is None:
        origin_id = next(q['origin']['origin_id'] for q in manifest['config']['queue']
                         if q['origin']['origin_id'] in available)
    arms = [u for u in units if u['origin_id'] == origin_id and u['policy_id'] != 'zero' and u.get('network_frames')]
    zeros = [u for u in units if u['origin_id'] == origin_id and u['policy_id'] == 'zero' and u.get('network_frames')]
    if not arms:
        raise ValueError('No saved active arm for this origin')
    arms = zeros + sorted(arms, key=lambda u: (u['unit']['package'], u['policy_id']))
    config = manifest['config']
    has_targeted_field = any(u["summaries"].get("effective_settings", {}).get("targeted_nodes") for u in arms)
    notes = ['Fixed positions from initial memory. Sampled frames do not certify the uninterrupted hold.']
    if has_targeted_field:
        notes.append('Stars indicate extra field strength, not stance, success or a permanent node type.')
    if not zeros:
        notes.append('No matched uncontrolled arm is available in this bundle; '
                     'this is not a controlled/uncontrolled comparison.')
    meta = dict(horizon=float(config['horizon']), release_deadline=float(config['release_deadline']),
                memory_mode=str(config.get('memory_mode', 'adaptive')),
                targeted_field=has_targeted_field, notes=notes,
                future_note=(f'Times run to {config["horizon"]:g}; release deadline '
                             f'{config["release_deadline"]:g}; target is 75% or more in both layers.'))
    payload, report = payload_from_units(units, style='control', origin_id=origin_id,
                                         max_frames=max_frames, meta=meta, max_bytes=None)
    out = Path(output_dir) if output_dir else bundle/'animation'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'network_progress.html'
    title = f'{origin_id} | {config.get("memory_mode", "adaptive")} memory | saved trajectories'
    write_player_html(payload, path, title=title)
    metadata = {'origin':origin_id,
                'origin_selection':'explicit' if explicit_origin else 'first declared with saved data',
                'arms':[{'policy':u['policy_id'],'package':u['unit']['package']} for u in arms],
                'frame_times_by_comparison': [
                    {'role': c['role'], 'matched_zero_present': c['zero'] != c['arm'],
                     'frame_times': list(payload['arms'][c['arm']]['t'])}
                    for c in payload['comparisons']],
                'source_manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
                'player_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'matched_zero_present':payload['matched_zero_present'],
                'player':'self-contained canvas page from saved frames; no resource is fetched',
                'byte_report':report,
                'imported_baselines':[u['imported_baseline'] for u in arms if 'imported_baseline' in u]}
    (out/'animation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path
