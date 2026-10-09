"""Readable reports from saved focused-study evidence; never run simulations."""
from __future__ import annotations

import base64
import hashlib
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment.baselines.reference import exact_binomial_lower_bound
from src.analytics.influence import calculate_influence_metrics
from src.control.evaluator import compute_top_10_percent_set
from src.visualization.network_layout import _seeded_spring_layout
from src.visualization.control_plots import _draw_control_network, _network_scales, _add_network_key


def target_residence(transitions, start: float, end: float) -> dict:
    """Integrate exact predicate intervals, retaining same-time violations.

    Duration ignores zero-length excursions, but longest uninterrupted hold
    resets on every false transition, including one at a tied timestamp.
    """
    if end < start:
        raise ValueError('Residence window ends before it starts')
    condition=False; previous=start; total=0.; longest=0.; run=0.
    for t,value in transitions:
        t=float(t)
        if t < start:
            condition=bool(value)
            continue
        if t>end:
            break
        dt=max(0.,t-previous)
        if condition:
            total+=dt;run+=dt;longest=max(longest,run)
        if not value:
            run=0.
        condition=bool(value);previous=t
    if condition:
        total+=end-previous;run+=end-previous;longest=max(longest,run)
    return {'duration':total,'longest_uninterrupted':longest,'window':[start,end]}


def _load(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text())
    for info in list(manifest['units'].values())+list(manifest['origins'].values()):
        p=bundle/info['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError(f'Checksum mismatch: {p}')
    raw={key:json.loads((bundle/u['path']).read_text()) for key,u in manifest['units'].items()}
    units=[]
    for key,value in raw.items():
        if 'reference_unit' in value:
            reference=raw[value['reference_unit']]
            if 'reference_unit' in reference or reference['unit']['policy']!='zero':
                raise ValueError('Invalid zero-reference chain')
            if reference['unit']['origin'] != value['unit']['origin']:
                raise ValueError('Zero reference crosses origins')
            units.append({**reference, **value})
        else:
            units.append(value)
    return manifest,units


def _load_with_baselines(bundle, baseline_bundle=None, *, source_review=None):
    """Attach matching saved zeros without changing either input bundle.

    Different source versions require an explicit review bound to both hashes.
    A review cannot bypass scientific configuration, origin or seed checks.
    """
    manifest, units = _load(bundle)
    if baseline_bundle is None:
        return manifest, units
    baseline_bundle = Path(baseline_bundle)
    previous, saved = _load(baseline_bundle)
    for key in ('N', 'profile', 'runtime', 'initial_conditions', 'memory_mode',
                'stage', 'smoke', 'target_kind', 'horizon', 'release_deadline', 'holding_duration'):
        if manifest['config'][key] != previous['config'][key]:
            raise ValueError(f'Baseline configuration mismatch: {key}')
    # Optional policy registration can change without changing the zero contract.
    contracts = [{k: v for k, v in m['contract'].items() if k != 'optional_policies'}
                 for m in (manifest, previous)]
    if contracts[0] != contracts[1]:
        raise ValueError('Baseline scientific contract mismatch')
    current_hash = manifest['config']['source_sha256']
    baseline_hash = previous['config']['source_sha256']
    if current_hash != baseline_hash:
        if (not source_review or source_review.get('active_source_sha256') != current_hash
                or source_review.get('baseline_source_sha256') != baseline_hash
                or not source_review.get('rationale', '').strip()):
            raise ValueError('Different source versions require a hash-bound compatibility review')
    imported = []
    present = {(u['origin_id'], u['unit']['package']) for u in units if u['policy_id'] == 'zero'}
    for active in units:
        origin, package = active['origin_id'], active['unit']['package']
        if active['policy_id'] == 'zero' or (origin, package) in present:
            continue
        matches = [u for u in saved if u['origin_id'] == origin
                   and u['unit']['package'] == package and u['policy_id'] == 'zero']
        if len(matches) != 1:
            raise ValueError(f'Expected one matched zero for {origin}/{package}')
        zero = matches[0]
        if (not zero['is_valid'] or zero.get('error_message') or zero['expenditure'] != 0
                or zero['distinct_nodes_used'] != 0 or zero['structural_modifications'] != 0):
            raise ValueError('Imported zero must be valid and unactuated')
        if manifest['origins'][origin]['sha256'] != previous['origins'][origin]['sha256']:
            raise ValueError('Baseline initial-origin checksum mismatch')
        if active['seeds'] != zero['seeds']:
            raise ValueError('Baseline role-seed mismatch')
        # Explicit cross-package zero aliases retain the canonical roster.
        # No actuation means that roster cannot alter the physical trajectory.
        if not zero.get('reference_unit') and active['accessible_roster'] != zero['accessible_roster']:
            raise ValueError('Baseline accessible-roster mismatch')
        if zero['network_frames'][-1]['t'] != manifest['config']['horizon']:
            raise ValueError('Imported baseline does not reach the declared horizon')
        for key in ('x', 'y', 'W'):
            if not np.array_equal(active['network_frames'][0][key], zero['network_frames'][0][key]):
                raise ValueError('Baseline initial frame mismatch')
        provenance = {'bundle': str(baseline_bundle.resolve()),
                      'manifest_sha256': hashlib.sha256((baseline_bundle/'manifest.json').read_bytes()).hexdigest(),
                      'unit_key': zero['unit']['key'], 'source_review': source_review}
        imported.append({**zero, 'imported_baseline': provenance})
        present.add((origin, package))
    return manifest, units + imported


def _image(path):
    return '<img style="width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def render_transformation_report(bundle_dir: str | Path, *, output_dir: str | Path | None = None,
                                  baseline_bundle=None, source_review=None) -> Path:
    bundle=Path(bundle_dir); manifest,units=_load_with_baselines(bundle, baseline_bundle, source_review=source_review)
    out=Path(output_dir) if output_dir is not None else bundle/'report'
    out.mkdir(parents=True, exist_ok=True)
    config=manifest['config'];end=config['horizon'];deadline=config['release_deadline']
    structural = config.get('target_kind', 'state') == 'structural'
    predicate = ('concentration at most half its initial value, active-row fraction ≥0.8 and largest weak component fraction ≥0.8' if structural else '≥75% destination support in each private/public layer')
    heading='SOFTWARE REHEARSAL — NOT SCIENTIFIC EVIDENCE' if config['smoke'] else ('Fresh-origin selection — not confirmation' if config.get('stage') == 'selection' else 'Development evidence — not confirmation')
    if not config['smoke'] and config.get('stage') == 'validation':
        heading = 'Frozen algorithm validation on new networks — repeated futures remain clustered'
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Persistent transformation</title>',
           '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid rgba(128,128,128,0.3)}h1{font-size:28px}h2{margin-top:32px}.notice{background:#fff0cc;padding:14px;color:#17232b}figure{margin:24px 0}code{overflow-wrap:anywhere}</style></head><body>',
           '<h1>Persistent transformation</h1><p class="notice">'+heading+'</p>',
           f'<p>Status: <b>{escape(manifest["status"])}</b>. {sum('imported_baseline' not in u for u in units)} / {len(config["queue"])} declared comparisons saved ({manifest.get("physical_continuations_completed",len(units))} physical continuations). Initial conditions: <b>{escape(config["initial_conditions"])}</b>. Memory: <b>{escape(config.get("memory_mode", "adaptive"))}</b>.</p>',
           f'<p>N={config["N"]}; field direction −1; certificate: {predicate}; uninterrupted post-release hold {config["holding_duration"]:g}; release by {deadline:g}; horizon {end:g}. Completion and persistence through the horizon are separate outcomes.</p>',
           '<p>Public reports are sampled, delayed and noisy. Private trajectories shown here are evaluation diagnostics, unavailable to controllers. Targeted and population-wide packages are separate intervention scenarios with imposed, uncalibrated resource limits. Population-wide success does not establish targeted feasibility. Attention concentration is not institutional authority.</p>']
    imports = [u['imported_baseline'] for u in units if 'imported_baseline' in u]
    if imports:
        parts.append(f'<p>{len(imports)} matched uncontrolled comparisons imported from a checksum-verified saved bundle; these are reused evidence, not new continuations.</p>')
    (out/'baseline_provenance.json').write_text(json.dumps(imports, indent=2)+'\n')
    rows=[]; grouped={}
    for u in units:
        policy=u['unit']['policy'];origin=u['unit']['origin']['origin_id'];package=u['unit']['package']
        grouped.setdefault((origin,package),[]).append(u)
        summary=u['summaries'];release=summary.get('actual_release_time')
        transitions=summary.get('target_condition_transitions',[])
        residence=target_residence(transitions,deadline,end)
        points=summary['recorder']['scalar_points']
        late=[p for p in points if p['t']>=.9*end]
        final=u['final_state'];sx=(1-np.mean(final['x']))/2;sy=(1-np.mean(final['y']))/2
        row={'origin':origin,'package':package,'policy':policy,'reused_zero_reference':u.get('reference_unit'),'imported_baseline':u.get('imported_baseline'),'eligible':u['is_eligible'],'valid':u['is_valid'],
             'completion_time':u['completion_time'],'relapsed':u['relapsed'],
             'success_through_horizon':u['success_through_horizon'],'exposure':u['expenditure'],
             'actual_release':release,'distinct_nodes':u['distinct_nodes_used'],
             'post_deadline_residence':residence,
             'final_private_support':float(sx),'final_public_support':float(sy),
             'final_private_public_disagreement':float(np.mean(np.asarray(final['x'])!=np.asarray(final['y']))),
             'initial_structure':u['initial_structure'],'final_structure':u['final_structure'],
             'ineligibility_reason':u.get('target_definition',{}).get('ineligibility_reason'),
             'late_sampled_private_support':float(np.mean([(1-p['m_x'])/2 for p in late])) if late else None,
             'late_sampled_public_support':float(np.mean([(1-p['m_y'])/2 for p in late])) if late else None}
        first = np.asarray(u['network_frames'][0]['W'])
        leaders0 = compute_top_10_percent_set(first)
        leaders1 = compute_top_10_percent_set(np.asarray(final['W']))
        row['original_leader_retention'] = len(leaders0 & leaders1)/len(leaders0) if leaders0 else None
        row['leader_jaccard'] = len(leaders0 & leaders1)/len(leaders0 | leaders1) if leaders0 | leaders1 else None
        rows.append(row)
    # Same-origin/package comparisons only; missing zero remains missing.
    for row in rows:
        zero=next((r for r in rows if r['origin']==row['origin'] and r['package']==row['package'] and r['policy']=='zero'),None)
        row['late_sampled_public_effect_vs_zero']=(row['late_sampled_public_support']-zero['late_sampled_public_support']) if zero and row['late_sampled_public_support'] is not None and zero['late_sampled_public_support'] is not None else None
        row['late_sampled_private_effect_vs_zero']=(row['late_sampled_private_support']-zero['late_sampled_private_support']) if zero and row['late_sampled_private_support'] is not None and zero['late_sampled_private_support'] is not None else None
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    cohort = []
    for package, policy in sorted({(q['package'], q['policy']) for q in config['queue']}):
        planned = {q['origin']['origin_id'] for q in config['queue'] if q['package']==package and q['policy']==policy}
        saved = [r for r in rows if r['package']==package and r['policy']==policy]
        usable = [r for r in saved if r['valid'] and r['eligible']]
        cohort.append({'package':package,'policy':policy,'planned_origins':len(planned),
                       'saved_origins':len(saved),'valid_eligible_origins':len(usable),
                       'completed_hold':sum(r['completion_time'] is not None for r in usable),
                       'persisted_through_horizon':sum(bool(r['success_through_horizon']) for r in usable)})
    for row in cohort:
        complete = (config.get('independent_origins', True) and
                    row['saved_origins'] == row['valid_eligible_origins'] == row['planned_origins'])
        row['uncertainty'] = {
            'method': 'one-sided exact binomial lower bound', 'confidence': 0.95,
            'interpretation': 'descriptive only; assumes independent origins, not confirmation',
            'status': ('withheld: repeated futures within networks' if not config.get('independent_origins', True)
                       else 'complete' if complete else 'withheld: incomplete, invalid or ineligible origins'),
            'hold_lower_bound': exact_binomial_lower_bound(row['completed_hold'], row['planned_origins']) if complete else None,
            'through_horizon_lower_bound': exact_binomial_lower_bound(row['persisted_through_horizon'], row['planned_origins']) if complete else None,
        }
    (out/'cohort_summary.json').write_text(json.dumps(cohort,indent=2)+'\n')
    if not config.get('independent_origins', True):
        parts.append('<p>Repeated audit futures share initial networks. Independent-origin confidence bounds are withheld; use the role table and aggregate by base network.</p>')
    parts.append('<h2>Across origins</h2><table><tr><th>Scenario / policy</th><th>Saved / planned</th><th>Valid and eligible</th><th>Completed hold</th><th>Persisted through H</th><th>95% lower bound: hold / through H</th></tr>')
    for r in cohort:
        parts.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
            (r['package']+' / '+r['policy'],str(r['saved_origins'])+' / '+str(r['planned_origins']),
             r['valid_eligible_origins'],r['completed_hold'],r['persisted_through_horizon'],
             ' / '.join('withheld' if r['uncertainty'][k] is None else f"{r['uncertainty'][k]:.3f}"
                        for k in ('hold_lower_bound', 'through_horizon_lower_bound'))))+'</tr>')
    parts.append('</table><p>Counts use distinct origins within each scenario/policy. Reused zero comparisons across scenarios are not independent replications. Missing, invalid or ineligible origins are shown in the denominators, not silently counted as successful trials. Bounds assume independent origins and are descriptive for development/selection, not confirmation. They are withheld when any declared origin is missing, invalid or ineligible. Holding and through-horizon persistence are separate endpoints; small samples do not certify the 0.8 reliability target.</p>')
    parts.append('<h2>Per-origin outcomes</h2><table><tr><th>Origin / package</th><th>Policy</th><th>Eligible</th><th>Complete at</th><th>Relapsed</th><th>Through H</th><th>Exposure</th><th>Nodes</th><th>Residence after deadline</th></tr>')
    for r in rows:
        parts.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (r['origin']+' / '+r['package'],r['policy'],r['eligible'],r['completion_time'],r['relapsed'],r['success_through_horizon'],round(r['exposure'],3),r['distinct_nodes'],round(r['post_deadline_residence']['duration'],3)))+'</tr>')
    parts.append('</table><p>Residence uses the exact target-predicate transitions in the common post-deadline window. The declared target predicate must hold continuously; separate sampled traces cannot prove holding. No reliability claim is made from this development sample.</p>')
    for (origin,package),arms in sorted(grouped.items()):
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for u in sorted(arms,key=lambda u:u['unit']['policy']):
            policy=u['unit']['policy'];points=u['summaries']['recorder']['scalar_points']
            t=[p['t'] for p in points]
            axes[0,0].plot(t,[(1-p['m_x'])/2 for p in points],label=policy,linewidth=1.5)
            axes[0,1].plot(t,[(1-p['m_y'])/2 for p in points],label=policy,linewidth=1.5)
            axes[1,1].step(t,[p['exposure'] for p in points],where='post',label=policy)
            frames=u['network_frames']
            axes[1,0].plot([f['t'] for f in frames],[calculate_influence_metrics(np.asarray(f['W']))['C_N'] for f in frames],label=policy)
        for ax,title in zip(axes.flat,['Private destination support','Public destination support','Attention concentration (sampled)','Cumulative exposure']):
            ax.set_title(title,fontsize=11);ax.set_xlabel('Time');ax.axvline(deadline,color='#777',linestyle=':',linewidth=1);ax.grid(alpha=.15)
        for ax in axes[0]:
            ax.axhline(.75,color='#222',linestyle='--',linewidth=1);ax.set_ylim(-.03,1.03)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle(f'{origin} · {package}',fontsize=13)
        path=out/f'{origin}__{package}.png';fig.savefig(path,dpi=130);plt.close(fig)
        parts+=['<h2>'+escape(origin+' / '+package)+'</h2>',_image(path),'<p>Dotted line: release deadline; actual release is recorded per arm in summary.json. Dashed line: 75% state threshold. Concentration curves are sampled; structural certificates, when selected, use the exact event-level predicate and connectivity/activity safeguards.</p>']
    candidates = [r for r in rows if r['policy']!='zero' and r['late_sampled_private_effect_vs_zero'] is not None]
    if candidates:
        ranked=sorted(candidates,key=lambda r:(r['late_sampled_private_effect_vs_zero'],r['origin'],r['package'],r['policy']))
        chosen=ranked[len(ranked)//2]
        arms=grouped[(chosen['origin'],chosen['package'])]
        selected=next(u for u in arms if u['unit']['policy']==chosen['policy'])
        zero=next(u for u in arms if u['unit']['policy']=='zero')
        first=selected['network_frames'][0]
        views=[('Initial', first), ('Matched zero at H', zero['final_state']),
               (chosen['policy']+' at H', selected['final_state'])]
        positions=_seeded_spring_layout(np.asarray(first['W']), layout_seed=42)
        fig,axes=plt.subplots(1,3,figsize=(16,6))
        fig.subplots_adjust(left=.02,right=.92,top=.86,bottom=.27,wspace=.12)
        cmap=plt.get_cmap('coolwarm');norm=matplotlib.colors.Normalize(vmin=-1,vmax=1)
        attention_max,tie_max=_network_scales([frame for _,frame in views])
        for ax,(title,frame) in zip(axes,views):
            _draw_control_network(ax,frame,positions,cmap,norm,attention_max,tie_max)
            ax.set_title(title,fontsize=11)
        _add_network_key(fig,cmap,norm)
        fig.suptitle('Matched network comparison — fixed positions from initial memory',fontsize=13)
        path=out/'representative_network.png';fig.savefig(path,dpi=130);plt.close(fig)
        parts+=['<h2>Representative network comparison</h2>',_image(path),
            '<p>Case chosen by upper-median late-window private-support effect versus matched zero, with origin/package/policy breaking ties: '+escape(chosen['origin']+' / '+chosen['package']+' / '+chosen['policy'])+'. Blue is −1; red is +1. Outer disks show public stance and centers private stance. Positions use the initial-memory spring layout (seed 42), fixed across panels; coordinates have no physical meaning. Node area reflects current received attention on a shared scale. Lines show symmetric remembered interaction volume W_ij + W_ji; direction is hidden. This display cannot certify connectivity or holding.</p>']
    else:
        parts.append('<p class="notice">Static network comparison unavailable: no active arm has a matched uncontrolled baseline in this bundle. This report does not fulfill the controlled/uncontrolled presentation requirement. Reuse a provenance-checked saved baseline; do not rerun a simulation merely to populate figures.</p>')
    player=out/'network_progress.html'
    try:
        from src.visualization.transformation_animation import render_transformation_animation
        render_transformation_animation(bundle, output_dir=out, source_review=source_review,
                                        baseline_bundle=baseline_bundle)
        parts+=['<h2>Playable network progress</h2>',
                '<p><a href="network_progress.html">Open the saved-trajectory player</a> — one '
                'self-contained page that steps through every saved arm of one declared origin. '
                'It reads the saved frames, fetches no stylesheet, font or script from the network, '
                'and opens offline from disk. Positions, sizes, colours and the printed stance '
                'fractions match the static panels above; sampled frames still do not certify an '
                'uninterrupted hold.</p>']
    except Exception as exc:
        parts.append('<p class="notice">Saved-trajectory player unavailable: '+escape(str(exc))+
                     '. This does not affect the figures or the checksum-verified records above.</p>')
    parts+=['<h2>Provenance and next decision</h2>',
            '<p>Review matched policy effects, spontaneous baseline drift, realized exposure and relapse before selecting follow-ups. A failed candidate is not proof of constrained infeasibility. Initial states were directly constructed, not equilibrated.</p>',
            '<p>Contract hash: <code>'+escape(config['contract_sha256'])+'</code><br>Source hash: <code>'+escape(config['source_sha256'])+'</code></p>',
            '<p>Exact configuration, frozen seeds, origin states, actions and final states are in the bundle. This report verifies their saved checksums before reading them.</p></body></html>']
    path=out/'index.html';path.write_text('\n'.join(parts))
    artifact_root = out if output_dir is not None else bundle
    artifacts={str(p.relative_to(artifact_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name != 'report_artifacts.json'}
    (artifact_root/'report_artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    return path
