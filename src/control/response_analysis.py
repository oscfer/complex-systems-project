"""Exact secondary endpoints and interpretation for the final weak-node study."""
from __future__ import annotations
import html,json
from pathlib import Path
import numpy as np
from src.control.evaluator import evaluate_holding_trace


def diagnostics(unit,plan,initial):
    x=np.array(initial['x']).reshape(-1).copy();y=np.array(initial['y']).reshape(-1).copy()
    minimum=plan['secondary_majority']['minimum_blue_nodes']
    trace=[(0.,bool(min((x<0).sum(),(y<0).sum())>=minimum))]
    peak=int(min((x<0).sum(),(y<0).sum()))
    for event in unit['summaries']['recorder']['nucleation']['flips']:
        state=x if event['layer']=='x' else y
        if state[event['node']]!=event['before']:raise ValueError('Exact flip replay mismatch')
        state[event['node']]=event['after'];joint=int(min((x<0).sum(),(y<0).sum()));peak=max(peak,joint)
        value=joint>=minimum
        if value!=trace[-1][1]:trace.append((event['t'],value))
    if x.tolist()!=np.array(unit['final_state']['x']).reshape(-1).tolist() or y.tolist()!=np.array(unit['final_state']['y']).reshape(-1).tolist():
        raise ValueError('Exact final-state replay mismatch')
    completion=evaluate_holding_trace(trace,unit['actual_release_time'],plan['hold'],plan['horizon'])
    persistent=completion is not None and not any(t>completion and not value for t,value in trace)
    logs=unit['controller_log'];latest_low=set();ranks=[]
    contexts={c['time']:c for c in unit['summaries']['recorder']['sequential_context']['action_contexts']}
    for log in logs:
        if log.get('sample_time') is not None:latest_low=set(log['low_nodes'])
        if len(log['nodes'])>1 or not set(log['nodes'])<=latest_low:raise ValueError('Single currently-weak recipient constraint violated')
        if log['reason']=='pulse_start':
            if any(i not in log.get('currently_red',log['nodes']) for i in log['nodes']):raise ValueError('Recruitment must use fresh red evidence')
            powers=contexts[log['time']]['true_received_attention']
            ranks.extend(sum(v<=powers[i]+1e-12 for v in powers)/len(powers) for i in log['nodes'])
    return dict(majority_hold=completion is not None,majority_persistent=bool(persistent),majority_completion=completion,
        peak_joint=peak,max_response_core=max((len(l.get('response_core',[])) for l in logs),default=0),
        restarts=max(0,max((l.get('episode',0) for l in logs),default=0)-1),
        growth_waits=sum(l['reason']=='wait_for_spontaneous_spread' for l in logs),
        mean_selected_true_percentile=float(np.mean(ranks)) if ranks else None)


def write_interpretation(data,out):
    """Self-contained overview; relative links keep the single bundle portable."""
    out=Path(out);s=data['summary'];status='COMPLETE' if data['complete'] else 'PARTIAL — comparisons withheld'
    title='Final response-guided weak-node comparison'
    intro=(f'{status}. Primary success means at least75% blue in BOTH layers for100 uninterrupted units '
        'after permanent release, with no later loss through500. The predeclared ordinary-majority endpoint '
        '(at least21/40 in each layer, same hold and horizon) is secondary. Neither a temporary blue group '
        'nor a successful public switch alone counts as transformation.')
    limits=('All main policies target at most one currently weak node. Weakness is a partial-report proxy, '
        'not knowledge of true influence; retrospective true ranks are reported separately. '
        'Response groups describe observed associations after treatment, not individually identified causal effects. '
        'Frozen attention and complete-public arms are diagnostic regimes. Comparisons use fresh paired networks, '
        'with futures clustered within networks. Secondary intervals are exploratory and not multiplicity-adjusted. '
        'A [0,0] bootstrap interval with all-zero outcomes does not establish equivalence or zero population probability. '
        'Small network samples cannot establish80% reliability or global optimality. Failures remain in all cost summaries. '
        'No next experiment or retuning is automatically selected.')
    primary=json.dumps(data['primary'],indent=2)
    cols=[c for c in ['role','runs','persistent','persistent_rate','majority_persistent','majority_rate','mean_exposure','mean_distinct','max_concurrent','mean_response_core','mean_restarts','mean_selected_true_percentile'] if c in s]
    table=s[cols].to_html(index=False,float_format=lambda x:f'{x:.3f}') if not s.empty else '<p>No audit outcomes available.</p>'
    links=''.join(f'<li><a href="{regime}/player.html">{label}</a></li>' for regime,label in [
        ('adaptive_restricted','Full main animations'),('frozen_restricted','Frozen-attention diagnostic animations'),
        ('adaptive_complete_public','Complete-public diagnostic animations')] if (out/regime/'player.html').exists())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;color:#182431;line-height:1.5}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #ccc}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:16px}}img{{max-width:100%}}</style>
<h1>{title}</h1><p>{intro}</p><h2>Animations and treatment</h2><ul>{links}</ul>
<p>Solid gold: receiving the field now. Dashed gold: previously treated, field off. Labels are node IDs. At t=0 no treatment ring is expected. Full files preserve every saved comparison and101 time points; only future0 has network frames. Exact events and decisions cover every future. Inline previews retain time resolution but may omit comparisons.</p>
<h2>Primary comparison</h2><pre>{html.escape(primary)}</pre><h2>All policies and costs</h2><div class="scroll">{table}</div>
<h2>Interpretation</h2><p>{limits}</p><p><a href="contrasts.json">All predeclared comparisons</a> · <a href="roles.csv">Every trajectory</a> · <a href="decisions.csv">Decisions and response groups</a> · <a href="contexts.csv">Retrospective true influence</a></p></html>'''
    (out/'report.html').write_text(page)
    (out/'FINDINGS.md').write_text(f'# {title}\n\n{intro}\n\nPrimary:\n```json\n{primary}\n```\n\n{limits}\n\nSee report.html, summary.csv, roles.csv and contrasts.json.\n')


def interactions(frame,plan,origins):
    results=[]
    for spec in plan.get('interactions',[]):
        for metric in ['persistent','majority_persistent']:
            w=frame.groupby(['origin','role'])[metric].mean().unstack().loc[origins]
            diff=(w[spec['a']]-w[spec['b']])-(w[spec['c']]-w[spec['d']])
            rng=np.random.default_rng(plan['bootstrap_seed'])
            ci=np.quantile(rng.choice(diff.to_numpy(),(plan['bootstrap_draws'],len(diff))).mean(1),[.025,.975])
            results.append(dict(**spec,metric=metric,effect=float(diff.mean()),interval=ci.tolist(),
                origin_differences=diff.to_dict(),networks=len(diff),status='complete exploratory interaction',
                definition='(a-b)-(c-d), paired within networks; unadjusted network bootstrap'))
    return results
