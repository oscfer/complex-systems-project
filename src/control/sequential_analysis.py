"""Read-only long-horizon outcomes, costs, power diagnostics and offline exports."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from src.control.evaluator import evaluate_holding_trace
from src.control.optimizers.campaign import _atomic_json
from src.control.study_analysis import load_raw


def paired(frame,a,b,metric,origins,plan):
    part=frame[frame.role.isin([a,b])];wide=part.groupby(['origin','role'])[metric].mean().unstack()
    if not origins or a not in wide or b not in wide or set(wide.index)!=set(origins) or wide.isna().any().any():
        return dict(a=a,b=b,metric=metric,status='unavailable',effect=None)
    d=wide[a]-wide[b];rng=np.random.default_rng(plan['bootstrap_seed']);ci=np.quantile(rng.choice(d.to_numpy(),(plan['bootstrap_draws'],len(d))).mean(1),[.025,.975])
    return dict(a=a,b=b,metric=metric,status='complete exploratory comparison',effect=float(d.mean()),interval=ci.tolist() if len(d)>1 else None,
        origin_differences=d.to_dict(),networks=len(d),uncertainty='Paired network bootstrap; not independent futures, not multiplicity-adjusted.')


def outcomes(u,plan):
    trace=u['summaries']['target_condition_transitions'];release=float(u['actual_release_time']);H=plan['horizon'];hold=plan['hold']
    if not 0<=release<=plan['release_deadline']+1e-9:raise ValueError('Release outside contract')
    c=evaluate_holding_trace(trace,release,hold,H);relapse=bool(c is not None and any(t>c and not v for t,v in trace));persist=c is not None and not relapse
    if bool(c is not None)!=u['success'] or relapse!=u['relapsed'] or persist!=u['success_through_horizon']:raise ValueError('Exact trace/outcome mismatch')
    if c is not None and not math.isclose(c,u['completion_time'],abs_tol=1e-8):raise ValueError('Completion mismatch')
    actions=u['summaries']['recorder']['action_history'];used=set();exposure=on_time=0.;last=0.;rate=0.;last_field_end=0.;concurrency=0
    for act in actions:
        t=float(act['decision_time']);dt=t-last
        if dt < -1e-9:raise ValueError('Action history time reversal')
        exposure+=max(0.,dt)*rate
        if rate>0:on_time+=max(0.,dt);last_field_end=t
        rate=float(act['spending_rate']);last=t;concurrency=max(concurrency,len(act['actuated_nodes']));used.update(act['actuated_nodes'])
    if rate>0:raise ValueError('Nonzero terminal field')
    if not math.isclose(exposure,u['expenditure'],abs_tol=1e-6):raise ValueError('Action exposure mismatch')
    spec=u['policy_spec']
    if exposure>spec['budget']+1e-6 or concurrency>spec['slots'] or last_field_end>plan['release_deadline']+1e-9:raise ValueError('Control constraint violation')
    if sorted(used)!=u['recipient_nodes']:raise ValueError('Recipient union mismatch')
    row=dict(valid=bool(u['is_valid'] and u['is_eligible']),hold=c is not None,persistent=persist,relapsed=relapse,
        completion=c,penalized_time=c if persist else H,release=release,last_field_end=last_field_end,exposure=exposure,
        on_time=on_time,max_concurrent=concurrency,count=len(used),nodes=sorted(used),first_entry=next((t for t,v in trace if v),None),
        final_private_blue=float(np.mean(np.array(u['final_state']['x'])<0)),final_public_blue=float(np.mean(np.array(u['final_state']['y'])<0)),
        pulses=sum(x['reason'] in ('pulse_start','simultaneous_start') for x in u['controller_log']),
        available=not any(x['reason']=='insufficient_initial_candidates' for x in u['controller_log']),
        frame_count=len(u['network_frames']))
    for offset in plan.get('persistence_offsets',[100,300,500,800]):
        end=release+offset;row[f'persistent_at_{offset}_after_release']=bool(end<=H and c is not None and c<=end and not any(c<t<=end and not v for t,v in trace)) if end<=H else None
    return row


def science_digest(manifest):
    stable={k:v for k,v in manifest.items() if k not in ('report_status','error')}
    return hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def analyze_bundle(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text());plan=m['identity']['plan']
    if plan['version'] not in ('sequential-low-power-v1','response-low-power-v1'):raise ValueError('Choose a sequential low-power bundle')
    initial=[]
    for origin,info in m['initial_states'].items():
        raw=(bundle/info['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Initial-state checksum mismatch')
        state=json.loads(raw)
        if origin in (m.get('allocation') or {}).get('origins',[]):
            if plan.get('initial_red_majority_required') and not (np.mean(np.array(state['x'])>0)>.5 and np.mean(np.array(state['y'])>0)>.5):raise ValueError('Initial state is not red-majority in both layers')
            initial.append(dict(origin=origin,private_blue=float(np.mean(np.array(state['x'])<0)),public_blue=float(np.mean(np.array(state['y'])<0))))
    rows=[];decisions=[];contexts=[];actions=[]
    for key,info in m['actual'].items():
        u=load_raw(bundle,key)
        if info['kind']!='audit':continue
        if key not in m['audit_roles']:continue
        spec=plan['policies'][info['role']]
        if spec!=u['policy_spec']:raise ValueError('Policy identity mismatch')
        r=outcomes(u,plan)
        if plan['version']=='response-low-power-v1':
            from src.control.response_analysis import diagnostics
            r.update(diagnostics(u,plan,json.loads((bundle/m['initial_states'][info['origin']]['path']).read_text())))
        r.update(physical_unit=key,origin=info['origin'],future=info['future'],role=info['role'],family=spec['name'],
            budget=spec['budget'],memory=spec['memory'],visibility=spec['visibility']);rows.append(r)
        tags=dict(physical_unit=key,origin=info['origin'],future=info['future'],role=info['role'])
        decisions.extend(dict(**tags,**v) for v in u['controller_log']);actions.extend(dict(**tags,**v) for v in u['summaries']['recorder']['action_history'])
        contexts.extend(dict(**tags,**v) for v in u['summaries']['recorder']['sequential_context']['action_contexts'])
    frame=pd.DataFrame(rows);allocation=m.get('allocation');expected=set(m['queue']);actual={k for k,v in m['actual'].items() if v['kind']=='audit'}
    complete=bool(m['status']=='complete' and m['selection_frozen'] and allocation and expected and set(m['audit_roles'])==expected==actual and len(frame)==len(expected) and frame.valid.all())
    summary=pd.DataFrame()
    if not frame.empty:
        group=['role','family','budget','memory','visibility']
        totals=frame.groupby(group,as_index=False).agg(runs=('persistent','size'),available=('available','sum'))
        usable=frame[frame.available].groupby(group,as_index=False).agg(hold=('hold','sum'),persistent=('persistent','sum'),persistent_rate=('persistent','mean'),mean_exposure=('exposure','mean'),mean_distinct=('count','mean'),max_concurrent=('max_concurrent','max'),mean_penalized_time=('penalized_time','mean'),mean_release=('release','mean'))
        summary=totals.merge(usable,on=group,how='left');summary['unavailable']=summary.runs-summary.available
    if not summary.empty and plan['version']=='response-low-power-v1':
        extra=frame.groupby('role',as_index=False).agg(majority_persistent=('majority_persistent','sum'),
            majority_rate=('majority_persistent','mean'),mean_peak_joint=('peak_joint','mean'),
            mean_response_core=('max_response_core','mean'),mean_restarts=('restarts','mean'),
            mean_growth_waits=('growth_waits','mean'),mean_selected_true_percentile=('mean_selected_true_percentile','mean'))
        summary=summary.merge(extra,on='role')
    primary=dict(status='withheld: incomplete allocation',effect=None);contrasts=[]
    if complete:
        origins=allocation['origins'];primary=paired(frame,**plan['primary'],origins=origins,plan=plan)
        if 'contrasts' in plan:
            metrics=['persistent','majority_persistent','exposure','count','penalized_time']
            for contrast in plan['contrasts']:
                for metric in metrics:
                    c=paired(frame,contrast['a'],contrast['b'],metric,origins,plan)
                    c['hypothesis']=contrast['hypothesis'];contrasts.append(c)
        else:
            for B in plan['budgets']:
                a=f'dynamic_low1_B{B}'
                for b in [f'initial_low1_B{B}',f'random1_B{B}',f'dynamic_low2_B{B}',f'fallen_low1_B{B}',f'simultaneous_low4_B{B}',f'simultaneous_attention4_B{B}',f'simultaneous_coverage4_B{B}']:
                    available=[o for o in origins if frame[(frame.origin==o)&frame.role.isin([a,b])].available.all()]
                    for metric in ['persistent','exposure','count','penalized_time']:
                        c=paired(frame[frame.origin.isin(available)],a,b,metric,available,plan);c['excluded_networks']=[o for o in origins if o not in available];contrasts.append(c)
            for a in ['no_pause_B80','long_pulse_B80','strict_low_B80','no_early_stop_B80','fallen_only_B80','frozen_low_B80','complete_low_B80']:
                for metric in ['persistent','exposure','count','penalized_time']:contrasts.append(paired(frame,a,'dynamic_low1_B80',metric,origins,plan))
            for a,b in [('zero_frozen','zero'),('zero_complete','zero')]:contrasts.append(paired(frame,a,b,'persistent',origins,plan))
    interaction_results=[]
    if complete and plan.get('interactions'):
        from src.control.response_analysis import interactions
        interaction_results=interactions(frame,plan,allocation['origins'])
    return dict(bundle=str(bundle),manifest=m,plan=plan,complete=complete,interactions=interaction_results,roles=frame,summary=summary,initial=pd.DataFrame(initial),
        primary=primary,contrasts=contrasts,decisions=pd.DataFrame(decisions),contexts=pd.DataFrame(contexts),actions=pd.DataFrame(actions),N=plan['N'])


def player_view(data,memory,visibility):
    view=copy.copy(data);r=data['roles'];r=r[(r.memory==memory)&(r.visibility==visibility)&(r.frame_count>0)].copy()
    zero='zero_complete' if visibility=='complete_public' else ('zero_frozen' if memory=='frozen' else 'zero');r.loc[r.role==zero,'role']='zero'
    zeros=set(zip(r[r.role=='zero'].origin,r[r.role=='zero'].future));r=r[[pair in zeros for pair in zip(r.origin,r.future)]]
    view['roles']=r;view['include_unactuated_policies']=True;return view


def build_saved_report(bundle,*,output_dir=None):
    from src.visualization.animation_player import prepare_comparisons,payload_from_data,write_player_html,write_player_metadata,INLINE_MAX_BYTES,PayloadBudgetError
    import matplotlib.pyplot as plt
    data=analyze_bundle(bundle);out=Path(output_dir) if output_dir is not None else Path(bundle)/'analysis';out.mkdir(parents=True,exist_ok=True)
    for name in ['roles','summary','initial','decisions','contexts','actions']:data[name].to_csv(out/f'{name}.csv',index=False)
    _atomic_json(out/'contrasts.json',dict(primary=data['primary'],secondary=data['contrasts'],interactions=data.get('interactions',[]),note='Exploratory secondary comparisons; no automatic selection or next run.'))
    figs=[]
    if not data['summary'].empty:
        s=data['summary'];fig,axs=plt.subplots(1,2,figsize=(15,max(7,.27*len(s))))
        for ax,col,label in [(axs[0],'persistent_rate',f"Persistent success through t={data['plan']['horizon']:g}"),(axs[1],'mean_exposure','Mean actual exposure, including failures')]:
            ax.barh(s.role,s[col],color='#3b7789');ax.invert_yaxis();ax.set_xlabel(label)
        axs[0].set_xlim(0,1);fig.suptitle('All declared policies — '+('complete allocation' if data['complete'] else 'PARTIAL: inference withheld'))
        fig.tight_layout();p=out/'success_and_cost.png';fig.savefig(p,dpi=130);plt.close(fig);figs.append(str(p))
        fig,ax=plt.subplots(figsize=(9,5))
        rr=data['roles'];main=rr[(rr.memory=='adaptive')&(rr.visibility=='restricted')&rr.family.isin(data['plan'].get('plot_families',['dynamic_low1','initial_low1','random1','fallen_low1','dynamic_low2']))]
        for family,q in main.groupby('family'):
            q=q.groupby('budget').agg(exposure=('exposure','mean'),success=('persistent','mean'));ax.plot(q.exposure,q.success,'o-',label=family)
        ax.set(xlabel='Mean realized exposure across all futures',ylabel='Persistent success fraction',ylim=(-.02,1.02),title='Success and cost together; empirical comparisons, not an optimality proof');ax.legend(fontsize=9);fig.tight_layout();p=out/'cost_success.png';fig.savefig(p,dpi=140);plt.close(fig);figs.append(str(p))
    players={}
    if not data['roles'].empty:
        for memory,visibility in [('adaptive','restricted'),('frozen','restricted'),('adaptive','complete_public')]:
            view=player_view(data,memory,visibility)
            if view['roles'].empty or not (view['roles'].role!='zero').any():continue
            label=f'{memory}_{visibility}';folder=out/label;folder.mkdir(exist_ok=True)
            ctx=prepare_comparisons(view,progress=False);ctx['meta']['future_note']='Only future0 retains frames; all futures retain exact flips and decisions.'
            payload,report=payload_from_data(view,prepared=ctx,max_frames=data['plan']['max_frames'],max_bytes=None)
            page=write_player_html(payload,folder/'player.html',title=f'{data["plan"].get("title","Sequential low-power pilot")} — {label}');write_player_metadata(payload,page,report=report,source=str(bundle))
            inline=None
            try:
                small,_=payload_from_data(view,prepared=ctx,max_frames=data['plan']['max_frames'],min_frames=data['plan']['max_frames'],max_bytes=INLINE_MAX_BYTES);inline=str(write_player_html(small,folder/'inline_player.html',title=f'Preview — {label}'))
            except PayloadBudgetError:pass
            players[label]=dict(path=str(page),inline=inline,comparisons=len(payload['comparisons']),dropped=report['dropped'],skipped=report['skipped'])
    report=dict(complete=data['complete'],science_status=data['manifest']['status'],allocation=data['manifest'].get('allocation'),primary=data['primary'],roles=len(data['roles']),
        figures=figs,players=players,science_sha256=science_digest(data['manifest']),source_manifest_sha256=hashlib.sha256((Path(bundle)/'manifest.json').read_bytes()).hexdigest(),
        horizon=data['plan']['horizon'],release_deadline=data['plan']['release_deadline'],no_automatic_followup=True)
    if data['plan']['version']=='response-low-power-v1':
        from src.control.response_analysis import write_interpretation
        write_interpretation(data,out)
    report['artifact_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='report_complete.json'}
    _atomic_json(out/'report_complete.json',report);return report


def read_saved_report(bundle):
    """Read or relocate one complete output folder; no simulation or regeneration."""
    bundle=Path(bundle);out=bundle/'analysis'
    m=json.loads((bundle/'manifest.json').read_text());marker=out/'report_complete.json'
    if m.get('report_status')!='complete' or not marker.exists():return None,out,m
    report=json.loads(marker.read_text())
    if report.get('science_sha256')!=science_digest(m):raise ValueError('Report belongs to a different scientific bundle or revision')
    if not report.get('artifact_sha256'):raise ValueError('Missing export checksums')
    for name,digest in report.get('artifact_sha256',{}).items():
        p=(out/name).resolve()
        if not p.is_relative_to(out.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'Export checksum mismatch: {name}')
    report['figures']=[str(out/Path(p).name) for p in report['figures']]
    for label,player in report['players'].items():
        player['path']=str(out/label/Path(player['path']).name)
        if player.get('inline'):player['inline']=str(out/label/Path(player['inline']).name)
    return report,out,m


def inspect_unit(bundle,origin,role,future=0,start=0.,end=200.):
    key=f'{origin}__{role}__future{future}';u=load_raw(bundle,key)
    decisions=pd.DataFrame([d for d in u['controller_log'] if start<=d['time']<=end])
    flips=pd.DataFrame([d for d in u['summaries']['recorder']['nucleation']['flips'] if start<=d['t']<=end])
    contexts=pd.DataFrame([d for d in u['summaries']['recorder']['sequential_context']['action_contexts'] if start<=d['time']<=end])
    return decisions,flips,contexts
