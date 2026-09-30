"""Read-only pilot analysis: speed with failure penalties, exact local traces."""
from __future__ import annotations
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.control.study_analysis import analyze_bundle,load_raw,export_analysis
from src.control.evaluator import evaluate_holding_trace


def _contrast(frame,a,b,metric,origins,plan):
    sub=frame[frame.role.isin([a,b]) & frame.origin.isin(origins)]
    if sub.empty:return dict(comparison=f'{a} minus {b}',status='unavailable',effect=None)
    wide=sub.groupby(['origin','role'])[metric].mean().unstack()
    if a not in wide or b not in wide or set(wide.index)!=set(origins) or wide.isna().any().any():
        return dict(comparison=f'{a} minus {b}',status='unavailable on one or more networks',effect=None)
    d=(wide[a]-wide[b]).to_numpy();rng=np.random.default_rng(plan['bootstrap_seed'])
    ci=np.quantile(rng.choice(d,(plan['bootstrap_draws'],len(d))).mean(axis=1),[.025,.975])
    return dict(comparison=f'{a} minus {b}',metric=metric,status='complete exploratory pilot',effect=float(d.mean()),
        interval=ci.tolist() if len(d)>1 else None,origin_differences=dict(zip(wide.index,d.tolist())),networks=len(d),
        uncertainty='Paired network bootstrap; limited clusters, not independent futures. Exploratory uncertainty only.')


def analyze_pilot(bundle):
    data=analyze_bundle(bundle);m=data['manifest'];plan=m['identity']['plan']
    if plan['version'] not in ('nucleation-pilot-v1','nucleation-pilot-v2'):raise ValueError('Choose a nucleation pilot bundle')
    for key,info in m['actual'].items():
        if info['kind']=='timing':load_raw(bundle,key)  # Verify provenance, never use timing outcomes in inference.
    for origin,groups in m.get('groups',{}).items():
        for row in data['unavailable'].to_dict('records'):
            if row['origin']==origin:
                data['unavailable'].loc[(data['unavailable'].origin==origin)&(data['unavailable'].role==row['role']),'reason']=groups[row['role']].get('reason','Unavailable by frozen selection rule')
    expected=m.get('planned_roles',{})
    data['complete']=bool(data['complete'] and expected and m['audit_roles']==expected and {k for k,v in m['actual'].items() if v['kind']=='audit'}==set(m['resolved_units']))
    cache={};samples=[];core_rows=[]
    for key in set(m['audit_roles'].values()):
        u=load_raw(bundle,key);info=m['actual'][key]
        if u['memory_mode']!=info['memory']:raise ValueError('Memory mode mismatch')
        if abs(u['expenditure']-38*len(u['recipient_nodes']))>1e-6:raise ValueError('Unexpected pilot exposure')
        diag=u['summaries']['recorder']['nucleation'];trace=u['summaries']['target_condition_transitions']
        cache[key]=dict(memory=info['memory'],frame_count=len(u.get('network_frames',[])),first_joint_entry=next((float(t) for t,v in trace if v),None),
            initially_joint_target=bool(trace and trace[0][0]==0 and trace[0][1]),flip_count=len(diag['flips']))
        samples.extend(dict(physical_unit=key,origin=info['base_origin'],future=info['future'],memory=info['memory'],**r) for r in diag.get('core_samples',[]))
        for name,local in diag['core_transitions'].items():
            completion=evaluate_holding_trace(local,20.,20.,320.)
            core_rows.append(dict(physical_unit=key,origin=info['base_origin'],future=info['future'],memory=info['memory'],core=name,
                initially_target=bool(local[0][1]),first_entry=next((t for t,v in local if v),None),
                local_hold_completion=completion,local_onset=completion-20 if completion is not None else None,
                local_relapsed=bool(completion is not None and any(t>completion and not v for t,v in local))))
    if not data['roles'].empty:
        add=pd.DataFrame([dict(physical_unit=k,**v) for k,v in cache.items()]);data['roles']=data['roles'].merge(add,on='physical_unit')
    data['core_outcomes']=pd.DataFrame(core_rows)
    data['core_samples']=pd.DataFrame(samples)
    data['primary']=dict(status='withheld: incomplete or missing allocation',effect=None)
    data['contrasts']=[]
    if data['complete']:
        origins=m['allocation']['origins'];frame=data['roles']
        data['primary']=_contrast(frame,'nucleus_bridge4','audience_coverage4','penalized_time',origins,plan)
        for k in plan['sizes']:
            for a,b in [(f'nucleus_bridge{k}',f'audience_coverage{k}'),(f'nucleus_bridge{k}',f'nucleus{k}'),
                        (f'nucleus{k}',f'attention{k}'),(f'nucleus_bridge{k}',f'random{k}'),
                        (f'nucleus_bridge{k}',f'nucleus_bridge{k}_frozen'),
                        (f'nucleus{k}',f'random_core{k}'),(f'nucleus_bridge{k}',f'random_bridge{k}'),
                        (f'nucleus{k}',f'nucleus{k}_frozen')]:
                for metric in ['penalized_time','persistent']:
                    available=[o for o in origins if all(m['groups'][o].get(r,{}).get('available',False) for r in (a,b))]
                    result=_contrast(frame,a,b,metric,available,plan)
                    result.update(scope='paired available networks',excluded_networks=[o for o in origins if o not in available])
                    data['contrasts'].append(result)
        data['contrasts'].append(_contrast(frame,'zero','zero_frozen','persistent',origins,plan))
        persistent=_contrast(frame,'nucleus_bridge4','audience_coverage4','persistent',origins,plan)
        winners=frame[(frame.role=='nucleus_bridge4')&frame.persistent].origin.nunique()
        data['decision']=dict(advance_hypothesis=bool(data['primary']['effect'] is not None and data['primary']['effect']<0
            and persistent['effect'] is not None and persistent['effect']>=0 and winners>=2),
            networks_with_bridge4_persistence=int(winners),rule=plan['decision'],no_automatic_run=True)
    else:data['decision']=dict(advance_hypothesis=None,reason='Finish the frozen allocation; missing is not failure.')
    data['recipient_characteristics']=recipient_characteristics(data)
    data['memory_interactions']=memory_interactions(data) if data['complete'] else []
    return data


def export_pilot(data,out_dir):
    out=export_analysis(data,out_dir)
    data['core_outcomes'].to_csv(out/'core_outcomes.csv',index=False)
    data['core_samples'].to_csv(out/'core_samples.csv',index=False)
    data['recipient_characteristics'].to_csv(out/'recipient_characteristics.csv',index=False)
    (out/'pilot_contrasts.json').write_text(json.dumps(dict(primary=data['primary'],contrasts=data['contrasts'],memory_interactions=data['memory_interactions'],decision=data['decision']),indent=2))
    return out


def memory_view(data,memory):
    """Keep frozen arms paired with frozen zero in the existing saved-data player."""
    view=copy.copy(data);frame=data['roles'];view['roles']=frame[frame.memory==memory].copy()
    if memory=='frozen':view['roles']['role']=view['roles'].role.str.removesuffix('_frozen')
    r=view['roles'];r=r[r.frame_count>0].copy()
    zeros=set(zip(r[r.role=='zero'].origin,r[r.role=='zero'].future))
    active=set(zip(r[r.role!='zero'].origin,r[r.role!='zero'].future))
    pairs=zeros & active
    view['roles']=r[[pair in pairs for pair in zip(r.origin,r.future)]]
    return view


def flip_window(data,origin,role,future=0,start=0,end=25):
    rows=data['roles'];row=rows[(rows.origin==origin)&(rows.role==role)&(rows.future==future)]
    if len(row)!=1:raise ValueError('Choose one saved role/origin/future')
    u=load_raw(data['bundle'],row.iloc[0].physical_unit)
    return pd.DataFrame([e for e in u['summaries']['recorder']['nucleation']['flips'] if start<=e['t']<=end])


def recipient_characteristics(data):
    """Full-state traits are retrospective descriptors, never proposal inputs."""
    rows=[];m=data['manifest'];bundle=Path(data['bundle'])
    for origin,groups in m['groups'].items():
        entry=m['initial_states'][origin];raw=(bundle/entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Initial state checksum mismatch')
        state=json.loads(raw)
        W=np.asarray(state['W'],float);A=np.divide(W,W.sum(1)[:,None],out=np.zeros_like(W),where=W.sum(1)[:,None]>0);r=A.sum(0)
        for role,g in groups.items():
            if not g['available']:continue
            k=len(g['nodes']);core=set(m['cores'][origin].get(f'core{k}',[]))
            for i in g['nodes']:
                rows.append(dict(origin=origin,role=role,memory=g['memory'],node=i,
                    location='outside_bridge' if i in g.get('outside_nodes',[]) else ('core' if i in core else 'other'),
                    initial_received_attention=float(r[i]),K=float(state['K'][i]),g=float(state['g'][i]),
                    private_bias=float(state['b'][i][0]),private_update_time=float(state['tau_x'][i]),
                    initially_private_blue=state['x'][i][0]<0,initially_public_blue=state['y'][i][0]<0))
    return pd.DataFrame(rows)


def memory_interactions(data):
    m=data['manifest'];plan=m['identity']['plan'];origins=m['allocation']['origins'];frame=data['roles'];results=[]
    for k in plan['sizes']:
        roles=[f'nucleus_bridge{k}',f'nucleus{k}',f'nucleus_bridge{k}_frozen',f'nucleus{k}_frozen']
        eligible=[o for o in origins if all(m['groups'][o].get(r,{}).get('available',False) for r in roles)]
        for metric in ['penalized_time','persistent']:
            part=frame[frame.origin.isin(eligible)&frame.role.isin(roles)]
            if not eligible or part.empty:continue
            wide=part.groupby(['origin','role'])[metric].mean().unstack()
            if any(r not in wide for r in roles) or wide[roles].isna().any().any():continue
            d=(wide[roles[0]]-wide[roles[1]])-(wide[roles[2]]-wide[roles[3]])
            rng=np.random.default_rng(plan['bootstrap_seed'])
            ci=np.quantile(rng.choice(d.to_numpy(),(plan['bootstrap_draws'],len(d))).mean(1),[.025,.975])
            results.append(dict(count=k,metric=metric,definition='(bridge - nucleus) adaptive minus (bridge - nucleus) frozen',
                effect=float(d.mean()),interval=ci.tolist() if len(d)>1 else None,networks=len(d),origin_differences=d.to_dict(),
                excluded_networks=[o for o in origins if o not in eligible],interpretation='Paired available-network interaction, exploratory; does not isolate a particular edge.'))
    return results


def build_saved_report(bundle):
    """Called inside the same hard-limited worker as the scientific run."""
    from src.visualization.study_figures import ensemble_figures
    from src.visualization.animation_player import (prepare_comparisons,payload_from_data,write_player_html,
        write_player_metadata,INLINE_MAX_BYTES,PayloadBudgetError)
    data=analyze_pilot(bundle);out=Path(str(bundle)+'_analysis');export_pilot(data,out)
    figures=ensemble_figures(dict(data,groups=data['groups'].iloc[0:0]),out)
    import matplotlib.pyplot as plt
    if not data['summary'].empty:
        summary=data['summary'];fig,ax=plt.subplots(figsize=(10,max(5,.3*len(summary))))
        ax.barh(summary.role,summary.mean_penalized_time.fillna(320));ax.invert_yaxis()
        ax.set(xlim=(0,320),xlabel='Completion time; failures and later relapses scored 320',title='Speed and persistence together — '+('complete' if data['complete'] else 'PARTIAL'))
        fig.tight_layout();path=out/'completion_times.png';fig.savefig(path,dpi=130);plt.close(fig);figures.append(path)
    players={}
    if not data['roles'].empty:
        for memory in ('adaptive','frozen'):
            view=memory_view(data,memory)
            if view['roles'].empty:continue
            folder=out/memory;folder.mkdir(parents=True,exist_ok=True);ctx=prepare_comparisons(view,progress=False)
            payload,report=payload_from_data(view,prepared=ctx,max_frames=161,max_bytes=None)
            page=write_player_html(payload,folder/'player.html',title=f'{memory.title()} — overnight nucleation pilot')
            write_player_metadata(payload,page,report=report,source=str(bundle))
            inline=None
            try:
                small,_=payload_from_data(view,prepared=ctx,max_bytes=INLINE_MAX_BYTES)
                inline=str(write_player_html(small,folder/'inline_player.html',title=f'{memory.title()} — notebook-sized preview'))
            except PayloadBudgetError:pass
            players[memory]=dict(path=str(page),inline=inline,comparisons=len(payload['comparisons']),full_export=True)
    overview=dict(complete=data['complete'],status=data['status'],physical_audit_units=data['physical_audit_units'],
        role_rows=len(data['roles']),primary=data['primary'],decision=data['decision'],
        allocation=data['manifest'].get('allocation'),timing_units=sum(v['kind']=='timing' for v in data['manifest']['actual'].values()),
        figures=[str(p) for p in figures],players=players,core_sample_rows=len(data['core_samples']))
    # Written last: a partial export cannot masquerade as a completed report.
    from src.control.optimizers.campaign import _atomic_json
    _atomic_json(out/'report_complete.json',overview)
    return overview
