"""Streaming verification, prespecified cluster summaries and offline players."""
import csv, html, json, math
from pathlib import Path
import numpy as np
from src.control.scale_extension import (read_gzip, _sha, _atomic_json, reconstruct_endpoint, digest_data)

NOTICE=('Constructed synthetic Tiered-population scenario, not equilibrated or empirically calibrated. '
    'Independent origins, not futures/nodes/roles, are the replication units. '
    'Frozen memory disables a bundle of mechanisms. Threshold breaches need not be opposing-basin returns. '
    'Sampled animations do not certify persistence. Bootstrap intervals are exploratory with few origins; '
    'degenerate all-zero intervals do not establish equivalence.')

def csv_write(path,rows):
    if not rows:path.write_text('');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def paired(rows,N,a,b,metric,draws,rng):
    origins=sorted({r['origin'] for r in rows if r['N']==N});values=[]
    for origin in origins:
        arms={role:[r[metric] for r in rows if r['origin']==origin and r['role']==role] for role in (a,b)}
        if not arms[a] or not arms[b]:return None
        values.append(float(np.mean(arms[a])-np.mean(arms[b])))
    if not values:return None
    v=np.array(values);samples=rng.choice(v,(draws,len(v))).mean(axis=1)
    return dict(N=N,a=a,b=b,metric=metric,effect=float(v.mean()),
        interval=np.quantile(samples,[.025,.975]).tolist(),per_origin=dict(zip(origins,values)),
        origins=len(v),degenerate=bool(np.ptp(samples)==0))

# Compatibility imports; rendering is owned by the visualization module.
from src.visualization.scale_animation import fixed_layout, player, export_animations, rebuild_animations

def export(bundle):
    m=json.loads((bundle/'manifest.json').read_text());out=bundle/'analysis';out.mkdir(exist_ok=True)
    m['export_status']='incomplete';_atomic_json(bundle/'manifest.json',m)
    if not m.get('prepared'):
        (out/'report.html').write_text('<h1>No completed initial-condition preparation</h1><p>Science incomplete. Preserve the manifest and source.</p>')
        m['export_status']='complete';_atomic_json(bundle/'manifest.json',m);return
    if _sha(bundle/'origins.json.gz')!=m['origins_sha256']:raise ValueError('Origin hash mismatch')
    data=read_gzip(bundle/'origins.json.gz');origins=data['origins'];rows=[];performance=[];links=[];initial=[]
    for origin in origins.values():
        s=origin['initial'];W=np.asarray(s['W']);N=s['N']
        initial.append(dict(origin=origin['id'],N=N,seed=origin['seed'],candidate=origin['candidate'],
            private_blue=int((np.asarray(s['x'])<0).sum()),public_blue=int((np.asarray(s['y'])<0).sum()),
            tokens_mean=float(W.sum(axis=1).mean()),degree_mean=float((W>0).sum(axis=1).mean()),
            density=float((W>0).sum()/(N*(N-1))),row_cap_min=min(s['row_caps']),row_cap_max=max(s['row_caps'])))
    roles_by_key={}
    for name,key in m.get('roles',{}).items():roles_by_key.setdefault(key,[]).append(name.split('__')[-1])
    zero_cache={}
    def zero_for(u,roles):
        """Matched zero arm: same origin, future and memory; None when absent."""
        if not roles or roles[0].split('_')[-2]=='zero':return None
        name=u['origin']+'__f'+str(u['future'])+'__zero_'+roles[0].rsplit('_',1)[1]
        zkey=m.get('roles',{}).get(name)
        if zkey is None or zkey not in m['units']:return None
        if zkey not in zero_cache:
            zu=read_gzip(bundle/m['units'][zkey]['path'])
            zero_cache[zkey]=zu if zu['frames'] else None
        return zero_cache[zkey]
    for key,info in m['units'].items():
        path=bundle/info['path']
        if _sha(path)!=info['sha256']:raise ValueError('Saved unit hash mismatch: '+key)
        u=read_gzip(path);origin=origins[u['origin']]
        endpoint=reconstruct_endpoint(origin['initial'],u)
        if endpoint!=u['endpoint']:raise AssertionError('Endpoint verification mismatch')
        # Independently replay every stored node-state frame, not just terminal support.
        states=[np.asarray(origin['initial'][layer]).reshape(-1).copy() for layer in ('x','y')]
        flips=iter(u['summaries']['recorder']['spin_changes']);change=next(flips,None)
        for f in u['frames']:
            while change is not None and change[0]<=f['t']:
                t,l,i,b,a=change;states[l][i]=a;change=next(flips,None)
            if not all(np.array_equal(states[l],f[layer]) for l,layer in enumerate(('x','y'))):raise AssertionError('Saved frame does not replay')
            W=np.zeros((u['N'],u['N']),dtype=np.int64)
            for i,j,v in f['edges']:W[i,j]=v
            if (np.diag(W)!=0).any() or (W<0).any() or (W.sum(axis=1)>np.asarray(origin['initial']['row_caps'])).any():raise AssertionError('Frame memory invariant')
        events=sum(u['event_counts'].values())
        performance.append(dict(unit=key,N=u['N'],phase=info['phase'],memory=u['group']['memory'],
            recipients=len(u['group']['nodes']),seconds=info['elapsed'],events=events,
            events_per_second=events/max(info['elapsed'],1e-9),compressed_bytes=path.stat().st_size,
            spin_changes=len(u['summaries']['recorder']['spin_changes']),frames=len(u['frames'])))
        if info['phase']!='audit':continue
        for role in roles_by_key.get(key,[]):
            row=dict(N=u['N'],origin=u['origin'],future=u['future'],role=role,physical_unit=key,
                recipients=len(u['group']['nodes']),amplitude=u['group']['amplitude'],exposure=u['exposure'],
                exposure_per_node=u['exposure']/u['N'],**{k:v for k,v in endpoint.items() if k!='hysteresis_sensitivity'})
            for threshold,value in endpoint['hysteresis_sensitivity'].items():row['hysteresis_'+threshold]=value
            rows.append(row)
        if u['frames']:
            roles_here=roles_by_key.get(key,[])
            filename=key+'.html';player(out/filename,u,origin,roles_here,zero=zero_for(u,roles_here));links.append((filename,u['origin']+' '+', '.join(roles_here)))
    csv_write(out/'outcomes.csv',rows);csv_write(out/'initial_conditions.csv',initial);csv_write(out/'performance.csv',performance)
    complete=bool(m['science_status']=='complete' and m.get('queue') and all(j['key'] in m['units'] for j in m['queue']))
    contrasts=[];size_contrasts=[];interactions=[];c=m['plan'];rng=np.random.default_rng(c['bootstrap_seed'])
    if complete:
        for N in c['sizes']:
            for memory in c['memory']:
                pairs=[(r+'_'+memory,'zero_'+memory) for r in c['roles'] if r!='zero']
                pairs +=[(a+'_'+memory,b+'_'+memory) for a,b in [('coverage10','random10'),('coverage25','coverage10'),('coverage10','coverage4'),('coverage10_equal_total','coverage4')]]
                for a,b in pairs:
                    for metric in ('persistent','penalized_time'):
                        result=paired(rows,N,a,b,metric,c['bootstrap_draws'],rng)
                        if result is not None:contrasts.append(result)
            for role in c['roles'][1:]:
                ad=paired(rows,N,role+'_adaptive','zero_adaptive','persistent',c['bootstrap_draws'],rng)
                fr=paired(rows,N,role+'_frozen','zero_frozen','persistent',c['bootstrap_draws'],rng)
                keys=sorted(ad['per_origin']);d=np.array([ad['per_origin'][k]-fr['per_origin'][k] for k in keys])
                boot=rng.choice(d,(c['bootstrap_draws'],len(d))).mean(axis=1)
                interactions.append(dict(N=N,role=role,definition='(treated-zero)adaptive minus (treated-zero)frozen',effect=float(d.mean()),interval=np.quantile(boot,[.025,.975]).tolist(),per_origin=dict(zip(keys,d.tolist()))))
        for role in c['roles'][1:]:
            for memory in c['memory']:
                vectors=[]
                for N in c['sizes']:
                    result=next(r for r in contrasts if r['N']==N and r['a']==role+'_'+memory and r['b']=='zero_'+memory and r['metric']=='persistent')
                    vectors.append(np.array(list(result['per_origin'].values())))
                a,b=vectors;draws=c['bootstrap_draws'];boot=rng.choice(b,(draws,len(b))).mean(axis=1)-rng.choice(a,(draws,len(a))).mean(axis=1)
                size_contrasts.append(dict(role=role,memory=memory,definition='N400 treatment-minus-control effect minus N40 effect; independent size cohorts',effect=float(b.mean()-a.mean()),interval=np.quantile(boot,[.025,.975]).tolist()))
    summary=[]
    for N in c['sizes']:
        for role in [r+'_'+m for m in c['memory'] for r in c['roles']]:
            rr=[r for r in rows if r['N']==N and r['role']==role]
            if rr:summary.append(dict(N=N,role=role,origins=len({r['origin'] for r in rr}),futures_completed=len(rr),successes=sum(r['persistent'] for r in rr),rate=float(np.mean([r['persistent'] for r in rr])),mean_exposure=float(np.mean([r['exposure'] for r in rr]))))
    csv_write(out/'summary.csv',summary)
    per_origin=[]
    for N in c['sizes']:
        for origin in sorted({r['origin'] for r in rows if r['N']==N}):
            for role in sorted({r['role'] for r in rows if r['origin']==origin}):
                rr=[r for r in rows if r['origin']==origin and r['role']==role]
                per_origin.append(dict(N=N,origin=origin,role=role,futures=len(rr),
                    successes=sum(r['persistent'] for r in rr),rate=float(np.mean([r['persistent'] for r in rr])),
                    final_private_mean=float(np.mean([r['final_private']/N for r in rr])),
                    final_public_mean=float(np.mean([r['final_public']/N for r in rr]))))
    csv_write(out/'per_origin.csv',per_origin)
    results=dict(complete=complete,notice=NOTICE,primary=c['primary'],contrasts=contrasts,memory_interactions=interactions,size_contrasts=size_contrasts,
        replay=m.get('replay',{}),inference_status='complete exploratory comparisons' if complete else 'withheld: incomplete frozen allocation',
        verified_physical_units=len(performance),role_rows=len(rows),players=len(links))
    _atomic_json(out/'analysis.json',results)
    table='<table><tr>'+''.join('<th>'+k+'</th>' for k in ('N','role','origins','futures_completed','successes','rate','mean_exposure'))+'</tr>'
    for r in summary:table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r.values())+'</tr>'
    table+='</table>'
    report='<!doctype html><meta charset="utf-8"><title>Modular 3 final extension</title><style>body{font:16px system-ui;margin:35px;max-width:1300px;line-height:1.5}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}a{color:#1465a0}</style><h1>N40 / N400 final extension</h1>'
    if m.get('software_only'):report+='<p><strong>SOFTWARE REHEARSAL ONLY — shortened horizon, no scientific conclusions.</strong></p>'
    report+='<p>'+html.escape(NOTICE)+'</p><p>Science: '+html.escape(m['science_status'])+'; '+results['inference_status']+'.</p>'
    report+='<p>Primary: '+html.escape(c['primary'])+'. N40 aliases share physical units; they are not additional evidence. All future0 audit players are included in queue order, never selected by outcome.</p>'+table
    report+='<h2>Primary comparisons</h2>'
    if not complete:report+='<p>Inferential comparisons withheld because the frozen allocation is incomplete.</p>'
    for r in contrasts:
        if r['a']=='coverage10_adaptive' and r['b']=='random10_adaptive' and r['metric']=='persistent':
            lo,hi=r['interval']
            report+=f"<p>N={r['N']}: coverage10% minus random10% = {100*r['effect']:.2f} percentage points; exploratory95% origin-bootstrap interval [{100*lo:.2f}, {100*hi:.2f}]; {r['origins']} independent origins.</p>"
            if r['degenerate']:report+='<p>The bootstrap distribution is degenerate. This does not establish equivalence or a zero uncertainty bound.</p>'
    report+='<h2>Computational scalability</h2><p>Measured end-to-end seconds and event throughput depend on treatment, memory mode and frame retention. Compare like-for-like units; these are measurements on this runtime, not a complexity theorem.</p>'
    for N in c['sizes']:
        for memory in c['memory']:
            rr=[r for r in performance if r['N']==N and r['memory']==memory and r['phase']=='audit' and r['frames']>0]
            if rr:
                report+=f"<p>N={N}, {memory}, {len(rr)} archived-frame units: median {np.median([r['seconds'] for r in rr]):.2f}s/unit; median {np.median([r['events_per_second'] for r in rr]):.1f} events/s; {sum(r['compressed_bytes'] for r in rr)/1e6:.2f}MB compressed unit archives.</p>"
    report+='<p><a href="per_origin.csv">Per-origin outcomes and final support</a></p>'
    report+='<p><a href="analysis.json">All prespecified contrasts, uncertainty, replay checks</a> · <a href="outcomes.csv">Exact outcomes and excursions</a> · <a href="performance.csv">Runtime / event / storage scalability</a> · <a href="initial_conditions.csv">Initial conditions</a></p><h2>Offline animations</h2><ul>'
    report+=''.join('<li><a href="'+html.escape(f)+'">'+html.escape(label)+'</a></li>' for f,label in links)+'</ul>'
    (out/'report.html').write_text(report)
    animation_result=export_animations(bundle)
    m['animation_players']=animation_result['players']
    report=report.replace('<h2>Offline animations</h2>',
        '<h2>Offline animations</h2><p id="ensemble-animations"><a href="animations/index.html">Browse all simulations in the established ensemble players</a></p>')
    (out/'report.html').write_text(report)
    exports={str(p.relative_to(bundle)):_sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    m['exports']=exports;m['export_status']='complete';_atomic_json(bundle/'manifest.json',m)
    print('Verified export:',len(performance),'physical units;',len(links),'players;',results['inference_status'],flush=True)
