"""Saved-data adapter for the final scale study's established animation player.

No simulation or statistical analysis runs here. Sparse archives feed the shared
renderer one matrix at a time, and each size/memory has its own ensemble page.
"""
import html, json
from pathlib import Path
import numpy as np
from src.control.scale_extension import read_gzip, _sha, _atomic_json, digest_data

_LAYOUTS={}

def fixed_layout(initial):
    """Initial-memory spring layout only; shared across all arms and frames."""
    import hashlib
    W=np.asarray(initial['W'],dtype=np.uint32)
    key=hashlib.sha256(W.tobytes()).hexdigest()
    if key not in _LAYOUTS:
        from src.visualization.network_layout import _seeded_spring_layout
        pos=_seeded_spring_layout(W,layout_seed=42,iterations=50)
        _LAYOUTS[key]=pos.tolist()
    return _LAYOUTS[key]

def _dense_frames(unit):
    """Saved sparse frames become the full-matrix frames the shared player encodes."""
    n=unit['N'];out=[]
    for f in unit['frames']:
        W=np.zeros((n,n),dtype=np.int64)
        for i,j,v in f['edges']:W[i,j]=v
        mask=np.zeros(n,dtype=bool)
        if f['controlled']:mask[np.asarray(f['controlled'],dtype=int)]=True
        out.append(dict(t=float(f['t']),W=W,x=np.asarray(f['x'],dtype=float).reshape(-1),
            y=np.asarray(f['y'],dtype=float).reshape(-1),controlled_mask=mask))
    return out

def player(path,u,origin,roles,zero=None):
    """The established animation-player page: treated arm beside its matched zero.

    Same page, controls and encodings as the previous notebooks' players; only
    the frames come from this study's sparse saved units. Positions reuse the
    study's frozen initial-memory layout, shared across time and matched arms.
    """
    from src.control.scale_extension_analysis import NOTICE
    from src.visualization.animation_player import _arms_at,_scene,assemble_payload,write_player_html
    from src.visualization.control_plots import _select_animation_indices
    n=u['N'];treated=_dense_frames(u);reference=_dense_frames(zero) if zero is not None else treated
    if zero is not None and ([f['t'] for f in reference]!=[f['t'] for f in treated]
        or any(not np.array_equal(reference[0][key],treated[0][key]) for key in ('W','x','y'))):
        raise ValueError('Matched arms must share time grids and initial state')
    first_treated={}
    for action in u['summaries']['recorder']['action_history']:
        if action.get('spending_rate',0)>0:
            for node in action['actuated_nodes']:first_treated.setdefault(str(node),float(action['decision_time']))
    if first_treated:treated[0]['first_treated']=first_treated
    role='+'.join(roles);nodes=[int(i) for i in u['group']['nodes']]
    label=f"{u['origin']} · {role} · future {u['future']}"
    def scenes_for(k):
        indices=_select_animation_indices(treated,k)
        # Without a matched zero the treatment is repeated, as payload_from_units
        # does, so the page renders one panel and flags the missing control.
        pairs=[('Matched zero','::zero',reference),(role,'::arm',treated)] if zero is not None \
            else [(role,'::arm',treated),(role,'::arm',treated)]
        arms=_arms_at(pairs,indices=indices,layouts={},
            layout=lambda fs:np.asarray(fixed_layout(origin['initial']),dtype=float))
        return [_scene(u['origin'],role,int(u['future']),nodes,label,arms,
            outcome=dict(persistent=bool(u['endpoint']['persistent']),relapsed=bool(u['endpoint']['relapsed']),
                valid=True,completion=None))]
    # The saved campaign players use the study encoding at both sizes.
    style='study'
    meta=dict(horizon=float(u['contract']['horizon']),release_deadline=float(u['contract']['release']),
        matched_zero_present=zero is not None,skipped=[],
        future_note='All '+str(len(u['frames']))+' recorded frames of this future0 audit unit are shown; '
            'nothing is re-simulated. Exact endpoint: '+(('persistent' if u['endpoint']['persistent'] else 'not persistent')
            +'; threshold '+str(u['endpoint']['threshold'])+'/'+str(n)+' in both layers; certification '
            +str(u['endpoint']['certification']))+'. '+NOTICE)
    payload,report=assemble_payload(scenes_for,n=n,style=style,meta=meta,
        max_frames=len(treated),min_frames=2,max_bytes=None)
    write_player_html(payload,path,title=f"Modular 3 — {u['origin']} · future {u['future']} · N={n}")

class _SavedFrames:
    """Expand one sparse frame at a time; N400 ensembles must not retain dense W copies."""
    def __init__(self,frames,n,first_treated,indices=None):
        self.frames=frames;self.n=n;self.first_treated=first_treated
        self.indices=list(range(len(frames))) if indices is None else list(indices)

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        f=self.frames[self.indices[index]]
        W=np.zeros((self.n,self.n),dtype=np.int64)
        for i,j,v in f['edges']:W[i,j]=v
        mask=np.zeros(self.n,dtype=bool);mask[np.asarray(f['controlled'],dtype=int)]=True
        result=dict(t=float(f['t']),W=W,x=np.asarray(f['x']).reshape(-1),
            y=np.asarray(f['y']).reshape(-1),controlled_mask=mask)
        if index==0:result['first_treated']=self.first_treated
        return result

def export_animations(bundle):
    """Write the established ensemble player per size/memory, using saved data only.

    Keep alias roles in the list, deduplicate their physical arms, preserve every
    frame in standalone pages, and cap only the explicitly labelled inline copy.
    """
    from src.control.scale_extension_analysis import NOTICE
    from src.visualization.animation_player import (assemble_payload,_scene,
        write_player_html,write_player_metadata,INLINE_MAX_BYTES,PayloadBudgetError)
    from src.visualization.control_plots import _select_animation_indices
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    if _sha(bundle/'origins.json.gz')!=m['origins_sha256']:raise ValueError('Origin hash mismatch')
    origins=read_gzip(bundle/'origins.json.gz')['origins']
    out=bundle/'analysis/animations';out.mkdir(parents=True,exist_ok=True)
    players={};all_skipped=[]
    for N in m['plan']['sizes']:
        for memory in m['plan']['memory']:
            cache={};comparisons=[];skipped=[]
            def load(key):
                if key not in cache:
                    info=m['units'].get(key)
                    if info is None:return None
                    path=bundle/info['path']
                    if _sha(path)!=info['sha256']:raise ValueError('Saved unit hash mismatch: '+key)
                    u=read_gzip(path)
                    if info['phase']!='audit' or not u['frames']:return None
                    if u['N']!=N or u['group']['memory']!=memory:
                        raise ValueError('Saved arm size/memory mismatch: '+key)
                    first_treated={}
                    for action in u['summaries']['recorder']['action_history']:
                        if action.get('spending_rate',0)>0:
                            for node in action['actuated_nodes']:
                                first_treated.setdefault(str(node),float(action['decision_time']))
                    frames=_SavedFrames(u['frames'],N,first_treated)
                    initial=frames[0]
                    cache[key]=dict(frames=frames,origin=u['origin'],future=u['future'],
                        endpoint=u['endpoint'],nodes=u['group']['nodes'],
                        times=[float(f['t']) for f in u['frames']],
                        initial=digest_data({k:initial[k].tolist() for k in ('W','x','y')}))
                return cache[key]
            for name,key in sorted(m.get('roles',{}).items()):
                origin,future_tag,role=name.split('__')
                policy,arm_memory=role.rsplit('_',1);future=int(future_tag[1:])
                if policy=='zero' or arm_memory!=memory or future!=0:continue
                if origins[origin]['initial']['N']!=N:continue
                arm=load(key);zero_key=m['roles'].get(f'{origin}__f{future}__zero_{memory}')
                zero=load(zero_key) if zero_key else None
                if arm is None or zero is None:
                    skipped.append(name+': missing completed matched zero or retained frames');continue
                if any(a['origin']!=origin or a['future']!=future for a in (zero,arm)):
                    raise ValueError('Saved arm origin/future mismatch: '+name)
                if zero['times']!=arm['times'] or zero['initial']!=arm['initial']:
                    raise ValueError('Matched arms must share time grids and initial state: '+name)
                comparisons.append((origin,future,role,key,zero_key))
            all_skipped.extend(skipped)
            if not comparisons:continue
            layouts={}
            def scenes_for(k):
                scenes=[]
                for origin,future,role,key,zero_key in comparisons:
                    arm=cache[key];frames=arm['frames']
                    # Selection only needs time and treatment masks, never dense matrices.
                    hints=[dict(t=f['t'],controlled_mask=[bool(f['controlled'])]) for f in frames.frames]
                    indices=_select_animation_indices(hints,k)
                    if origin not in layouts:layouts[origin]=np.asarray(fixed_layout(origins[origin]['initial']))
                    arms=[]
                    for label,unit_key in [('Matched zero',zero_key),(role,key)]:
                        saved=cache[unit_key]['frames']
                        arms.append(dict(key=unit_key+'::frames='+','.join(map(str,indices)),label=label,
                            frames=_SavedFrames(saved.frames,N,saved.first_treated,indices),
                            pos=layouts[origin],targeted=[]))
                    endpoint=arm['endpoint']
                    scenes.append(_scene(origin,role,future,arm['nodes'],f'{origin} · {role} · future {future}',arms,
                        outcome=dict(persistent=bool(endpoint['persistent']),relapsed=bool(endpoint['relapsed']),
                            valid=True,completion=endpoint.get('certification'))))
                return scenes
            meta=dict(horizon=m['plan']['horizon'],release_deadline=m['plan']['release'],
                matched_zero_present=True,skipped=skipped,
                source_manifest_sha256=_sha(bundle/'manifest.json'),
                future_note='All retained frames of every available future 0 comparison. Alias roles share stored trajectories. '+NOTICE)
            frames=max(len(v['frames']) for v in cache.values())
            payload,report=assemble_payload(scenes_for,n=N,style='study',meta=meta,max_frames=frames,min_frames=2,max_bytes=None)
            folder=out/f'N{N}_{memory}';title=f'Modular 3 final scale — N{N} · {memory}'
            page=write_player_html(payload,folder/'player.html',title=title)
            write_player_metadata(payload,page,source=str(bundle),report=report)
            inline=page
            if report['bytes']>INLINE_MAX_BYTES:
                try:
                    preview_meta=dict(meta,future_note='Notebook preview of saved future 0 comparisons. The standalone player preserves every available comparison and frame. '+NOTICE)
                    small,_=assemble_payload(scenes_for,n=N,style='study',meta=preview_meta,max_frames=min(24,frames),
                        min_frames=min(10,frames),max_bytes=INLINE_MAX_BYTES)
                    inline=write_player_html(small,folder/'inline_player.html',title=title+' — notebook preview')
                except PayloadBudgetError:inline=None
            players[f'N{N}_{memory}']=dict(path=str(page.relative_to(bundle)),
                inline=str(inline.relative_to(bundle)) if inline else None,N=N,memory=memory,
                comparisons=len(payload['comparisons']),stored_arms=len(payload['arms']),frames=frames)
            print(f'Animation N{N} {memory}: {len(payload["comparisons"])} comparisons, {len(payload["arms"])} stored arms',flush=True)
            del payload,cache
    result=dict(schema='scale-animations-v1',players=players,skipped=all_skipped,
        renderer_sha256=_sha(Path(__file__).with_name('animation_player.py')))
    _atomic_json(out/'animation_export.json',result)
    links=''.join('<li><a href="'+html.escape(str(Path(p['path']).relative_to('analysis/animations')))+'">'
        +html.escape(name)+f' — {p["comparisons"]} saved comparisons</a></li>' for name,p in players.items())
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Final scale animations</title>'
        '<style>body{font:16px system-ui;margin:35px;line-height:1.6}</style><h1>N40 / N400 saved animations</h1>'
        '<p>Same ensemble player as the previous notebooks. All available future 0 comparisons; '
        'matched controls have the same origin, future and memory. No simulations rerun.</p><ul>'+links+'</ul>'
        +('<p>Unavailable comparisons: '+html.escape('; '.join(all_skipped))+'</p>' if all_skipped else ''))
    return result

def rebuild_animations(bundle):
    """Repair derived animations without recomputing tables, inference or science."""
    bundle=Path(bundle);result=export_animations(bundle)
    m=json.loads((bundle/'manifest.json').read_text())
    origins=read_gzip(bundle/'origins.json.gz')['origins'];roles_by_key={}
    for name,key in m.get('roles',{}).items():roles_by_key.setdefault(key,[]).append(name.split('__')[-1])
    for key,info in m['units'].items():
        if info['phase']!='audit':continue
        if _sha(bundle/info['path'])!=info['sha256']:raise ValueError('Saved unit hash mismatch: '+key)
        u=read_gzip(bundle/info['path'])
        if not u['frames']:continue
        roles=roles_by_key.get(key,[])
        name=f'{u["origin"]}__f{u["future"]}__zero_{u["group"]["memory"]}'
        zkey=m.get('roles',{}).get(name);zero=None
        if zkey and zkey!=key and zkey in m['units']:
            zinfo=m['units'][zkey]
            if _sha(bundle/zinfo['path'])!=zinfo['sha256']:raise ValueError('Saved unit hash mismatch: '+zkey)
            zero=read_gzip(bundle/zinfo['path'])
            if not zero['frames']:zero=None
        player(bundle/'analysis'/f'{key}.html',u,origins[u['origin']],roles,zero)
    report=bundle/'analysis/report.html'
    if report.exists() and 'id="ensemble-animations"' not in report.read_text():
        text=report.read_text();link='<p id="ensemble-animations"><a href="animations/index.html">Browse all simulations in the established ensemble players (N40 / N400, adaptive / frozen)</a></p>'
        at=text.find('</h1>')
        report.write_text(text[:at+5]+link+text[at+5:] if at>=0 else link+text)
    m['animation_players']=result['players']
    m['exports']={str(p.relative_to(bundle)):_sha(p) for p in sorted((bundle/'analysis').rglob('*')) if p.is_file()}
    _atomic_json(bundle/'manifest.json',m)
    return result

