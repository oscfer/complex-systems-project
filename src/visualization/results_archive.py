"""One offline viewer for historical and final results, built only from saved data.

Each physical arm is verified, encoded once, compressed, and decoded on demand
in the browser. All saved frames and scalar audit outcomes remain available.
"""
from __future__ import annotations
import base64, csv, gzip, hashlib, html, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from .animation_player import _encode_arm, _encode_layout, _serialise, write_player_html
from .network_layout import _seeded_spring_layout
from .result_labels import LABEL_KEY, STUDIES, policy_label

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())

def _resolve(bundle,relative):
    """Resolve split Google Drive exports without importing scientific analysis."""
    bundle=Path(bundle).resolve();parts=[bundle]
    if bundle.parent.name=='results_tiered':parts+=sorted(bundle.parent.parent.parent.glob('*/results_tiered/'+bundle.name))
    hits=[]
    for part in dict.fromkeys(parts):
        p=(part/relative).resolve()
        if not p.is_relative_to(part.resolve()):raise ValueError('Unit path escapes bundle')
        if p.is_file():hits.append(p)
    if not hits:raise FileNotFoundError(relative+' missing from '+bundle.name)
    if len(hits)>1 and len({_sha(p) for p in hits})!=1:raise ValueError('Conflicting split bundle: '+relative)
    return hits[0]

def discover_sources(roots):
    found={};excluded=[]
    for root in roots:
        root=Path(root).expanduser()
        manifests=[root/'manifest.json'] if (root/'manifest.json').is_file() else sorted(root.rglob('manifest.json'))
        for p in manifests:
            if p.parent.name in ('study','analysis'):continue  # mirrors, not new evidence
            m=_read(p);name=p.parent.name
            if not name.startswith(('transformation_','campaign_','sparse_','nucleation_','sequential_','response_','final_scale_','main_study_')):continue
            status=m.get('science_status',m.get('status'))
            software_only=m.get('software_only') or m.get('config',{}).get('smoke') or m.get('plan',{}).get('software_only')
            if status!='complete' or software_only:
                excluded.append(dict(bundle=name,reason='Incomplete or software-only run',status=status));continue
            digest=_sha(p)
            if name in found:
                if found[name]['sha256']!=digest:raise ValueError('Conflicting manifests for '+name)
                continue
            found[name]=dict(bundle=p.parent,manifest=m,sha256=digest)
    return [found[n] for n in sorted(found,reverse=True)],excluded

def _study_title(name):
    curated=STUDIES.get(name.rsplit('_',1)[-1])
    if curated:
        date=re.search(r'20\d{6}',name)[0]
        return curated[0]+f' · {date[:4]}-{date[4:6]}-{date[6:]}'
    for prefix,label in [('final_scale','Scale comparison'),('response_low','Response-guided'),('sequential_low','Sequential weak recipients'),('nucleation','Nucleation and bridges'),('sparse_validation','Independent validation'),('sparse_discovery','Sparse discovery'),('campaign','Recipient search development'),('transformation','Early development'),('main_study','Main study')]:
        if name.startswith(prefix):
            stamp=re.search(r'(20\d{6})_(\d{6})',name)
            date=stamp[1] if stamp else ''
            when=f'{date[:4]}-{date[4:6]}-{date[6:]}' if date else ''
            extra=' '+stamp[2][:2]+':'+stamp[2][2:4] if prefix=='transformation' and stamp else ''
            return label+' · '+when+extra
    return name.replace('_',' ')

def _role_rows(m):
    if m.get('schema')=='final-scale-v1':
        for name,key in m['roles'].items():
            origin,future,role=name.split('__');yield key,origin,int(future[1:]),role
    elif m.get('audit_roles'):
        for name,key in m['audit_roles'].items():
            origin,role,future=name.split('__');yield key,origin,int(future.removeprefix('future')),role
    else:
        for key,info in m.get('actual',m.get('units',{})).items():
            if info.get('kind')=='timing':continue
            origin=info.get('base_origin',info.get('origin_id',key.split('__')[0]))
            role=next((r for r in info.get('roles',[]) if r!='representative'),key.split('__')[-1])
            yield key,origin,int(info.get('future',0)),role

def build_results_archive(roots,output,*,progress=True,force=False):
    roots=[roots] if isinstance(roots,(str,Path)) else list(roots)
    output=Path(output);sources,excluded=discover_sources(roots)
    if not sources:raise ValueError('No completed saved result bundles found. Check RESULTS_ROOTS.')
    source_key=hashlib.sha256(_serialise(dict(sources=[(s['bundle'].name,s['sha256']) for s in sources],renderer=_sha(Path(__file__).with_name('animation_player.py')),adapter=_sha(__file__),labels=_sha(Path(__file__).with_name('result_labels.py')))).encode()).hexdigest()
    inventory_path=output.with_suffix('.inventory.json')
    if not force and output.exists() and inventory_path.exists():
        previous=_read(inventory_path)
        if previous.get('source_key')==source_key and previous.get('player_sha256')==_sha(output):return dict(page=output,inventory=previous)
    payload=dict(schema='modular-results-archive-v1',N=40,style='study',comparisons=[],arms=[],layouts=[],studies=[],records=[],label_key=LABEL_KEY,dropped=[],skipped=[],horizon=None,release_deadline=None)
    from .animation_player import _coolwarm_lut
    payload['coolwarm']=_coolwarm_lut()
    arm_ids={};layout_ids={};missing=[];frame_total=0;inventory=[]
    for study_index,source in enumerate(sources):
        b=source['bundle'];m=source['manifest'];sid='study-'+str(study_index)
        plan=m.get('plan',m.get('identity',{}).get('plan',m.get('contract',{})))
        label=_study_title(b.name);grouped=defaultdict(list)
        for key,origin,future,role in _role_rows(m):grouped[key].append((origin,future,role))
        group_labels={role:f'Saved group {i+1:02d}' for i,role in enumerate(sorted({role for aliases in grouped.values() for _,_,role in aliases if re.fullmatch(r'\d+(?:_\d+)+',role)}))}
        members=[];before=len(payload['comparisons']);rows_before=len(payload['records']);saved_frames=0
        for unit_index,(key,aliases) in enumerate(sorted(grouped.items())):
            info=m.get('actual',m.get('units',{})).get(key)
            if info is None:missing.append(dict(study=label,unit=key,reason='Role references missing completed unit'));continue
            p=_resolve(b,info['path'])
            if _sha(p)!=info['sha256']:raise ValueError('Saved unit checksum mismatch: '+str(p))
            u=_read(p);scale=m.get('schema')=='final-scale-v1'
            alias_package=u.get('package',u.get('unit',{}).get('package','default'))
            resolved_sha=info['sha256'];reference_seen={key}
            while 'reference_unit' in u:
                reference=u['reference_unit']
                if reference in reference_seen:raise ValueError('Cyclic unit reference: '+key)
                reference_seen.add(reference)
                ref_info=m.get('actual',m.get('units',{}))[reference]
                ref_path=_resolve(b,ref_info['path'])
                if _sha(ref_path)!=ref_info['sha256']:raise ValueError('Referenced unit checksum mismatch: '+reference)
                u=_read(ref_path);resolved_sha=ref_info['sha256']
            frames=u.get('frames',[]) if scale else u.get('network_frames',[])
            if scale:
                N=u['N'];memory=u['group']['memory'];visibility='partial';ep=u['endpoint'];valid=True
                nodes=u['group']['nodes'];exposure=u['exposure'];completion=ep['certification'];persistent=ep['persistent'];relapsed=ep['relapsed'];held=completion is not None
                target=u['contract'];horizon=target['horizon'];release=target['release'];hold=target['hold'];threshold=.75
            else:
                target=u['target_definition'];N=int(u.get('final_state',{}).get('N',len(frames[0]['W']) if frames else m.get('identity',{}).get('profile',{}).get('N',40)))
                memory=u.get('memory_mode',info.get('memory','adaptive'));visibility='complete' if u.get('visibility')=='complete_public' else 'partial'
                nodes=u.get('recipient_nodes',sorted({i for f in frames for i,v in enumerate(f['controlled_mask']) if v}))
                exposure=u['expenditure'];completion=u.get('completion_time');valid=bool(u.get('is_valid') and u.get('is_eligible'));persistent=bool(u.get('success_through_horizon'));relapsed=bool(u.get('relapsed'));held=bool(u.get('success'))
                horizon=target['horizon'];release=target['release_deadline'];hold=target['holding_duration'];threshold=(1+target['threshold'])/2
            frame_id=None;layout_id=None;initial_digest=None;times=[]
            if frames:
                first_treated={}
                for action in u.get('summaries',{}).get('recorder',{}).get('action_history',[]):
                    if action.get('spending_rate',0)>0:
                        for node in action.get('actuated_nodes',[]):first_treated.setdefault(str(node),float(action['decision_time']))
                if scale:
                    from .scale_animation import _SavedFrames
                    fs=_SavedFrames(frames,N,first_treated)
                else:
                    fs=frames
                    if first_treated:fs[0]['first_treated']=first_treated
                f0=fs[0];initial_digest=hashlib.sha256(_serialise({k:np.asarray(f0[k]).tolist() for k in ('W','x','y')}).encode()).hexdigest()
                layout_key=(N,initial_digest)
                if layout_key not in layout_ids:
                    layout_ids[layout_key]=len(payload['layouts']);payload['layouts'].append(_encode_layout(_seeded_spring_layout(np.asarray(f0['W']),iterations=50)))
                layout_id=layout_ids[layout_key]
                arm_key=(resolved_sha,N)
                if arm_key not in arm_ids:
                    record=dict(label='Saved trajectory',targeted=[],**_encode_arm(fs,N))
                    compressed=base64.b64encode(gzip.compress(_serialise(record).encode(),mtime=0)).decode()
                    arm_ids[arm_key]=len(payload['arms']);payload['arms'].append(dict(N=N,gzip=compressed))
                    frame_total+=len(frames)
                frame_id=arm_ids[arm_key];times=[float(f['t']) for f in frames];saved_frames+=len(frames)
            package=alias_package
            for origin,future,role in aliases:
                readable=group_labels.get(role,policy_label(role))
                if scale:
                    base=role.rsplit('_',1)[0]
                    if base in ('random4','coverage4'):readable+=' · fixed 4'
                    elif base in ('random10','coverage10','coverage25'):readable+=' · '+('25%' if base=='coverage25' else '10%')+' of nodes'
                if not m.get('audit_roles') and not scale and package!='default':readable+=' · '+{'original':'original access limits','wider':'wider access','broad_diagnostic':'full-access diagnostic'}.get(package,package.replace('_',' '))
                row=dict(study=sid,N=N,origin=origin,future=future,role=role,policy_label=readable,memory=memory,visibility=visibility,package=package,physical_unit=key,physical_sha256=resolved_sha,unit_sha256=info['sha256'],count=len(nodes),exposure=exposure,completion=completion,persistent=persistent,relapsed=relapsed,hold=held,valid=valid,frames=len(frames),horizon=horizon,release_deadline=release,holding_duration=hold,threshold=threshold)
                payload['records'].append(row)
                members.append(dict(**row,frame_id=frame_id,layout_id=layout_id,nodes=nodes,initial_digest=initial_digest,times=times))
        networks={}
        for N in sorted({r['N'] for r in members}):
            for i,origin in enumerate(sorted({r['origin'] for r in members if r['N']==N})):networks[(N,origin)]=f'Network {i+1:02d}'
        zeros={}
        for r in members:
            if r['role'].startswith('zero') and r['frame_id'] is not None:zeros[(r['N'],r['origin'],r['future'],r['memory'],r['visibility'],r['package'])]=r
        for r in members:
            if r['role'].startswith('zero') or r['frame_id'] is None:continue
            zero=zeros.get((r['N'],r['origin'],r['future'],r['memory'],r['visibility'],r['package']))
            # Do not fabricate a matched control for an unpaired development package.
            matched=zero is not None
            if matched and (zero['times']!=r['times'] or zero['initial_digest']!=r['initial_digest']):
                missing.append(dict(study=label,unit=r['physical_unit'],reason='Unsynchronized or different initial state; displayed as unpaired'));matched=False
            z=zero if matched else r
            payload['comparisons'].append(dict(study=sid,study_label=label,N=r['N'],style='study',origin=r['origin'],network_label=networks[(r['N'],r['origin'])],future=r['future'],role=r['role'],policy_label=r['policy_label'],nodes=r['nodes'],count=r['count'],memory=r['memory'],visibility=r['visibility'],label=r['policy_label'],zero=z['frame_id'],arm=r['frame_id'],layouts=[z['layout_id'],r['layout_id']],matched=matched,horizon=r['horizon'],release_deadline=r['release_deadline'],holding_duration=r['holding_duration'],threshold=r['threshold'],exposure=r['exposure'],outcome=dict(persistent=r['persistent'],relapsed=r['relapsed'],hold=r['hold'],valid=r['valid'],completion=r['completion'])))
        retained_reason=STUDIES.get(b.name.rsplit('_',1)[-1],('', 'Completed scientific comparison; retained independently of outcome.'))[1]
        item=dict(id=sid,label=label,bundle=b.name,manifest_sha256=source['sha256'],status='complete',retained_reason=retained_reason,comparisons=len(payload['comparisons'])-before,outcome_rows=len(payload['records'])-rows_before,physical_units=len(grouped),saved_frames=saved_frames,primary=str(plan.get('primary',plan.get('primary_comparison','Development / diagnostic comparisons'))))
        inventory.append(item);payload['studies'].append(item)
        if progress:print(label,':',item['comparisons'],'animations;',item['outcome_rows'],'outcomes',flush=True)
    if not payload['comparisons']:raise ValueError('No completed saved trajectories have retained animation frames')
    # Newest experiment first; selection is independent of outcomes.
    rank={s['id']:i for i,s in enumerate(sorted(payload['studies'],key=lambda s:s['bundle'].split('_20')[-1],reverse=True))}
    payload['comparisons'].sort(key=lambda c:(rank[c['study']],c['N'],c['memory'],c['origin'],c['future'],c['role']))
    payload['studies'].sort(key=lambda s:rank[s['id']])
    payload['archive_notes']=dict(excluded=excluded,issues=missing,frames=frame_total)
    page=write_player_html(payload,output,title='Tiered-population · saved results')
    report=dict(schema=payload['schema'],source_key=source_key,player_sha256=_sha(page),studies=inventory,comparisons=len(payload['comparisons']),outcome_rows=len(payload['records']),physical_arms=len(payload['arms']),retained_frames=frame_total,excluded=excluded,issues=missing,frame_policy='Every retained frame; no budget truncation, no simulation')
    inventory_path.write_text(json.dumps(report,indent=2)+'\n')
    with output.with_suffix('.outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(payload['records'][0]));writer.writeheader();writer.writerows(payload['records'])
    return dict(page=page,inventory=report)

def show_results_archive(page):
    """Embed one complete viewer without storing megabytes of outputs in notebooks."""
    from IPython.display import HTML,JSON,display
    page=Path(page).resolve()
    import sys
    if 'google.colab' in sys.modules:
        from google.colab import output
        def chunk(offset):
            with page.open('rb') as f:
                f.seek(int(offset))
                return JSON({'chunk':base64.b64encode(f.read(512*1024)).decode()})
        output.register_callback('modular3.results_chunk',chunk)
        script="""<div id="archive-loader">Loading saved results…</div><script>
(async function(){try{let parts=[],offset=0;while(true){let r=await google.colab.kernel.invokeFunction('modular3.results_chunk',[offset],{});let s=r.data['application/json'].chunk;let bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));if(!bytes.length)break;parts.push(bytes);offset+=bytes.length;document.getElementById('archive-loader').textContent='Loading saved results… '+Math.round(100*offset/__SIZE__)+'%';}
let url=URL.createObjectURL(new Blob(parts,{type:'text/html'}));let box=document.getElementById('archive-loader');box.textContent='';let link=document.createElement('a');link.href=url;link.download='tiered_results.html';link.textContent='Save the complete offline viewer';box.appendChild(link);let frame=document.createElement('iframe');frame.src=url;frame.style='width:100%;height:960px;border:0';frame.allow='fullscreen';frame.setAttribute('allowfullscreen','');box.appendChild(frame);}catch(e){document.getElementById('archive-loader').textContent='Could not load viewer: '+e.message+'. Rerun this cell with the runtime connected.';}})();</script>"""
        display(HTML(script.replace('__SIZE__',str(page.stat().st_size))))
    else:
        # The compact HTML lives on the Jupyter server; the notebook stores its URL only.
        try:
            import jupyter_server.serverapp as serverapp
            servers=list(serverapp.list_running_servers())
            server=next(s for s in servers if page.is_relative_to(Path(s['root_dir']).resolve()))
            from urllib.parse import quote
            url=server.get('base_url','/')+'files/'+quote(str(page.relative_to(Path(server['root_dir']).resolve())))
            display(HTML('<iframe title="All saved Tiered-population results" src="'+html.escape(url,quote=True)+'" style="width:100%;height:960px;border:0" allow="fullscreen" allowfullscreen></iframe>'))
        except (ImportError,KeyError,StopIteration):
            display(HTML('<a href="'+html.escape(page.as_uri(),quote=True)+'">Open the complete offline results viewer</a>'))
    return page
