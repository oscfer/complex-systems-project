"""Read-only ensemble curation for frozen sparse batches and main studies.

Never imports a simulator. Raw bundles remain immutable; exports go elsewhere.
Repeated policy roles retain provenance but do not inflate physical-unit counts.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def bundle_file(bundle, relative):
    """Resolve a file across sibling parts of an extracted results download."""
    bundle = Path(bundle).resolve()
    parts = [bundle]
    if bundle.parent.name == 'results_tiered':
        parts.extend(sorted(bundle.parent.parent.parent.glob(
            f'*/results_tiered/{bundle.name}')))
    matches = []
    for part in dict.fromkeys(parts):
        path = (part / relative).resolve()
        if not path.is_relative_to(part.resolve()):
            raise ValueError('Unit path escapes bundle')
        if path.is_file():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f'{relative} absent from all parts of {bundle.name}')
    if len(matches) > 1:
        checksums = {hashlib.sha256(p.read_bytes()).hexdigest() for p in matches}
        if len(checksums) > 1:
            raise ValueError(f'Conflicting split-bundle copies: {relative}')
    return matches[0]


def discover_bundles(root):
    root=Path(root)
    result=[]
    if not root.exists():return result
    for p in sorted(root.rglob('manifest.json')):
        # Only batch-level manifests; study-format mirrors are not new evidence.
        if p.parent.name in ('study','analysis'):continue
        try:m=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if 'audit_roles' in m and 'actual' in m:
            result.append(dict(path=str(p.parent),status=m.get('status'),
                N=m.get('identity',{}).get('profile',{}).get('N'),
                version=m.get('identity',{}).get('plan',{}).get('version'),
                units=len(m['actual']),audit_roles=len(m['audit_roles'])))
    return result


def load_raw(bundle, key):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text());info=m['actual'][key]
    path=bundle_file(bundle,info['path'])
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError(f'Checksum mismatch: {key}')
    return json.loads(raw)


def analyze_bundle(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    if 'audit_roles' not in m:raise ValueError('Choose a sparse discovery/validation or main-study bundle')
    plan=m['identity']['plan'];profile=m['identity']['profile'];N=profile['N']
    allocation=m.get('allocation') or {}
    origins=allocation.get('origins',plan.get('origins',list(plan.get('origin_seeds',{}))))
    futures=allocation.get('audit_futures',plan.get('audit_futures',0))
    historical={}
    try:
        char=bundle_file(bundle,'group_characteristics.json')
    except FileNotFoundError:
        char=None
    if char is not None:
        for x in json.loads(char.read_text()):
            if x['phase']=='audit':historical[(x['origin'],tuple(x['nodes']))]=x
    unit_rows={};initial=dict(m.get('origin_info',{}))
    # Verify each physical audit unit once. Search/timing outcomes are separate.
    for key in sorted(set(m['audit_roles'].values())):
        u=load_raw(bundle,key);info=m['actual'][key]
        if info['kind']!='audit':raise ValueError('Non-audit unit assigned an audit role')
        base=info['base_origin'];nodes=tuple(info['recipient_nodes'])
        if sorted(u['recipient_nodes'])!=list(nodes):raise ValueError('Manifest membership differs from raw unit')
        valid=bool(u['is_valid'] and u['is_eligible'])
        unit_rows[key]=dict(physical_unit=key,origin=base,future=info['future'],nodes=list(nodes),count=len(nodes),
            valid=valid,hold=bool(u['success']),persistent=bool(u['success_through_horizon']),
            relapsed=bool(u['relapsed']),completion=u['completion_time'],
            penalized_time=u['completion_time'] if u['success_through_horizon'] else 320.,exposure=u['expenditure'])
        if base not in initial:
            metrics=u['target_definition']['initial_metrics']
            initial[base]=dict(N=N,private_blue=(1-metrics['m_x_0'])/2,public_blue=(1-metrics['m_y_0'])/2)
    roles=[]
    for role,key in m['audit_roles'].items():
        base,name,future=role.split('__');r=dict(unit_rows[key],role=name)
        if base!=r['origin'] or int(future.removeprefix('future'))!=r['future']:
            raise ValueError('Role does not match physical origin/future')
        g=m.get('groups',{}).get(base,{}).get(name) or historical.get((base,tuple(r['nodes'])),{})
        for prefix,field in [('true','full_network_initial'),('observed','observed_initial')]:
            for metric,value in g.get(field,{}).items():r[f'{prefix}_{metric}']=value
        roles.append(r)
    frame=pd.DataFrame(roles)
    expected={}
    unavailable=[]
    if 'groups' in m:
        for base,groups in m['groups'].items():
            for role,g in groups.items():
                if g['available']:
                    expected.setdefault(role,[]).extend((base,f) for f in range(futures))
                else:unavailable.append(dict(origin=base,role=role,reason='insufficient positive-score candidates below ceiling'))
    else:
        for label,x in m.get('strata',{}).items():
            base,role=label.split('__')
            if x['available']:expected.setdefault(role,[]).extend((base,f) for f in range(futures))
            else:unavailable.append(dict(origin=base,role=role,reason='infeasible frozen stratum'))
        for base,groups in m.get('benchmark_groups',{}).items():
            for role in groups:expected.setdefault(role,[]).extend((base,f) for f in range(futures))
        if not m.get('benchmark_groups'):
            for size in plan['sizes']:
                expected[f'random{size}']=[(b,f) for b in origins for f in range(futures)]
        expected['zero']=[(b,f) for b in origins for f in range(futures)]
    summaries=[]
    for role,pairs in expected.items():
        rs=frame[frame.role==role] if not frame.empty else frame
        valid=rs[rs.valid] if not rs.empty else rs
        actual=set(zip(rs.origin,rs.future)) if not rs.empty else set()
        complete=(actual==set(pairs) and len(valid)==len(pairs))
        summaries.append(dict(role=role,expected=len(pairs),observed=len(rs),valid=len(valid),
            origins=len({b for b,f in pairs}),hold=int(valid.hold.sum()) if len(valid) else 0,
            persistent=int(valid.persistent.sum()) if len(valid) else 0,
            persistent_rate=float(valid.persistent.mean()) if len(valid) else None,
            mean_penalized_time=float(valid.penalized_time.mean()) if len(valid) else None,
            complete=complete))
    summary=pd.DataFrame(summaries)
    complete=(m['status']=='complete' and m.get('selection_frozen',False) and
              (not 'groups' in m or set(m['groups'])==set(origins)) and
              len(summary)>0 and bool(summary.complete.all()))
    primary=dict(status='withheld: incomplete allocation',effect=None,interval=None)
    main=m.get('schema_version')=='main-study-v1'
    a='audience_coverage10' if main else 'n10_q1';b='random10'
    if complete and a in expected and b in expected:
        wide=frame[frame.role.isin([a,b])].groupby(['origin','role']).persistent.mean().unstack()
        if len(wide)==len(origins) and not wide.isna().any().any():
            d=(wide[a]-wide[b]).to_numpy();rng=np.random.default_rng(271828)
            ci=np.quantile(rng.choice(d,(10000,len(d))).mean(axis=1),[.025,.975])
            primary=dict(status='complete',comparison=f'{a} minus {b}',networks=len(d),
                effect=float(d.mean()),interval=ci.tolist(),origin_differences=dict(zip(wide.index,d.tolist())),
                bootstrap='paired networks, 10000 draws, seed 271828')
    groups=frame[frame.valid].groupby(['origin','role'],as_index=False).agg(
        persistent_rate=('persistent','mean'),futures=('future','count')) if not frame.empty else pd.DataFrame()
    if not groups.empty:
        features=[c for c in frame if c.startswith(('true_','observed_'))]+['count','nodes','exposure']
        groups=groups.merge(frame.groupby(['origin','role'],as_index=False)[features].first(),on=['origin','role'])
    return dict(bundle=str(bundle),manifest=m,N=N,status=m['status'],complete=complete,
        roles=frame,summary=summary,groups=groups,initial=pd.DataFrame([dict(origin=k,**v) for k,v in initial.items()]),
        unavailable=pd.DataFrame(unavailable),physical_audit_units=len(unit_rows),primary=primary)


def export_analysis(data, output_dir):
    out=Path(output_dir);source=Path(data['bundle']).resolve()
    if out.resolve()==source or out.resolve().is_relative_to(source):
        raise ValueError('Use a separate derived-output folder; keep raw bundle immutable')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('roles','summary','groups','initial','unavailable'):data[name].to_csv(out/f'{name}.csv',index=False)
    meta=dict(status=data['status'],complete=data['complete'],N=data['N'],
        source_bundle=str(source),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        physical_audit_units=data['physical_audit_units'],role_rows=len(data['roles']),primary=data['primary'],
        interpretation='Repeated futures cluster by network. Partial observed rates are descriptive, not completed inference. Feature associations are exploratory.')
    (out/'analysis.json').write_text(json.dumps(meta,indent=2)+'\n')
    return out
