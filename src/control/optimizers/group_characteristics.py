"""Retrospective descriptors, never inputs to discovery decisions.

Features describe association, not causal mechanisms. Observed and full-state
views remain separate; missing public opinions are excluded with coverage shown.
"""
import numpy as np


def describe_group(W, public, nodes, *, groups=None, private=None):
    W=np.asarray(W,dtype=float);public=np.asarray(public,dtype=float).reshape(-1)
    N=len(public);nodes=np.asarray(sorted(set(nodes)),dtype=int)
    if W.shape!=(N,N) or not len(nodes) or np.any(nodes<0) or np.any(nodes>=N):
        raise ValueError('Invalid group/network dimensions')
    totals=W.sum(axis=1,keepdims=True)
    A=np.divide(W,totals,out=np.zeros_like(W),where=totals>0)
    received=A.sum(axis=0)
    untreated=np.setdiff1d(np.arange(N),nodes)
    # Directed audience: rows i listening to at least one recipient j.
    audience=untreated[np.any(W[np.ix_(untreated,nodes)]>0,axis=1)]
    reaches=(W[np.ix_(untreated,nodes)]>0).sum(axis=0)
    finite=np.isfinite(public[nodes]);neighbor_finite=np.isfinite(public[audience])
    result={'recipient_count':len(nodes),'received_attention_mean':float(received[nodes].mean()),
        'received_attention_max':float(received[nodes].max()),
        'received_attention_sum':float(received[nodes].sum()),
        'external_audience_count':len(audience),
        'audience_overlap_fraction':float(1-len(audience)/reaches.sum()) if reaches.sum() else None,
        'within_group_attention_sum':float(A[np.ix_(nodes,nodes)].sum()),
        'untreated_attention_to_group_mean':float(A[np.ix_(untreated,nodes)].sum(axis=1).mean()) if len(untreated) else None,
        'recipient_public_observed_count':int(finite.sum()),
        'recipient_public_blue_fraction':float(np.mean(public[nodes][finite]<0)) if finite.any() else None,
        'audience_public_observed_count':int(neighbor_finite.sum()),
        'audience_public_blue_fraction':float(np.mean(public[audience][neighbor_finite]<0)) if neighbor_finite.any() else None}
    if groups is not None:
        groups=np.asarray(groups)
        result['recipient_group_count']=len(np.unique(groups[nodes]))
        result['audience_group_count']=len(np.unique(groups[audience]))
    if private is not None:
        private=np.asarray(private).reshape(-1)
        result['recipient_private_blue_fraction']=float(np.mean(private[nodes]<0))
    return result


def observed_group(observation,nodes):
    N=len(observation.public_states);W=np.zeros((N,N))
    for (i,j),count in observation.observed_edges.items():W[i,j]=count
    public=observation.public_states[:,0].copy()
    public[~observation.panel_mask]=np.nan
    return describe_group(W,public,nodes,groups=observation.group_labels)


def state_group(state,nodes):
    return describe_group(state.W,state.y[:,0],nodes,groups=state.opportunity_groups,private=state.x[:,0])


def export_characteristics(bundle):
    """Analyze every completed group after selection; never modify selection."""
    import json
    from pathlib import Path
    from collections import defaultdict
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.control.optimizers.campaign import _origin_state,_case_problem,reconstruct_t0_observation,_atomic_json
    from src.control.transformation_study import load_transformation_contract
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    seeds=m.get('identity',{}).get('plan',{}).get('origin_seeds') or {r['origin_id']:r['seed'] for r in load_transformation_contract()['development']}
    views={};aggregates=defaultdict(list);dynamics=[]
    for info in m['actual'].values():
        nodes=tuple(info['recipient_nodes'])
        if not nodes:continue
        base=info['base_origin']
        if base not in views:
            state=_origin_state(LIGHT_TIERED,seeds[base])
            p=_case_problem(LIGHT_TIERED,seeds[base],'broad_diagnostic',-1,base)
            obs=reconstruct_t0_observation(LIGHT_TIERED,state,seeds[base],p.accessible_roster)
            views[base]=(state,obs)
        u=json.loads((bundle/info['path']).read_text())
        aggregates[(base,nodes,info['kind'])].append(u)
        frames=u.get('network_frames',[])
        if frames:
            chosen={min(range(len(frames)),key=lambda i:abs(float(frames[i]['t'])-t)) for t in (0,20,120,320)}
            for i in sorted(chosen):
                frame=frames[i];W=np.asarray(frame['W']);y=np.asarray(frame['y']).reshape(-1)
                outside=np.setdiff1d(np.arange(len(y)),nodes)
                feature=describe_group(W,y,nodes,private=frame['x'])
                dynamics.append({'origin':base,'nodes':list(nodes),'phase':info['kind'],
                    'future':info['future'],'time':float(frame['t']),
                    'no_relapse_success':u['success_through_horizon'],
                    'recipient_attention_sum':feature['received_attention_sum'],
                    'recipient_public_blue_fraction':feature['recipient_public_blue_fraction'],
                    'recipient_private_blue_fraction':feature['recipient_private_blue_fraction'],
                    'untreated_public_blue_fraction':float(np.mean(y[outside]<0)),
                    'untreated_private_blue_fraction':float(np.mean(np.asarray(frame['x']).reshape(-1)[outside]<0))})
    rows=[]
    for (base,nodes,phase),units in aggregates.items():
        state,obs=views[base]
        rows.append({'origin':base,'nodes':list(nodes),'phase':phase,'futures':len(units),
            'valid_futures':sum(u['is_valid'] and u['is_eligible'] for u in units),
            'no_relapse_successes':sum(u['success_through_horizon'] for u in units),
            'observed_initial':observed_group(obs,nodes),'full_network_initial':state_group(state,nodes)})
    _atomic_json(bundle/'group_characteristics.json',rows)
    _atomic_json(bundle/'group_dynamics.json',dynamics)
    # Comparable candidate pairs: same origin, size, phase; nearest observed max
    # and total power. These pairs describe associations, not independent tests.
    pairs=[]
    for i,row in enumerate(rows):
        rate=row['no_relapse_successes']/row['futures']
        if rate==0:continue
        candidates=[(j,r) for j,r in enumerate(rows) if j!=i and r['origin']==row['origin']
                    and r['phase']==row['phase'] and len(r['nodes'])==len(row['nodes'])
                    and r['no_relapse_successes']/r['futures']<rate]
        if not candidates:continue
        fields=('received_attention_max','received_attention_sum')
        j,other=min(candidates,key=lambda item:(sum(abs(item[1]['observed_initial'][f]-row['observed_initial'][f]) for f in fields),tuple(item[1]['nodes'])))
        pairs.append({'higher_success_row':i,'lower_success_row':j,
                      'observed_power_absolute_differences':{f:abs(other['observed_initial'][f]-row['observed_initial'][f]) for f in fields}})
    _atomic_json(bundle/'characteristic_comparisons.json',{'rows_file':'group_characteristics.json','pairs':pairs,
        'interpretation':'Exploratory same-network/same-size comparisons; inspect matching distances. Search and audit outcomes remain separate.'})
    return rows
