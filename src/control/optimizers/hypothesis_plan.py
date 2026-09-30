"""Pre-final development contrasts. No hidden graph enters recipient selection."""
import hashlib
import numpy as np
from src.control.optimizers.observed_subsets import _proposal_orders


def future_seed(origin, future):
    return int.from_bytes(hashlib.sha256(
        f'modular3-prefinal-v1:{origin}:{future}'.encode()).digest()[:8],'big')


def restricted_low_coverage(observation, size, seed):
    """Coverage within the lower half of POSITIVE reported attention scores.

    Median cutoff includes ties; missing scores are excluded. Too few eligible
    recipients means unavailable, not expansion to powerful/unmeasured nodes.
    Selection uses only reported interactions, never actual initial influence.
    """
    roster,incoming,_,_= _proposal_orders(observation,np.random.default_rng(seed))
    scored=[n for n in roster if incoming[n]>0]
    cutoff=float(np.median(incoming[scored])) if scored else None
    eligible=[n for n in scored if incoming[n]<=cutoff] if scored else []
    if len(eligible)<size:
        return {'available':False,'nodes':[], 'cutoff':cutoff,'eligible_count':len(eligible)}
    from dataclasses import replace
    restricted=replace(observation,accessible_roster=tuple(eligible))
    orders=dict(_proposal_orders(restricted,np.random.default_rng(seed))[3])
    return {'available':True,'nodes':sorted(int(n) for n in orders['audience_coverage'][:size]),
            'cutoff':cutoff,'eligible_count':len(eligible)}


def hypothesis_plan():
    return {'version':'prefinal-v1','development_origins':[f'development_{i:02d}' for i in range(4)],
            'futures_per_origin':40,'strategy_roles':['random4','coverage4','restricted_low_coverage4','random10','coverage10','zero'],
            'physical_ceiling':960,'forecast_simulations':0,
            'recipient_source':'Reuse Batch1 random/coverage memberships; new restricted-low coverage uses the same delivered report.',
            'scope':'Development: paired future repeats conditional on four fixed origins; not160 independent networks.',
            'primary_endpoint':'completed100-unit hold without subsequent relapse through320',
            'secondary_endpoints':['hold completion time','hold completed irrespective of later relapse','initial recipient influence','exposure'],
            'fixed':{'N':40,'strength':2,'release':20,'hold':100,'horizon':320,'support':.75},
            'unavailable_rule':'Record insufficient eligible low-score recipients; do not substitute a different group.',
            'final_validation':'Eight validation origins remain untouched; no final run until this batch is analyzed.'}
