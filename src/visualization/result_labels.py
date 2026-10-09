"""Presentation names for saved policy identifiers; never changes their meaning."""
import re

# Curated scientific questions, checked against the experiment findings ledger.
# Failed hypotheses remain useful evidence; software rehearsals do not.
STUDIES = {
    'f92ee7c9': ('Reference pilot', 'Valid development evidence: original four-recipient constraints and feedback withdrawal.'),
    '5c86dac2': ('Reach and dose', 'Contrasts original, wider and whole-population reach; establishes a broad positive control.'),
    '47a5b08e': ('Equal exposure', 'Distributed weak versus concentrated strong treatment at equal cumulative exposure.'),
    '29493474': ('Fixed-memory control', 'Tests whether broad treatment persists without memory adaptation.'),
    '2713e0a1': ('Broadcast + targeted boost', 'A valid negative comparison: mixed allocation did not improve on uniform treatment.'),
    '091ffb42': ('Held-out selection', 'Reserved selection networks test the earlier broad and original-access candidates.'),
    '2e2c3b37': ('Two-recipient targeting', 'Valid negative evidence on propagation from two high- or low-attention recipients.'),
    '040e5357': ('Random 50% coverage', 'Partial-population feasibility and strength comparison on reused development networks.'),
    '7ab3cf6f': ('Random 25% coverage', 'Equal-exposure comparison with 50% coverage; completed holds can still relapse.'),
    '87a7fc8d': ('Recipient search development', 'Actual continuations evaluate selected groups and simple selectors; forecast simulations are excluded.'),
    '8bef2af4': ('Sparse discovery', 'Exploratory recipient-count and observed-power tradeoffs; not independent confirmation.'),
    'dc816b6f': ('Independent validation', 'Fresh networks evaluate frozen search rules against matched simple selectors.'),
    '522c5820': ('Nucleation and bridges', 'Tests local cores, bridge selection and fixed-memory diagnostics.'),
    '109c33d4': ('Sequential weak recipients', 'Valid negative evidence for the strict endpoint, including timing and budget hypotheses.'),
    '8de67a4d': ('Response-guided', 'Valid negative evidence for response, reinforcement and expansion hypotheses.'),
    '1079cee0': ('Scale comparison', 'Completed N40/N400 coverage, random-targeting, exposure and memory contrasts.'),
}

NAMES = {
    'zero':'No intervention', 'random':'Random recipients', 'attention':'High observed attention',
    'coverage':'Audience coverage', 'audience_coverage':'Audience coverage',
    'low_observed_attention':'Low observed attention', 'nucleus':'Local nucleus',
    'nucleus_bridge':'Nucleus + selected bridges', 'random_core':'Random within nucleus',
    'random_bridge':'Nucleus + random bridges',
    'dynamic_low1':'Current weak · one at a time', 'dynamic_low2':'Current weak · two at a time',
    'fallen_low1':'Newly weakened · one at a time', 'initial_low1':'Initially weak · one at a time',
    'random1':'Random · one at a time', 'simultaneous_low4':'Weak recipients · four at once',
    'simultaneous_attention4':'High attention · four at once', 'simultaneous_coverage4':'Audience coverage · four at once',
    'complete_low':'Current weak', 'frozen_low':'Current weak', 'fallen_only':'Newly weakened only',
    'long_pulse':'Current weak · longer pulse', 'no_early_stop':'Current weak · no early stop',
    'no_pause':'Current weak · no pauses', 'strict_low':'Current weak · strict weakness rule',
    'response':'Response-guided', 'weakest':'Weakest-first', 'random_low':'Random among weak',
    'reinforce':'Reinforce converted group', 'expand':'Expand converted group',
    'topology':'Response-guided · topology rule', 'no_growth_wait':'Response-guided · no growth wait',
    'no_restart':'Response-guided · no restart', 'weakest_seed':'Response-guided · weakest seed',
    'all_blue_core':'Response-guided · all converted nodes form core',
    'bottom_quarter':'Response-guided · weakest quarter', 'fallen_first':'Response-guided · newly weakened first',
    'pulse10':'Response-guided · pulse 10', 'pulse15':'Response-guided · pulse 15',
}

def policy_label(role):
    """Expand budgets, fractions, power ceilings and selector names consistently."""
    name=str(role)
    if name.startswith(('frozen_low_','complete_low_')):name=re.sub(r'^(frozen|complete)_low_', 'dynamic_low1_', name)
    name=re.sub(r'_(adaptive|frozen)$','',name)
    name=re.sub(r'^(frozen|complete)_','',name)
    if name in ('zero','zero_complete'):return 'No intervention'
    early=re.fullmatch(r'(attention|random|high_attention_two|low_attention_two|random_half|random_quarter)_(sustained|shutoff)_(\d+)',name)
    if early:
        selector={'attention':'High observed attention','random':'Random recipients','high_attention_two':'High observed attention · two recipients','low_attention_two':'Low observed attention · two recipients','random_half':'Random recipients · 50% of nodes','random_quarter':'Random recipients · 25% of nodes'}[early[1]]
        return selector+' · field strength '+early[3]+(' · early withdrawal' if early[2]=='shutoff' else '')
    if name=='combined_broadcast_08_boost_2':return 'Broadcast strength 0.8 + attention boost strength 2'
    suffix=[]
    budget=re.search(r'_B(\d+)$',name)
    if budget:suffix.append('budget ≤ '+budget[1]);name=name[:budget.start()]
    search=re.fullmatch(r'n(\d+)_q([\d.]+)',name)
    if search:return f'Simulator search · {search[1]} recipients · observed-power ceiling {float(search[2])*100:g}%'
    equal='_equal_total' in name
    name=name.replace('_equal_total','').removeprefix('selector_')
    size=re.fullmatch(r'(coverage|random|attention|audience_coverage|nucleus_bridge|nucleus|random_core|random_bridge|low_observed_attention)_?(\d+)',name)
    if size:name=size[1]
    label=NAMES.get(name)
    if label is None:
        label=name.replace('shutoff','withdrawn').replace('sustained','sustained field').replace('_',' ').capitalize()
    if equal:suffix.append('same total exposure as 4 recipients')
    return ' · '.join([label]+suffix)

LABEL_KEY = [
    ['Study', 'A separate saved experiment. Different studies are not pooled.'],
    ['N', 'Number of simulated nodes; N400 is completed, N500 has no returned efficacy evidence.'],
    ['Network 01', 'An independent starting condition within the selected study and size. Network numbers restart in each study.'],
    ['Future 01', 'One stochastic continuation of that network; futures are nested replicates, not new networks.'],
    ['Recipients', 'Distinct nodes that received intervention at any time; sequential policies may treat them at different times.'],
    ['Budget ≤ B', 'Declared maximum cumulative field exposure, not recipient count or money.'],
    ['Field strength', 'Intervention amplitude in early development studies; different from recipient count or cumulative exposure.'],
    ['Saved group', 'A recorded recipient set in an early search experiment; the original set identifier remains under Source identifiers.'],
    ['Adaptive / Fixed memory', 'Memory evolves / encounter updates and decay are disabled.'],
    ['Partial / Complete public reports', 'Restricted noisy observations / complete-public diagnostic observations. Neither exposes private opinions.'],
    ['Persistent / Threshold breach / Hold not completed', 'Saved event-level outcome: certified and sustained / certified then exited / never completed the required hold.'],
    ['Audience coverage', 'Recipients chosen to cover distinct observed audiences; not a claim of optimal targeting.'],
    ['Observed-power ceiling', 'Search restricts recipients to a declared quantile of positive observed attention scores; unknown scores are excluded.'],
    ['Gold rings', 'Solid: field active now. Dashed: treated earlier, field now off. Node numbers are intentionally hidden.'],
]
