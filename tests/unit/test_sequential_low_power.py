"""Focused controller/accounting tests; only tiny N6/H18 physical executions."""
import copy,json,sys,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from src.control.optimizers.sequential_low_power import pilot_plan,policy_spec,SequentialLowPowerController,choose_allocation
from src.control.contracts import Observation,ControlProblem,TargetType
from src.control.ledger import BudgetLedger
from src.control import sequential_batch as batch
from src.control import sequential_analysis as analysis


def obs(weights=(1,2,3,4,6,9),t=1,y=None):
    N=len(weights);W=np.tile(np.array(weights,float),(N,1));np.fill_diagonal(W,0)
    return Observation(time=t,public_states=np.ones((N,1)) if y is None else np.array(y,float).reshape(N,1),panel_mask=np.ones(N,bool),
        observed_edges={(i,j):v for (i,j),v in np.ndenumerate(W) if v>0},accessible_roster=tuple(range(N)),sample_time=t-1,delivery_time=t)


def controller(spec=None):
    spec=spec or policy_spec('test');p=ControlProblem(problem_id='test',target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1,'threshold':.5},N=6,D=1,k=6,exposure_cap=spec['budget'],amplitude_cap=2,accessible_roster=tuple(range(6)),release_deadline=200,holding_duration=100,horizon=500)
    c=SequentialLowPowerController(spec);c.reset(p,obs(),np.random.default_rng(1));l=BudgetLedger(N=6,D=1,k=6,exposure_cap=spec['budget'],amplitude_cap=2,accessible_roster=tuple(range(6)),release_deadline=200)
    return c,l


def deliver(c,l,o):
    l.advance_to(o.time);a=c.act(o,l);l.apply_action(a,current_time=o.time);return a


def test_pulses_pause_retarget_and_no_hidden_release():
    c,l=controller();a=deliver(c,l,obs());assert a.actuated_nodes()=={0} and a.intended_end_time==6
    a=deliver(c,l,obs(t=6));assert not a.entries and not a.release and not l.is_released and l.integrated_exposure==10
    a=deliver(c,l,obs((30,1,3,4,6,9),t=11));assert 0 not in a.actuated_nodes() and len(a.entries)==1
    assert l.integrated_exposure==10
    deliver(c,l,obs(t=201));assert l.is_released and not l.current_field.any()


def test_declining_power_not_color_triggers_fallen_eligibility():
    c,l=controller(policy_spec('fallen',ranking='fallen_only'));deliver(c,l,obs());assert not l.current_field.any()
    for t in (6,11,16,21):deliver(c,l,obs((1,2,3,4,6,.01),t=t))
    fallen=[x for x in c.log if 5 in x['fallen_nodes']];assert fallen and any(5 in x['nodes'] for x in fallen)
    assert all(x['public_blue_estimate']==0 for x in c.log)
    # Blue alone, with no power decline, does not create a fallen-power candidate.
    c,l=controller(policy_spec('fallen',ranking='fallen_only'))
    for t in (1,6,11):deliver(c,l,obs(t=t,y=[1,1,1,1,1,-1]))
    assert all(not x['fallen_nodes'] for x in c.log) and l.integrated_exposure==0


def test_unknown_scores_and_hidden_payloads_are_rejected():
    c,l=controller();o=replace(obs(),observed_edges={(0,1):1.});deliver(c,l,o)
    assert c.log[-1]['nodes']==[1] and set(c.log[-1]['scores'])=={'1'}
    with pytest.raises(ValueError,match='hidden'):c.act(replace(obs(t=6),private_states=np.ones((6,1))),l)
    with pytest.raises(ValueError,match='Undelivered'):c.act(replace(obs(t=6),sample_time=7),l)


def test_duplicate_reports_do_not_confirm_color_or_decline():
    c,l=controller();deliver(c,l,obs());deliver(c,l,obs(t=6,y=[-1]*6));n=len(c.history)
    c.act(obs(t=6,y=[-1]*6),l);assert len(c.history)==n and c.public_stop_streak==1
    deliver(c,l,obs(t=11,y=[-1]*6));assert l.is_released and c.log[-1]['reason']=='public_stop_confirmed'


def test_mid_pulse_power_promotion_and_budget_exhaustion():
    c,l=controller(policy_spec('long',pulse=10.));deliver(c,l,obs());deliver(c,l,obs((90,1,2,3,4,5),t=6))
    assert not l.current_field.any() and not l.is_released and c.next_start==11
    c,l=controller(policy_spec('small',budget=3.));a=deliver(c,l,obs());assert a.intended_end_time==2.5
    l.advance_to(2.5);assert l.integrated_exposure==3 and l.is_released


def test_initial_ranking_baseline_and_simultaneous_fixed_membership():
    c,l=controller(policy_spec('initial',ranking='initial_low'));deliver(c,l,obs());deliver(c,l,obs(t=6));a=deliver(c,l,obs((90,1,2,3,4,5),t=11));assert a.actuated_nodes()=={0}
    c,l=controller(policy_spec('batch',kind='simultaneous',ranking='attention',slots=4,budget=80.));a=deliver(c,l,obs());members=a.actuated_nodes();assert len(members)==4 and a.intended_end_time==11
    a=deliver(c,l,obs((90,1,2,3,4,5),t=6,y=[-1]*6));assert a.actuated_nodes()==members and a.intended_end_time==11
    l.advance_to(11);assert l.is_released and l.integrated_exposure==80


def test_plan_multiple_hypotheses_and_conservative_tiers(tmp_path):
    p=pilot_plan();assert (p['horizon'],p['release_deadline'],p['hold'])==(500,200,100)
    assert len(p['policies'])==34 and p['budgets']==[40,80,152]
    assert p['max_frames']==101 and p['persistence_offsets']==[100,200,300]
    assert p['horizon']-p['release_deadline']==300
    a=choose_allocation(p,[10,10,10],28000);assert len(a['origins'])==8 and a['futures']==4
    a=choose_allocation(p,[50]*3,28000);assert len(a['origins'])==4 and a['futures']==2
    assert choose_allocation(p,[100]*3,28000) is None
    assert choose_allocation(p,[10]*3,1201) is None
    assert batch.run_pilot(tmp_path,plan_only=True)==p
    with pytest.raises(RuntimeError,match='Colab-only'):batch.run_pilot(tmp_path)


def tiny_state():
    from src.core.state import State,StateMode
    return State(N=6,D=1,x=np.ones((6,1)),y=np.ones((6,1)),W=np.ones((6,6),dtype=int)-np.eye(6,dtype=int),
        n=np.zeros(6),Q=np.zeros((6,1,2)),state_mode=StateMode.BINARY,K=np.full(6,.3),b=np.full((6,1),-.15),g=np.ones(6),tau_x=np.ones(6),nu=np.ones(6))


def test_tiny_physics_recorder_passivity_and_later_pulse_accounting():
    from src.control.sequential_runner import run_unit,SequentialRecorder
    from src.control.recording import RecorderConfig,RolloutRecorder
    from src.config.tiered_profiles import LIGHT_TIERED
    p=pilot_plan();p.update(N=6,horizon=18.,release_deadline=12.,hold=2.,scalar_cadence=1.,frame_interval=1.,max_frames=19)
    profile=LIGHT_TIERED.model_copy(update={'N':6,'panel_fraction':1.,'edge_reveal_prob':1.,'binary_flip_prob':0.})
    spec=policy_spec('tiny',early_stop=False);cfg=RecorderConfig(cadence=1.,record_network_frames=True,max_network_frames=19)
    u=run_unit(profile,p,'tiny',42,0,spec,initial_state=tiny_state(),recorder=SequentialRecorder(cfg,frame_interval=1.))
    v=run_unit(profile,p,'tiny',42,0,spec,initial_state=tiny_state(),recorder=RolloutRecorder(cfg))
    assert u['is_valid'] and v['is_valid']
    for k in ['x','y','W']:assert u['final_state'][k]==v['final_state'][k]
    assert u['event_counts']==v['event_counts'];assert u['summaries']['target_condition_transitions']==v['summaries']['target_condition_transitions']
    r=analysis.outcomes(u,p);assert r['exposure']<=80 and r['last_field_end']<=12
    assert [f['t'] for f in u['network_frames']]==list(range(19))
    replay={'x':np.ones((6,1)),'y':np.ones((6,1))}
    for e in u['summaries']['recorder']['nucleation']['flips']:
        a=replay[e['layer']];assert a[e['node'],0]==e['before'];a[e['node'],0]=e['after']
    for k,a in replay.items():assert np.array_equal(a,np.array(u['final_state'][k]))


@pytest.fixture
def fake(monkeypatch,tmp_path):
    p=pilot_plan();p.update(N=6,origins=['one','two'],origin_seeds={'one':1,'two':2},allocation_tiers=[[2,2]])
    monkeypatch.setattr(batch,'identity_for',lambda plan:dict(plan=plan,profile={'N':6},source={'test':True}))
    monkeypatch.setattr(batch,'_origin_state',lambda *args:tiny_state())
    calls=[];interrupt={'once':True}
    def unit(profile,plan,base,seed,future,spec):
        if interrupt['once'] and len(calls)==4:interrupt['once']=False;raise RuntimeError('interrupt')
        calls.append((base,future,spec['name'],spec['budget']));active=spec['kind']!='zero';success=active and spec['name']!='random1'
        state=tiny_state();final=tiny_state()
        if success:final.x[:]=-1;final.y[:]=-1
        from src.control.checkpointing import serialize_state
        hist=[dict(decision_time=0.,spending_rate=0.,actuated_nodes=[])]
        if active:hist += [dict(decision_time=1.,spending_rate=2.,actuated_nodes=[0]),dict(decision_time=6.,spending_rate=0.,actuated_nodes=[])]
        release=11. if active else 0.;trace=[[0.,False]]+([[10.,True]] if success else [])
        return dict(policy_spec=spec,memory_mode=spec['memory'],visibility=spec['visibility'],is_valid=True,is_eligible=True,recipient_nodes=[0] if active else [],
            success=success,relapsed=False,success_through_horizon=success,completion_time=111. if success else None,actual_release_time=release,
            expenditure=10. if active else 0.,elapsed_seconds_this_invocation=.01,controller_log=[dict(time=1.,reason='pulse_start',nodes=[0])] if active else [],
            final_state=serialize_state(final),summaries={'target_condition_transitions':trace,'recorder':{'action_history':hist,'nucleation':{'flips':[]},'sequential_context':{'action_contexts':[]}}},
            network_frames=[dict(t=t,x=s.x.tolist(),y=s.y.tolist(),W=s.W.tolist(),controlled_mask=[False]*6) for t,s in [(0.,state),(plan['horizon'],final)]] if future==0 else [])
    monkeypatch.setattr(batch,'run_unit',unit);return p,calls


def test_durable_resume_frozen_allocation_and_paired_analysis(fake,tmp_path):
    p,calls=fake
    with pytest.raises(RuntimeError,match='interrupt'):batch._run(tmp_path,p)
    b=next(tmp_path.iterdir());d=analysis.analyze_bundle(b);assert not d['complete'] and d['primary']['effect'] is None
    batch._run(tmp_path,p,b);n=len(calls);batch._run(tmp_path,p,b);assert len(calls)==n==len(set(calls))
    raw=(b/'manifest.json').read_bytes();d=analysis.analyze_bundle(b);assert d['complete'] and d['primary']['networks']==2
    assert (b/'manifest.json').read_bytes()==raw and len(d['roles'])==136
    changed=copy.deepcopy(p);changed['horizon']=1000
    with pytest.raises(ValueError,match='identical'):batch._run(tmp_path,changed,b)
    info=next(iter(d['manifest']['actual'].values()));(b/info['path']).write_text('{}')
    with pytest.raises(ValueError,match='Checksum'):analysis.analyze_bundle(b)


def test_timing_only_and_no_false_complete(fake,tmp_path):
    p,calls=fake;p['planning_seconds']=0;b=batch._run(tmp_path,p);report=analysis.build_saved_report(b)
    assert len(calls)==3 and not report['complete'] and not report['players']


def test_supervisor_stops_export_without_invalidating_science(monkeypatch,tmp_path):
    import multiprocessing as mp
    if 'fork' not in mp.get_all_start_methods():pytest.skip('needs fork')
    context=mp.get_context('fork');monkeypatch.setattr(batch.mp,'get_context',lambda method:context);monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace())
    def sleeper(out,plan,resume,events,budget):
        p=Path(out)/'saved';p.mkdir();(p/'manifest.json').write_text('{"status":"complete"}');events.put(('bundle',str(p)));events.put(('compute_done',str(p)));time.sleep(10)
    monkeypatch.setattr(batch,'_worker',sleeper);begin=time.monotonic();b=batch.run_pilot(tmp_path,wall_time_limit=.3)
    assert time.monotonic()-begin<3;m=json.loads((b/'manifest.json').read_text());assert m['status']=='complete' and m['report_status']=='incomplete_hard_time_limit'


def test_no_actuation_can_stop_on_confirmed_public_majority():
    c,l=controller();deliver(c,l,obs(t=1,y=[-1]*6));assert not l.is_released
    deliver(c,l,obs(t=6,y=[-1]*6));assert l.is_released and l.integrated_exposure==0


def test_long_horizon_late_relapse_and_final_withdrawal(fake):
    p,_=fake;spec=p['policies']['dynamic_low1_B80'];u=batch.run_unit(None,p,'one',1,0,spec)
    u.update(actual_release_time=200.,completion_time=300.)
    u['summaries']['target_condition_transitions']=[[0.,False],[10.,True],[400.,False],[450.,True]]
    u.update(relapsed=True,success_through_horizon=False)
    r=analysis.outcomes(u,p)
    assert r['hold'] and not r['persistent'] and r['penalized_time']==500
    assert r['persistent_at_100_after_release'] and not r['persistent_at_200_after_release']
    assert not r['persistent_at_300_after_release']
    u['summaries']['target_condition_transitions']=[[0.,False],[10.,True],[199.,False],[250.,True]]
    u.update(completion_time=350.,relapsed=False,success_through_horizon=True)
    r=analysis.outcomes(u,p);assert r['persistent'] and not r['persistent_at_100_after_release']
    assert r['persistent_at_300_after_release']


def complete_fake(fake,tmp_path):
    p,_=fake
    try:batch._run(tmp_path,p)
    except RuntimeError as e:assert str(e)=='interrupt'
    b=next(p for p in tmp_path.iterdir() if p.is_dir())
    batch._run(tmp_path,p,b);return b




def test_saved_reports_offline_players_relocation(fake,tmp_path):
    import shutil
    b=complete_fake(fake,tmp_path);report=analysis.build_saved_report(b)
    assert report['complete'] and report['horizon']==500 and report['release_deadline']==200
    assert set(report['players'])=={'adaptive_restricted','frozen_restricted','adaptive_complete_public'}
    for player in report['players'].values():
        assert player['comparisons']>0 and not player['dropped'] and not player['skipped']
        path=Path(player['path']);html=path.read_text();meta=json.loads(path.with_name('player_metadata.json').read_text())
        assert '<html' in html and '<script src=' not in html
        assert meta['horizon']==500 and meta['release_deadline']==200
    m=json.loads((b/'manifest.json').read_text());m['report_status']='complete';(b/'manifest.json').write_text(json.dumps(m))
    assert not Path(str(b)+'_analysis').exists()
    raw=tmp_path/'downloaded_run';derived=raw/'analysis'
    shutil.copytree(b,raw)
    saved,out,_=analysis.read_saved_report(raw);assert saved['complete']
    assert all(Path(p).is_relative_to(derived) for p in saved['figures'])
    bad=json.loads((raw/'manifest.json').read_text());bad['identity']['plan']['hold']=99;(raw/'manifest.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='different scientific'):analysis.read_saved_report(raw)
    (raw/'manifest.json').write_text(json.dumps(m));(derived/'summary.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='Export checksum'):analysis.read_saved_report(raw)


def test_unavailable_baseline_denominators_and_zero_spend_comparisons(fake,tmp_path):
    from src.control.optimizers.campaign import _atomic_json,_sha
    from src.visualization.animation_player import prepare_comparisons
    b=complete_fake(fake,tmp_path);m=json.loads((b/'manifest.json').read_text())
    for key,info in m['actual'].items():
        if info['kind']=='audit' and info['role']=='simultaneous_low4_B80':
            u=json.loads((b/info['path']).read_text());u['controller_log'].append(dict(time=1.,reason='insufficient_initial_candidates'))
            _atomic_json(b/info['path'],u);info['sha256']=_sha(b/info['path'])
    _atomic_json(b/'manifest.json',m);d=analysis.analyze_bundle(b)
    row=d['summary'].set_index('role').loc['simultaneous_low4_B80']
    assert row['available']==0 and row['unavailable']==4 and np.isnan(row['persistent_rate'])
    contrast=next(c for c in d['contrasts'] if c['b']=='simultaneous_low4_B80')
    assert contrast['status']=='unavailable' and contrast['excluded_networks']==['one','two']
    d['roles'].loc[d['roles'].role=='fallen_only_B80','count']=0
    view=analysis.player_view(d,'adaptive','restricted');ctx=prepare_comparisons(view,progress=False)
    total=ctx['comparison_count'] if 'comparison_count' in ctx else len(ctx['scenes_for'](2))
    view['include_unactuated_policies']=False;old=prepare_comparisons(view,progress=False)
    oldtotal=old['comparison_count'] if 'comparison_count' in old else len(old['scenes_for'](2))
    assert total==oldtotal+2


def test_supervisor_stops_science_and_withholds_report(monkeypatch,tmp_path):
    import multiprocessing as mp
    if 'fork' not in mp.get_all_start_methods():pytest.skip('needs fork')
    context=mp.get_context('fork');monkeypatch.setattr(batch.mp,'get_context',lambda method:context);monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace())
    def sleeper(out,plan,resume,events,budget):
        p=Path(out)/'saved';p.mkdir();(p/'manifest.json').write_text('{"status":"incomplete"}');events.put(('bundle',str(p)));time.sleep(10)
    monkeypatch.setattr(batch,'_worker',sleeper);b=batch.run_pilot(tmp_path,wall_time_limit=.3)
    m=json.loads((b/'manifest.json').read_text());assert m['status']=='incomplete_hard_time_limit' and m['report_status']=='incomplete_hard_time_limit'




def test_real_engine_allows_pulses_after_old_t20_cutoff():
    from src.control.sequential_runner import run_unit
    from src.config.tiered_profiles import LIGHT_TIERED
    p=pilot_plan();p.update(N=6,horizon=42.,release_deadline=32.,hold=2.,frame_interval=1.,max_frames=43,scalar_cadence=1.)
    profile=LIGHT_TIERED.model_copy(update={'N':6,'panel_fraction':1.,'edge_reveal_prob':1.,'binary_flip_prob':0.})
    state=tiny_state();state.b[:]=10.  # Stable red toy system exposes scheduling independently of spread.
    u=run_unit(profile,p,'late-pulse-test',42,0,policy_spec('late',early_stop=False),initial_state=state)
    starts=[d['time'] for d in u['controller_log'] if d['reason']=='pulse_start']
    assert any(20<t<32 for t in starts)
    r=analysis.outcomes(u,p);assert r['last_field_end']==32 and u['actual_release_time']==32
    assert u['target_definition']['horizon']==42 and u['target_definition']['holding_duration']==2
    assert u['target_definition']['s_star']==-1 and u['target_definition']['threshold']==.5
    assert all(not any(f['controlled_mask']) for f in u['network_frames'] if f['t']>=32)


def test_blue_majority_origin_rejected_before_audit(fake,tmp_path,monkeypatch):
    p,_=fake
    def blue(*args):
        state=tiny_state();state.x[:]=-1;state.y[:]=-1;return state
    monkeypatch.setattr(batch,'_origin_state',blue)
    with pytest.raises(ValueError,match='initial red majority'):batch._run(tmp_path,p)
    m=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
    assert all(u['kind']=='timing' for u in m['actual'].values()) and not m['selection_frozen']




@pytest.mark.parametrize('role,delay', [('zero',1.),('zero_frozen',1.),('zero_complete',1.),('dynamic_low2_B152',1.),('complete_low_B80',1.),('dynamic_low1_B80',0.)])
def test_real_startup_zero_controls_and_timing_roles(role,delay):
    from src.control.sequential_runner import run_unit,SequentialRecorder
    from src.control.recording import RecorderConfig,RolloutRecorder
    from src.config.tiered_profiles import LIGHT_TIERED
    p=pilot_plan();p.update(N=6,horizon=6.,release_deadline=4.,hold=1.,scalar_cadence=1.,frame_interval=1.,max_frames=7)
    profile=LIGHT_TIERED.model_copy(update={'N':6,'panel_fraction':1.,'edge_reveal_prob':1.,'binary_flip_prob':0.,'delivery_delay_tau_public':delay})
    spec=p['policies'][role];cfg=RecorderConfig(cadence=1.,record_network_frames=True,max_network_frames=7)
    u=run_unit(profile,p,'startup',42,0,spec,initial_state=tiny_state(),recorder=SequentialRecorder(cfg,frame_interval=1.))
    v=run_unit(profile,p,'startup',42,0,spec,initial_state=tiny_state(),recorder=RolloutRecorder(cfg))
    assert u['is_valid'] and u['is_eligible']
    assert u['final_state']==v['final_state'] and u['event_counts']==v['event_counts']
    assert u['summaries']['target_condition_transitions']==v['summaries']['target_condition_transitions']
    actions=u['summaries']['recorder']['action_history'];contexts=u['summaries']['recorder']['sequential_context']['action_contexts']
    if spec['kind']=='zero' or delay==0:
        assert actions[0]['decision_time']==contexts[0]['time']==0.
        assert contexts[0]['private_blue']==contexts[0]['public_blue']==0
        assert contexts[0]['true_received_attention']==pytest.approx([1.]*6)
        assert contexts[0]['nodes']==actions[0]['actuated_nodes']
    if spec['kind']=='zero':
        assert u['actual_release_time']==0. and u['expenditure']==0. and not u['recipient_nodes']
    analysis.outcomes(u,p)
