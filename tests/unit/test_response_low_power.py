"""Behavioral tests and bounded N6 engine checks; never nominal study runs."""
import copy,json,sys,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from src.control.optimizers.response_low_power import final_plan,response_spec,ResponseLowPowerController
from src.control.optimizers.sequential_low_power import choose_allocation
from src.control import sequential_batch as batch,sequential_analysis as analysis
from src.control.response_analysis import diagnostics
from src.control.response_batch import run_final
from src.control.contracts import ControlProblem,TargetType
from src.control.ledger import BudgetLedger
from test_sequential_low_power import obs,deliver,tiny_state


def controller(spec=None):
    s=spec or response_spec('test');p=ControlProblem(problem_id='response-test',target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1,'threshold':.5},
        N=6,D=1,k=6,exposure_cap=s['budget'],amplitude_cap=2,accessible_roster=tuple(range(6)),release_deadline=200,holding_duration=100,horizon=500)
    c=ResponseLowPowerController(s);c.reset(p,obs(),np.random.default_rng(1))
    l=BudgetLedger(N=6,D=1,k=6,exposure_cap=s['budget'],amplitude_cap=2,accessible_roster=tuple(range(6)),release_deadline=200)
    return c,l


def view():
    A=np.zeros((6,6));A[4,1]=.9;A[1,4]=.7;A[2,4]=.1;A[3,2]=1.;A[0,2]=1.;A[5,3]=10.
    return dict(A=A,low={0,1,2},currently_red={0,1,2,3,5},observed=list(range(6)),scores={i:i+1. for i in range(6)},fallen=set(),sample=10.,public_blue=1/6)


def test_plan_all_weak_single_recipient_and_predeclared_comparisons(tmp_path):
    p=final_plan();assert len(p['policies'])==p['roles_per_block']==31
    assert p['horizon']==500 and p['release_deadline']==200 and p['hold']==100
    assert p['origin_candidate_indices']==[0,1,2,3,4,5,7,8]
    assert all(s['slots']==1 and s['kind'] in ('response','zero') for s in p['policies'].values())
    assert len({s['controller_seed_group'] for s in p['policies'].values()})==1
    assert all(c['a'] in p['policies'] and c['b'] in p['policies'] for c in p['contrasts'])
    assert p['secondary_majority']['minimum_blue_nodes']==21
    assert len(choose_allocation(p,[29.2]*3,28000)['origins'])==8
    assert choose_allocation(p,[29.2]*3,28000)['futures']==2
    assert run_final(tmp_path,plan_only=True)==p
    with pytest.raises(RuntimeError,match='Colab-only'):run_final(tmp_path)


def test_next_target_uses_response_and_never_promotes_powerful_candidates():
    c,l=controller();v=view();c._new_episode(v,l);c.core={4};c.episode_nodes={4}
    assert c._rank_response(v,l)[0]==2
    c.spec['ranking']='reinforce';assert c._rank_response(v,l)[0]==1
    c.spec['ranking']='expand';assert c._rank_response(v,l)[0]==2
    assert 3 not in c._rank_response(v,l)  # Has huge reach, but is not weak.
    v['low']={0};assert c._rank_response(v,l)==[]  # Weak but disconnected: wait.


def test_same_seed_across_response_ablations_and_weakest_seed_difference():
    choices=[]
    for ranking in ['hybrid','reinforce','expand']:
        c,l=controller(response_spec(ranking,ranking=ranking));choices.append(c._rank_response(view(),l)[0])
    assert choices==[2,2,2]
    c,l=controller(response_spec('weakest_seed',seed_rule='weakest'));assert c._rank_response(view(),l)[0]==0


def test_untreated_confirmation_excludes_treatment_samples_and_duplicate_reports():
    c,l=controller();a=deliver(c,l,obs());node=next(iter(a.actuated_nodes()));assert a.intended_end_time==6
    colors=[1]*6;colors[node]=-1
    deliver(c,l,obs(t=6,y=colors));assert c.off_reports==0 and not c.core
    deliver(c,l,obs(t=11,y=colors));assert c.off_reports==1 and not c.core
    c.act(obs(t=11,y=colors),l);assert c.off_reports==1
    deliver(c,l,obs(t=16,y=colors));assert node in c.core and c.log[-1]['reason']=='wait_for_spontaneous_spread'
    assert not l.current_field.any() and not l.is_released


def test_originally_blue_does_not_become_response_without_treatment():
    c,l=controller();v=view();v['currently_red']={0,1,2,3,5};c._new_episode(v,l);c.episode_nodes={0};c.last_end=5
    c._feedback(v);c._feedback(dict(v,sample=15));assert 4 not in c.core
    c.spec['core_rule']='all_blue';c._feedback(dict(v,sample=20));assert 4 in c.core


def test_new_response_changes_which_weak_node_is_selected():
    c,l=controller();v=view();c._new_episode(v,l);c.core={4};c.episode_nodes={4}
    first=c._rank_response(v,l)[0]
    # A different surviving group favors another recipient; power scores unchanged.
    v2=copy.deepcopy(v);v2['A'][5,0]=5.;c.core={5}
    assert first==2 and c._rank_response(v2,l)[0]==0


def test_budget_bounded_restart_and_no_restart_ablation():
    for restart in (True,False):
        c,l=controller(response_spec('restart',restart=restart));v=view();c._new_episode(v,l)
        c.seed_node=2;c.episode_nodes={2};c.attempted_seeds={2};c.off_reports=2
        l.integrated_exposure=10.;assert c._rank_response(v,l)==[2]
        l.integrated_exposure=20.;rank=c._rank_response(v,l)
        assert (bool(rank),c.episode)==((True,2) if restart else (False,1))
        if restart:assert 2 not in rank


def test_power_decline_is_eligible_not_blue_switch_and_promoted_active_stops():
    c,l=controller(response_spec('long',pulse=15.,seed_rule='weakest'))
    a=deliver(c,l,obs());assert a.actuated_nodes()=={0}
    deliver(c,l,obs((90,1,2,3,4,5),t=6));assert not l.current_field.any()
    assert c.log[-1]['reason']=='recipient_no_longer_weak'
    c,l=controller(response_spec('fallen',fallen_priority=True));v=view();c._new_episode(v,l);c.core={4};v['fallen']={1}
    assert c._rank_response(v,l)[0]==1
    c,l=controller()
    for t in (1,6,11):deliver(c,l,obs(t=t,y=[1,1,1,1,1,-1]))
    assert all(5 not in log['fallen_nodes'] for log in c.log)


def test_hidden_unknown_and_future_information_rejected():
    c,l=controller()
    for kwargs in [dict(private_states=np.ones((6,1))),dict(sample_time=7),dict(is_oracle=True)]:
        with pytest.raises(ValueError):c.act(replace(obs(t=6),**kwargs),l)
    a=deliver(c,l,replace(obs(),observed_edges={}))
    assert not a.entries and not l.is_released


@pytest.mark.parametrize('role',list(final_plan()['policies']))
def test_every_role_real_startup_constraints_and_exact_replay(role):
    from src.control.sequential_runner import run_unit
    from src.control.checkpointing import serialize_state
    from src.config.tiered_profiles import LIGHT_TIERED
    p=final_plan();p.update(N=6,horizon=20.,release_deadline=17.,hold=2.,scalar_cadence=1.,frame_interval=1.,max_frames=21)
    p['secondary_majority']=dict(minimum_blue_nodes=4)
    profile=LIGHT_TIERED.model_copy(update={'N':6,'panel_fraction':1.,'edge_reveal_prob':1.,'binary_flip_prob':0.})
    state=tiny_state();u=run_unit(profile,p,'tiny-response',42,0,p['policies'][role],initial_state=state)
    assert u['is_valid'] and u['is_eligible']
    r=analysis.outcomes(u,p);d=diagnostics(u,p,serialize_state(state))
    assert r['max_concurrent']<=1 and r['exposure']<=p['policies'][role]['budget'] and r['last_field_end']<=17
    assert [f['t'] for f in u['network_frames']]==list(range(21))
    assert all(not any(f['controlled_mask']) for f in u['network_frames'] if f['t']>=17)
    assert d['peak_joint']<=6


def test_response_recorder_does_not_change_physics():
    from src.control.sequential_runner import run_unit,SequentialRecorder
    from src.control.recording import RecorderConfig,RolloutRecorder
    from src.config.tiered_profiles import LIGHT_TIERED
    p=final_plan();p.update(N=6,horizon=20.,release_deadline=17.,hold=2.,scalar_cadence=1.,frame_interval=1.,max_frames=21)
    profile=LIGHT_TIERED.model_copy(update={'N':6});cfg=RecorderConfig(cadence=1.,record_network_frames=True,max_network_frames=21)
    u=run_unit(profile,p,'passivity',42,0,p['policies']['response_B80'],initial_state=tiny_state(),recorder=SequentialRecorder(cfg,frame_interval=1.))
    v=run_unit(profile,p,'passivity',42,0,p['policies']['response_B80'],initial_state=tiny_state(),recorder=RolloutRecorder(cfg))
    assert u['final_state']==v['final_state'] and u['event_counts']==v['event_counts']
    assert u['summaries']['target_condition_transitions']==v['summaries']['target_condition_transitions']


def fake_campaign(monkeypatch,tmp_path):
    from src.control.checkpointing import serialize_state
    p=final_plan();p.update(N=6,origins=['one','two'],origin_seeds={'one':1,'two':2},allocation_tiers=[[2,2]])
    p['secondary_majority']=dict(minimum_blue_nodes=4)
    monkeypatch.setattr(batch,'identity_for',lambda plan:dict(plan=plan,profile={'N':6},source={'test':True}))
    monkeypatch.setattr(batch,'_origin_state',lambda *args:tiny_state())
    calls=[];interrupt=[True]
    def unit(profile,plan,base,seed,future,spec):
        if len(calls)==5 and interrupt[0]:interrupt[0]=False;raise RuntimeError('injected interruption')
        calls.append((base,future,spec['name'],spec['budget']));state=tiny_state()
        return dict(policy_spec=spec,memory_mode=spec['memory'],visibility=spec['visibility'],is_valid=True,is_eligible=True,
            recipient_nodes=[],success=False,relapsed=False,success_through_horizon=False,completion_time=None,actual_release_time=200.,
            expenditure=0.,controller_log=[],final_state=serialize_state(state),
            summaries={'target_condition_transitions':[[0.,False]],'recorder':{'action_history':[dict(decision_time=0.,spending_rate=0.,actuated_nodes=[])],
                'nucleation':{'flips':[]},'sequential_context':{'action_contexts':[]}}},
            network_frames=[dict(t=t,x=state.x.tolist(),y=state.y.tolist(),W=state.W.tolist(),controlled_mask=[False]*6) for t in range(0,501,5)] if future==0 else [])
    monkeypatch.setattr(batch,'run_unit',unit)
    with pytest.raises(RuntimeError,match='injected'):batch._run(tmp_path,p)
    b=next(tmp_path.glob('response_low_power_v1_*'));d=analysis.analyze_bundle(b);assert not d['complete'] and d['primary']['effect'] is None
    batch._run(tmp_path,p,b);n=len(calls);batch._run(tmp_path,p,b);assert len(calls)==n==len(set(calls))
    with pytest.raises(ValueError,match='identical'):batch._run(tmp_path,dict(p,hold=99),b)
    return b,p






def test_real_exports_all_hypotheses_preserved_and_relocatable(monkeypatch,tmp_path):
    import shutil
    b,p=fake_campaign(monkeypatch,tmp_path);d=analysis.analyze_bundle(b)
    assert d['complete'] and len(d['roles'])==124 and len(d['contrasts'])==5*len(p['contrasts'])
    assert d['summary']['majority_persistent'].sum()==0
    r=analysis.build_saved_report(b);assert r['complete']
    assert (b/'analysis/report.html').exists() and (b/'analysis/FINDINGS.md').exists()
    for player in r['players'].values():
        assert not player['dropped'] and not player['skipped']
        page=Path(player['path']).read_text();assert 'Solid gold' in page and 'first_treated' in page and '<script src=' not in page
        import re
        payload=json.loads(re.search(r'<script id="player-data" type="application/json">(.*?)</script>',page,re.S).group(1))
        assert all(len(a['t'])==101 for a in payload['arms'])
    m=json.loads((b/'manifest.json').read_text());m['report_status']='complete';(b/'manifest.json').write_text(json.dumps(m))
    moved=tmp_path/'relocated';shutil.copytree(b,moved);assert analysis.read_saved_report(moved)[0]['complete']
    (moved/'analysis/summary.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='Export checksum'):analysis.read_saved_report(moved)


def test_secondary_majority_is_not_primary_and_late_relapse_is_failure():
    from src.control.checkpointing import serialize_state
    p=final_plan();p['secondary_majority']=dict(minimum_blue_nodes=4)
    state=tiny_state();initial=serialize_state(state)
    flips=[dict(t=100.,layer=layer,node=i,before=1.,after=-1.) for layer in ('x','y') for i in range(4)]
    state.x[:4]=-1;state.y[:4]=-1
    u=dict(actual_release_time=200.,controller_log=[],final_state=serialize_state(state),
        summaries={'recorder':{'nucleation':{'flips':flips},'sequential_context':{'action_contexts':[]}}})
    d=diagnostics(u,p,initial);assert d['majority_persistent'] and d['majority_completion']==300 and d['peak_joint']==4
    flips.extend([dict(t=350.,layer='x',node=0,before=-1.,after=1.),dict(t=400.,layer='x',node=0,before=1.,after=-1.)])
    d=diagnostics(u,p,initial);assert d['majority_hold'] and not d['majority_persistent']


def test_final_policy_can_act_after_t20_and_stops_at_release_deadline():
    from src.control.sequential_runner import run_unit
    from src.config.tiered_profiles import LIGHT_TIERED
    p=final_plan();p.update(N=6,horizon=42.,release_deadline=32.,hold=2.,scalar_cadence=1.,frame_interval=1.,max_frames=43)
    profile=LIGHT_TIERED.model_copy(update={'N':6,'panel_fraction':1.,'edge_reveal_prob':1.,'binary_flip_prob':0.})
    state=tiny_state();state.b[:]=10.
    u=run_unit(profile,p,'late-response',42,0,p['policies']['response_B80'],initial_state=state)
    assert any(20<log['time']<32 for log in u['controller_log'] if log['reason']=='pulse_start')
    assert u['actual_release_time']==32 and all(not any(f['controlled_mask']) for f in u['network_frames'] if f['t']>=32)


