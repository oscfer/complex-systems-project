"""Focused new checks for combined field accounting and checkpoint persistence."""
import json
from dataclasses import replace
import numpy as np
import pytest
from src.control.contracts import ControlProblem, Observation, TargetType
from src.control.ledger import BudgetLedger
from src.control.transformation_study import make_policy, run_transformation_study


def setup(budget=760, cap=4, k=40):
    p=ControlProblem(problem_id='combined',target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':-1,'threshold':.5},
        N=40,k=k,exposure_cap=budget,amplitude_cap=cap,accessible_roster=tuple(range(40)),
        release_deadline=20,holding_duration=100,horizon=320)
    o=Observation(time=1,public_states=np.ones((40,1)),panel_mask=np.ones(40,dtype=bool),
                  observed_edges={(i,(i+1)%40):i+1 for i in range(40)},accessible_roster=tuple(range(40)))
    ctl=make_policy('combined_broadcast_08_boost_2');ctl.reset(p,o,np.random.default_rng(2))
    ledger=BudgetLedger(N=40,k=k,exposure_cap=budget,amplitude_cap=cap,accessible_roster=set(range(40)),release_deadline=20)
    return ctl,ledger,o


def test_equal_dose_combined_fields_and_release():
    ctl,ledger,o=setup();action=ctl.act(o,ledger)
    assert len(action.entries)==40
    assert sum(np.isclose(e.field,-2.8) for e in action.entries)==4
    assert sum(np.isclose(e.field,-.8) for e in action.entries)==36
    assert action.spending_rate==pytest.approx(40)
    ledger.apply_action(action,1);ledger.advance_to(20)
    assert ledger.integrated_exposure==pytest.approx(760)
    assert ctl.act(replace(o,time=20),ledger).release


@pytest.mark.parametrize('budget,cap',[(380,4),(760,1)])
def test_combined_field_scales_both_components_to_permissions(budget,cap):
    ctl,ledger,o=setup(budget,cap);action=ctl.act(o,ledger)
    assert max(abs(e.field) for e in action.entries)<=cap+1e-12
    assert action.spending_rate*19<=budget+1e-9
    assert ctl.effective_settings['targeted_increment']/ctl.effective_settings['broadcast_amplitude']==pytest.approx(2.5)
    assert ledger.validate_action(action,current_time=1)[0]


def test_broadcast_cannot_bypass_distinct_node_limit():
    with pytest.raises(ValueError,match='distinct-node permission'):
        setup(k=4)


def test_combined_checkpoint_parity(tmp_path):
    kw=dict(smoke=True,initial_conditions='tiered_reference',policy_packages=(('combined_broadcast_08_boost_2','broad_diagnostic'),),include_zero_baselines=False,progress=False)
    bundles=[run_transformation_study(tmp_path,checkpoint_interval=cut,**kw) for cut in (100,1.5)]
    units=[]
    for b in bundles:
        m=json.loads((b/'manifest.json').read_text());units.append(json.loads((b/next(iter(m['units'].values()))['path']).read_text()))
    for key in ('final_state','expenditure','event_counts','completion_time','success_through_horizon'):
        assert units[0][key]==units[1][key]
    assert units[0]['summaries']['effective_settings']==units[1]['summaries']['effective_settings']
    assert units[0]['is_valid'] and units[1]['is_valid']
