"""New declared-task contract; no campaign or established-suite execution."""
import numpy as np
import pytest

from src.control.contracts import ControlProblem, Observation, TargetType
from src.control.controllers import ObservableSelectorPolicy
from src.control.evaluator import TargetEvaluator
from src.control.ledger import BudgetLedger
from src.core.state import State


def problem(sign=-1):
    return ControlProblem(problem_id='declared_v1', target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode':'declared','declared_destination_sign':sign,'threshold':.5},
        N=8,k=2,exposure_cap=160,amplitude_cap=4,accessible_roster=tuple(range(8)),
        holding_duration=100,release_deadline=20,horizon=320, config_version='persistent-transformation-v1')


def observation(value=0.,t=1.):
    return Observation(time=t,public_states=np.full((8,1),value),panel_mask=np.ones(8,dtype=bool),
        observed_edges={(0,1):4,(0,2):3},accessible_roster=tuple(range(8)))


def test_declared_evaluator_does_not_derive_direction_from_initial_majority():
    state=State.create_empty(N=8)
    state.x.fill(1)
    state.y.fill(-1)  # Legacy reversal would choose +1, explicitly declared task chooses -1.
    e=TargetEvaluator.create_for_state(TargetType.REVERSAL,state,target_params=problem().target_params)
    assert e.defn.is_eligible
    assert e.defn.s_star == -1
    state.x[:6]=-1
    assert e.evaluate_target_condition(state)


@pytest.mark.parametrize('sign',[0,2,float('nan')])
def test_invalid_declared_direction_rejected(sign):
    with pytest.raises(ValueError,match='declared_destination_sign'):
        TargetEvaluator.create_for_state(TargetType.REVERSAL,State.create_empty(N=8),target_params=problem(sign).target_params)


def test_declared_policy_acts_on_tied_report_and_records_task_provenance():
    p=problem()
    ctl=ObservableSelectorPolicy(initial_recruitment_fraction=1,nominal_amplitude=1,duration=20)
    ctl.reset(p,observation(),rng=np.random.default_rng(54))
    ledger=BudgetLedger(N=8,k=2,exposure_cap=160,accessible_roster=set(range(8)))
    action=ctl.act(observation(),ledger)
    assert len(action.entries)==2
    assert all(e.field == -1 for e in action.entries)
    assert ctl.effective_settings['direction_source']=='declared_task'
    assert 'controller_inferred_target_sign' not in ctl.effective_settings
