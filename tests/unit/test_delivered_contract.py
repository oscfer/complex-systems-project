"""Bounded checks of shipped artifacts; runner imports are intercepted in-module."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.tiered_profiles import LIGHT_TIERED, EXTENDED_TIERED, LIGHT_TIERED_SMOKE
from src.control.evaluator import TargetEvaluator
from src.control.contracts import TargetType
from src.core.state import State, StateMode

ROOT = Path(__file__).resolve().parents[2]




@pytest.mark.parametrize('initial_sign', [-1., 1.])
@pytest.mark.parametrize('private_count,public_count,expected', [(30,30,True),(29,30,False),(30,29,False),(40,40,True)])
def test_75_percent_requires_both_layers(initial_sign, private_count, public_count, expected):
    state = State.create_empty(N=40, D=1, state_mode=StateMode.BINARY)
    state.x.fill(initial_sign)
    state.y.fill(initial_sign)
    evaluator = TargetEvaluator.create_for_state(
        TargetType.REVERSAL, state,
        holding_duration=LIGHT_TIERED.holding_duration,
        horizon=LIGHT_TIERED.horizon,
        release_deadline=LIGHT_TIERED.release_deadline,
        target_params={'threshold': LIGHT_TIERED.target_threshold, 'eligibility_mode': 'public_majority'},
    )
    state.x[:private_count] = -initial_sign
    state.y[:public_count] = -initial_sign
    assert evaluator.evaluate_target_condition(state) is expected
