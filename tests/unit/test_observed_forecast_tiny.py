"""Tiny real-engine forecast smoke for the restricted search path.

One short-horizon forecast rollout at N=8 through the shared engine on an
estimated world: software check only, not scientific evidence and not a
validation of forecast quality.
"""
import numpy as np
import pytest

from src.control.contracts import ControlProblem, TargetType
from src.control.optimizers.observed_forecasts import ForecastAssumptions, ObservedInitialForecasts
from src.control.optimizers.observed_subsets import SubsetSearchConfig, search_observed_subsets
from src.control.observation import CompletePublicObservationModel
from src.config.scenarios import tiered_origin_provider


def _tiny_state(N=8):
    from src.config.tiered_profiles import LIGHT_TIERED
    profile = LIGHT_TIERED.model_copy(update={'N': N})
    return tiered_origin_provider(profile, 7)


def test_tiny_forecast_search_smoke():
    state = _tiny_state()
    N = state.N
    problem = ControlProblem(
        problem_id='smoke_search', target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode': 'declared', 'declared_destination_sign': -1,
                       'threshold': 0.5}, N=N, D=1, k=4, exposure_cap=16.0,
        amplitude_cap=4.0, accessible_roster=tuple(range(4)),
        release_deadline=4.0, holding_duration=1.0, horizon=6.0)
    observation = CompletePublicObservationModel(
        N, accessible_roster=tuple(range(4))).sample_observation(state=state, current_time=0.0)
    evaluator = ObservedInitialForecasts(
        observation, problem, target_sign=-1, assumptions=ForecastAssumptions(
            row_memory=4, public_error_probability=0.05, private_public_agreement=0.8,
            tau_private=5.0, encounter_rate=2.0))
    config = SubsetSearchConfig(
        sizes=(2, 4), amplitude=1.0, duration=3.0, candidate_limit=8, beam_width=1,
        search_futures=2, validation_futures=2, reliability_target=0.8, seed=3,
        representative_limit=2)
    result = search_observed_subsets(observation, evaluator, target_sign=-1, config=config)
    # Accounting is internally consistent: search calls (7 unique candidates
    # after dedup) plus one validation batch per representative, each with the
    # declared future count.
    n_search_calls = result.forecast_calls - len(result.representatives)
    assert result.forecast_rollouts == (result.forecast_calls - len(result.representatives)) * 2 \
        + len(result.representatives) * 2
    assert len(result.candidates) <= config.candidate_limit
    for summary in result.validation_summaries:
        assert summary['evidence_kind'] == 'forecast_only'
    assert len(result.representatives) == 2
    # Validation is audit-only and labeled as such.
    assert all(s.get('used_for_selection') is False for s in result.validation_summaries)
