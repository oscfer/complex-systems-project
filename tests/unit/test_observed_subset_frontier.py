"""Focused checks for the estimated-frontier subset search (no simulation)."""
import numpy as np
import pytest

from src.control.contracts import Observation
from src.control.optimizers.observed_subsets import (
    SubsetSearchConfig, _dominates, search_observed_subsets,
)


def _observation(N=10, roster=None, edges=None, **kwargs):
    roster = tuple(roster) if roster is not None else tuple(range(N))
    return Observation(
        time=0.0, public_states=np.full((N, 1), 1.0), panel_mask=np.ones(N, bool),
        observed_edges=edges or {(0, 1): 3, (0, 2): 5, (1, 3): 2, (2, 3): 7},
        accessible_roster=roster, group_labels=None, sample_time=0.0, **kwargs
    )


class MockRollout:
    def __init__(self, success):
        self.is_valid = True
        self.is_eligible = True
        self.success = success
        self.relapsed = False
        self.restricted_time = 10.0 if success else 100.0
        self.terminal_target_violation = 0.1
        self.expenditure = 50.0
        self.summaries = {'horizon': 320.0}


class MockEvaluator:
    """Deterministic mock: success iff the group contains node 0 (size 4)
    or covers >= 3 low-influence nodes (size 10)."""

    def __init__(self, good_four):
        self.good_four = good_four
        self.calls = 0

    def __call__(self, candidate, seeds):
        self.calls += 1
        nodes = set(candidate.target_nodes)
        if len(nodes) == 4:
            outcome = nodes == set(self.good_four)
        else:
            outcome = len(nodes - {0, 1, 2, 3}) >= 8
        return [MockRollout(outcome) for _ in seeds]


def test_dominates_basic():
    a = (0.5, 10.0, 4.0, 3.0, 3.0)
    assert _dominates(a, (0.4, 12.0, 4.0, 5.0, 5.0))
    assert not _dominates(a, (0.5, 10.0, 4.0, 3.0, 3.0))  # equal is not domination
    assert not _dominates((0.5, 10.0, 4.0, 3.0, 3.0), (0.6, 9.0, 3.0, 4.0, 4.0))


def test_frontier_and_representatives():
    obs = _observation()
    evaluator = MockEvaluator(good_four=(0, 1, 2, 3))
    config = SubsetSearchConfig(
        sizes=(4, 10), amplitude=2.0, duration=19.0, candidate_limit=8,
        beam_width=1, search_futures=2, validation_futures=2,
        reliability_target=0.8, seed=5, representative_limit=2)
    result = search_observed_subsets(obs, evaluator, target_sign=-1, config=config)

    assert result.budget_exhausted and len(result.candidates) == 8
    assert len(result.representatives) == 2
    assert result.selected in result.representatives
    # Every representative got one audit-only validation batch.
    assert len(result.validation_summaries) == 2
    for summary in result.validation_summaries:
        assert summary['used_for_selection'] is False
    # Forecast accounting counts search plus validation rollouts, per call.
    assert result.forecast_calls == 8 + 2
    assert result.forecast_rollouts == 8 * 2 + 2 * 2
    # Positive observed attention: 2,3,5,9 map to costs1,2,3,4 respectively.
    assert result.influence_ranks[3] == 3  # observed totals are 3,5,9
    assert result.influence_ranks[1] == 1
    assert result.influence_ranks[0] > result.influence_ranks[3]  # unknown is not low-power
    assert len(result.frontier) >= 1
    # Summaries carry the frozen influence objectives.
    for summary in result.summaries:
        assert {'mean_influence_rank', 'max_influence_rank'} <= set(summary)
    # Hidden payloads never enter the search.
    hidden = _observation(private_states=np.zeros((10, 1)))
    with pytest.raises(ValueError):
        search_observed_subsets(hidden, evaluator, target_sign=-1, config=config)


def test_power_cost_ties_and_unknowns():
    from src.control.optimizers.observed_subsets import _influence_ranks
    ranks = _influence_ranks([0,1,2,3], np.array([0,2,2,9]), {0:0,1:1,2:2,3:3})
    assert ranks[1] == ranks[2] < ranks[3] < ranks[0]
    assert _dominates((1,120,4,ranks[1],ranks[1]),(1,120,4,ranks[3],ranks[3]))
