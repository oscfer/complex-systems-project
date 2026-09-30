"""Focused checks for the batch1 campaign module (no long-running simulation).

Covers: plan-only queue counting, the forecast-gate metric on synthetic cases,
and gate-case loader verification/exclusion against a fake saved bundle. A
tiny N8 smoke of the actual-unit runner lives with the batch notebook checks.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.control.optimizers.campaign import (
    CampaignConfig, _actual_queue, _spearman, load_gate_cases, run_batch1_campaign,
    run_forecast_gate,
)
from src.config.tiered_profiles import LIGHT_TIERED


def test_spearman_known_values():
    assert _spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # Ties handled with average ranks: near-agreement despite one tie.
    assert _spearman([1, 1, 3], [1, 2, 3]) > 0.8
    # Fewer than two points is not a correlation.
    assert np.isnan(_spearman([1.0], [1.0]))


def _profile():
    return LIGHT_TIERED.model_copy(update={'N': 8})


def test_plan_only_counts_match_config():
    config = CampaignConfig(
        search_origins=('dev_00', 'dev_01'), candidate_limit=8,
        search_futures=2, validation_futures=2, representative_limit=2,
        gate_cases=({'origin_id': 'x', 'policy': 'p'}, {'origin_id': 'y', 'policy': 'q'}),
        gate_forecast_futures=2)
    plan = run_batch1_campaign(
        out_dir='unused', config=config, profile=_profile(), contract={},
        gate_cases=[], origin_providers={'dev_00': 1, 'dev_01': 2}, plan_only=True)
    assert plan['gate_forecast_simulations'] == 4
    assert plan['search_forecast_simulations'] == 2 * (8 * 2 + 2 * 2)
    assert plan['forecast_ceiling'] == 4 + 2 * (8 * 2 + 2 * 2)
    # Two representatives + up to four simple selectors + one matched zero per
    # package, per origin.
    assert plan['planned_actual_units'] == 2 * (2 + 8 + 1)
    assert plan['planned_actual_breakdown']['matched_zeros'] == 2
    assert plan['fallback_branch'].startswith('simple_selector')


def test_package_permission_validation():
    with pytest.raises(ValueError):
        # Size 10 under the original package (k=4) is infeasible.
        CampaignConfig(search_origins=('dev_00',),
                       package_by_size=('original', 'original'))


def test_fallback_queue_needs_no_surrogate():
    config = CampaignConfig(search_origins=('dev_00',), candidate_limit=8)
    profile = LIGHT_TIERED
    # Gate-failure fallback: heuristic selectors computed from the restricted
    # t0 observation only, plus matched zeros; no search record is consulted.
    queue = _actual_queue(config, {'fallback': True}, profile, 'dev_00', 2678435231)
    kinds = [kind for *_, kind in queue]
    assert sum(k.startswith('selector_') for k in kinds) == 8
    # No size-10 arms in the fallback, so only the selector package needs a zero.
    assert sorted(k for k in kinds if k.startswith('zero_')) == ['zero_broad_diagnostic']
    # Actuated entries are capped at the predeclared selector size (the tiny
    # N8 test roster has fewer accessible nodes than the real N40 roster).
    assert all(0 < len(nodes) in config.sizes for nodes, *_ in queue if nodes)
    # Normal path: representatives first, then selectors and zeros.
    record = {'representatives': [{'target_nodes': (0, 1, 2, 3)}],
              'candidates': [{'target_nodes': (5, 6, 7, 8),
                              'summary': {'heuristic_name': 'random'}}]}
    queue = _actual_queue(config, record, profile, 'dev_00', 2678435231)
    kinds = [kind for *_, kind in queue]
    assert kinds[0] == 'representative' and 'selector_random_4' in kinds
    # The size-4 representative uses the original package, so only that zero.
    assert 'zero_broad_diagnostic' in kinds


def _write_fake_bundle(bundle: Path):
    (bundle/'units').mkdir(parents=True)
    # Saved arm: two nodes at amplitude 4 for 8 duration units = 64 exposure.
    unit = {'unit': {'origin': {'origin_id': 'development_01', 'seed': 11},
                     'policy': 'test_arm', 'package': 'original'},
            'is_valid': True, 'is_eligible': True, 'success': True,
            'relapsed': False, 'expenditure': 64.0, 'summaries': {'effective_settings': {'initial_nodes': [0,1]}}}
    (bundle/'units'/'u1.json').write_text(json.dumps(unit))
    manifest = {'units': {'u1': {'path': 'units/u1.json', 'sha256': __import__('hashlib').sha256((bundle/'units'/'u1.json').read_bytes()).hexdigest()}}}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    return [
        {'origin_id': 'development_01', 'seed': 11, 'policy': 'test_arm',
         'package_id': 'original', 'recipient_nodes': [0, 1],
         'amplitude': 4.0, 'duration': 8.0, 'provenance': 'audit'},
        {'origin_id': 'development_01', 'seed': 11, 'policy': 'missing_arm',
         'package_id': 'original', 'recipient_nodes': [0], 'amplitude': 1.0,
         'duration': 8.0, 'provenance': 'audit'},
        # Same policy/arm but a wrong declared dose must be caught.
        {'origin_id': 'development_01', 'seed': 11, 'policy': 'test_arm',
         'package_id': 'original', 'recipient_nodes': [0, 1],
         'amplitude': 4.0, 'duration': 9.0, 'provenance': 'x'},
    ]


def test_gate_case_loader_verifies_and_excludes():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)/'fake_bundle'
        rows = _write_fake_bundle(bundle)
        cases = load_gate_cases(rows, [bundle])
    matched = [c for c in cases if c.get('provenance', '').startswith('audit')]
    assert len(matched) == 1
    assert matched[0]['excluded'] is False
    assert matched[0]['actual_hold_success'] is True
    # Dose mismatch is excluded, never silently relabeled.
    mismatched = [c for c in cases if c.get('exclusion_reason', '').startswith('expenditure')]
    assert len(mismatched) == 1 and mismatched[0]['excluded'] is True
    # A row with no matching saved unit is excluded, never fabricated.
    missing = [c for c in cases if c['policy'] == 'missing_arm']
    assert len(missing) == 1 and missing[0]['excluded'] is True
    assert 'no matching' in missing[0]['exclusion_reason']
