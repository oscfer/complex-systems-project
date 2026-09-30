"""Tiny, non-simulating checks of the v1 scientific design."""
import numpy as np
import pytest

from src.config.tiered_profiles import LIGHT_TIERED
from src.config.transformation import build_transformation_origin, load_transformation_contract, resolve_resource_package


def test_frozen_allocation_has_disjoint_seeds_and_declared_cells():
    c = load_transformation_contract()
    rows = c['development'] + c['selection']
    assert len({r['seed'] for r in rows}) == len(rows)
    assert len({r['origin_id'] for r in rows}) == len(rows)
    assert {r['cell_id'] for r in c['development']} == {x['id'] for x in c['cells']}
    assert c['state_mean_threshold'] == 2*c['state_support_fraction']-1 == .5
    assert (c['holding_duration'], c['release_deadline'], c['horizon']) == (100,20,320)


@pytest.mark.parametrize('cell', ['c0','c1','c2','c3'])
def test_origin_replay_and_initialization_invariants(cell):
    state = build_transformation_origin(LIGHT_TIERED, 601234, cell)
    repeat = build_transformation_origin(LIGHT_TIERED, 601234, cell)
    for name in ('x','y','W','K','tau_x','nu','g','b','opportunity_groups','opportunity_matrix'):
        np.testing.assert_array_equal(getattr(state,name),getattr(repeat,name))
    assert np.count_nonzero(state.y==1) == 32
    assert np.count_nonzero(state.x!=state.y) == 4
    assert np.all(state.W.sum(axis=1)==20)
    assert np.all(np.diag(state.W)==0)
    assert np.all(np.diag(state.opportunity_matrix)==0)
    assert len(state.scenario_metadata['core_indices'])==8
    assert state.t==0


def test_factor_changes_preserve_stance_traits_and_core():
    a = build_transformation_origin(LIGHT_TIERED, 982341,'c0')
    for cell in ['c1','c2','c3']:
        b = build_transformation_origin(LIGHT_TIERED, 982341,cell)
        for name in ('x','y','K','tau_x','nu','g','b','opportunity_groups'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        assert a.scenario_metadata['core_indices']==b.scenario_metadata['core_indices']
        assert not np.array_equal(a.W,b.W)


@pytest.mark.parametrize('name,k,access,exposure', [('original',4,10,320),('wider',10,10,800),('broad_diagnostic',40,40,3200)])
def test_package_geometry_and_scaling(name,k,access,exposure):
    p = resolve_resource_package(name,40)
    assert (p['k'],p['accessible_count'],p['exposure_cap'])==(k,access,exposure)
    large = resolve_resource_package(name,500)
    assert large['k']==round(k*12.5)
    assert large['accessible_count']==round(access*12.5)
    assert large['exposure_cap']==exposure*12.5
