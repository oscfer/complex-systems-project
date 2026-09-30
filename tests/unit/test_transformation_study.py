"""New focused-runner checks with tiny explicitly shortened N=8 rehearsals."""
import json

import numpy as np
import pytest

from src.control.transformation_study import run_transformation_study


def read_units(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    return m, {k:json.loads((bundle/v['path']).read_text()) for k,v in m['units'].items()}


def test_smoke_matched_rosters_and_durable_resume(tmp_path):
    bundle=run_transformation_study(tmp_path,smoke=True,checkpoint_interval=2.5,max_units=1,progress=False)
    m,units=read_units(bundle)
    assert m['status']=='incomplete'
    zero_info=next(iter(m['units'].values()))
    original_zero=(bundle/zero_info['path']).read_bytes()
    run_transformation_study(tmp_path,smoke=True,checkpoint_interval=2.5,resume_bundle=bundle,progress=False)
    m,units=read_units(bundle)
    assert m['status']=='complete' and len(units)==5
    assert (bundle/zero_info['path']).read_bytes()==original_zero
    arms={u['unit']['policy']:u for u in units.values()}
    attention=[arms[p] for p in ('attention_sustained_1','attention_sustained_4','attention_shutoff_1')]
    assert len({tuple(u['summaries']['effective_settings']['initial_nodes']) for u in attention})==1
    for u in units.values():
        assert u['final_state']['t']==6
        assert u['expenditure']<=u['geometry']['exposure_cap']+1e-10
        assert u['distinct_nodes_used']<=u['geometry']['k']
        assert u['network_frames']
    assert not list((bundle/'checkpoints').glob('*.json'))


def test_new_policy_checkpoint_parity(tmp_path):
    pairs=(('attention_shutoff_1','original'),)
    a=run_transformation_study(tmp_path,smoke=True,policy_packages=pairs,checkpoint_interval=100,progress=False)
    b=run_transformation_study(tmp_path,smoke=True,policy_packages=pairs,checkpoint_interval=2.5,progress=False)
    _,ua=read_units(a); _,ub=read_units(b)
    for key in ua:
        for field in ('final_state','expenditure','distinct_nodes_used','completion_time','success','relapsed','event_counts'):
            assert ua[key][field]==ub[key][field],field
        assert ua[key]['summaries']['effective_settings']==ub[key]['summaries']['effective_settings']
        assert ua[key]['summaries']['target_condition_transitions']==ub[key]['summaries']['target_condition_transitions']


def test_selection_rejects_implicit_search_and_source_mismatch(tmp_path):
    with pytest.raises(ValueError,match='frozen policy_packages'):
        run_transformation_study(tmp_path,stage='selection')
    bundle=run_transformation_study(tmp_path,smoke=True,policy_packages=(('zero','original'),),progress=False)
    p=bundle/'manifest.json'; m=json.loads(p.read_text());m['config']['source_sha256']='wrong';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='mismatch'):
        run_transformation_study(tmp_path,smoke=True,policy_packages=(('zero','original'),),resume_bundle=bundle,progress=False)


def test_reference_family_preserves_provider_initialization(tmp_path):
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.config.scenarios import tiered_origin_provider
    from src.control.checkpointing import serialize_state
    bundle=run_transformation_study(tmp_path,smoke=True,initial_conditions='tiered_reference',
        policy_packages=(('zero','original'),),progress=False)
    manifest,units=read_units(bundle)
    row=manifest['config']['queue'][0]['origin']
    expected=tiered_origin_provider(LIGHT_TIERED.model_copy(update={'N':8}),row['seed'])
    saved=json.loads((bundle/manifest['origins'][row['origin_id']]['path']).read_text())
    assert saved==serialize_state(expected)
    assert manifest['config']['initial_conditions']=='tiered_reference'


def test_structural_target_changes_certificate_not_field_policy(tmp_path):
    pairs=(('attention_sustained_4','original'),)
    a=run_transformation_study(tmp_path,smoke=True,policy_packages=pairs,checkpoint_interval=2.5,progress=False)
    b=run_transformation_study(tmp_path,smoke=True,policy_packages=pairs,target_kind='structural',checkpoint_interval=2.5,progress=False)
    _,ua=read_units(a); mb,ub=read_units(b)
    assert mb['config']['target_kind']=='structural'
    for key in ua:
        for field in ('x','y','W'):
            np.testing.assert_array_equal(ua[key]['final_state'][field],ub[key]['final_state'][field])
        assert ua[key]['expenditure']==ub[key]['expenditure']
        assert ub[key]['target_definition']['target_type']=='decentralization'
