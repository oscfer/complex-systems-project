"""New report calculations and repaired metadata lookup, without simulation."""
from types import SimpleNamespace

from src.visualization.transformation_report import target_residence
from src.visualization.control_plots import _selection_reason_lines


def test_exact_residence_separates_duration_and_uninterrupted_hold():
    r=target_residence([(0,False),(20,True),(50,False),(60,True)],20,100)
    assert r['duration']==70
    assert r['longest_uninterrupted']==40
    tied=target_residence([(0,True),(5,False),(5,True)],0,10)
    assert tied['duration']==10
    assert tied['longest_uninterrupted']==5


def test_saved_controller_caption_reads_result_summary():
    b=SimpleNamespace(manifest={'selection_summary':{'selected_candidate_id':'p'}},
        work_units={'u':{'summaries':{'effective_settings':{'controller_inferred_target_sign':-1,'inference_source':'full_panel_mean'}}}})
    lines=_selection_reason_lines(b,'u',True)
    assert any('controller_inferred_target_sign' in line and '-1' in line for line in lines)
    assert not any('not recorded' in line for line in lines)
