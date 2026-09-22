import pytest
import os
import shutil
from src.config.tiered_profiles import LIGHT_TIERED
from src.control.campaign import run_campaign

def test_tiered_light_campaign_smoke(tmp_path):
    output_dir = tmp_path / "tiered_test_out"
    
    # Run the control campaign using the tiered_light profile
    # We override some origins to be very small for a fast smoke test
    test_profile = LIGHT_TIERED.model_copy(update={
        "N": 40,
        "n_origins": 3,
        "training_origins": 1,
        "selection_origins": 1,
        "evaluation_origins": 1,
        "grid_origins": 1,
        "diagnostic_origins": 1,
        "candidate_cap": 2,
    })
    
    from src.config.scenarios import tiered_origin_provider

    res = run_campaign(
        profile=test_profile,
        out_dir=str(output_dir),
        origin_provider=tiered_origin_provider,
        resume=False
    )
    
    assert res is not None
    manifests = list(output_dir.glob("**/manifest.json"))
    assert len(manifests) > 0, "No manifest.json found in output directory"
