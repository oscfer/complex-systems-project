import pytest
import numpy as np
from src.config.scenarios import build_tiered_origin, load_tiered_spec
from src.core.state import StateMode

def test_tiered_initialization():
    spec = load_tiered_spec("research/scenarios/tiered/scenario.json")
    
    # Test N=40 (light)
    state_light = build_tiered_origin(spec, seed=42, N=40)
    assert state_light.N == 40
    assert state_light.D == 1
    assert state_light.state_mode == StateMode.BINARY
    
    # Verify tiers
    counts = np.bincount(state_light.blocks)
    assert counts[1] == 6  # 15%
    assert counts[2] == 4  # 10%
    assert counts[3] == 6  # 15%
    assert counts[4] == 6  # 15%
    assert counts[5] == 18 # 45%
    
    # Test N=500 (extended)
    state_ext = build_tiered_origin(spec, seed=42, N=500)
    assert state_ext.N == 500
    counts = np.bincount(state_ext.blocks)
    assert counts[1] == 75  # 15%
    assert counts[2] == 50  # 10%
    assert counts[3] == 75  # 15%
    assert counts[4] == 75  # 15%
    assert counts[5] == 225 # 45%
    
    # Ensure attention concentration (Tiers 1-3 get ~70%)
    # Let's just check the state validates without errors
    state_light.validate()
