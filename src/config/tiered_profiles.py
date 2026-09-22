from pydantic import Field
from src.config.control_profiles import ControlProfile

class TieredProfile(ControlProfile):
    """Profile for Tiered-population scenarios."""
    target_type: str = "reversal"
    # Signed binary mean: 0.5 means 75% destination support in EACH layer.
    target_threshold: float = 0.5
    holding_duration: float = 100.0
    release_deadline: float = 20.0
    horizon: float = 320.0
    reliability_target: float = 0.8
    
    target_component: int = 0
    eligibility_mode: str = "public_majority"
    
    accessible_fraction: float = 0.25
    k_distinct_nodes_fraction: float = 0.1
    exposure_cap_normalized: float = 2.0
    amplitude_cap_over_J: float = 4.0
    
    # Specifics for tiered
    scenario_spec_path: str = "research/scenarios/tiered/scenario.json"
    grid_origins: int = Field(default=2, ge=1)
    diagnostic_origins: int = Field(default=2, ge=1)

LIGHT_TIERED = TieredProfile(
    topic="tiered",
    tier="light",
    N=40,
    n_origins=6,
    training_origins=2,
    training_futures_per_origin=1,
    selection_candidates=2,
    selection_origins=1,
    selection_futures_per_origin=1,
    evaluation_origins=3,
    evaluation_futures_per_origin=1,
    grid_origins=2,
    diagnostic_origins=2,
)

# Explicit tiny-scale pipeline rehearsal (audit F1a). This profile is a SMOKE
# run for software validation only: it shrinks the population AND the declared
# time/target contract (short horizon, short holding, early release), so it
# must never be presented as, or silently substituted for, the nominal light
# study above. The nominal LIGHT_TIERED profile is the only light-tier study
# profile; notebooks may use LIGHT_TIERED or LIGHT_TIERED_SMOKE, never an
# unnamed inline reduction of LIGHT_TIERED.
LIGHT_TIERED_SMOKE = TieredProfile(
    topic="tiered",
    tier="light",
    # Tiny population and shortened declared contract (pipeline rehearsal).
    N=12,
    D=1,
    horizon=24.0,
    establishment_time=8.0,
    n_origins=8,
    seed_base=8800,
    holding_duration=4.0,
    release_deadline=8.0,
    k_distinct_nodes_fraction=0.25,
    exposure_cap_normalized=0.5,
    candidate_cap=4,
    selectors=("reported_attention", "random"),
    observed_slot_quotas=(0.0, 1.0),
    feedback_choices=(False, True),
    training_origins=2,
    training_futures_per_origin=1,
    selection_candidates=2,
    selection_origins=2,
    selection_futures_per_origin=2,
    evaluation_origins=4,
    evaluation_futures_per_origin=1,
    grid_origins=1,
    diagnostic_origins=1,
)

EXTENDED_TIERED = TieredProfile(
    topic="tiered",
    tier="extended",
    N=500,
    n_origins=54,
    training_origins=12,
    training_futures_per_origin=2,
    selection_candidates=4,
    selection_origins=12,
    selection_futures_per_origin=2,
    evaluation_origins=30,
    evaluation_futures_per_origin=1,
    grid_origins=4,
    diagnostic_origins=2,
)
