"""Experiment design, interventions, and reference baselines.

Provides:
- Initial network topology & belief state generators (generators.py):
    Tabula rasa emergence, decentralized, modular, concentrated hub, aligned states.
- Scheduled experimental interventions (interventions.py):
    Field pulse shocks, structural tie disruptions, multi-population interface schedules, speed rescaling.

Simulation Stage Mapping:
    - Stage 0: Initial state creation for timing benchmarks (generate_emergence_state).
    - Stage 1: Initial condition generation for 450 pilot trajectories.
    - Stage 2: Extension campaigns (6 experimental perturbation families).
"""

from src.experiment.generators import (
    generate_aligned_state,
    generate_aligned_states,
    generate_decentralized_network,
    generate_decentralized_state,
    generate_emergence_state,
    generate_endpoint_continuous,
    generate_hub_network,
    generate_hub_state,
    generate_modular_network,
    generate_modular_state,
    generate_opposing_populations_state,
)
from src.experiment.interventions import (
    FieldPulseIntervention,
    InterfaceSchedule,
    TieDisruptionIntervention,
    apply_tie_disruption,
    create_targeted_pulse,
    create_uniform_pulse,
    rescale_mechanism_speed,
)
from src.experiment.baselines import (
    BaselineComparison,
    BaselineCondition,
    BaselineDynamics,
    BaselineExperimentConfig,
    ConstructedOriginSpec,
    ReferenceOrigin,
    ReliabilityEstimand,
    ReliabilitySummary,
    build_constructed_origin,
    run_reference_baselines,
)

__all__ = [
    # Generators
    "generate_emergence_state",
    "generate_decentralized_network",
    "generate_decentralized_state",
    "generate_modular_network",
    "generate_modular_state",
    "generate_hub_network",
    "generate_hub_state",
    "generate_aligned_states",
    "generate_aligned_state",
    "generate_endpoint_continuous",
    "generate_opposing_populations_state",
    # Interventions
    "FieldPulseIntervention",
    "TieDisruptionIntervention",
    "apply_tie_disruption",
    "create_targeted_pulse",
    "create_uniform_pulse",
    "InterfaceSchedule",
    "rescale_mechanism_speed",
    # reference experiments
    "BaselineComparison",
    "BaselineCondition",
    "BaselineDynamics",
    "BaselineExperimentConfig",
    "ConstructedOriginSpec",
    "ReferenceOrigin",
    "ReliabilityEstimand",
    "ReliabilitySummary",
    "build_constructed_origin",
    "run_reference_baselines",
]
