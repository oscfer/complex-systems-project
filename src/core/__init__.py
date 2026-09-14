"""Core mathematical models, state containers, and continuous-time update kernels.

Simulation Stages Mapping:
-------------------------
Stage 0 (Timing & Calibration): scheduler.py, heat_baths.py, memory.py (kernel benchmarking).
Stage 1 (Core Emergence): state.py, fields.py, heat_baths.py, memory.py, couplings.py.
Stage 2 (Extension Campaigns): participation.py (coordination games), fields.py (pulses).
Stage 3 (Adaptive Learning): q_learning.py, participation.py.
Stage 4 (Diagnostics): state.py (discrepancy property d_2).
"""

from src.core.state import (
    State,
    StateMode,
)
from src.core.fields import (
    FieldPulseSpec,
    compute_attention_matrix,
    compute_external_field,
    compute_private_fields,
    compute_public_fields,
    compute_received_attention,
    compute_single_private_field,
    compute_single_public_field,
    compute_single_row_attention,
)
from src.core.heat_baths import (
    binary_glauber_prob_plus,
    calibrate_continuous_temperature,
    sample_binary_glauber_scalar,
    sample_binary_glauber_vector,
    sample_continuous_inverse_cdf_scalar,
    sample_continuous_inverse_cdf_vector,
)
from src.core.couplings import (
    calculate_internal_fields,
    calculate_isolated_internal_energy,
    enumerate_triad_energy_spectrum,
    get_coupling_matrix,
    validate_coupling_matrix,
)
from src.core.memory import (
    EncounterResult,
    MemoryManager,
)
from src.core.participation import (
    ParticipationParams,
    ParticipationTracker,
    compute_participation_potential,
)
from src.core.q_learning import (
    QLearningParams,
    action_to_index,
    compute_immediate_reward,
    index_to_action,
    init_q_table,
    select_action,
    softmax_probabilities,
    step_q_learning,
    update_chosen_q_value,
)
from src.core.scheduler import (
    PRIORITY_HOLDING,
    PRIORITY_HORIZON,
    PRIORITY_INTERVENTION,
    PRIORITY_RESTORE,
    PRIORITY_SNAPSHOT,
    DeterministicEvent,
    PoissonScheduler,
    SchedulerStats,
    calculate_total_event_rate,
)
from src.core.heterogeneity import (
    DEFAULT_BIAS_BOUNDS_OVER_J,
    DEFAULT_BIAS_SD_OVER_J,
    DEFAULT_CROSS_GROUP_ELIGIBILITY,
    DEFAULT_MULTIPLIER_BOUNDS,
    DEFAULT_NUM_OPPORTUNITY_GROUPS,
    DEFAULT_ROW_CAP,
    DEFAULT_SUSCEPTIBILITY_BOUNDS,
    DEFAULT_WITHIN_GROUP_ELIGIBILITY,
    QuenchedHeterogeneity,
    calculate_cross_group_exploration_probability,
    clipped_lognormal_multiplier,
    generate_opportunity_groups,
    generate_quenched_heterogeneity,
    generate_susceptibilities,
    generate_truncated_normal_biases,
)

__all__ = [
    # State
    "State",
    "StateMode",
    # Fields
    "compute_attention_matrix",
    "compute_received_attention",
    "compute_single_row_attention",
    "compute_public_fields",
    "compute_single_public_field",
    "compute_private_fields",
    "compute_single_private_field",
    "FieldPulseSpec",
    "compute_external_field",
    # Heat baths
    "binary_glauber_prob_plus",
    "sample_binary_glauber_scalar",
    "sample_binary_glauber_vector",
    "sample_continuous_inverse_cdf_scalar",
    "sample_continuous_inverse_cdf_vector",
    "calibrate_continuous_temperature",
    # Couplings
    "get_coupling_matrix",
    "validate_coupling_matrix",
    "calculate_internal_fields",
    "calculate_isolated_internal_energy",
    "enumerate_triad_energy_spectrum",
    # Memory
    "EncounterResult",
    "MemoryManager",
    # Participation
    "ParticipationParams",
    "ParticipationTracker",
    "compute_participation_potential",
    # Q-learning
    "QLearningParams",
    "action_to_index",
    "index_to_action",
    "init_q_table",
    "softmax_probabilities",
    "select_action",
    "compute_immediate_reward",
    "update_chosen_q_value",
    "step_q_learning",
    # Scheduler
    "PRIORITY_RESTORE",
    "PRIORITY_INTERVENTION",
    "PRIORITY_HOLDING",
    "PRIORITY_SNAPSHOT",
    "PRIORITY_HORIZON",
    "DeterministicEvent",
    "SchedulerStats",
    "PoissonScheduler",
    "calculate_total_event_rate",
    # Heterogeneity & Realism
    "DEFAULT_MULTIPLIER_BOUNDS",
    "DEFAULT_BIAS_SD_OVER_J",
    "DEFAULT_BIAS_BOUNDS_OVER_J",
    "DEFAULT_SUSCEPTIBILITY_BOUNDS",
    "DEFAULT_NUM_OPPORTUNITY_GROUPS",
    "DEFAULT_CROSS_GROUP_ELIGIBILITY",
    "DEFAULT_WITHIN_GROUP_ELIGIBILITY",
    "DEFAULT_ROW_CAP",
    "QuenchedHeterogeneity",
    "clipped_lognormal_multiplier",
    "generate_truncated_normal_biases",
    "generate_susceptibilities",
    "generate_quenched_heterogeneity",
    "generate_opportunity_groups",
    "calculate_cross_group_exploration_probability",
]
