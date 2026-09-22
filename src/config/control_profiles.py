"""Control profiles extending realism settings with intervention and optimization parameters.

LIGHT and EXTENDED are singletons representing the two supported execution
tiers for the campaign.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pydantic import Field, field_validator, model_validator
from src.config.profiles import RunProfile

class ControlProfile(RunProfile):
    """Profile adding control constraints, targets, and optimization allocation."""
    
    # Target block
    target_type: str = "reversal"
    target_threshold: float = 0.5
    holding_duration: float
    release_deadline: float = 20.0  # tau_y units
    
    # Observation block
    panel_fraction: float = 0.5
    edge_reveal_prob: float = 0.5
    binary_flip_prob: float = 0.05
    sample_interval_tau_public: float = 5.0
    delivery_delay_tau_public: float = 1.0
    private_states_visible: bool = False
    
    # Intervention block
    accessible_fraction: float = 0.25
    k_distinct_nodes_fraction: float
    exposure_cap_normalized: float
    amplitude_cap_over_J: float = 4.0
    candidate_node_fractions: Tuple[float, ...] = (0.05, 0.1)
    nominal_amplitudes_over_J: Tuple[float, ...] = (0.25, 1.0, 4.0)
    durations_tau_public: Tuple[float, ...] = (1.0, 5.0, 20.0)
    
    # Optimizer allocation block
    candidate_cap: int = 24
    selectors: Tuple[str, ...] = ("random", "reported_attention", "reported_outgoing_degree", "reported_cross_group", "group_balanced")
    observed_slot_quotas: Tuple[float, ...] = (0.0, 0.5, 1.0)
    feedback_choices: Tuple[bool, ...] = (False, True)
    
    training_origins: int
    training_futures_per_origin: int
    selection_candidates: int
    selection_origins: int
    selection_futures_per_origin: int
    evaluation_origins: int
    evaluation_futures_per_origin: int
    
    reliability_target: float = 0.8
    
    @field_validator(
        "panel_fraction", "edge_reveal_prob", "accessible_fraction",
        "k_distinct_nodes_fraction", "reliability_target"
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("Fraction must be in (0, 1]")
        return v

    @field_validator("exposure_cap_normalized")
    @classmethod
    def _validate_positive_exposure_ratio(cls, v: float) -> float:
        # Exposure integrates amplitude over time, so this normalized budget
        # may exceed one even though node/access fractions cannot.
        if not (0.0 < v < float("inf")):
            raise ValueError("Normalized exposure cap must be finite and positive")
        return v

    @model_validator(mode="after")
    def _validate_horizon(self) -> ControlProfile:
        if self.horizon <= self.release_deadline:
            raise ValueError("horizon must be strictly greater than release_deadline")
        return self


LIGHT_CONTROL_PROFILE = ControlProfile(
    topic="campaign",
    tier="light",
    # Physics (matching light campaign except for extended horizons required by control rules)
    N=100, D=1, horizon=320.0, establishment_time=10.0, n_origins=12, seed_base=400,
    # Control params
    holding_duration=100.0,
    release_deadline=20.0,
    k_distinct_nodes_fraction=0.1,
    exposure_cap_normalized=0.5,
    training_origins=2,
    training_futures_per_origin=1,
    selection_candidates=3,
    selection_origins=2,
    selection_futures_per_origin=2,
    evaluation_origins=4,
    evaluation_futures_per_origin=1
)

EXTENDED_CONTROL_PROFILE = ControlProfile(
    topic="campaign",
    tier="extended",
    # Physics (matching extended campaign except for extended horizons required by control rules)
    N=500, D=1, horizon=320.0, establishment_time=20.0, n_origins=54, seed_base=4000,
    # Control params
    holding_duration=100.0,
    release_deadline=20.0,
    k_distinct_nodes_fraction=0.1,
    exposure_cap_normalized=0.5,
    training_origins=12,
    training_futures_per_origin=2,
    selection_candidates=3,
    selection_origins=12,
    selection_futures_per_origin=4,
    evaluation_origins=30,
    evaluation_futures_per_origin=1
)


def preflight_control_profile(profile: Any) -> Dict[str, Any]:
    """Validate campaign allocation and intervention geometry before simulation.

    Deliberately implemented as a plain function, not a Pydantic validator:
    `ControlProfile.model_copy(update=...)` bypasses model validation, so
    copied/edited profiles (notebook smoke profiles, tests, user edits) must be
    re-checked here before any runner is built. Raises ``ValueError`` with a
    specific message on the first infeasible setting.

    Checks:
    - allocation consistency: ``n_origins >= training+selection+evaluation``
      and all counts/futures/candidate caps strictly positive;
    - holding feasibility: the release deadline plus the persistence window
      must fit inside the declared horizon;
    - positive observation intervals and fractions in (0, 1];
    - access/node-cap compatibility: roster and distinct-node cap derive to
      at least one (and at most N) actuable node; amplitude/exposure caps and
      declared amplitude/duration/selector/quota/feedback choices are positive
      and non-empty. Short non-feedback pulses (duration below the delivery
      delay) are legitimate and are NOT rejected.

    Returns the derived intervention geometry for the run record.
    """
    tier = str(getattr(profile, "tier", "?"))

    def _fail(msg: str) -> None:
        raise ValueError(f"control-profile preflight failed (tier={tier}): {msg}")

    # --- allocation block (origin pools and optimizer allocation) -------------
    n_prepare = int(profile.n_origins)
    n_train = int(profile.training_origins)
    n_sel = int(profile.selection_origins)
    n_eval = int(profile.evaluation_origins)
    n_needed = n_train + n_sel + n_eval
    if n_train < 1 or n_sel < 1 or n_eval < 1:
        _fail(
            f"training_origins={n_train}, selection_origins={n_sel}, "
            f"evaluation_origins={n_eval}: all origin counts must be >= 1"
        )
    for name in (
        "training_futures_per_origin",
        "selection_futures_per_origin",
        "evaluation_futures_per_origin",
    ):
        if int(getattr(profile, name)) < 1:
            _fail(f"{name}={int(getattr(profile, name))} must be >= 1")
    if int(profile.selection_candidates) < 1:
        _fail(f"selection_candidates={int(profile.selection_candidates)} must be >= 1")
    if int(profile.candidate_cap) < 1:
        _fail(f"candidate_cap={int(profile.candidate_cap)} must be >= 1")
    if n_prepare < n_needed:
        _fail(
            f"allocation requires {n_needed} prepared origins "
            f"(training {n_train} + selection {n_sel} + evaluation {n_eval}) "
            f"but n_origins={n_prepare}: internally inconsistent allocation"
        )

    # --- holding feasibility ---------------------------------------------------
    holding = float(profile.holding_duration)
    deadline = float(profile.release_deadline)
    horizon = float(profile.horizon)
    if holding <= 0:
        _fail(f"holding_duration={holding} must be > 0")
    if deadline <= 0:
        _fail(f"release_deadline={deadline} must be > 0")
    if horizon <= 0:
        _fail(f"horizon={horizon} must be > 0")
    if deadline + holding > horizon:
        _fail(
            f"holding infeasible: release_deadline ({deadline}) + holding_duration "
            f"({holding}) exceeds horizon ({horizon}); the persistence window can "
            "never complete"
        )

    # --- observation block ------------------------------------------------------
    for name in ("panel_fraction", "edge_reveal_prob"):
        v = float(getattr(profile, name))
        if not (0.0 < v <= 1.0):
            _fail(f"{name}={v} must be in (0, 1]")
    if float(profile.sample_interval_tau_public) <= 0:
        _fail("sample_interval_tau_public must be > 0")
    if float(profile.delivery_delay_tau_public) < 0:
        _fail("delivery_delay_tau_public must be >= 0")

    # --- intervention geometry (mirrors _campaign_geometry conventions) --------
    N = int(profile.N)
    J = float(profile.J)
    if N < 1:
        _fail(f"N={N} must be >= 1")
    if J <= 0:
        _fail(f"J={J} must be > 0")
    k = max(1, int(round(float(profile.k_distinct_nodes_fraction) * N)))
    if k > N:
        _fail(
            f"k_distinct_nodes_fraction={float(profile.k_distinct_nodes_fraction)} "
            f"derives to k={k} > N={N}"
        )
    roster_n = max(1, int(round(float(profile.accessible_fraction) * N)))
    if roster_n > N:
        _fail(
            f"accessible_fraction={float(profile.accessible_fraction)} derives to a "
            f"roster of {roster_n} > N={N}"
        )
    amplitude_cap = float(profile.amplitude_cap_over_J) * J
    if amplitude_cap <= 0:
        _fail(f"amplitude cap derives to {amplitude_cap}, must be > 0")
    exposure_cap = float(profile.exposure_cap_normalized) * N * amplitude_cap
    if exposure_cap <= 0:
        _fail(f"exposure cap derives to {exposure_cap}, must be > 0")
    nominal_amplitudes = tuple(float(a) for a in profile.nominal_amplitudes_over_J)
    if not nominal_amplitudes or any(a <= 0 for a in nominal_amplitudes):
        _fail("nominal_amplitudes_over_J must be non-empty with all values > 0")
    durations = tuple(float(d) for d in profile.durations_tau_public)
    if not durations or any(d <= 0 for d in durations):
        _fail("durations_tau_public must be non-empty with all values > 0")
    node_fractions = tuple(float(f) for f in profile.candidate_node_fractions)
    if not node_fractions or any(not (0.0 < f <= 1.0) for f in node_fractions):
        _fail("candidate_node_fractions must be non-empty with values in (0, 1]")
    selectors = tuple(str(s) for s in profile.selectors)
    quotas = tuple(float(q) for q in profile.observed_slot_quotas)
    feedback = tuple(bool(b) for b in profile.feedback_choices)
    if not selectors:
        _fail("selectors must be non-empty")
    if not quotas or any(not (0.0 <= q <= 1.0) for q in quotas):
        _fail("observed_slot_quotas must be non-empty with values in [0, 1]")
    if not feedback:
        _fail("feedback_choices must be non-empty")
    if float(profile.reliability_target) <= 0 or float(profile.reliability_target) >= 1:
        _fail("reliability_target must be in (0, 1)")

    return {
        "k": k,
        "accessible_roster_size": roster_n,
        "amplitude_cap": amplitude_cap,
        "exposure_cap": exposure_cap,
        "n_prepare": n_prepare,
        "n_needed": n_needed,
        "n_unassigned_spare": n_prepare - n_needed,
    }
