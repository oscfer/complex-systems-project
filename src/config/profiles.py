"""Experiment profiles binding the pilot design to runner kwargs.

A profile is immutable; LIGHT tiers are representative smoke runs, EXTENDED
tiers are the committed multi-hour campaigns on Colab. All physics constants
come from the model specification, not from any completed run.

Heterogeneity, opportunity groups, and row capacity are materialized through
State.create_realism_state; the TrajectoryRunner adopts state-level traits
automatically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunProfile(BaseModel):
    """Declarative profile for emergence/transformation/belief/campaign topics."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(..., description="emergence | communities | beliefs | campaign")
    tier: str = Field(..., description="'light' or 'extended'")
    # Model constants
    J: float = 1.0
    K_nominal: float = 0.5
    tau_public: float = 1.0
    tau_private_nominal: float = 5.0
    temperature: float = 0.3
    nu_nominal: float = 2.0
    delta: float = 0.1
    exploration: float = 0.1
    prestige_global: float = 0.0
    matching: float = 1.0
    closure: float = 0.3
    memory_capacity_per_row: int = 40
    state_mode: str = "binary"

    # Run settings
    N: int = Field(..., ge=2)
    D: int = 1
    horizon: float = Field(..., gt=0.0)
    establishment_time: float = Field(5.0, gt=0.0)
    sample_interval: float = 0.5
    animation_interval: float = 1.0
    n_origins: int = Field(8, ge=1)
    seed_base: int = 0

    @field_validator("state_mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in ("binary", "continuous"):
            raise ValueError("state_mode must be 'binary' or 'continuous'")
        return v

    @field_validator("D")
    @classmethod
    def _dim(cls, v: int) -> int:
        if v not in (1, 3):
            raise ValueError("D must be 1 or 3")
        return v

    def to_manifest(self) -> Dict[str, Any]:
        return self.model_dump()

    def profile_id(self) -> str:
        m = self.to_manifest()
        import hashlib
        import json
        return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:12]


# Representative (minutes-scale) profiles for validity checks.
LIGHT_PROFILES: Dict[str, RunProfile] = {
    "emergence": RunProfile(topic="emergence", tier="light", N=200, D=1, horizon=30.0,
                            establishment_time=15.0, n_origins=8, seed_base=100),
    "communities": RunProfile(topic="communities", tier="light", N=200, D=1, horizon=30.0,
                              establishment_time=15.0, n_origins=8, seed_base=200),
    "beliefs": RunProfile(topic="beliefs", tier="light", N=150, D=3, horizon=30.0,
                          establishment_time=15.0, n_origins=8, seed_base=300),
    "campaign": RunProfile(topic="campaign", tier="light", N=100, D=1, horizon=20.0,
                           establishment_time=10.0, n_origins=12, seed_base=400),
}

# Committed (multi-hour) campaigns on Colab: N≈1000, 30 independent origins.
EXTENDED_PROFILES: Dict[str, RunProfile] = {
    "emergence": RunProfile(topic="emergence", tier="extended", N=1000, D=1, horizon=60.0,
                            establishment_time=30.0, n_origins=30, seed_base=1000),
    "communities": RunProfile(topic="communities", tier="extended", N=1000, D=1, horizon=60.0,
                              establishment_time=30.0, n_origins=30, seed_base=2000),
    "beliefs": RunProfile(topic="beliefs", tier="extended", N=600, D=3, horizon=60.0,
                          establishment_time=30.0, n_origins=30, seed_base=3000),
    "campaign": RunProfile(topic="campaign", tier="extended", N=500, D=1, horizon=40.0,
                           establishment_time=20.0, n_origins=30, seed_base=4000),
}


def get_profile(topic: str, tier: str = "light") -> RunProfile:
    table = LIGHT_PROFILES if tier == "light" else EXTENDED_PROFILES
    if topic not in table:
        raise KeyError(f"Unknown topic {topic!r}; choose from {sorted(table)}")
    return table[topic]


def build_realism_runner(profile: RunProfile, origin_seed: int, **overrides: Any):
    """Build a TrajectoryRunner on a realism state for the given profile.

    The state carries quenched traits (K_i, tau_x_i, nu_i, g_i, b), row caps,
    and four opportunity groups; the runner adopts them automatically.
    """
    from src.core.state import State, StateMode
    from src.execution.runner import TrajectoryRunner

    rng = np_rng = __import__("numpy").random.default_rng(origin_seed)
    state = State.create_realism_state(
        N=profile.N,
        D=profile.D,
        state_mode=StateMode(profile.state_mode),
        J=profile.J,
        K_nominal=profile.K_nominal,
        tau_x_nominal=profile.tau_private_nominal,
        nu_nominal=profile.nu_nominal,
        enable_heterogeneity=True,
        row_cap=profile.memory_capacity_per_row,
        enable_opportunity_groups=True,
        num_groups=4,
        between_eligibility=0.05,
        within_eligibility=1.0,
        rng=rng,
    )
    kwargs: Dict[str, Any] = dict(
        state=state,
        J=profile.J,
        T=profile.temperature,
        delta=profile.delta,
        epsilon=profile.exploration,
        a=profile.prestige_global,
        c=profile.matching,
        chi=profile.closure,
        tau_y=profile.tau_public,
        sample_interval=profile.sample_interval,
        record_network_frames=True,
        animation_interval=profile.animation_interval,
        seed=origin_seed,
    )
    kwargs.update(overrides)
    return TrajectoryRunner(**kwargs)