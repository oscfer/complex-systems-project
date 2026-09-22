"""Exact, reproducible origin definitions for reference experiments.

Generated checkpoints and constructed controls remain distinct ensembles. The
constructed defaults combine the accepted concentrated-attention control with
the governing realism traits. ``heterogeneous=False`` retains the homogeneous
reference without changing downstream baseline interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict

import numpy as np

from src.core.heterogeneity import generate_opportunity_groups, generate_quenched_heterogeneity
from src.core.state import State, StateMode
from src.experiment.generators import generate_aligned_states, generate_hub_network


@dataclass(frozen=True)
class ConstructedOriginSpec:
    """Fully declared concentrated-attention reference ensemble."""

    N: int = 64
    D: int = 1
    memory_draws_per_row: int = 20
    hub_mixture: float = 0.1
    public_positive_probability: float = 0.9
    private_public_flip_probability: float = 0.0
    random_global_sign: bool = True
    heterogeneous: bool = True
    constrained_opportunities: bool = True
    opportunity_groups: int = 4
    within_group_eligibility: float = 1.0
    between_group_eligibility: float = 0.05
    memory_capacity_per_row: int = 40
    J: float = 1.0
    K_nominal: float = 0.5
    tau_private_nominal: float = 5.0
    nu_nominal: float = 2.0

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError("Constructed origins require N >= 2")
        if self.D not in (1, 3):
            raise ValueError("D must be 1 or 3")
        if self.memory_draws_per_row < 0:
            raise ValueError("memory_draws_per_row must be non-negative")
        if not 0.0 <= self.hub_mixture <= 1.0:
            raise ValueError("hub_mixture must be in [0, 1]")
        if not 0.0 <= self.public_positive_probability <= 1.0:
            raise ValueError("public_positive_probability must be in [0, 1]")
        if not 0.0 <= self.private_public_flip_probability <= 1.0:
            raise ValueError("private_public_flip_probability must be in [0, 1]")
        if self.memory_capacity_per_row < self.memory_draws_per_row:
            raise ValueError("memory capacity cannot be smaller than constructed row mass")
        if self.constrained_opportunities and self.N < self.opportunity_groups:
            raise ValueError("N cannot be smaller than opportunity_groups")
        if self.opportunity_groups < 1:
            raise ValueError("opportunity_groups must be positive")
        if self.within_group_eligibility < 0.0 or self.between_group_eligibility < 0.0:
            raise ValueError("opportunity eligibility weights must be non-negative")

    @property
    def ensemble_id(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return "constructed_concentrated_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class ReferenceOrigin:
    """A complete branchable state plus provenance metadata."""

    origin_id: str
    ensemble_id: str
    source: str
    state: State
    metadata: Dict[str, Any] = field(default_factory=dict)
    between_eligibility: float = 0.05
    within_eligibility: float = 1.0

    def __post_init__(self) -> None:
        if not self.origin_id or not self.ensemble_id:
            raise ValueError("origin_id and ensemble_id must be non-empty")
        if self.between_eligibility < 0.0 or self.within_eligibility < 0.0:
            raise ValueError("opportunity eligibility weights must be non-negative")

    def branch(self) -> State:
        return self.state.copy(deep=True)


def _attach_traits(state: State, spec: ConstructedOriginSpec, rng: np.random.Generator) -> None:
    if spec.heterogeneous:
        traits = generate_quenched_heterogeneity(
            N=spec.N,
            D=spec.D,
            J=spec.J,
            K_nominal=spec.K_nominal,
            tau_x_nominal=spec.tau_private_nominal,
            nu_nominal=spec.nu_nominal,
            rng=rng,
        )
        state.K = traits.K
        state.tau_x = traits.tau_x
        state.nu = traits.nu
        state.g = traits.g
        state.b = traits.b
    else:
        state.K = np.full(spec.N, spec.K_nominal, dtype=np.float64)
        state.tau_x = np.full(spec.N, spec.tau_private_nominal, dtype=np.float64)
        state.nu = np.full(spec.N, spec.nu_nominal, dtype=np.float64)
        state.g = np.ones(spec.N, dtype=np.float64)
        state.b = np.zeros((spec.N, spec.D), dtype=np.float64)

    if spec.constrained_opportunities:
        groups, _, _ = generate_opportunity_groups(
            N=spec.N,
            num_groups=spec.opportunity_groups,
            between_eligibility=spec.between_group_eligibility,
            within_eligibility=spec.within_group_eligibility,
            rng=rng,
        )
        state.opportunity_groups = groups
        state.blocks = groups.copy()
    state.row_caps = np.full(spec.N, spec.memory_capacity_per_row, dtype=np.uint32)


def build_constructed_origin(
    origin_index: int,
    master_seed: int = 1592026,
    spec: ConstructedOriginSpec | None = None,
) -> ReferenceOrigin:
    """Build one deterministic constructed origin from a stable seed sequence."""

    if origin_index < 0:
        raise ValueError("origin_index must be non-negative")
    spec = spec or ConstructedOriginSpec()
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), int(origin_index), 5]))

    W, hub = generate_hub_network(
        N=spec.N,
        m=spec.memory_draws_per_row,
        eta=spec.hub_mixture,
        rng=rng,
    )
    x, y = generate_aligned_states(
        N=spec.N,
        D=spec.D,
        p_g=spec.public_positive_probability,
        q=spec.private_public_flip_probability,
        state_mode=StateMode.BINARY,
        rng=rng,
    )
    orientation = 1
    if spec.random_global_sign and bool(rng.integers(0, 2)):
        x = -x
        y = -y
        orientation = -1

    state = State.create_empty(spec.N, spec.D, StateMode.BINARY)
    state.x = x
    state.y = y
    state.W = W
    state.hub_idx = hub
    _attach_traits(state, spec, rng)
    state.validate()

    origin_token = hashlib.sha256(
        f"{master_seed}|{origin_index}".encode("utf-8")
    ).hexdigest()[:12]
    origin_id = f"{spec.ensemble_id}_{origin_token}"
    return ReferenceOrigin(
        origin_id=origin_id,
        ensemble_id=spec.ensemble_id,
        source="constructed_concentrated",
        state=state,
        metadata={
            "origin_index": int(origin_index),
            "master_seed": int(master_seed),
            "global_orientation": orientation,
            "hub": int(hub),
            "spec": dict(spec.__dict__),
        },
        between_eligibility=spec.between_group_eligibility,
        within_eligibility=spec.within_group_eligibility,
    )


def register_generated_checkpoint(
    state: State,
    origin_id: str,
    establishment_time: float | None = None,
    source_manifest: str | None = None,
    between_eligibility: float = 0.05,
    within_eligibility: float = 1.0,
) -> ReferenceOrigin:
    """Register a complete generated checkpoint without altering its physical state."""

    if not origin_id:
        raise ValueError("origin_id must be non-empty")
    state.validate()
    metadata: Dict[str, Any] = {"physical_age": float(state.t)}
    if establishment_time is not None:
        metadata["establishment_time"] = float(establishment_time)
    if source_manifest is not None:
        metadata["source_manifest"] = str(source_manifest)
    return ReferenceOrigin(
        origin_id=origin_id,
        ensemble_id="generated_checkpoint",
        source="generated_checkpoint",
        state=state.copy(deep=True),
        metadata=metadata,
        between_eligibility=float(between_eligibility),
        within_eligibility=float(within_eligibility),
    )
