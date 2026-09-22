"""Configuration schemas for emergent order simulations (Stages 0–4).

Defines Pydantic models enforcing strict parameter types, domain bounds, and decoupling
between physical model parameters and execution settings across all simulation stages:
    - Stage 0: Cost Calibration & Timing Benchmarks
    - Stage 1: Core Emergence & Pilot Exploration
    - Stage 2: Extension Campaigns & Controlled Perturbations
    - Stage 3: Adaptive Learning & Q-Learning Comparisons
    - Stage 4: Measurement, Analytics & Neural Diagnostics
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SimulationConfig(BaseModel):
    """Configuration schema for single-trajectory simulation runs (Stages 0–3)."""

    model_config = ConfigDict(extra="forbid")

    N: int = Field(..., ge=1, description="Population size")
    D: int = Field(1, description="Belief dimensionality (1 or 3)")
    T: float = Field(..., gt=0.0, description="Thermal noise temperature (strictly positive)")
    delta: float = Field(..., ge=0.0, description="Memory decay rate")
    nu: float = Field(..., ge=0.0, description="Encounter attempt rate")

    J: float = Field(1.0, ge=0.0, description="Social coupling strength")
    K: float = Field(0.5, description="Private-public coupling strength")
    L: float = Field(1.0, description="Internal belief coupling strength")

    tau_x: float = Field(1.0, gt=0.0, description="Private stance relaxation timescale")
    tau_y: float = Field(1.0, gt=0.0, description="Public stance relaxation timescale")
    tau_n: Optional[float] = Field(None, description="Participation relaxation timescale")

    epsilon: float = Field(0.05, ge=0.0, le=1.0, description="Encounter exploration probability")
    a: float = Field(0.0, description="Prestige bias sensitivity")
    c: float = Field(0.0, description="Matching belief homophily bias")
    chi: float = Field(0.0, ge=0.0, le=1.0, description="Triadic closure probability")

    reciprocal: bool = Field(False, description="Whether reciprocal memory pairing is active")
    state_mode: str = Field("binary", description="State representation: 'binary' or 'continuous'")
    seed: Optional[int] = Field(None, description="Random seed")
    horizon: float = Field(100.0, gt=0.0, description="Simulation horizon time")
    H_0: Optional[float] = Field(None, description="Initial observation horizon")
    name: Optional[str] = Field(None, description="Optional experiment name")

    T_n: Optional[float] = Field(None, gt=0.0, description="Participation temperature (strictly positive)")
    b_act: Optional[float] = Field(None, description="Grievance participation incentive")
    g: Optional[float] = Field(None, description="Social coordination incentive")
    c_act: Optional[float] = Field(None, description="Action cost")
    theta: Optional[float] = Field(None, ge=0.0, le=1.0, description="Institutional threshold")
    t_hold: Optional[float] = Field(None, ge=0.0, description="Holding duration")
    eta_Q: Optional[float] = Field(None, ge=0.0, description="Q-learning rate")
    T_Q: Optional[float] = Field(None, gt=0.0, description="Q-learning temperature")

    @field_validator("D")
    @classmethod
    def validate_dimensionality(cls, v: int) -> int:
        if v not in (1, 3):
            raise ValueError(f"Belief dimensionality D must be 1 or 3, got {v}")
        return v

    @field_validator("state_mode")
    @classmethod
    def validate_state_mode(cls, v: str) -> str:
        if v.lower() not in ("binary", "continuous"):
            raise ValueError(f"state_mode must be 'binary' or 'continuous', got {v}")
        return v.lower()

    @field_validator("tau_n")
    @classmethod
    def validate_tau_n(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0.0:
            raise ValueError(f"Participation timescale tau_n must be strictly positive, got {v}")
        return v


class PhysicsConfig(BaseModel):
    """Model physics parameters decoupled from execution settings."""

    model_config = ConfigDict(extra="forbid")

    J: float = Field(1.0, ge=0.0, description="Social coupling strength")
    K: float = Field(0.5, description="Private-public coupling strength")
    T: float = Field(1.0, gt=0.0, description="Temperature (strictly positive)")
    L: float = Field(1.0, description="Internal belief coupling strength")
    delta: float = Field(0.1, ge=0.0, description="Memory decay rate")
    nu: float = Field(2.0, ge=0.0, description="Encounter rate")

    epsilon: float = Field(0.05, ge=0.0, le=1.0, description="Encounter exploration probability")
    a: float = Field(0.0, description="Prestige bias sensitivity")
    c: float = Field(0.0, description="Matching belief homophily bias")
    chi: float = Field(0.0, ge=0.0, le=1.0, description="Triadic closure probability")
    tau_x: float = Field(1.0, gt=0.0, description="Private stance relaxation timescale")
    tau_y: float = Field(1.0, gt=0.0, description="Public stance relaxation timescale")
    tau_n: Optional[float] = Field(None, description="Participation relaxation timescale")
    reciprocal: bool = Field(False, description="Whether reciprocal memory pairing is active")

    T_n: Optional[float] = Field(None, gt=0.0, description="Participation temperature (strictly positive)")
    b_act: Optional[float] = Field(None, description="Grievance participation incentive")
    g: Optional[float] = Field(None, description="Social coordination incentive")
    c_act: Optional[float] = Field(None, description="Action cost")
    theta: Optional[float] = Field(None, ge=0.0, le=1.0, description="Institutional threshold")
    t_hold: Optional[float] = Field(None, ge=0.0, description="Holding duration")
    eta_Q: Optional[float] = Field(None, ge=0.0, description="Q-learning rate")
    T_Q: Optional[float] = Field(None, gt=0.0, description="Q-learning temperature")

    @field_validator("tau_n")
    @classmethod
    def validate_tau_n(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0.0:
            raise ValueError(f"Participation timescale tau_n must be strictly positive, got {v}")
        return v


class ExecutionConfig(BaseModel):
    """Computational execution settings decoupled from physics."""

    model_config = ConfigDict(extra="forbid")

    num_workers: int = Field(1, ge=1, description="Worker count for multiprocessing")
    seed: Optional[int] = Field(None, description="Master execution PRNG seed")
    output_dir: Optional[str] = Field(None, description="Artifact and checkpoint output directory")
    log_level: str = Field("INFO", description="Logging verbosity level")
    batch_size: int = Field(1, ge=1, description="Batch size for parallel execution")
    checkpoint_interval: Optional[float] = Field(None, gt=0.0, description="Time interval between state checkpoints")
    replicates: int = Field(1, ge=1, description="Number of replicates per condition")


class RootConfig(BaseModel):
    """Root configuration container separating experiment, physics, and execution sections."""

    model_config = ConfigDict(extra="allow")

    experiment: Optional[SimulationConfig] = None
    physics: Optional[PhysicsConfig] = None
    execution: Optional[ExecutionConfig] = None

    def is_empty(self) -> bool:
        """Check if all configuration sections are empty/None."""
        return self.experiment is None and self.physics is None and self.execution is None


def build_realization_id(condition_id: str, replicate_index: int) -> str:
    """Build standardized realization identifier string.

    Format: {condition_id}_rep{replicate_index:02d}

    Args:
        condition_id: Unique experimental condition identifier (e.g. 'T0.3_d0.1').
        replicate_index: Replicate integer index >= 0.

    Returns:
        Formatted realization identifier string.
    """
    if not condition_id or not isinstance(condition_id, str):
        raise ValueError("condition_id must be a non-empty string")
    if int(replicate_index) < 0:
        raise ValueError(f"replicate_index must be non-negative, got {replicate_index}")
    return f"{condition_id}_rep{int(replicate_index):02d}"
