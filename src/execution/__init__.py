"""Execution campaigns and trajectory runners.

Provides:
- TrajectoryRunner: Single-trajectory simulation runner with bit-for-bit checkpointing.
- CampaignRunner: Parallel multiprocess campaign executor with deterministic seed isolation.
- Resource profiles: 'lightweight' / 'small', 'standard', 'extended'.
- Checkpoint persistence: save_checkpoint, load_checkpoint, branch_checkpoint.
"""

from __future__ import annotations

from src.execution.runner import (
    Trajectory,
    TrajectoryRunner,
    branch_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.execution.campaign import (
    CampaignResult,
    CampaignRunner,
    ResourceProfile,
    get_profile,
)

__all__ = [
    "Trajectory",
    "TrajectoryRunner",
    "CampaignResult",
    "CampaignRunner",
    "ResourceProfile",
    "get_profile",
    "save_checkpoint",
    "load_checkpoint",
    "branch_checkpoint",
]
