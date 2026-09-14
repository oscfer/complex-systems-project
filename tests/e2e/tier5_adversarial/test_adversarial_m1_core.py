"""Tier 5 Adversarial and Stress Suite for the simulator core.

Tests extreme floating point bounds, large memory matrices, Poisson scheduler distribution,
and institutional holding oscillations.
"""

from tests.unit.test_adversarial_core_stress import TestAdversarialCoreStress

__all__ = ["TestAdversarialCoreStress"]
