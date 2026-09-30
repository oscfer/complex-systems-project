"""Policy Optimizers and Algorithmic Search Module.

Implements:
1. Degree-based and influence-based heuristic scoring and subset discovery.
2. Greedy budget allocation enforcing k distinct-node and B exposure limits.
3. Heuristic subset search optimizer for static snapshots and reference ensembles.
4. Noise-resilient feedback-dependent target shift algorithms adhering to the Observation contract.
"""

from src.control.optimizers.feedback_shift import (
    FeedbackTargetShiftController,
    ShiftStrategy,
)
from src.control.optimizers.heuristics import (
    allocate_greedy_budget,
    compute_degree_scores,
    compute_influence_scores,
    select_heuristic_candidates,
)
from src.control.optimizers.search import (
    HeuristicSubsetSearchOptimizer,
    OptimizationResult,
    PolicyCandidate,
    compute_candidate_score,
)

__all__ = [
    "allocate_greedy_budget",
    "compute_degree_scores",
    "compute_influence_scores",
    "select_heuristic_candidates",
    "HeuristicSubsetSearchOptimizer",
    "OptimizationResult",
    "PolicyCandidate",
    "compute_candidate_score",
    "FeedbackTargetShiftController",
    "ShiftStrategy",
]
