import pytest
import numpy as np
from src.core.state import State, StateMode
from src.control.evaluator import TargetEvaluator, TargetType

def test_majority_target_eligibility():
    # Tie state
    N = 4
    x = np.array([[1], [1], [-1], [-1]], dtype=float)
    y = np.array([[1], [1], [-1], [-1]], dtype=float)
    W = np.zeros((N, N), dtype=int)
    n = np.zeros(N, dtype=int)
    Q = np.zeros((N, 1, 2), dtype=float)
    state = State(N=N, D=1, state_mode=StateMode.BINARY, x=x, y=y, W=W, n=n, Q=Q, t=0.0)

    # Tie should be ineligible
    evaluator_tie = TargetEvaluator.create_for_state(
        TargetType.REVERSAL, state, 100, 320, 20, 
        "t1", "p1", "o1", 0,
        target_params={"threshold": 0.5, "eligibility_mode": "public_majority"}
    )
    assert not evaluator_tie.defn.is_eligible
    assert "is a tie" in evaluator_tie.defn.ineligibility_reason

    # Weak public majority: mean = 0.5 (so 3 out of 4 is +1), but private might disagree
    y_weak = np.array([[1], [1], [1], [-1]], dtype=float)
    x_weak = np.array([[-1], [-1], [-1], [1]], dtype=float)
    state_weak = State(N=N, D=1, state_mode=StateMode.BINARY, x=x_weak, y=y_weak, W=W, n=n, Q=Q, t=0.0)
    
    # Should be eligible for public_majority
    evaluator_weak = TargetEvaluator.create_for_state(
        TargetType.REVERSAL, state_weak, 100, 320, 20, 
        "t2", "p1", "o1", 0,
        target_params={"threshold": 0.5, "eligibility_mode": "public_majority"}
    )
    assert evaluator_weak.defn.is_eligible
    assert evaluator_weak.defn.s_star == -1.0
    
    # But ineligible for legacy_strong
    evaluator_weak_legacy = TargetEvaluator.create_for_state(
        TargetType.REVERSAL, state_weak, 100, 320, 20, 
        "t3", "p1", "o1", 0,
        target_params={"threshold": 0.5, "eligibility_mode": "legacy_strong"}
    )
    assert not evaluator_weak_legacy.defn.is_eligible

