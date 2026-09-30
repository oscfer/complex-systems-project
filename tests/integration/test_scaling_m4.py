import time
from src.control.contracts import ControlProblem, TargetType
from src.control.optimizers.search import ObservablePolicySearchOptimizer
from src.core.state import State
import numpy as np

def test_observable_optimizer_scaling():
    start_time = time.time()
    
    problem = ControlProblem(
        problem_id="scaling_test",
        target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.5},
        N=100,
        D=1,
        k=10,
        exposure_cap=100.0,
        amplitude_cap=4.0,
        release_deadline=10.0,
        holding_duration=5.0,
        horizon=20.0,
        accessible_roster=tuple(range(25))
    )
    
    optimizer = ObservablePolicySearchOptimizer(
        problem=problem,
        candidate_cap=10000,
        futures_per_candidate=1,
    )
    
    candidates = optimizer.generate_candidates()
    elapsed = time.time() - start_time
    
    assert len(candidates) <= 10000, f"Generated {len(candidates)} candidates, expected <= 10000"
    assert elapsed < 1.0, f"Candidate generation took {elapsed:.2f}s, expected < 1.0s"
    print(f"Generated {len(candidates)} candidates in {elapsed:.4f}s")
    
if __name__ == "__main__":
    test_observable_optimizer_scaling()
