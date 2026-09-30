import pytest
import numpy as np

from src.control.optimizers.search import ObservablePolicySearchOptimizer, ObservablePolicyCandidate
from src.control.contracts import ControlProblem, TargetType, RolloutResult
from src.core.state import State
from src.experiment.baselines.ensembles import ReferenceOrigin

def test_selector_coverage_and_cap_handling():
    """Verify explicit coverage of configured selectors and sensible cap handling."""
    problem = ControlProblem(
        problem_id="test_prob",
        target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.5},
        N=10, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0,
        release_deadline=5.0, holding_duration=2.0, horizon=10.0
    )
    
    # Configure 5 selectors, cap = 4. 
    # Must include zero baseline + 3 distinct selectors.
    selectors = ("random", "reported_attention", "reported_outgoing_degree", "reported_cross_group", "group_balanced")
    
    optimizer = ObservablePolicySearchOptimizer(
        problem=problem,
        candidate_cap=4,
        search_seed=42,
        selectors=selectors,
    )
    cands = optimizer.generate_candidates()
    
    assert len(cands) == 4, f"Expected 4 candidates, got {len(cands)}"
    assert cands[0].candidate_id == "observable_zero"
    
    sel_set = set(c.selector for c in cands[1:])
    assert len(sel_set) == 3, f"Expected 3 distinct selectors due to cap, got {len(sel_set)}: {sel_set}"
    
    # Reproducibility check
    opt2 = ObservablePolicySearchOptimizer(
        problem=problem, candidate_cap=4, search_seed=42, selectors=selectors
    )
    cands2 = opt2.generate_candidates()
    assert [c.signature for c in cands] == [c.signature for c in cands2]
    
    # Cap 24 -> Should cover all 5 and fill the rest up to 24
    opt3 = ObservablePolicySearchOptimizer(
        problem=problem, candidate_cap=24, search_seed=99, selectors=selectors
    )
    cands3 = opt3.generate_candidates()
    assert len(cands3) == 24
    sel_set3 = set(c.selector for c in cands3[1:])
    assert len(sel_set3) == 5, f"Expected all 5 selectors covered, got {len(sel_set3)}: {sel_set3}"
    assert len(set(c.signature for c in cands3)) == 24, "Candidates should be distinct"

def test_distinct_origin_identities():
    """Verify origin_id uses distinct deterministic IDs in optimize()."""
    problem = ControlProblem(
        problem_id="test_prob", target_type=TargetType.REVERSAL, target_params={"threshold": 0.5},
        N=10, D=1, k=2, exposure_cap=10.0, amplitude_cap=2.0, release_deadline=5.0, holding_duration=2.0, horizon=10.0
    )
    optimizer = ObservablePolicySearchOptimizer(
        problem=problem, candidate_cap=1, futures_per_candidate=1, search_seed=42, selectors=("random",)
    )
    
    # Create two distinct raw State objects
    s1 = State(N=10, D=1, state_mode="continuous", n=1, Q=np.zeros((1, 1)), y=np.zeros((10, 1)), x=np.zeros((10, 1)), W=np.zeros((10, 10)), opportunity_groups=None)
    s2 = State(N=10, D=1, state_mode="continuous", n=1, Q=np.zeros((1, 1)), y=np.ones((10, 1)), x=np.zeros((10, 1)), W=np.zeros((10, 10)), opportunity_groups=None)
    
    origins = [s1, s2]
    
    # Patch run_controlled_simulation to track the evaluator's origin_id
    tracked_origins = []
    import src.control.optimizers.search
    original_run = src.control.optimizers.search.run_controlled_simulation
    
    def fake_run(*args, target_evaluator, **kwargs):
        tracked_origins.append(target_evaluator.origin_id)
        # return a dummy rollout result
        res = RolloutResult(
            problem_id="test_prob", policy_id="obs_cand_1", origin_id=target_evaluator.origin_id, future_id=0,
            is_valid=True, is_eligible=True, success=True, completion_time=1.0, 
            restricted_time=1.0, terminal_target_violation=0.0, expenditure=1.0, 
            distinct_nodes_used=2, relapsed=False, event_counts={}
        )
        return res, None, None
    
    src.control.optimizers.search.run_controlled_simulation = fake_run
    try:
        optimizer.optimize(training_origins=origins)
        assert len(tracked_origins) == 2
        assert tracked_origins[0] != tracked_origins[1], "Origin IDs should be distinct"
    finally:
        src.control.optimizers.search.run_controlled_simulation = original_run

def test_actual_node_count_tie_break():
    """Verify compute_candidate_score uses actual rollout distinct_nodes_used for observable candidates."""
    from src.control.optimizers.search import compute_candidate_score, PolicyCandidate
    
    cand = ObservablePolicyCandidate(
        candidate_id="obs_test",
        selector="reported_attention",
        observed_slot_quota=0.0,
        feedback=False,
        initial_recruitment_fraction=1.0,
        nominal_amplitude=1.0,
        duration=1.0
    )
    
    rollouts = [
        RolloutResult(
            problem_id="p1", policy_id="c1", origin_id="o1", future_id=0,
            is_valid=True, is_eligible=True, success=True, completion_time=1.0, 
            restricted_time=1.0, terminal_target_violation=0.0, expenditure=1.0, 
            distinct_nodes_used=5, relapsed=False, event_counts={}
        ),
        RolloutResult(
            problem_id="p1", policy_id="c1", origin_id="o1", future_id=1,
            is_valid=True, is_eligible=True, success=True, completion_time=1.0, 
            restricted_time=1.0, terminal_target_violation=0.0, expenditure=1.0, 
            distinct_nodes_used=7, relapsed=False, event_counts={}
        )
    ]
    
    score, summ = compute_candidate_score(cand, rollouts, p0=0.8) # type: ignore
    
    # n_nodes is the 5th element in the score tuple (0-indexed: 0:feasible, 1:time, 2:-p, 3:d_G, 4:exp, 5:nodes, 6:id)
    assert score[5] == 6.0, f"Expected average distinct_nodes_used to be 6.0, got {score[5]}"
    assert summ["selector"] == "reported_attention"
    
    # Oracle behavior preservation
    oracle_cand = PolicyCandidate(
        candidate_id="oracle_test", target_nodes=(1, 2, 3), signs=((1, 1), (2, 1), (3, 1)), nominal_amplitude=1.0, duration=1.0
    )
    score_or, summ_or = compute_candidate_score(oracle_cand, rollouts, p0=0.8) # type: ignore
    assert score_or[5] == 3, f"Expected len(target_nodes) = 3 for oracle, got {score_or[5]}"
    assert summ_or["selector"] is None
