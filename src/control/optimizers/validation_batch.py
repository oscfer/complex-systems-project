"""Frozen algorithm validation on eight previously uninspected origins.

The algorithm may use per-origin search rollouts; only separate audit futures
measure its performance. No audit-dependent tuning. This is offline simulator
access, not a partially observed online controller.
"""
import hashlib
from src.control.optimizers.sparse_discovery import discovery_plan


def validation_origins():
    return {f'validation_{i:02d}': int.from_bytes(hashlib.sha256(
        f'modular3-report-validation-v1:{i}'.encode()).digest()[:4], 'big') for i in range(8)}


def validation_plan():
    plan = discovery_plan()
    plan.update(version='sparse-validation-v1', origins=list(validation_origins()),
        origin_seeds=validation_origins(), sizes=[4, 10], power_quantiles=[.5, 1.],
        benchmark_strategies=['random', 'attention', 'audience_coverage'],
        search_ceiling=960, audit_ceiling=640, zero_ceiling=88,
        physical_ceiling=1688, forecast_simulations=0,
        primary_comparison='n10_q1 versus random10: origin-level mean difference in persistent audit success',
        secondary_comparisons=['n4_q1 versus random4', 'search versus attention and audience coverage at matched count',
            'n4/n10_q0.5 versus q1: individual observed-power ceiling, skip unavailable strata',
            'initial mean/max full-network influence, exposure, completion and later relapse'],
        scope='Frozen offline simulator-assisted algorithm on eight new networks; three search futures per candidate and eight separate audit futures. Aggregate by network, not by trajectory.',
        study_label='independent_network_offline_algorithm_validation',
        interpretation='Four and ten recipients retain sparse feasibility and the strongest development result. Half-score ceiling retains the weak-recipient question; no retuning after audit. Full-state influence is retrospective only.')
    return plan


def run_validation_batch(out_dir, *, resume_bundle=None, plan_only=False):
    from src.control.optimizers.sparse_discovery_batch import run_sparse_discovery
    return run_sparse_discovery(out_dir, resume_bundle=resume_bundle,
                                plan_only=plan_only, validation=True)
