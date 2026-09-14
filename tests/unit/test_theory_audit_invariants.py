"""Analytical invariants tying the simulator to exact analytical results.

Checks:
1. Uncapped homogeneous memory: <S(t)> -> N nu / delta (Eq. memorymean).
2. Capacity: per-row truncated Poisson p(s) = (lambda^s / s!) / sum_r (lambda^r / r!).
3. Isolated weak-field slopes: binary d<m>/dF -> 1/T, continuous -> 1/(3T).
4. Attempt-rate accounting: event counts match R_x + R_y over a fixed horizon.
5. Wilson intervals contain the true p for large n.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.heat_baths import (
    binary_glauber_prob_plus,
    calibrate_continuous_temperature,
    sample_continuous_inverse_cdf_vector,
)
from src.core.state import State, StateMode
from src.execution.runner import TrajectoryRunner
from src.experiment.baselines.reference import wilson_interval


def _homogeneous_runner(nu=1.0, delta=0.05, N=60, seed=0):
    st = State.create_unbiased_emergence(N=N, D=1, rng=np.random.default_rng(seed))
    return TrajectoryRunner(state=st, T=1e6, J=0.0, K=0.0, L=0.0,
                            nu=nu, delta=delta, tau_x=1e9, tau_y=1e9,
                            sample_interval=1e9, seed=seed)


def test_uncapped_memory_stationarity():
    """<S> -> N nu / delta; per-row marginal approx Poisson(nu/delta)."""
    nu, delta, N = 1.0, 0.05, 60
    r = _homogeneous_runner(nu=nu, delta=delta, N=N, seed=11)
    r.run(horizon=120.0)  # ~6 relaxation times
    S = r.state.total_memory
    mean = N * nu / delta
    assert abs(S - mean) < 0.15 * mean


def test_capacity_truncated_poisson_row_marginal():
    """Per-row count distribution matches the truncated Poisson balance law."""
    nu, delta, N, C = 1.0, 0.2, 40, 40
    st = State.create_unbiased_emergence(N=N, D=1, rng=np.random.default_rng(5))
    r = TrajectoryRunner(state=st, T=1e6, J=0.0, K=0.0, L=0.0, nu=nu, delta=delta,
                         tau_x=1e9, tau_y=1e9, sample_interval=1e9, seed=5)
    r.state.row_caps = np.full(N, C, dtype=np.uint32)
    r.run(horizon=150.0)
    lam = nu / delta
    from math import factorial, exp, log
    def pois(s):
        return exp(-lam + s * log(lam) - log(factorial(s)))
    denom = sum(pois(s) for s in range(C + 1))
    expected_mean = sum(s * pois(s) / denom for s in range(C + 1))
    row_counts = np.sum(r.state.W, axis=1)
    simulated_mean = float(np.mean(row_counts))
    assert abs(simulated_mean - expected_mean) < 0.15 * expected_mean


def test_isolated_weak_field_slopes():
    """Binary: P(+1|kappa) slope at 0 is 1; continuous T_cal gives slope 1/3."""
    eps = 1e-4
    # <s> = 2 P(+1) - 1, so d<s>/d(kappa) = 2 * dP/d(kappa) = 1 at kappa = 0.
    p_plus = binary_glauber_prob_plus(eps)
    p_minus = binary_glauber_prob_plus(-eps)
    slope_binary = 2.0 * (p_plus - p_minus) / (2 * eps)
    assert abs(slope_binary - 1.0) < 1e-3

    # Continuous: E[s | kappa] ~ kappa / 3 under the same T (before calibration);
    # calibrate_continuous_temperature maps T_bin -> T_bin/3 to match 1/T_bin.
    assert abs(calibrate_continuous_temperature(0.3) - 0.1) < 1e-12
    rng = np.random.default_rng(7)
    for kappa in (0.05, 0.1):
        z = sample_continuous_inverse_cdf_vector(np.full(200000, kappa), rng)
        assert abs(float(np.mean(z)) - kappa / 3.0) < 0.005


def test_state_attempt_rate_accounting():
    """Private+public event counts match D*N*(1/tau_x + 1/tau_y)*horizon."""
    N, D, tau_x, tau_y, horizon = 30, 3, 2.0, 1.0, 40.0
    st = State.create_unbiased_emergence(N=N, D=D, rng=np.random.default_rng(2))
    r = TrajectoryRunner(state=st, T=1e6, J=0.0, K=0.0, L=0.0, nu=0.0, delta=0.0,
                         tau_x=tau_x, tau_y=tau_y, sample_interval=1e9, seed=2)
    r.run(horizon=horizon)
    events = r.scheduler.stats.private_updates + r.scheduler.stats.public_updates
    expected = D * N * (1.0 / tau_x + 1.0 / tau_y) * horizon
    assert abs(events - expected) < 0.08 * expected


def test_wilson_interval_contains_true_p():
    rng = np.random.default_rng(13)
    n, p_true = 2000, 0.7
    draws = rng.binomial(n, p_true, size=200)
    covered = sum(1 for k in draws if wilson_interval(k, n)[0] <= p_true <= wilson_interval(k, n)[1])
    assert covered / len(draws) > 0.90  # nominal 0.95 with tolerance


def test_one_component_per_event_d3():
    """D=3 fixed point: with zero fields and huge T, each component flips at rate 1/tau.

    After the audit fix (one component per event), the flip rate per component is
    N/tau per unit time in each of x and y channels; previously it was D x larger.
    """
    N, D, tau, horizon = 20, 3, 1.0, 30.0
    st = State.create_unbiased_emergence(N=N, D=D, rng=np.random.default_rng(4))
    r = TrajectoryRunner(state=st, T=1e6, J=0.0, K=0.0, L=0.0, nu=0.0, delta=0.0,
                         tau_x=tau, tau_y=tau, sample_interval=1e9, seed=4)
    r.run(horizon=horizon)
    events = r.scheduler.stats.private_updates + r.scheduler.stats.public_updates
    expected = D * N * (2.0 / tau) * horizon
    assert abs(events - expected) < 0.08 * expected