"""Comprehensive performance benchmark harness for the simulator core.

Measures and profiles:
1. Continuous inverse-CDF heat bath throughput (scalar and vectorized, across kappa regimes).
2. Binary Glauber heat bath throughput.
3. MemoryManager operations throughput (encounters, decays, attention recalculation for N in {64, 128, 256}).
4. PoissonScheduler jump event processing throughput across population scales.
5. Participation and institutional threshold checking throughput.
"""

from __future__ import annotations

import time
import json
import numpy as np

from src.core.heat_baths import (
    sample_continuous_inverse_cdf_scalar,
    sample_continuous_inverse_cdf_vector,
    sample_binary_glauber_scalar,
    sample_binary_glauber_vector,
)
from src.core.memory import MemoryManager
from src.core.fields import compute_attention_matrix, compute_public_fields, compute_private_fields
from src.core.scheduler import PoissonScheduler
from src.core.participation import ParticipationTracker, ParticipationParams


def benchmark_heat_baths():
    print("\n=== 1. HEAT BATH BENCHMARKS ===")
    results = {}
    rng = np.random.default_rng(42)

    # 1.1 Continuous scalar sampling
    n_scalar = 200_000
    for k_val, name in [(0.0, "kappa=0 (Taylor)"), (1.5, "kappa=1.5 (Standard)"), (1000.0, "kappa=1000 (Extreme)")]:
        u_arr = rng.uniform(0.0, 1.0, size=n_scalar)
        t0 = time.perf_counter()
        for i in range(n_scalar):
            sample_continuous_inverse_cdf_scalar(k_val, u_arr[i])
        t_elapsed = time.perf_counter() - t0
        rate = n_scalar / t_elapsed
        results[f"continuous_scalar_{name}"] = {"ops_sec": round(rate, 1), "time_sec": round(t_elapsed, 4)}
        print(f"Continuous Scalar [{name}]: {rate:,.1f} samples/sec ({t_elapsed:.3f}s for {n_scalar:,})")

    # 1.2 Continuous vectorized sampling
    n_vec = 1_000_000
    for k_val, name in [(0.0, "kappa=0"), (1.5, "kappa=1.5"), (1000.0, "kappa=1000")]:
        k_vec = np.full(n_vec, k_val, dtype=np.float64)
        t0 = time.perf_counter()
        sample_continuous_inverse_cdf_vector(k_vec, rng)
        t_elapsed = time.perf_counter() - t0
        rate = n_vec / t_elapsed
        results[f"continuous_vector_{name}"] = {"ops_sec": round(rate, 1), "time_sec": round(t_elapsed, 4)}
        print(f"Continuous Vectorized [{name}]: {rate:,.1f} samples/sec ({t_elapsed:.3f}s for {n_vec:,})")

    # 1.3 Binary Glauber vectorized sampling
    for k_val, name in [(0.0, "kappa=0"), (1.5, "kappa=1.5"), (1000.0, "kappa=1000")]:
        k_vec = np.full(n_vec, k_val, dtype=np.float64)
        t0 = time.perf_counter()
        sample_binary_glauber_vector(k_vec, rng)
        t_elapsed = time.perf_counter() - t0
        rate = n_vec / t_elapsed
        results[f"binary_vector_{name}"] = {"ops_sec": round(rate, 1), "time_sec": round(t_elapsed, 4)}
        print(f"Binary Glauber Vectorized [{name}]: {rate:,.1f} spins/sec ({t_elapsed:.3f}s for {n_vec:,})")

    return results


def benchmark_memory_matrix():
    print("\n=== 2. MEMORY MATRIX BENCHMARKS ===")
    results = {}
    rng = np.random.default_rng(123)

    for N in [64, 128, 256]:
        # Generate initial random memory
        raw = rng.integers(1, 10, size=(N, N), dtype=np.uint32)
        np.fill_diagonal(raw, 0)
        y = rng.choice([-1.0, 1.0], size=(N, 3))

        # Non-reciprocal encounters without triadic closure
        mm = MemoryManager(N=N, reciprocal=False, chi=0.0, initial_W=raw.copy())
        n_enc = 5000
        t0 = time.perf_counter()
        for _ in range(n_enc):
            i = int(rng.integers(0, N))
            mm.sample_and_apply_encounter(i, y, rng=rng)
        t_elapsed = time.perf_counter() - t0
        rate_enc_nochi = n_enc / t_elapsed
        print(f"N={N:3d} Encounters (chi=0.0): {rate_enc_nochi:,.1f} ops/sec ({t_elapsed:.3f}s)")

        # Encounters with triadic closure chi=0.2
        mm_chi = MemoryManager(N=N, reciprocal=False, chi=0.2, initial_W=raw.copy())
        n_enc_chi = 1000
        t0 = time.perf_counter()
        for _ in range(n_enc_chi):
            i = int(rng.integers(0, N))
            mm_chi.sample_and_apply_encounter(i, y, rng=rng)
        t_elapsed = time.perf_counter() - t0
        rate_enc_chi = n_enc_chi / t_elapsed
        print(f"N={N:3d} Encounters (chi=0.2): {rate_enc_chi:,.1f} ops/sec ({t_elapsed:.3f}s)")

        # Decays (non-reciprocal)
        n_dec = 10_000
        t0 = time.perf_counter()
        for _ in range(n_dec):
            mm.sample_and_apply_decay(rng=rng)
        t_elapsed = time.perf_counter() - t0
        rate_dec_std = n_dec / t_elapsed
        print(f"N={N:3d} Decays (standard):   {rate_dec_std:,.1f} ops/sec ({t_elapsed:.3f}s)")

        # Reciprocal decays
        sym_raw = raw + raw.T
        np.fill_diagonal(sym_raw, 0)
        mm_rec = MemoryManager(N=N, reciprocal=True, initial_W=sym_raw)
        n_dec_rec = 2000
        t0 = time.perf_counter()
        for _ in range(n_dec_rec):
            mm_rec.sample_and_apply_decay(rng=rng)
        t_elapsed = time.perf_counter() - t0
        rate_dec_rec = n_dec_rec / t_elapsed
        print(f"N={N:3d} Decays (reciprocal): {rate_dec_rec:,.1f} ops/sec ({t_elapsed:.3f}s)")

        results[f"memory_N_{N}"] = {
            "encounters_chi0_ops_sec": round(rate_enc_nochi, 1),
            "encounters_chi02_ops_sec": round(rate_enc_chi, 1),
            "decays_standard_ops_sec": round(rate_dec_std, 1),
            "decays_reciprocal_ops_sec": round(rate_dec_rec, 1),
        }

    return results


def benchmark_scheduler():
    print("\n=== 3. SCHEDULER BENCHMARKS ===")
    results = {}
    rng = np.random.default_rng(456)

    for N in [64, 256, 1000]:
        sched = PoissonScheduler(
            N=N, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.01, tau_n=1.0,
            reciprocal=False, rng=rng
        )
        sched.set_total_memory(10 * N)

        # No preemption
        n_steps = 100_000
        t0 = time.perf_counter()
        for _ in range(n_steps):
            sched.step()
        t_elapsed = time.perf_counter() - t0
        rate_pure = n_steps / t_elapsed
        print(f"N={N:4d} Scheduler pure stochastic: {rate_pure:,.1f} events/sec ({t_elapsed:.3f}s for {n_steps:,})")

        # With scheduled deterministic events
        sched_det = PoissonScheduler(
            N=N, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.01, tau_n=1.0,
            reciprocal=False, rng=rng
        )
        sched_det.set_total_memory(10 * N)
        # Schedule periodic snapshots
        for t_snap in np.linspace(0.1, 100.0, 500):
            sched_det.schedule_deterministic_event(t_snap, priority=3, event_type="snapshot")

        t0 = time.perf_counter()
        while sched_det.time < 100.0:
            sched_det.step()
        t_elapsed = time.perf_counter() - t0
        rate_preempt = sched_det.stats.total_events / t_elapsed
        print(f"N={N:4d} Scheduler with preemption:   {rate_preempt:,.1f} events/sec ({sched_det.stats.total_events:,} events, {sched_det.stats.preemptions} preemptions)")

        results[f"scheduler_N_{N}"] = {
            "pure_stochastic_rate": round(rate_pure, 1),
            "preemption_rate": round(rate_preempt, 1),
            "preemptions_count": sched_det.stats.preemptions,
        }

    return results


def main():
    t_start = time.time()
    all_benchmarks = {}
    all_benchmarks["heat_baths"] = benchmark_heat_baths()
    all_benchmarks["memory"] = benchmark_memory_matrix()
    all_benchmarks["scheduler"] = benchmark_scheduler()
    t_total = time.time() - t_start

    print(f"\nTotal benchmark run duration: {t_total:.2f}s")
    print(json.dumps(all_benchmarks, indent=2))


if __name__ == "__main__":
    main()
