"""Stage 0 timing and cost calibration campaign.

Profiles setup vs kernel separation across population scales:
- Measures setup time (state and scheduler allocation).
- Measures kernel execution time (continuous Poisson jump stepping).
- Tracks total events processed and calculates event throughput (events/sec).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import numpy as np

from src.core.fields import compute_single_private_field, compute_single_public_field
from src.core.heat_baths import sample_binary_glauber_scalar
from src.core.memory import MemoryManager
from src.core.scheduler import PoissonScheduler
from src.core.state import State, StateMode
from src.execution.runner import TrajectoryRunner


class Stage0TimingCampaign:
    """Stage 0 benchmark verifying setup vs kernel separation."""

    def __init__(
        self,
        N: int = 256,
        D: int = 1,
        horizon: float = 10.0,
        num_runs: int = 1,
        T: float = 0.3,
        delta: float = 0.1,
        nu: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.N = int(N)
        self.D = int(D)
        self.horizon = float(horizon)
        self.num_runs = max(1, int(num_runs))
        self.T = float(T)
        self.delta = float(delta)
        self.nu = float(nu)
        self.seed = int(seed)

    def execute(self) -> Dict[str, Any]:
        """Execute timing benchmark across specified number of runs."""
        setup_times = []
        kernel_times = []
        event_counts = []

        for run_idx in range(self.num_runs):
            run_seed = self.seed + run_idx * 10007
            # 1. Benchmark setup
            t_setup_start = time.perf_counter()
            runner = TrajectoryRunner(
                N=self.N,
                D=self.D,
                T=self.T,
                nu=self.nu,
                delta=self.delta,
                seed=run_seed,
            )
            t_setup_end = time.perf_counter()
            setup_times.append(t_setup_end - t_setup_start)

            # 2. Benchmark kernel execution
            t_kernel_start = time.perf_counter()
            trajectory = runner.run(self.horizon)
            t_kernel_end = time.perf_counter()
            kernel_times.append(t_kernel_end - t_kernel_start)
            event_counts.append(int(trajectory.data["stats"]["total_events"]))

        total_setup = float(np.mean(setup_times))
        total_kernel = float(np.mean(kernel_times))
        total_events = int(np.mean(event_counts))
        throughput = total_events / max(total_kernel, 1e-9)

        return {
            "setup_time": total_setup,
            "kernel_time": total_kernel,
            "total_events": total_events,
            "events_per_sec": throughput,
            "total_time": total_setup + total_kernel,
            "N": self.N,
            "horizon": self.horizon,
            "num_runs": self.num_runs,
        }
