"""Parallel Campaign Execution Engine and Resumable Checkpointing.

Strictly decouples execution orchestration from physical models and controllers:
- ResourceProfile: Declarative compute budget ('lightweight', 'standard', 'extended').
- CampaignRunner: Multiprocess parallel executor with deterministic seed isolation.
- Resumable Checkpointing: Incremental growth executing only missing realizations.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.analytics.discrepancy import calculate_population_discrepancy
from src.analytics.influence import calculate_influence_metrics
from src.execution.runner import (
    NumpyEncoder,
    TrajectoryRunner,
    _atomic_write_json,
    _canonical_json_bytes,
    load_checkpoint,
)


RESULT_SCHEMA_VERSION = 1


def _stable_digest(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_completed_result(path: Path, expected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a verified completed result, or None for stale/incomplete data."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        payload = envelope["payload"]
        checksum = envelope["sha256"]
        if checksum != _stable_digest(payload):
            return None
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION or payload.get("status") != "complete":
            return None
        for key in ("experiment_id", "realization_id", "seed", "horizon"):
            if payload.get(key) != expected.get(key):
                return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class ResourceProfile:
    """Resource and scale profile for simulation campaigns."""

    name: str
    N: int
    D: int
    horizon: float
    num_realizations: int
    max_workers: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined standard resource profiles
PROFILES: Dict[str, ResourceProfile] = {
    "lightweight": ResourceProfile(
        name="lightweight",
        N=10,
        D=1,
        horizon=2.0,
        num_realizations=2,
        max_workers=2,
        description="Miniature toy campaign for fast local pipeline validation and unit tests.",
    ),
    "small": ResourceProfile(
        name="small",
        N=10,
        D=1,
        horizon=2.0,
        num_realizations=2,
        max_workers=2,
        description="Miniature toy campaign for fast local pipeline validation and unit tests (alias for lightweight).",
    ),
    "standard": ResourceProfile(
        name="standard",
        N=30,
        D=1,
        horizon=20.0,
        num_realizations=10,
        max_workers=4,
        description="Standard exploratory campaign profile with moderate network size and duration.",
    ),
    "extended": ResourceProfile(
        name="extended",
        N=100,
        D=3,
        horizon=50.0,
        num_realizations=30,
        max_workers=8,
        description="Confirmatory multi-component campaign profile with high statistical replication.",
    ),
}


def get_profile(name: str) -> ResourceProfile:
    """Retrieve predefined resource profile by name."""
    clean_name = name.strip().lower()
    if clean_name not in PROFILES:
        raise ValueError(f"Unknown resource profile '{name}'. Choose from: {list(PROFILES.keys())}")
    return PROFILES[clean_name]


@dataclass
class RealizationResult:
    """Outcome and telemetry of an individual realization."""

    realization_id: str
    seed: int
    horizon: float
    final_time: float
    m_x_mean: List[float]
    m_y_mean: List[float]
    C_N: Optional[float]
    N_eff: Optional[float]
    d2: float
    total_memory: int
    f_active: float
    checkpoint_path: Optional[str] = None
    is_cached: bool = False
    resumed_from_partial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignResult:
    """Consolidated campaign results across all realizations."""

    campaign_id: str
    profile_name: str
    total_realizations: int
    executed_count: int
    skipped_count: int
    realizations: List[RealizationResult]
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_C_N(self) -> float:
        """Average normalized concentration across completed realizations."""
        vals = [r.C_N for r in self.realizations if r.C_N is not None and np.isfinite(r.C_N)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_d2(self) -> float:
        """Average belief discrepancy across completed realizations."""
        vals = [r.d2 for r in self.realizations if not np.isnan(r.d2)]
        return float(np.mean(vals)) if vals else 0.0


def _run_single_realization(work_item: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function executed across parallel worker processes.

    Strictly deterministic: seed assignment depends only on realization index,
    completely independent of worker scheduling order.
    """
    realization_id = work_item["realization_id"]
    seed = work_item["seed"]
    horizon = work_item["horizon"]
    output_path = Path(work_item["output_path"])
    partial_path = Path(work_item["partial_path"])
    params = work_item["params"]

    # 1. A race-safe cache check validates identity, completeness, and checksum.
    if output_path.is_file():
        data = _read_completed_result(output_path, work_item)
        if data is not None:
            data["is_cached"] = True
            return data

    # 2. Resume a verified in-progress realization, or start it deterministically.
    runner: TrajectoryRunner
    resumed_from_partial = False
    if partial_path.is_file():
        try:
            loaded = load_checkpoint(partial_path)
            if all(loaded.get(key) == work_item.get(key) for key in ("experiment_id", "realization_id", "seed", "horizon")):
                runner = TrajectoryRunner.from_checkpoint(loaded)
                resumed_from_partial = True
            else:
                runner = TrajectoryRunner(**params, seed=seed)
        except (ValueError, OSError, KeyError, TypeError):
            runner = TrajectoryRunner(**params, seed=seed)
    else:
        runner = TrajectoryRunner(**params, seed=seed)

    interval = float(work_item.get("checkpoint_interval_seconds", 60.0))
    next_checkpoint_at = time.monotonic() + max(0.0, interval)
    simulated_chunk = max(0.1, horizon / 100.0)
    traj = None
    while runner.state.t < horizon - 1e-12:
        traj = runner.run(min(horizon, runner.state.t + simulated_chunk))
        if runner.state.t < horizon - 1e-12 and time.monotonic() >= next_checkpoint_at:
            runner.get_checkpoint(
                partial_path,
                extra={
                    "status": "partial",
                    "experiment_id": work_item["experiment_id"],
                    "realization_id": realization_id,
                    "seed": seed,
                    "horizon": horizon,
                },
            )
            next_checkpoint_at = time.monotonic() + max(0.0, interval)
    assert traj is not None
    final_state = traj.final_state

    # 3. Compute analytics
    influence = calculate_influence_metrics(final_state.W)
    C_N = float(influence["C_N"]) if np.isfinite(influence["C_N"]) else None
    eff_raw = influence.get("effective_leaders", influence.get("N_eff", 1.0))
    N_eff = float(eff_raw) if np.isfinite(eff_raw) else None

    d2_val = calculate_population_discrepancy(final_state.x, final_state.y)
    d2 = float(d2_val) if not np.isnan(d2_val) else 0.0

    m_x_mean = np.mean(final_state.x, axis=0).tolist()
    m_y_mean = np.mean(final_state.y, axis=0).tolist()
    total_mem = int(np.sum(final_state.W))
    f_active = float(influence["f_active"])

    # 4. Save checkpoint to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": work_item["experiment_id"],
        "realization_id": realization_id,
        "seed": seed,
        "horizon": horizon,
        "final_time": float(final_state.t),
        "m_x_mean": m_x_mean,
        "m_y_mean": m_y_mean,
        "C_N": C_N,
        "N_eff": N_eff,
        "d2": d2,
        "total_memory": total_mem,
        "f_active": f_active,
        "checkpoint_path": str(output_path),
        "is_cached": False,
        "resumed_from_partial": resumed_from_partial,
        "final_state": final_state.to_dict(),
        "trajectory_checkpoint": {
            key: value for key, value in runner.get_checkpoint().items() if key != "rng"
        },
    }

    _atomic_write_json(output_path, {"payload": result_record, "sha256": _stable_digest(result_record)})
    if partial_path.is_file():
        partial_path.unlink()

    return result_record


class CampaignRunner:
    """Orchestrates parallel campaigns with resumability and incremental growth.

    Guarantees:
    - Stable realization identifiers.
    - Seed assignment strictly independent of worker scheduling order.
    - Resumability: running twice skips already completed realizations.
    - Incremental growth: increasing num_realizations only runs the new additions.
    """

    def __init__(
        self,
        profile: Optional[Union[str, ResourceProfile]] = None,
        *,
        master_seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        campaign_id: str = "campaign",
        max_workers: Optional[int] = None,
        checkpoint_interval_seconds: float = 60.0,
        N: Optional[int] = None,
        D: Optional[int] = None,
        horizon: Optional[float] = None,
        num_realizations: Optional[int] = None,
        T: float = 0.3,
        delta: float = 0.1,
        nu: float = 2.0,
        J: float = 1.0,
        K: Union[float, np.ndarray] = 0.5,
        **extra_params: Any,
    ) -> None:
        if profile is not None:
            self.profile = get_profile(profile) if isinstance(profile, str) else profile
        else:
            self.profile = get_profile("lightweight")

        self.master_seed = int(master_seed)
        self.campaign_id = campaign_id
        self.output_dir = Path(output_dir) if output_dir is not None else Path(f"./campaign_output_{campaign_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.N = N if N is not None else self.profile.N
        self.D = D if D is not None else self.profile.D
        self.horizon = float(horizon) if horizon is not None else self.profile.horizon
        self.num_realizations = int(num_realizations) if num_realizations is not None else self.profile.num_realizations

        if max_workers is not None:
            self.max_workers = max(1, int(max_workers))
        else:
            self.max_workers = max(1, self.profile.max_workers)

        self.sim_params = {
            "N": self.N,
            "D": self.D,
            "T": T,
            "delta": delta,
            "nu": nu,
            "J": J,
            "K": K,
        }
        self.sim_params.update(extra_params)
        self.checkpoint_interval_seconds = max(0.0, float(checkpoint_interval_seconds))

        self.definition = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "master_seed": self.master_seed,
            "N": self.N,
            "D": self.D,
            "horizon": self.horizon,
            "sim_params": self.sim_params,
        }
        self.experiment_id = _stable_digest(self.definition)

        # Write or update campaign manifest
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Record campaign metadata and parameter manifest."""
        manifest = {
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "schema_version": RESULT_SCHEMA_VERSION,
            "profile": self.profile.name,
            "master_seed": self.master_seed,
            "N": self.N,
            "D": self.D,
            "horizon": self.horizon,
            "num_realizations": self.num_realizations,
            "max_workers": self.max_workers,
            "sim_params": self.sim_params,
        }
        manifest_path = self.output_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)

    def run(self) -> CampaignResult:
        """Execute the campaign, skipping already completed realizations.

        Returns:
            CampaignResult containing all completed realizations.
        """
        checkpoints_dir = self.output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Build list of work items
        work_items: List[Dict[str, Any]] = []
        pending_items: List[Dict[str, Any]] = []
        cached_results: Dict[str, Dict[str, Any]] = {}

        for idx in range(self.num_realizations):
            realization_id = f"{self.campaign_id}_{self.experiment_id[:12]}_r{idx:04d}"
            # Canonical SHA-256 identity -> SeedSequence -> named PCG64 seed word.
            seed_identity = {
                "master_seed": self.master_seed,
                "experiment_id": self.experiment_id,
                "realization_index": idx,
                "stream_role": "physical_simulation",
            }
            digest = hashlib.sha256(_canonical_json_bytes(seed_identity)).digest()
            entropy = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
            realization_seed = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
            out_file = checkpoints_dir / f"{realization_id}.json"

            item = {
                "realization_id": realization_id,
                "experiment_id": self.experiment_id,
                "seed": realization_seed,
                "horizon": self.horizon,
                "output_path": str(out_file),
                "partial_path": str(out_file.with_suffix(".partial.json")),
                "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
                "params": self.sim_params,
            }
            work_items.append(item)

            # Check if realization was already executed and saved to disk
            if out_file.is_file():
                data = _read_completed_result(out_file, item)
                if data is not None:
                    data["is_cached"] = True
                    cached_results[realization_id] = data
                else:
                    pending_items.append(item)
            else:
                pending_items.append(item)

        executed_count = 0
        newly_executed_results: Dict[str, Dict[str, Any]] = {}

        if pending_items:
            # Parallel execution via multiprocessing spawn context
            if self.max_workers > 1 and len(pending_items) > 1:
                ctx = get_context("spawn")
                pool_size = min(self.max_workers, len(pending_items))
                with ctx.Pool(processes=pool_size) as pool:
                    raw_results = pool.map(_run_single_realization, pending_items)
                for res in raw_results:
                    newly_executed_results[res["realization_id"]] = res
                    executed_count += 1
            else:
                for item in pending_items:
                    res = _run_single_realization(item)
                    newly_executed_results[res["realization_id"]] = res
                    executed_count += 1

        # Combine cached and newly executed results in stable index order
        all_realizations: List[RealizationResult] = []
        for idx in range(self.num_realizations):
            r_id = f"{self.campaign_id}_{self.experiment_id[:12]}_r{idx:04d}"
            if r_id in cached_results:
                rec = cached_results[r_id]
            else:
                rec = newly_executed_results[r_id]

            r_res = RealizationResult(
                realization_id=rec["realization_id"],
                seed=rec["seed"],
                horizon=rec["horizon"],
                final_time=rec["final_time"],
                m_x_mean=rec["m_x_mean"],
                m_y_mean=rec["m_y_mean"],
                C_N=rec["C_N"],
                N_eff=rec["N_eff"],
                d2=rec["d2"],
                total_memory=rec["total_memory"],
                f_active=rec["f_active"],
                checkpoint_path=rec.get("checkpoint_path"),
                is_cached=rec.get("is_cached", False),
                resumed_from_partial=rec.get("resumed_from_partial", False),
            )
            all_realizations.append(r_res)

        skipped_count = len(cached_results)

        manifest_path = self.output_dir / "manifest.json"
        manifest_data = {}
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        return CampaignResult(
            campaign_id=self.campaign_id,
            profile_name=self.profile.name,
            total_realizations=self.num_realizations,
            executed_count=executed_count,
            skipped_count=skipped_count,
            realizations=all_realizations,
            manifest=manifest_data,
        )
