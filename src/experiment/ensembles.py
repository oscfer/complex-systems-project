"""Origin-level ensembles: independent origins, paired branches, aggregation.

Statistical units are independent origins (frames/repeated futures are not
independent origins). Each origin draws its realism state once from
a seed derived deterministically from (seed_base, origin_index); perturbation
branches reuse the SAME origin state and continue from the establishment
checkpoint with seed offsets, so effects are origin-paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config.profiles import RunProfile, build_realism_runner
from src.analytics.influence import calculate_influence_metrics
from src.analytics.discrepancy import calculate_population_discrepancy


def derive_origin_seed(seed_base: int, origin_index: int) -> int:
    """Deterministic per-origin seed independent of execution order."""
    import hashlib
    import json
    payload = json.dumps({"seed_base": int(seed_base), "origin": int(origin_index)}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).digest()
    return int(np.frombuffer(digest[:4], dtype=np.uint32)[0])


def freeze_memory_dynamics(runner) -> None:
    """Genuinely freeze W: disable encounter attempts AND token decay.

    Setting only nu=0 leaves forgetting active, which still reshapes W.
    Both the runner attributes and the scheduler channels are zeroed so no
    encounter or decay event can fire after freezing.
    """
    runner.nu = 0.0
    runner.delta = 0.0
    runner.scheduler.nu = 0.0
    runner.scheduler.delta = 0.0


def _origin_seed_offset(base_seed: int, branch: str) -> int:
    """Distinct continuation seed per branch so futures differ across arms."""
    import hashlib
    digest = hashlib.sha256(f"branch::{branch}".encode()).digest()
    return int(np.frombuffer(digest[:4], dtype=np.uint32)[0])


@dataclass
class OriginResult:
    origin_index: int
    seed: int
    branch: str
    final_m_y: List[float]
    final_m_x: List[float]
    C_N: Optional[float]
    N_eff: Optional[float]
    d2: float
    total_memory: int
    f_active: float
    modularity_Q: Optional[float] = None
    num_communities: Optional[int] = None
    times: List[float] = field(default_factory=list)
    m_y_series: List[List[float]] = field(default_factory=list)


@dataclass
class EnsembleResult:
    profile_name: str
    tier: str
    branch_results: Dict[str, List[OriginResult]]
    profile_manifest: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for branch, results in self.branch_results.items():
            c_vals = np.array([r.C_N for r in results if r.C_N is not None and np.isfinite(r.C_N)])
            d_vals = np.array([r.d2 for r in results])
            my = np.array([r.final_m_y[0] for r in results]) if results else np.array([])
            q_vals = np.array([r.modularity_Q for r in results if r.modularity_Q is not None])
            out[branch] = {
                "n_origins": len(results),
                "mean_m_y": float(np.mean(my)) if len(my) else None,
                "mean_C_N": float(np.mean(c_vals)) if len(c_vals) else None,
                "mean_d2": float(np.mean(d_vals)) if len(d_vals) else None,
                "mean_modularity_Q": float(np.mean(q_vals)) if len(q_vals) else None,
            }
        return out

    def paired_effect(self, branch_a: str, branch_b: str) -> Dict[str, Any]:
        """Origin-level paired differences of final m_y with a bootstrap CI."""
        a = self.branch_results.get(branch_a, [])
        b = self.branch_results.get(branch_b, [])
        common = sorted(set(r.origin_index for r in a) & set(r.origin_index for r in b))
        if not common:
            return {"n_pairs": 0}
        da = {r.origin_index: r for r in a}
        db = {r.origin_index: r for r in b}
        diffs = np.array([da[k].final_m_y[0] - db[k].final_m_y[0] for k in common])
        rng = np.random.default_rng(0)
        boots = [float(np.mean(rng.choice(diffs, size=len(diffs)))) for _ in range(2000)]
        return {
            "n_pairs": len(common),
            "mean_effect": float(np.mean(diffs)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }


def run_single_origin(
    profile: RunProfile,
    origin_index: int,
    branches: Sequence[str] = ("unperturbed",),
    progress: bool = False,
) -> List[OriginResult]:
    """Run one independent origin, optionally with paired branches.

    Branch 'unperturbed' establishes the origin and continues to the horizon.
    Branch 'perturbed' re-establishes the same origin state, then installs a
    targeted commanded field opposing the majority sign during half of the
    post-establishment window (charge accounted as commanded u).
    Branch 'frozen' continues with encounter attempts disabled (nu = 0), i.e.
    the adaptive contact mechanism switched off after establishment.
    """
    seed = derive_origin_seed(profile.seed_base, origin_index)
    t_est = profile.establishment_time
    results: List[OriginResult] = []
    if progress:
        print(f"  origin {origin_index + 1} (seed {seed})")

    for branch in branches:
        runner = build_realism_runner(profile, seed)
        runner.run(horizon=t_est)
        checkpoint = runner.get_checkpoint()

        branch_seed = (seed + _origin_seed_offset(seed, branch)) % (2**31)
        cont = type(runner).from_checkpoint(checkpoint)
        fresh_rng = np.random.default_rng(branch_seed)
        cont.rng = fresh_rng
        cont.scheduler.rng = fresh_rng
        cont.memory_mgr.rng = fresh_rng

        if branch == "frozen":
            freeze_memory_dynamics(cont)
        elif branch == "perturbed":
            from src.experiment.interventions import CommandedFieldIntervention
            majority_sign = float(np.sign(np.mean(cont.state.y[:, 0])) or 1.0)
            targets = np.zeros(profile.N, dtype=bool)
            targets[: max(1, profile.N // 10)] = True  # top ~10% access panel
            cont.interventions = [
                CommandedFieldIntervention(
                    start_time=t_est,
                    duration=max(1.0, (profile.horizon - t_est) * 0.5),
                    amplitude=-majority_sign * 3.0,
                    component=0,
                    target_mask=targets,
                )
            ]
            cont._schedule_interventions()

        traj = cont.run(horizon=profile.horizon)
        final = traj.final_state
        influence = calculate_influence_metrics(final.W)
        mod = None
        k = None
        if origin_index % 5 == 0:  # community metrics on a subset of origins (cost)
            from src.analytics.communities import detect_communities
            res = detect_communities(final.W, gamma=1.0, restarts=5, rng=int(seed))
            mod = res["modularity_Q"]
            k = res["num_communities"]
        c_val = influence.get("C")
        results.append(
            OriginResult(
                origin_index=origin_index,
                seed=seed,
                branch=branch,
                final_m_y=list(np.mean(final.y, axis=0)),
                final_m_x=list(np.mean(final.x, axis=0)),
                C_N=float(influence["C_N"]) if influence["C_N"] == influence["C_N"] else None,
                N_eff=(1.0 / c_val if c_val not in (None, 0) and c_val == c_val else None),
                d2=float(calculate_population_discrepancy(final.x, final.y)),
                total_memory=int(np.sum(final.W)),
                f_active=float(influence["f_active"]),
                modularity_Q=float(mod) if mod is not None and mod == mod else None,
                num_communities=k,
                times=[float(s["t"]) for s in traj.timeseries],
                m_y_series=[list(s["m_y"]) for s in traj.timeseries],
            )
        )
    return results


def run_ensemble(
    profile: RunProfile,
    branches: Sequence[str] = ("unperturbed",),
    n_origins: Optional[int] = None,
    progress: bool = True,
) -> EnsembleResult:
    """Run an origin-level ensemble under the profile."""
    n = n_origins if n_origins is not None else profile.n_origins
    branch_results: Dict[str, List[OriginResult]] = {b: [] for b in branches}
    for idx in range(n):
        results = run_single_origin(profile, idx, branches=branches, progress=progress)
        for r in results:
            branch_results[r.branch].append(r)
    return EnsembleResult(
        profile_name=f"{profile.topic}_{profile.tier}",
        tier=profile.tier,
        branch_results=branch_results,
        profile_manifest=profile.to_manifest(),
    )