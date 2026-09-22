"""Standardized scientific output bundles for ensembles and campaigns.

Every experiment writes a directory `results/<profile_name>_<timestamp>/`
containing:
- timeseries.csv      -- macro observables per origin/branch over time (long format)
- origins_summary.csv -- one row per origin/branch (endpoint statistics)
- observables.npz     -- per-origin m_y response curves (for 2D/3D response plots)
- manifest.json       -- profile, summary statistics, code versions, checksum
Final networks (W, y) are exported separately by the notebooks via
export_networks (they are large; the ensemble layer keeps aggregates).
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.experiment.ensembles import EnsembleResult


def _repo_root() -> Path:
    """Repository root containing src/ and tools/ (this file lives in src/io/)."""
    return Path(__file__).resolve().parents[2]


def _git_head(repo_root: Path) -> Optional[str]:
    """Current git HEAD commit hash, or None when git is unavailable/dirty-free info fails."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            head = out.stdout.strip()
            return head or None
    except Exception:
        pass
    return None


def _source_digest(repo_root: Path) -> Optional[str]:
    """sha256 over the sorted per-file sha256 digests of tracked files under src/ and tools/.

    Uses `git ls-files src tools` when available so untracked scratch files are
    excluded (dirty state is captured by which files are tracked plus their
    current contents); falls back to walking src/ and tools/ without
    __pycache__ when git is unavailable. Returns None only when no source
    files can be hashed at all.
    """
    files: List[Path] = []
    try:
        out = subprocess.run(
            ["git", "ls-files", "src", "tools"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            files = [repo_root / rel for rel in out.stdout.split() if rel]
    except Exception:
        files = []
    if not files:
        for sub in ("src", "tools"):
            base = repo_root / sub
            if base.is_dir():
                files.extend(
                    p for p in sorted(base.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts
                )
        files = sorted(set(files))
    file_lines: List[str] = []
    for p in sorted(files):
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
        try:
            rel = p.relative_to(repo_root).as_posix()
        except ValueError:
            rel = str(p)
        file_lines.append(f"{rel}:{digest}")
    if not file_lines:
        return None
    return hashlib.sha256("\n".join(file_lines).encode("utf-8")).hexdigest()


def _runtime_versions() -> Dict[str, str]:
    """Runtime library versions relevant to reproducing an executed bundle."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import importlib.metadata as _md
    except Exception:
        _md = None  # type: ignore[assignment]
    for name in ("networkx", "matplotlib"):
        try:
            versions[name] = _md.version(name) if _md is not None else "unknown"
        except Exception:
            versions[name] = "unknown"
    return versions


def _executable_provenance() -> Dict[str, Any]:
    """Executable-source provenance for NEW bundle manifests.

    model_hash (problem configuration identity) is recorded separately in
    manifest['problem']; these fields add code-revision and runtime identity.
    Old bundles without these fields remain fully readable: all loaders use
    .get defaults and never require them.
    """
    root = _repo_root()
    return {
        "git_head": _git_head(root),
        "source_digest": _source_digest(root),
        "runtime_versions": _runtime_versions(),
    }


def export_bundle(result: EnsembleResult, out_dir: str = "results") -> str:
    """Write the output bundle for an ensemble; returns the output path."""
    root = Path(out_dir)
    path = root / f"{result.profile_name}_{_time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)

    # 1. timeseries (long format)
    dim = 0
    for results in result.branch_results.values():
        if results:
            dim = len(results[0].final_m_y)
            break
    dim = max(dim, 1)
    header = "branch,origin,t," + ",".join(f"m_y_{a}" for a in range(dim))
    with open(path / "timeseries.csv", "w") as f:
        f.write(header + "\n")
        for branch, results in result.branch_results.items():
            for r in results:
                for t, m_y in zip(r.times, r.m_y_series):
                    vals = ",".join(f"{v:.6g}" for v in m_y)
                    f.write(f"{branch},{r.origin_index},{t:.6g},{vals}\n")

    # 2. origin summary
    header2 = ("branch,origin,seed,m_y_0,m_x_0,C_N,N_eff,d2,total_memory,f_active,"
               "modularity_Q,num_communities")
    with open(path / "origins_summary.csv", "w") as f:
        f.write(header2 + "\n")
        for branch, results in result.branch_results.items():
            for r in results:
                f.write(",".join([
                    branch, str(r.origin_index), str(r.seed),
                    f"{r.final_m_y[0]:.6g}", f"{r.final_m_x[0]:.6g}",
                    "" if r.C_N is None else f"{r.C_N:.6g}",
                    "" if r.N_eff is None else f"{r.N_eff:.6g}",
                    f"{r.d2:.6g}", str(r.total_memory), f"{r.f_active:.6g}",
                    "" if r.modularity_Q is None else f"{r.modularity_Q:.6g}",
                    "" if r.num_communities is None else str(r.num_communities),
                ]) + "\n")

    # 3. per-branch m_y response curves (npz, used for 2D/3D response plots)
    series = [np.asarray(r.m_y_series) for br in result.branch_results.values() for r in br]
    tags = [f"{branch}_origin{r.origin_index}"
            for branch, results in result.branch_results.items() for r in results]
    if series:
        np.savez_compressed(
            path / "observables.npz",
            tags=np.array(tags),
            m_y_series=np.array(series, dtype=object),
        )

    # 4. manifest with checksum
    manifest = {
        "profile": result.profile_manifest,
        "profile_name": result.profile_name,
        "tier": result.tier,
        "summary": result.summary(),
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return str(path)


def export_networks(trajectories_by_tag, out_path):
    """Export final networks: {tag: {'W': ndarray, 'y': ndarray}} to npz."""
    arrays = {}
    for tag, item in trajectories_by_tag.items():
        arrays[f"{tag}__W"] = np.asarray(item["W"])
        arrays[f"{tag}__y"] = np.asarray(item["y"])
    np.savez_compressed(out_path, **arrays)
    return out_path


def _json_default(obj: Any):
    """JSON-serialize numpy scalars/arrays; NaN -> None."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return str(obj)


def make_campaign_bundle_dir(
    out_dir: Union[str, Path], campaign_id: str, timestamp: Optional[str] = None,
    *, resume: bool = False,
) -> Path:
    """Create an isolated bundle, or identify one existing bundle for resumption."""
    root = Path(out_dir)
    prefix = f"control_{campaign_id}_"
    is_bundle_path = root.name.startswith(prefix)
    if resume:
        if is_bundle_path:
            if not root.is_dir():
                raise FileNotFoundError(f"Campaign bundle does not exist: {root}")
            path = root
        else:
            matches = sorted(
                d for d in root.glob(f"{prefix}*") if d.is_dir()
            ) if root.exists() else []
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple campaign bundles match {campaign_id}; pass the exact bundle path to resume"
                )
            path = matches[0] if matches else None
            if path is None:
                # No previous run exists; a resume-enabled first call creates one.
                return make_campaign_bundle_dir(root, campaign_id, timestamp, resume=False)
    else:
        if is_bundle_path:
            raise ValueError(f"Fresh campaign needs an output parent, not an existing bundle path: {root}")
        ts = timestamp or _time.strftime("%Y%m%d_%H%M%S")
        base = root / f"{prefix}{ts}"
        path = base
        suffix = 1
        while path.exists():
            path = root / f"{prefix}{ts}_{suffix:02d}"
            suffix += 1
        path.mkdir(parents=True, exist_ok=False)
    (path / "snapshots").mkdir(parents=True, exist_ok=True)
    return path


def export_control_bundle(result, out_dir: str = "results") -> str:
    """Write the unified output bundle for a control campaign; returns the output path.

    Writes into a single unified directory `results/control_<campaign_id>_<timestamp>/`:
    - rollout_results.csv       -- per origin/future: success, T_G, restricted
                                   time, expenditure, distinct nodes, relapse,
                                   violation, seeds (all campaign arms)
    - adaptive_frozen_pairs.csv -- matched adaptive/frozen branch pairs on
                                   identical fresh origins and seeds
    - policy_schedules.json     -- selected candidate parameters + baseline
                                   schedules (reusable intervention definitions)
    - search_summary.json       -- candidate scores, rollout counts, search cost
    - preparation_attempts.json -- all attempted, accepted, and rejected origins
    - trajectories.npz          -- scalar diagnostic trajectories (t, m_x, m_y,
                                   distance, predicate, hold, exposure, spending)
    - diagnostic_trajectories.json -- summary of recorded trajectories
    - action_history.json       -- realized action schedules and clampings
    - network_frames.npz        -- bounded network snapshots (W, y, x, masks, inputs)
    - retention_summary.json    -- original-state retention evaluations
    - manifest.json             -- declared spec, problem hash, seeds, config
                                   version, preparation/selection/reliability
                                   summaries, provenance, file inventory, checksum
    """
    # Imported lazily to keep the io -> control dependency cycle-free.
    from src.control.campaign import ControlCampaignResult

    if not isinstance(result, ControlCampaignResult):
        raise TypeError(f"expected ControlCampaignResult, got {type(result)!r}")

    root = Path(out_dir)
    if getattr(result, "bundle_path", None) and Path(result.bundle_path).exists() and (
        out_dir == "results" or Path(out_dir) == Path(result.bundle_path) or Path(out_dir) in Path(result.bundle_path).parents
    ):
        path = Path(result.bundle_path)
    elif (root / "manifest.json").exists() or (root.name.startswith("control_") and root.is_dir()):
        path = root
    else:
        campaign_id = result.manifest.get("campaign_id", "campaign")
        path = root / f"control_{campaign_id}_{_time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshots").mkdir(parents=True, exist_ok=True)

    # 1. rollout_results.csv (long format over every executed arm)
    header = ("group,policy_id,dynamics,origin_id,future_id,is_eligible,success,"
              "completion_time,restricted_time,expenditure,distinct_nodes_used,"
              "relapsed,sustained_success,terminal_target_violation,structural_modifications,"
              "sim_seed,ctl_seed")
    with open(path / "rollout_results.csv", "w") as f:
        f.write(header + "\n")
        for group, arms in (("baseline", result.baseline_results.items()),
                            ("evaluation", result.evaluation_results.items())):
            for arm_key, rollouts in arms:
                policy_id = rollouts[0].policy_id if rollouts else arm_key
                dynamics = rollouts[0].summaries.get("dynamics", "") if rollouts else ""
                for r in rollouts:
                    sim_seed = r.summaries.get("sim_seed", "")
                    ctl_seed = r.summaries.get("ctl_seed", "")
                    f.write(",".join([
                        group, r.policy_id, str(r.summaries.get("dynamics", dynamics)),
                        r.origin_id, str(r.future_id),
                        str(r.is_eligible), str(r.success),
                        "" if r.completion_time is None else f"{r.completion_time:.6g}",
                        f"{r.restricted_time:.6g}", f"{r.expenditure:.6g}",
                        str(r.distinct_nodes_used), str(r.relapsed),
                        str(bool(r.success and not r.relapsed)),
                        f"{r.terminal_target_violation:.6g}",
                        str(r.structural_modifications),
                        str(sim_seed), str(ctl_seed),
                    ]) + "\n")

    # 2. adaptive_frozen_pairs.csv (matched on identical origins and seeds)
    header2 = ("origin_id,future_id,sim_seed,ctl_seed,"
               "adaptive_success,adaptive_restricted_time,adaptive_expenditure,"
               "frozen_success,frozen_restricted_time,frozen_expenditure")
    with open(path / "adaptive_frozen_pairs.csv", "w") as f:
        f.write(header2 + "\n")
        for pair in result.adaptive_vs_frozen_pairs:
            ra, rf = pair["adaptive"], pair["frozen"]
            f.write(",".join([
                pair["origin_id"], str(pair["future_id"]),
                str(pair["sim_seed"]), str(pair["ctl_seed"]),
                "" if ra is None else str(ra.success),
                "" if ra is None else f"{ra.restricted_time:.6g}",
                "" if ra is None else f"{ra.expenditure:.6g}",
                str(rf.success), f"{rf.restricted_time:.6g}", f"{rf.expenditure:.6g}",
            ]) + "\n")

    # 3. policy_schedules.json (selected intervention + baseline schedules)
    policy_schedules = {
        "selected_policy_id": result.selection_summary.get("selected_candidate_id"),
        "selected_candidate": result.selection_summary.get("selected_candidate_params"),
        "baseline_schedule": result.manifest.get("baseline_schedule"),
        "declared_geometry": result.manifest.get("geometry"),
    }
    with open(path / "policy_schedules.json", "w") as f:
        json.dump(policy_schedules, f, indent=2, sort_keys=True, default=_json_default)

    # 4. search_summary.json (candidate scores, rollout counts, search cost)
    search = result.search_result
    search_summary = {
        "search_incumbent": result.selection_summary.get("search_incumbent"),
        "ranked_search_candidates": result.selection_summary.get(
            "ranked_search_candidates", []
        ),
        "total_rollouts": getattr(search, "total_rollouts", None),
        "search_exhausted": getattr(search, "search_exhausted", None),
        "candidate_scores": {
            cid: list(score) for cid, score in getattr(search, "candidate_scores", {}).items()
        },
        "candidate_summaries": getattr(search, "candidate_summaries", {}),
        "selection_summary": result.selection_summary,
    }
    with open(path / "search_summary.json", "w") as f:
        json.dump(search_summary, f, indent=2, sort_keys=True, default=_json_default)

    # 5. preparation_attempts.json
    prep_attempts = result.preparation_summary.get("attempts", [])
    if hasattr(result, "preparation_attempts") and result.preparation_attempts:
        prep_attempts = result.preparation_attempts
    with open(path / "preparation_attempts.json", "w") as f:
        json.dump(prep_attempts, f, indent=2, sort_keys=True, default=_json_default)

    # 5b. accepted_snapshots.json
    accepted_snaps = {
        "ensemble_id": result.preparation_summary.get("ensemble_id"),
        "origin_ids": result.preparation_summary.get("origin_ids", []),
        "accepted_origin_seeds": result.preparation_summary.get("accepted_origin_seeds", []),
        "accepted_snapshot_paths": result.preparation_summary.get("accepted_snapshot_paths", []),
        "pool_membership": result.preparation_summary.get("pool_membership", {}),
    }
    with open(path / "accepted_snapshots.json", "w") as f:
        json.dump(accepted_snaps, f, indent=2, sort_keys=True, default=_json_default)

    # 6. Extract diagnostic trajectories, actions, and network frames
    trajectories_npz: Dict[str, np.ndarray] = {}
    action_history_json: Dict[str, List[Dict[str, Any]]] = {}
    network_frames_npz: Dict[str, np.ndarray] = {}
    diagnostic_summary: Dict[str, Any] = {}

    if getattr(result, "diagnostic_trajectories", None):
        for k, v in result.diagnostic_trajectories.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    trajectories_npz[f"{k}__{sub_k}"] = np.asarray(sub_v)
                t_v = v.get("t", [])
                my_v = v.get("m_y", [])
                mx_v = v.get("m_x", [])
                exp_v = v.get("exposure", [0.0])
                if len(t_v) > 0:
                    diagnostic_summary[k] = {
                        "n_points": len(t_v),
                        "t_span": [float(t_v[0]), float(t_v[-1])],
                        "final_m_y": float(my_v[-1]) if len(my_v) > 0 else 0.0,
                        "final_m_x": float(mx_v[-1]) if len(mx_v) > 0 else 0.0,
                        "final_exposure": float(exp_v[-1]) if len(exp_v) > 0 else 0.0,
                    }
            elif isinstance(v, np.ndarray):
                trajectories_npz[k] = v

    if getattr(result, "action_histories", None):
        action_history_json.update(result.action_histories)

    if getattr(result, "network_frames", None):
        for k, frames in result.network_frames.items():
            if frames:
                network_frames_npz[f"{k}__t"] = np.array([f.get("t", 0.0) if isinstance(f, dict) else getattr(f, "t", 0.0) for f in frames], dtype=np.float64)
                network_frames_npz[f"{k}__W"] = np.array([f["W"] if isinstance(f, dict) else f.W for f in frames], dtype=np.float64)
                network_frames_npz[f"{k}__y"] = np.array([f["y"] if isinstance(f, dict) else f.y for f in frames], dtype=np.float64)
                network_frames_npz[f"{k}__x"] = np.array([f["x"] if isinstance(f, dict) else f.x for f in frames], dtype=np.float64)
                network_frames_npz[f"{k}__controlled_mask"] = np.array([f["controlled_mask"] if isinstance(f, dict) else f.controlled_mask for f in frames], dtype=bool)
                network_frames_npz[f"{k}__influence_inputs"] = np.array([f["influence_inputs"] if isinstance(f, dict) else f.influence_inputs for f in frames], dtype=np.float64)

    for group, arms in (("baseline", result.baseline_results.items()),
                        ("evaluation", result.evaluation_results.items())):
        for arm_key, rollouts in arms:
            for r in rollouts:
                rk = f"{group}__{r.policy_id}__{r.origin_id}__fut{r.future_id}__{r.summaries.get('dynamics', 'adaptive')}"
                rec = getattr(r, "recorder", None)
                rec_dict = r.summaries.get("recorder")

                pts = None
                if rec is not None and getattr(rec, "scalar_points", None):
                    pts = [p.to_dict() for p in rec.scalar_points]
                elif rec_dict and "scalar_points" in rec_dict:
                    pts = rec_dict["scalar_points"]

                if pts and f"{rk}__t" not in trajectories_npz:
                    t_arr = np.array([p["t"] for p in pts], dtype=np.float64)
                    mx_arr = np.array([p["m_x"] for p in pts], dtype=np.float64)
                    my_arr = np.array([p["m_y"] for p in pts], dtype=np.float64)
                    dist_arr = np.array([p["distance_to_target"] for p in pts], dtype=np.float64)
                    pred_arr = np.array([p["target_predicate"] for p in pts], dtype=bool)
                    hold_arr = np.array([p["hold_progress"] for p in pts], dtype=np.float64)
                    rel_arr = np.array([p["actual_release"] for p in pts], dtype=bool)
                    exp_arr = np.array([p["exposure"] for p in pts], dtype=np.float64)
                    rate_arr = np.array([p["spending_rate"] for p in pts], dtype=np.float64)
                    nodes_arr = np.array([p["distinct_nodes_used"] for p in pts], dtype=np.int32)

                    trajectories_npz[f"{rk}__t"] = t_arr
                    trajectories_npz[f"{rk}__m_x"] = mx_arr
                    trajectories_npz[f"{rk}__m_y"] = my_arr
                    trajectories_npz[f"{rk}__distance_to_target"] = dist_arr
                    trajectories_npz[f"{rk}__target_predicate"] = pred_arr
                    trajectories_npz[f"{rk}__hold_progress"] = hold_arr
                    trajectories_npz[f"{rk}__actual_release"] = rel_arr
                    trajectories_npz[f"{rk}__exposure"] = exp_arr
                    trajectories_npz[f"{rk}__spending_rate"] = rate_arr
                    trajectories_npz[f"{rk}__distinct_nodes_used"] = nodes_arr

                    diagnostic_summary[rk] = {
                        "n_points": len(pts),
                        "t_span": [float(t_arr[0]), float(t_arr[-1])],
                        "final_m_y": float(my_arr[-1]),
                        "final_m_x": float(mx_arr[-1]),
                        "final_exposure": float(exp_arr[-1]),
                    }

                actions = None
                if rec is not None and getattr(rec, "action_history", None):
                    actions = [a.to_dict() for a in rec.action_history]
                elif rec_dict and "action_history" in rec_dict:
                    actions = rec_dict["action_history"]
                if actions and rk not in action_history_json:
                    action_history_json[rk] = actions

                frames = None
                if rec is not None and getattr(rec, "network_frames", None):
                    frames = rec.network_frames
                if frames and f"{rk}__t" not in network_frames_npz:
                    t_f = np.array([f.t for f in frames], dtype=np.float64)
                    x_f = np.array([f.x for f in frames], dtype=np.float64)
                    y_f = np.array([f.y for f in frames], dtype=np.float64)
                    W_f = np.array([f.W for f in frames], dtype=np.float64)
                    mask_f = np.array([f.controlled_mask for f in frames], dtype=bool)
                    inp_f = np.array([f.influence_inputs for f in frames], dtype=np.float64)

                    network_frames_npz[f"{rk}__t"] = t_f
                    network_frames_npz[f"{rk}__x"] = x_f
                    network_frames_npz[f"{rk}__y"] = y_f
                    network_frames_npz[f"{rk}__W"] = W_f
                    network_frames_npz[f"{rk}__controlled_mask"] = mask_f
                    network_frames_npz[f"{rk}__influence_inputs"] = inp_f

    if trajectories_npz:
        np.savez_compressed(path / "trajectories.npz", **trajectories_npz)
    with open(path / "diagnostic_trajectories.json", "w") as f:
        json.dump(diagnostic_summary, f, indent=2, sort_keys=True, default=_json_default)
    if action_history_json:

        with open(path / "action_history.json", "w") as f:
            json.dump(action_history_json, f, indent=2, sort_keys=True, default=_json_default)
    if network_frames_npz:
        np.savez_compressed(path / "network_frames.npz", **network_frames_npz)

    # 7. retention_summary.json
    retention_data = {
        "uncontrolled_persistence": result.manifest.get("uncontrolled_persistence", {}),
        "retention_summaries": getattr(result, "retention_summaries", {}),
    }
    with open(path / "retention_summary.json", "w") as f:
        json.dump(retention_data, f, indent=2, sort_keys=True, default=_json_default)

    # 7b. paired_comparisons.json
    paired_comp = getattr(result, "paired_comparisons", {})
    with open(path / "paired_comparisons.json", "w") as f:
        json.dump(paired_comp, f, indent=2, sort_keys=True, default=_json_default)

    # 8. manifest with checksum and file inventory
    manifest = dict(result.manifest)
    manifest["reliability_summaries"] = result.reliability_summaries
    manifest["paired_comparisons"] = paired_comp
    manifest["adaptive_vs_frozen_pairs"] = [
        {
            "origin_id": p["origin_id"],
            "future_id": p["future_id"],
            "sim_seed": p["sim_seed"],
            "ctl_seed": p["ctl_seed"],
            "adaptive_success": None if p["adaptive"] is None else p["adaptive"].success,
            "frozen_success": p["frozen"].success,
        }
        for p in result.adaptive_vs_frozen_pairs
    ]
    inventory = []
    checksums = {}
    for p_file in sorted(path.rglob("*")):
        if p_file.is_file() and p_file.name != "manifest.json":
            rel = str(p_file.relative_to(path))
            inventory.append(rel)
            with open(p_file, "rb") as f_in:
                checksums[rel] = hashlib.sha256(f_in.read()).hexdigest()

    manifest["file_inventory"] = inventory
    manifest["checksums"] = checksums
    manifest["bundle_path"] = str(path)
    manifest["bundle_files"] = inventory
    manifest["created"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["python"] = platform.python_version()
    manifest["numpy"] = np.__version__
    # Executable-source provenance for NEW bundles: code revision, source
    # digest, and runtime versions. model_hash in manifest['problem'] remains
    # the configuration identity; these fields add executable identity. Old
    # bundles without them keep loading unchanged.
    manifest.update(_executable_provenance())
    payload = json.dumps(manifest, sort_keys=True, indent=2, default=_json_default)
    manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)

    return str(path)


def update_bundle_manifest(
    bundle_dir: Union[str, Path],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update manifest.json in bundle_dir with fresh file inventory, checksums, and metadata."""
    path = Path(bundle_dir)
    manifest_path = path / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    if extra_fields:
        manifest.update(extra_fields)

    inventory = []
    checksums = {}
    for p_file in sorted(path.rglob("*")):
        if p_file.is_file() and p_file.name != "manifest.json":
            rel = str(p_file.relative_to(path))
            inventory.append(rel)
            with open(p_file, "rb") as f_in:
                checksums[rel] = hashlib.sha256(f_in.read()).hexdigest()

    manifest["file_inventory"] = inventory
    manifest["checksums"] = checksums
    manifest["bundle_path"] = str(path)
    manifest["bundle_files"] = inventory
    manifest["updated"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    # Refresh executable-source provenance when a bundle manifest is rewritten
    # (visualization registration etc.). Absent in legacy manifests, which
    # remain fully readable.
    manifest.update(_executable_provenance())

    payload = json.dumps(manifest, sort_keys=True, indent=2, default=_json_default)
    manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)

    return manifest


class LoadedControlBundle:
    """Reconstructed campaign bundle loaded from disk without a live simulation object."""

    def __init__(self, bundle_dir: Union[str, Path]) -> None:
        self.bundle_dir = Path(bundle_dir)
        if not (self.bundle_dir / "manifest.json").exists():
            raise FileNotFoundError(f"Manifest not found in bundle directory: {self.bundle_dir}")

        with open(self.bundle_dir / "manifest.json", "r") as f:
            self.manifest: Dict[str, Any] = json.load(f)

        self.rollout_results: List[Dict[str, Any]] = []
        if (self.bundle_dir / "rollout_results.csv").exists():
            with open(self.bundle_dir / "rollout_results.csv", "r") as f:
                self.rollout_results = list(csv.DictReader(f))
        self.rollouts = self.rollout_results

        self.accepted_snapshots: Dict[str, Any] = {}
        if (self.bundle_dir / "accepted_snapshots.json").exists():
            with open(self.bundle_dir / "accepted_snapshots.json", "r") as f:
                self.accepted_snapshots = json.load(f)


        self.adaptive_frozen_pairs: List[Dict[str, Any]] = []
        if (self.bundle_dir / "adaptive_frozen_pairs.csv").exists():
            with open(self.bundle_dir / "adaptive_frozen_pairs.csv", "r") as f:
                self.adaptive_frozen_pairs = list(csv.DictReader(f))

        self.policy_schedules: Dict[str, Any] = {}
        if (self.bundle_dir / "policy_schedules.json").exists():
            with open(self.bundle_dir / "policy_schedules.json", "r") as f:
                self.policy_schedules = json.load(f)

        self.search_summary: Dict[str, Any] = {}
        if (self.bundle_dir / "search_summary.json").exists():
            with open(self.bundle_dir / "search_summary.json", "r") as f:
                self.search_summary = json.load(f)

        self.preparation_attempts: List[Dict[str, Any]] = []
        if (self.bundle_dir / "preparation_attempts.json").exists():
            with open(self.bundle_dir / "preparation_attempts.json", "r") as f:
                self.preparation_attempts = json.load(f)

        # Optional per-arm work-unit records (controller effective_settings
        # exports such as controller_inferred_target_sign). Legacy bundles
        # without a work_units directory simply get an empty mapping.
        self.work_units: Dict[str, Any] = {}
        work_units_dir = self.bundle_dir / "work_units"
        if work_units_dir.is_dir():
            for unit_path in sorted(work_units_dir.glob("*.json")):
                try:
                    with open(unit_path, "r") as f:
                        unit_doc = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                key = unit_doc.get("key", unit_path.stem) if isinstance(unit_doc, dict) else unit_path.stem
                self.work_units[key] = unit_doc.get("payload", unit_doc) if isinstance(unit_doc, dict) else unit_doc

        self.trajectories: Dict[str, Dict[str, np.ndarray]] = {}
        if (self.bundle_dir / "trajectories.npz").exists():
            data = np.load(self.bundle_dir / "trajectories.npz", allow_pickle=False)
            rollout_keys = set()
            for k in data.files:
                if "__" in k:
                    rollout_keys.add(k.rsplit("__", 1)[0])
            for rk in rollout_keys:
                self.trajectories[rk] = {}
                for field_name in (
                    "t", "m_x", "m_y", "distance_to_target", "target_predicate",
                    "hold_progress", "actual_release", "exposure", "spending_rate",
                    "distinct_nodes_used",
                ):
                    full_k = f"{rk}__{field_name}"
                    if full_k in data:
                        self.trajectories[rk][field_name] = data[full_k]

        self.diagnostic_summary: Dict[str, Any] = {}
        if (self.bundle_dir / "diagnostic_trajectories.json").exists():
            with open(self.bundle_dir / "diagnostic_trajectories.json", "r") as f:
                self.diagnostic_summary = json.load(f)

        self.action_history: Dict[str, List[Dict[str, Any]]] = {}
        if (self.bundle_dir / "action_history.json").exists():
            with open(self.bundle_dir / "action_history.json", "r") as f:
                self.action_history = json.load(f)

        self.network_frames: Dict[str, List[Dict[str, Any]]] = {}
        if (self.bundle_dir / "network_frames.npz").exists():
            nf_data = np.load(self.bundle_dir / "network_frames.npz", allow_pickle=False)
            rollout_keys = set()
            for k in nf_data.files:
                if "__" in k:
                    rollout_keys.add(k.rsplit("__", 1)[0])
            for rk in rollout_keys:
                t_arr = nf_data.get(f"{rk}__t", [])
                W_arr = nf_data.get(f"{rk}__W", [])
                y_arr = nf_data.get(f"{rk}__y", [])
                x_arr = nf_data.get(f"{rk}__x", [])
                mask_arr = nf_data.get(f"{rk}__controlled_mask", [])
                inp_arr = nf_data.get(f"{rk}__influence_inputs", [])
                frames = []
                for idx in range(len(t_arr)):
                    frames.append({
                        "t": float(t_arr[idx]),
                        "W": W_arr[idx] if idx < len(W_arr) else None,
                        "y": y_arr[idx] if idx < len(y_arr) else None,
                        "x": x_arr[idx] if idx < len(x_arr) else None,
                        "controlled_mask": mask_arr[idx] if idx < len(mask_arr) else None,
                        "influence_inputs": inp_arr[idx] if idx < len(inp_arr) else None,
                    })
                self.network_frames[rk] = frames

        self.retention_summary: Dict[str, Any] = {}
        if (self.bundle_dir / "retention_summary.json").exists():
            with open(self.bundle_dir / "retention_summary.json", "r") as f:
                self.retention_summary = json.load(f)

        self.paired_comparisons: Dict[str, Any] = self.manifest.get("paired_comparisons", {})
        if not self.paired_comparisons and (self.bundle_dir / "paired_comparisons.json").exists():
            with open(self.bundle_dir / "paired_comparisons.json", "r") as f:
                self.paired_comparisons = json.load(f)

        self.visualizations: Dict[str, Any] = self.manifest.get("visualizations", {})

    def reconstruct_plot_data(self) -> Dict[str, Any]:
        """Reconstruct plot and visualization input data without a live campaign object."""
        timeseries_by_arm: Dict[str, List[Dict[str, Any]]] = {}
        for rk, traj in self.trajectories.items():
            pts = []
            n_pts = len(traj.get("t", []))
            for i in range(n_pts):
                pts.append({
                    "t": float(traj["t"][i]),
                    "m_y": [float(traj["m_y"][i])],
                    "m_x": [float(traj["m_x"][i])],
                    "distance": float(traj["distance_to_target"][i]) if "distance_to_target" in traj else 0.0,
                    "exposure": float(traj["exposure"][i]) if "exposure" in traj else 0.0,
                    "spending_rate": float(traj["spending_rate"][i]) if "spending_rate" in traj else 0.0,
                    "hold_progress": float(traj["hold_progress"][i]) if "hold_progress" in traj else 0.0,
                    "predicate": bool(traj["target_predicate"][i]) if "target_predicate" in traj else False,
                    "is_released": bool(traj["actual_release"][i]) if "actual_release" in traj else False,
                })
            timeseries_by_arm[rk] = pts

        budget_consumption: Dict[str, Any] = {
            "exposure_cap": self.manifest.get("geometry", {}).get("exposure_cap", 200.0),
            "amplitude_cap": self.manifest.get("geometry", {}).get("amplitude_cap", 4.0),
            "k_nodes_cap": self.manifest.get("geometry", {}).get("k", 10),
            "trajectories": {},
        }
        for rk, pts in timeseries_by_arm.items():
            budget_consumption["trajectories"][rk] = {
                "times": [p["t"] for p in pts],
                "cumulative_exposure": [p["exposure"] for p in pts],
                "spending_rate": [p["spending_rate"] for p in pts],
            }

        return {
            "trajectories": timeseries_by_arm,
            "budget_consumption": budget_consumption,
            "policy_comparisons": self.manifest.get("reliability_summaries", {}),
            "network_frames": self.network_frames,
            "adaptive_frozen_pairs": self.adaptive_frozen_pairs,
            "paired_comparisons": self.paired_comparisons,
            "action_history": self.action_history,
            "visualizations": self.visualizations,
            "retention": self.retention_summary or self.manifest.get("uncontrolled_persistence", {}),
            "work_units": self.work_units,
            "manifest": self.manifest,
        }


def load_control_bundle(bundle_dir: Union[str, Path]) -> LoadedControlBundle:
    """Load an exported campaign bundle from disk without a live simulation object."""
    return LoadedControlBundle(bundle_dir)
