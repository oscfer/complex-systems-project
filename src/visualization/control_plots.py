"""Scientific visualization and rendering suite for control-first transformation benchmarks.

Implementation:
All functions consume saved bundle data (LoadedControlBundle or bundle directory)
WITHOUT running any physical simulations.

Provides:
1. plot_scalar_trajectories: Private/public trajectories with initial orientation,
   target thresholds, forcing interval, actual release, holding window, and
   certification marker where achieved (or failure penalty marker).
2. plot_intervention_and_budget: Realized intervention schedules, spending rates,
   and cumulative budget consumption against exposure, amplitude, and node caps.
3. plot_policy_reliability_and_time: Success probability with uncertainty,
   restricted completion time distinguished from successful completion, and
   conditional relapse with explicit N/A handling.
4. plot_paired_adaptive_frozen_comparison: Paired adaptive/frozen diagnostics on
   matched origins/seeds and uncertainty for policy differences, with honest
   all-failure notices.
5. plot_network_views_before_during_after: Synchronized before/during/after network
   panels with consistent node layout derived from origin W_0, consistent
   color/size scales, influence scaling, and highlighted controlled agents.
6. render_network_animation: Playable matched controlled/uncontrolled animation
   (HTML5/JS video and GIF) with fixed layout, shared time grid, and consistent scales.
7. render_bundle_visualizations: Standalone pipeline rendering all plots and
   animations from disk, saving them under the unified bundle directory, and
   registering them in manifest.json with checksum verification.
"""

from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Set safe MPL config directory before matplotlib import
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_modular3")

import numpy as np

try:
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    animation = None

from src.analytics.influence import calculate_attention_matrix, calculate_influence_metrics
from src.experiment.baselines.reference import wilson_interval
from src.visualization.network_layout import _seeded_spring_layout
from src.visualization.animation_player import inline_player_fixups
from src.io.results import LoadedControlBundle, load_control_bundle, update_bundle_manifest


def _resolve_bundle(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
) -> Tuple[LoadedControlBundle, Optional[Path]]:
    """Resolve a bundle argument into a LoadedControlBundle and its directory path."""
    if isinstance(bundle_or_dir, (str, Path)):
        p = Path(bundle_or_dir)
        loaded = load_control_bundle(p)
        return loaded, p
    if isinstance(bundle_or_dir, LoadedControlBundle):
        return bundle_or_dir, getattr(bundle_or_dir, "bundle_dir", None)
    if hasattr(bundle_or_dir, "bundle_path") and bundle_or_dir.bundle_path:
        p = Path(bundle_or_dir.bundle_path)
        return load_control_bundle(p), p
    raise TypeError(f"Expected LoadedControlBundle, Path, or str; got {type(bundle_or_dir)!r}")


def _has_recorded_actuation(bundle: LoadedControlBundle, key: str) -> bool:
    """Whether the saved rollout contains an actual, nonzero intervention."""
    for frame in bundle.network_frames.get(key, []):
        if np.any(frame.get("controlled_mask", [])):
            return True
    for action in bundle.action_history.get(key, []):
        if float(action.get("spending_rate", 0.0) or 0.0) > 0.0:
            return True
    return False


def _find_representative_rollout_key(
    bundle: LoadedControlBundle,
    prefer_group: str = "evaluation",
) -> Optional[str]:
    """Find a representative rollout key from bundle trajectories."""
    if not bundle.trajectories:
        return None

    # Prefer a measured, nonzero evaluation intervention when one exists.
    # An all-zero selected policy remains a valid negative result, but showing
    # its identical-to-baseline path as the only representative hides what the
    # tested active policy actually did.
    eval_keys = sorted((k for k in bundle.trajectories if k.startswith(f"{prefer_group}__")),
                       key=lambda k: (not k.endswith("__adaptive"), k))
    active_eval = [k for k in eval_keys if _has_recorded_actuation(bundle, k)]
    representative = active_eval or eval_keys
    rep_fut0 = [k for k in representative if "__fut0" in k]
    if rep_fut0:
        return rep_fut0[0]
    if representative:
        return representative[0]

    # 4. Fallback to baseline representative
    base_keys = [k for k in bundle.trajectories if k.startswith("baseline__")]
    base_rep = [k for k in base_keys if "__origin_0__" in k and "fut0" in k]
    if base_rep:
        return base_rep[0]
    if base_keys:
        return base_keys[0]

    # 5. First available key
    return next(iter(bundle.trajectories.keys()))


# -----------------------------------------------------------------------------
# 1. Trajectory Plots
# -----------------------------------------------------------------------------

def plot_scalar_trajectories(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    rollout_key: Optional[str] = None,
    ax: Optional[Any] = None,
    save_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "",
) -> Dict[str, Any]:
    """Plot private and public stance trajectories with target thresholds and milestones.

    R4 Requirements:
    - Private (m_x) and public (m_y) trajectories over time.
    - Initial orientation baseline / indicator.
    - Target threshold lines for private and public means.
    - Active forcing / intervention interval shading.
    - Actual release vertical line / indicator.
    - Holding window shading where achieved.
    - Certification marker where achieved (or failure penalty marker).
    - Truthful handling of all-failure runs without crashing.
    """
    bundle, bundle_path = _resolve_bundle(bundle_or_dir)

    if not bundle.trajectories:
        raise ValueError("No scalar trajectory data available in bundle.")

    rk = rollout_key or _find_representative_rollout_key(bundle)
    if rk is None or rk not in bundle.trajectories:
        raise ValueError(f"Rollout key {rollout_key!r} not found in bundle trajectories.")

    traj = bundle.trajectories[rk]
    t = np.asarray(traj.get("t", []))
    if len(t) == 0:
        raise ValueError(f"Trajectory for rollout {rk} has 0 time points.")

    m_x = np.asarray(traj.get("m_x", []))
    m_y = np.asarray(traj.get("m_y", []))
    predicate = np.asarray(traj.get("target_predicate", []))
    hold_progress = np.asarray(traj.get("hold_progress", []))
    actual_release = np.asarray(traj.get("actual_release", []))
    exposure = np.asarray(traj.get("exposure", []))
    spending_rate = np.asarray(traj.get("spending_rate", []))
    dist_to_target = np.asarray(traj.get("distance_to_target", []))

    # Target specification and geometry from manifest
    spec = bundle.manifest.get("spec", {})
    target_params = spec.get("target_params", {})
    threshold = float(target_params.get("threshold", 0.5))
    holding_duration = float(spec.get("holding_duration", 4.0))
    horizon = float(spec.get("horizon", float(t[-1]) if len(t) > 0 else 24.0))
    release_deadline = float(spec.get("release_deadline", 8.0))

    # Initial orientation and target sign
    m_y0 = float(m_y[0]) if len(m_y) > 0 else 1.0
    m_x0 = float(m_x[0]) if len(m_x) > 0 else 1.0
    s_0 = 1.0 if m_y0 >= 0.0 else -1.0
    s_star = -s_0  # Reversal target sign
    target_threshold_y = s_star * threshold

    # Outcome evaluation from rollout_results.csv or trajectory
    success = False
    completion_time = None
    restricted_time = horizon
    for r in bundle.rollout_results:
        # Match arm / rollout
        r_pol = r.get("policy_id", r.get("policy", ""))
        r_orig = r.get("origin_id", "")
        r_fut = r.get("future_id", "")
        r_dyn = r.get("dynamics", "")
        r_group = r.get("group", "")
        expected_key = f"{r_group}__{r_pol}__{r_orig}__fut{r_fut}"
        if (r_pol and r_orig and rk in {expected_key, f"{expected_key}__{r_dyn}"}):
            success = str(r.get("success", "")).lower() in ("true", "1")
            ct = r.get("completion_time")
            if ct not in (None, "", "None"):
                completion_time = float(ct)
            rt = r.get("restricted_time")
            if rt not in (None, "", "None"):
                restricted_time = float(rt)
            break

    # If not found in CSV, check predicate and hold progress
    if completion_time is None and len(hold_progress) > 0:
        achieved_idx = np.where(hold_progress >= holding_duration - 1e-6)[0]
        if len(achieved_idx) > 0:
            success = True
            completion_time = float(t[achieved_idx[0]])

    # Forcing interval: where spending_rate > 0 or action history
    forcing_intervals: List[Tuple[float, float]] = []
    actions = bundle.action_history.get(rk, [])
    if actions:
        for act in actions:
            start = float(act.get("decision_time", 0.0))
            end_value = act.get("effective_end")
            if act.get("field_entries") and end_value is not None:
                end = float(end_value)
                if end > start and float(act.get("spending_rate", 0.0)) > 0.0:
                    forcing_intervals.append((start, end))
    elif len(spending_rate) > 0:
        active_idx = np.where(spending_rate > 1e-6)[0]
        if len(active_idx) > 0:
            forcing_intervals.append((float(t[active_idx[0]]), float(t[active_idx[-1]])))

    # Combine overlapping action pulses so repeated feedback decisions do not
    # darken the same time span multiple times in the shaded plot.
    if forcing_intervals:
        merged: List[Tuple[float, float]] = []
        for start, end in sorted(forcing_intervals):
            if merged and start <= merged[-1][1] + 1e-12:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        forcing_intervals = merged

    # Actual release time
    t_release: Optional[float] = None
    if len(actual_release) > 0 and np.any(actual_release):
        rel_indices = np.where(actual_release)[0]
        t_release = float(t[rel_indices[0]])
    # A field shutting off does not by itself certify formal release.

    plot_data = {
        "rollout_key": rk,
        "t": t.tolist(),
        "m_x": m_x.tolist(),
        "m_y": m_y.tolist(),
        # For the declared binary D=1 model, mean = 2 * plus share - 1.
        "private_plus_share": ((m_x + 1.0) / 2.0).tolist(),
        "public_plus_share": ((m_y + 1.0) / 2.0).tolist(),
        "s_0": s_0,
        "s_star": s_star,
        "threshold": threshold,
        "target_threshold_y": target_threshold_y,
        "holding_duration": holding_duration,
        "horizon": horizon,
        "release_deadline": release_deadline,
        "t_release": t_release,
        "forcing_intervals": forcing_intervals,
        "success": success,
        "completion_time": completion_time,
        "restricted_time": restricted_time,
    }

    if not HAS_MATPLOTLIB:
        return plot_data

    fig = None
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})
        ax_top, ax_bot = axes[0], axes[1]
    else:
        ax_top = ax
        ax_bot = None

    # Top panel: Stance trajectories
    ax_top.plot(t, m_y, label=r"Public stance mean $m_y(t)$", color="#1f77b4", linewidth=2.2, zorder=4)
    ax_top.plot(t, m_x, label=r"Private stance mean $m_x(t)$", color="#2ca02c", linewidth=1.8, linestyle="--", zorder=4)

    # Initial orientation marker
    ax_top.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=0.8, alpha=0.7)
    ax_top.scatter([t[0]], [m_y0], color="#1f77b4", s=60, zorder=5, label=f"Origin stance ($s_0={s_0:+.0f}$)")

    # Target threshold lines
    ax_top.axhline(
        target_threshold_y, color="#d62728", linestyle="-.", linewidth=1.8,
        label=rf"Reversal target threshold ($s^* \cdot \theta = {target_threshold_y:+.2f}$)",
        zorder=3,
    )

    # Forcing interval shading
    for idx_f, (f_start, f_end) in enumerate(forcing_intervals):
        ax_top.axvspan(
            f_start, f_end, color="#ffeb3b", alpha=0.25, zorder=1,
            label="Active intervention" if idx_f == 0 else None,
        )

    # Actual release line
    if t_release is not None:
        ax_top.axvline(
            t_release, color="#ff9800", linestyle="--", linewidth=1.6, zorder=3,
            label=rf"Actual release ($t={t_release:.1f}$)",
        )

    # Holding window and certification / failure marker
    if success and completion_time is not None:
        t_hold_start = max(0.0, completion_time - holding_duration)
        ax_top.axvspan(
            t_hold_start, completion_time, color="#81c784", alpha=0.35, zorder=2,
            label=rf"Holding window ($\Delta={holding_duration:.1f}$)",
        )
        ax_top.scatter(
            [completion_time], [target_threshold_y], marker="*", s=260, color="#ffd700",
            edgecolors="#1a1a1a", linewidths=1.5, zorder=6,
            label=rf"Target certified ($T_G={completion_time:.1f}$)",
        )
    else:
        # Honest all-failure or failed rollout indicator at horizon
        ax_top.scatter(
            [horizon], [target_threshold_y], marker="X", s=180, color="#d32f2f",
            edgecolors="#1a1a1a", linewidths=1.2, zorder=6,
            label=rf"Target not achieved (penalty $T={restricted_time:.1f}$)",
        )

    ax_top.set_ylabel("Stance mean", fontsize=11, fontweight="bold")
    status_str = "CERTIFIED SUCCESS" if success else "UNSUCCESSFUL (FAILURE PENALTY H)"
    title_text = f"Macro Trajectories: {rk}\nStatus: {status_str} | Target: Reversal (Δ={holding_duration}, H={horizon})"
    if title_suffix:
        title_text += f" | {title_suffix}"
    ax_top.set_title(title_text, fontsize=12, fontweight="bold", pad=8)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.set_ylim(-1.15, 1.15)
    if bundle.manifest.get("profile", {}).get("D") == 1:
        share_axis = ax_top.secondary_yaxis(
            "right", functions=(lambda mean: (mean + 1.0) / 2.0,
                                lambda share: 2.0 * share - 1.0)
        )
        share_axis.set_ylabel("+1 opinion share (binary D=1)", fontsize=9)
        share_axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_top.legend(loc="best", fontsize=8.5, framealpha=0.9)

    # Bottom panel: Exposure & distance to target
    if ax_bot is not None:
        if len(exposure) > 0:
            color_exp = "#9c27b0"
            ax_bot.plot(t, exposure, color=color_exp, linewidth=1.8, label=r"Cumulative exposure $E(t)$")
            ax_bot.set_ylabel("Exposure", color=color_exp, fontsize=10, fontweight="bold")
            ax_bot.tick_params(axis="y", labelcolor=color_exp)

        if len(dist_to_target) > 0:
            ax_bot_twin = ax_bot.twinx()
            color_dist = "#e65100"
            ax_bot_twin.plot(t, dist_to_target, color=color_dist, linestyle=":", linewidth=1.5, label="Distance to target")
            ax_bot_twin.set_ylabel("Distance", color=color_dist, fontsize=10, fontweight="bold")
            ax_bot_twin.tick_params(axis="y", labelcolor=color_dist)

        ax_bot.set_xlabel("Time $t$", fontsize=11, fontweight="bold")
        ax_bot.grid(True, linestyle="--", alpha=0.5)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_p, dpi=120, bbox_inches="tight")
        plot_data["fig"] = fig
        plot_data["axes"] = (ax_top, ax_bot)

    return plot_data


# -----------------------------------------------------------------------------
# 2. Intervention & Budget Consumption
# -----------------------------------------------------------------------------

def plot_intervention_and_budget(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Plot actual intervention schedules and cumulative budget consumption against caps.

    R4 Requirements:
    - Actual intervention schedules (realized timing and spending rate).
    - Cumulative budget consumption trajectories.
    - Exposure cap lines (B), amplitude cap (|u| <= U_max), and node cap (k).
    - Distinguish requested parameters from realized schedules.
    """
    bundle, bundle_path = _resolve_bundle(bundle_or_dir)

    manifest = bundle.manifest
    geom = manifest.get("geometry", {})
    prob = manifest.get("problem", {})
    exposure_cap = float(prob.get("exposure_cap", geom.get("exposure_cap", 200.0)))
    amplitude_cap = float(prob.get("amplitude_cap", geom.get("amplitude_cap", 4.0)))
    k_cap = int(prob.get("k", geom.get("k", 10)))

    # Gather exposure trajectories across all recorded rollouts
    rollout_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for rk, traj in bundle.trajectories.items():
        t = np.asarray(traj.get("t", []))
        exp = np.asarray(traj.get("exposure", []))
        rate = np.asarray(traj.get("spending_rate", []))
        nodes = np.asarray(traj.get("distinct_nodes_used", []))
        if len(t) > 0 and len(exp) > 0:
            rollout_curves[rk] = {
                "t": t,
                "exposure": exp,
                "rate": rate,
                "distinct_nodes": nodes,
            }

    # Action history analysis: clamping and realized schedules
    clamped_records: List[Dict[str, Any]] = []
    action_schedules: Dict[str, List[Dict[str, Any]]] = {}
    for rk, acts in bundle.action_history.items():
        action_schedules[rk] = acts
        for a in acts:
            if a.get("budget_clamped") or "budget" in str(a.get("reason", "")).lower():
                clamped_records.append({"rollout": rk, **a})

    plot_data = {
        "exposure_cap": exposure_cap,
        "amplitude_cap": amplitude_cap,
        "k_cap": k_cap,
        "num_rollouts_recorded": len(rollout_curves),
        "clamped_events_count": len(clamped_records),
    }

    if not HAS_MATPLOTLIB:
        return plot_data

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Cumulative Budget Consumption E(t)
    ax_exp = axes[0]
    eval_colors = {"evaluation": "#1f77b4", "baseline": "#7f7f7f"}
    max_t = 1.0

    for rk, data in rollout_curves.items():
        t = data["t"]
        exp = data["exposure"]
        if len(t) > 0:
            max_t = max(max_t, float(t[-1]))
        is_eval = rk.startswith("evaluation__")
        col = "#1f77b4" if is_eval else "#a0aec0"
        alpha = 0.85 if is_eval else 0.45
        lw = 1.8 if is_eval else 1.2
        lbl = "Evaluation arms" if (is_eval and "Evaluation arms" not in ax_exp.get_legend_handles_labels()[1]) else None
        if not is_eval and "Baselines" not in ax_exp.get_legend_handles_labels()[1]:
            lbl = "Baselines"
        ax_exp.plot(t, exp, color=col, alpha=alpha, linewidth=lw, label=lbl)

    # Exposure cap line
    ax_exp.axhline(
        exposure_cap, color="#d32f2f", linestyle="--", linewidth=2.0, zorder=5,
        label=rf"Budget cap ($B={exposure_cap:.1f}$)",
    )
    ax_exp.set_title("Cumulative Exposure vs Cap", fontsize=11, fontweight="bold")
    ax_exp.set_xlabel("Time $t$", fontsize=10)
    ax_exp.set_ylabel("Integrated Exposure $E(t)$", fontsize=10)
    ax_exp.set_ylim(-2.0, max(exposure_cap * 1.15, 10.0))
    ax_exp.grid(True, linestyle="--", alpha=0.5)
    ax_exp.legend(loc="center right", fontsize=8.5)

    # Panel 2: Realized Intervention Spending Rates & Clamping
    ax_rate = axes[1]
    has_clamped = len(clamped_records) > 0
    for rk, data in rollout_curves.items():
        if rk.startswith("evaluation__"):
            t = data["t"]
            rate = data["rate"]
            if len(rate) > 0:
                ax_rate.plot(t, rate, color="#2b6cb0", alpha=0.7, linewidth=1.5)

    if has_clamped:
        for c in clamped_records:
            t_clamp = float(c.get("decision_time", c.get("delivery_time", 0.0)))
            ax_rate.scatter([t_clamp], [0.0], marker="^", color="#d32f2f", s=80, zorder=6, label="Budget clamped")
    
    ax_rate.set_title(f"Realized Actuation Rate (Clamped: {len(clamped_records)})", fontsize=11, fontweight="bold")
    ax_rate.set_xlabel("Time $t$", fontsize=10)
    ax_rate.set_ylabel(r"Spending rate $\sum_i |u_i(t)|$", fontsize=10)
    ax_rate.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Distinct Controlled Nodes vs Cap k
    ax_nodes = axes[2]
    for rk, data in rollout_curves.items():
        if rk.startswith("evaluation__"):
            t = data["t"]
            nds = data["distinct_nodes"]
            if len(nds) > 0:
                ax_nodes.step(t, nds, color="#2e7d32", where="post", alpha=0.7, linewidth=1.6)

    ax_nodes.axhline(k_cap, color="#d32f2f", linestyle="--", linewidth=1.8, label=rf"Distinct node cap ($k={k_cap}$)")
    ax_nodes.set_title("Controlled Nodes vs Cap", fontsize=11, fontweight="bold")
    ax_nodes.set_xlabel("Time $t$", fontsize=10)
    ax_nodes.set_ylabel("Distinct nodes count", fontsize=10)
    ax_nodes.set_ylim(0, max(k_cap + 2, 5))
    ax_nodes.grid(True, linestyle="--", alpha=0.5)
    ax_nodes.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=120, bbox_inches="tight")

    plot_data["fig"] = fig
    plot_data["axes"] = axes
    return plot_data


# -----------------------------------------------------------------------------
# 3. Policy Success & Reliability
# -----------------------------------------------------------------------------

def plot_policy_reliability_and_time(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Plot policy success probabilities, restricted vs completion times, and conditional relapse.

    R4 Requirements:
    - Policy success with uncertainty and sample counts.
    - Restricted completion time distinguished from successful completion.
    - Conditional relapse with explicit N/A handling.
    - Truthful representation of all-failure runs.
    """
    bundle, bundle_path = _resolve_bundle(bundle_or_dir)

    summaries = bundle.manifest.get("reliability_summaries", {})
    p0 = float(bundle.manifest.get("spec", {}).get("reliability_target", 0.8))
    horizon = float(bundle.manifest.get("spec", {}).get("horizon", 320.0))

    arms = sorted(summaries.keys())
    if not arms:
        # Fallback to compute basic stats from rollout results if empty
        arm_groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in bundle.rollout_results:
            arm_name = f"{r.get('group', '')}_{r.get('policy', '')}_{r.get('dynamics', '')}".strip("_")
            arm_groups.setdefault(arm_name, []).append(r)
        for arm_name, rollouts in arm_groups.items():
            succ = sum(1 for r in rollouts if str(r.get("success", "")).lower() in ("true", "1"))
            n = len(rollouts)
            summaries[arm_name] = {
                "eligible_trials": n,
                "successes": succ,
                "success_probability": succ / n if n > 0 else 0.0,
                "mean_restricted_time": horizon if succ == 0 else horizon / 2.0,
                "mean_restricted_time_on_successes": None if succ == 0 else horizon / 2.0,
                "relapse_rate_conditional_on_success": None,
                "status": "certification_unavailable_legacy_summary",
                "uncertainty_method": "rollout_wilson_unverified_origin_independence",
            }
        arms = sorted(summaries.keys())

    arm_labels = []
    p_hats = []
    cis_low = []
    cis_high = []
    sample_counts = []
    mean_restricted_times = []
    success_completion_times = []
    conditional_relapses: List[Optional[float]] = []
    statuses = []
    interval_methods = []

    for arm in arms:
        s = summaries[arm]
        n = int(s.get("eligible_trials", 0))
        k = int(s.get("successes", 0))
        p = float(s.get("success_probability", 0.0))
        status = s.get("status", "unknown")

        # Prefer stored origin-aware descriptive intervals. Legacy fallback has
        # no defensible origin-level certification without its raw provenance.
        if s.get("descriptive_interval_low") is not None and s.get("descriptive_interval_high") is not None:
            w_lo, w_hi = float(s["descriptive_interval_low"]), float(s["descriptive_interval_high"])
        elif s.get("wilson_low") is not None and s.get("wilson_high") is not None:
            w_lo, w_hi = float(s["wilson_low"]), float(s["wilson_high"])
        elif n > 0:
            w_lo, w_hi = wilson_interval(k, n)
        else:
            w_lo, w_hi = 0.0, 0.0
        interval_methods.append(s.get("uncertainty_method", "legacy_rollout_wilson"))

        t_res = float(s.get("mean_restricted_time", horizon))
        t_succ = s.get("mean_restricted_time_on_successes")
        rel_cond = s.get("relapse_rate_conditional_on_success")

        arm_clean = arm.replace("evaluation_", "eval:").replace("baseline_", "base:")
        arm_labels.append(arm_clean)
        p_hats.append(p)
        cis_low.append(w_lo)
        cis_high.append(w_hi)
        sample_counts.append(f"{k}/{n} rollouts; {s.get('n_origins', '?')} origins")
        mean_restricted_times.append(t_res)
        success_completion_times.append(float(t_succ) if t_succ is not None else None)
        conditional_relapses.append(float(rel_cond) if rel_cond is not None else None)
        statuses.append(status)

    plot_data = {
        "arms": arms,
        "arm_labels": arm_labels,
        "p_hats": p_hats,
        "sample_counts": sample_counts,
        "interval_methods": interval_methods,
        "mean_restricted_times": mean_restricted_times,
        "success_completion_times": success_completion_times,
        "conditional_relapses": conditional_relapses,
    }

    if not HAS_MATPLOTLIB or not arms:
        return plot_data

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    y_pos = np.arange(len(arms))

    # Panel 1: Success probability with uncertainty & sample counts
    ax_p = axes[0]
    err_low = np.maximum(0.0, np.array(p_hats) - np.array(cis_low))
    err_high = np.maximum(0.0, np.array(cis_high) - np.array(p_hats))

    bar_colors = []
    for st in statuses:
        if st == "reliability_certified":
            bar_colors.append("#2e7d32")
        elif st == "estimated_feasible":
            bar_colors.append("#f57c00")
        else:
            bar_colors.append("#546e7a")

    ax_p.barh(y_pos, p_hats, xerr=[err_low, err_high], color=bar_colors, alpha=0.85, capsize=4, edgecolor="#263238")
    ax_p.axvline(p0, color="#d32f2f", linestyle="--", linewidth=1.8, label=rf"Target $p_0={p0:.2f}$")

    # Add sample counts text
    for idx, cnt in enumerate(sample_counts):
        # Keep counts clear of the horizontal confidence interval and its cap.
        ax_p.text(1.03, idx, cnt, va="center", fontsize=8.5, fontweight="bold")

    ax_p.set_yticks(y_pos)
    ax_p.set_yticklabels(arm_labels, fontsize=9)
    ax_p.set_xlim(-0.05, 1.35)
    ax_p.set_xlabel(r"Success Probability $\hat{p}$", fontsize=10)
    ax_p.set_title("Policy Success & Descriptive 95% Interval", fontsize=11, fontweight="bold")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                fontsize=8.5, frameon=True)
    ax_p.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Panel 2: Completion time vs Failure penalty
    ax_t = axes[1]
    # Hatched bars for restricted time (which includes failure penalties)
    ax_t.barh(y_pos, mean_restricted_times, color="#b0bec5", alpha=0.7, edgecolor="#37474f", hatch="//", label="Restricted Time (Failures = H)")

    # Scatter points for success-conditional completion time (where achieved)
    succ_y, succ_t = [], []
    for idx, t_s in enumerate(success_completion_times):
        if t_s is not None:
            succ_y.append(idx)
            succ_t.append(t_s)

    if succ_y:
        ax_t.scatter(succ_t, succ_y, color="#1565c0", marker="D", s=70, zorder=5, label=r"Successful Completion $T_G$")

    ax_t.axvline(horizon, color="#d32f2f", linestyle=":", linewidth=1.5, label=rf"Failure penalty ($H={horizon:.0f}$)")
    ax_t.set_yticks(y_pos)
    ax_t.set_yticklabels([])
    ax_t.set_xlabel("Model time $t$", fontsize=10)
    time_title = "Completion vs Failure Penalty Time"
    if not succ_y:
        time_title += "\nNo successes; failures assigned H"
    ax_t.set_title(time_title, fontsize=11, fontweight="bold")
    # Reserve space below the panel so the legend does not cover a data bar.
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5,
                ncol=1, frameon=True)
    ax_t.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Panel 3: Conditional Relapse (with explicit N/A handling)
    ax_r = axes[2]
    rel_vals = []
    rel_colors = []
    for idx, r_val in enumerate(conditional_relapses):
        if r_val is None:
            rel_vals.append(0.0)
            rel_colors.append("#eceff1")
            ax_r.text(0.05, idx, "N/A (0 successes)", va="center", color="#78909c", fontsize=8.5, fontstyle="italic")
        else:
            rel_vals.append(r_val)
            rel_colors.append("#e53935")
            ax_r.text(r_val + 0.03, idx, f"{r_val:.2f}", va="center", fontsize=8.5, fontweight="bold")

    ax_r.barh(y_pos, rel_vals, color=rel_colors, edgecolor="#b0bec5", alpha=0.85)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([])
    ax_r.set_xlim(-0.05, 1.15)
    ax_r.set_xlabel("Relapse Rate | Success", fontsize=10)
    ax_r.set_title("Conditional Relapse (N/A Handling)", fontsize=11, fontweight="bold")
    ax_r.grid(True, linestyle="--", alpha=0.5, axis="x")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=120, bbox_inches="tight")

    plot_data["fig"] = fig
    plot_data["axes"] = axes
    return plot_data


# -----------------------------------------------------------------------------
# 4. Paired Adaptive / Frozen Diagnostics
# -----------------------------------------------------------------------------

def plot_paired_adaptive_frozen_comparison(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Plot paired adaptive vs frozen comparisons and uncertainty for policy differences.

    R4 Requirements:
    - Matched adaptive/frozen diagnostics on identical fresh origins and seeds.
    - Uncertainty for policy differences (paired cluster bootstrap).
    - Scientific Integrity: NEVER assert equivalence from equal failures.
    """
    bundle, bundle_path = _resolve_bundle(bundle_or_dir)

    pairs = bundle.adaptive_frozen_pairs
    paired_comp = bundle.manifest.get("paired_comparisons", {})

    diff_success = []
    diff_restricted_time = []
    diff_expenditure = []
    adapt_successes = 0
    frozen_successes = 0

    for p in pairs:
        a_succ = str(p.get("adaptive_success", "")).lower() in ("true", "1")
        f_succ = str(p.get("frozen_success", "")).lower() in ("true", "1")
        if a_succ:
            adapt_successes += 1
        if f_succ:
            frozen_successes += 1

        diff_success.append(float(a_succ) - float(f_succ))

        a_time = p.get("adaptive_restricted_time")
        f_time = p.get("frozen_restricted_time")
        if a_time not in (None, "") and f_time not in (None, ""):
            diff_restricted_time.append(float(a_time) - float(f_time))

        a_exp = p.get("adaptive_expenditure")
        f_exp = p.get("frozen_expenditure")
        if a_exp not in (None, "") and f_exp not in (None, ""):
            diff_expenditure.append(float(a_exp) - float(f_exp))

    both_failed = (adapt_successes == 0 and frozen_successes == 0)

    plot_data = {
        "n_pairs": len(pairs),
        "adapt_successes": adapt_successes,
        "frozen_successes": frozen_successes,
        "both_failed": both_failed,
        "mean_diff_success": float(np.mean(diff_success)) if diff_success else 0.0,
        "mean_diff_time": float(np.mean(diff_restricted_time)) if diff_restricted_time else 0.0,
        "mean_diff_expenditure": float(np.mean(diff_expenditure)) if diff_expenditure else 0.0,
    }

    if not HAS_MATPLOTLIB:
        return plot_data

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Panel 1: Paired differences on matched origins
    ax_diff = axes[0]
    metrics = [r"$\Delta$ Success", r"$\Delta$ Time ($T_G$)", r"$\Delta$ Expenditure"]
    diff_means = [
        plot_data["mean_diff_success"],
        plot_data["mean_diff_time"],
        plot_data["mean_diff_expenditure"],
    ]

    # CI bounds from paired_comparisons or standard error
    y_pos = np.arange(len(metrics))
    ci_lows = [0.0, 0.0, 0.0]
    ci_highs = [0.0, 0.0, 0.0]

    # Inspect paired_comparisons dictionary if populated
    for family, stat in paired_comp.items():
        if isinstance(stat, dict):
            s_ci = stat.get("success_diff_ci95")
            if s_ci:
                ci_lows[0], ci_highs[0] = s_ci[0], s_ci[1]
            t_ci = stat.get("restricted_time_diff_ci95")
            if t_ci:
                ci_lows[1], ci_highs[1] = t_ci[0], t_ci[1]
            e_ci = stat.get("expenditure_diff_ci95")
            if e_ci:
                ci_lows[2], ci_highs[2] = e_ci[0], e_ci[1]
            break

    xerr_l = np.maximum(0.0, np.array(diff_means) - np.array(ci_lows))
    xerr_h = np.maximum(0.0, np.array(ci_highs) - np.array(diff_means))

    ax_diff.barh(y_pos, diff_means, xerr=[xerr_l, xerr_h], color="#3949ab", alpha=0.8, capsize=4, edgecolor="#1a237e")
    ax_diff.axvline(0.0, color="#d32f2f", linestyle="--", linewidth=1.5)
    ax_diff.set_yticks(y_pos)
    ax_diff.set_yticklabels(metrics, fontsize=10, fontweight="bold")
    ax_diff.set_xlabel(r"Paired Difference ($\Delta = \mathrm{Adaptive} - \mathrm{Frozen}$)", fontsize=10)
    ax_diff.set_title(f"Paired Policy Differences ({len(pairs)} matched origins/seeds)", fontsize=11, fontweight="bold")
    ax_diff.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Panel 2: Scientific notice and breakdown
    ax_notice = axes[1]
    ax_notice.axis("off")

    if both_failed:
        box_text = (
            "SCIENTIFIC INTEGRITY NOTICE:\n"
            "-----------------------------------------\n"
            f"- Matched origins evaluated: {len(pairs)}\n"
            f"- Adaptive successes: {adapt_successes}\n"
            f"- Frozen successes:   {frozen_successes}\n\n"
            "CRITICAL CAUTION:\n"
            "Both branches failed to achieve reversal on all\n"
            "evaluated origins. Equal failure rates (both 0.0)\n"
            "do NOT demonstrate dynamical equivalence between\n"
            "adaptive and genuinely frozen memory.\n\n"
            "In this bounded sample, neither branch reached\n"
            "the declared reversal target. More origins or a\n"
            "different permitted policy may give another result."
        )
        box_color = "#fff3e0"
        border_color = "#e65100"
    else:
        box_text = (
            "PAIRED COMPARISON SUMMARY:\n"
            "-----------------------------------------\n"
            f"- Matched pairs: {len(pairs)}\n"
            f"- Adaptive successes: {adapt_successes} ({adapt_successes / max(1, len(pairs)):.1%})\n"
            f"- Frozen successes:   {frozen_successes} ({frozen_successes / max(1, len(pairs)):.1%})\n\n"
            f"- Mean Success Difference: {plot_data['mean_diff_success']:+.3f}\n"
            f"- Mean Time Difference:    {plot_data['mean_diff_time']:+.1f}\n"
            f"- Mean Spending Difference: {plot_data['mean_diff_expenditure']:+.2f}\n"
        )
        box_color = "#e8f5e9"
        border_color = "#2e7d32"

    ax_notice.text(
        0.05, 0.5, box_text, transform=ax_notice.transAxes, va="center", ha="left",
        fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=box_color, edgecolor=border_color, linewidth=1.5),
    )

    fig.tight_layout()
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=120, bbox_inches="tight")

    plot_data["fig"] = fig
    plot_data["axes"] = axes
    return plot_data


# -----------------------------------------------------------------------------
# 5. Network Views & Animation
# -----------------------------------------------------------------------------

def _network_arm_label(key: str) -> str:
    """Compact policy label for a network panel or animation subtitle."""
    arm = key.split("__origin_", 1)[0]
    label = arm.removeprefix("evaluation__").removeprefix("baseline__")
    dynamics = key.rsplit("__", 1)[-1]
    return f"{label} ({dynamics})" if dynamics in {"adaptive", "frozen"} else label


def _network_horizon(bundle: LoadedControlBundle, last_time: float) -> float:
    """Read the declared horizon; old bundles can end their saved frames early."""
    manifest = bundle.manifest or {}
    for field in ("spec", "profile", "problem"):
        value = manifest.get(field, {})
        if isinstance(value, dict) and value.get("horizon") is not None:
            return float(value["horizon"])
    return float(last_time)


def _stance_counts(values: np.ndarray) -> Dict[str, Any]:
    """Exact red/blue counts and percentages computed from a saved stance layer.

    Red means the +1 side of the binary stance, blue the -1 side. The returned
    numbers are computed directly from the saved array passed in, so a figure
    built from them matches the saved arrays exactly.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    red = int(np.sum(v > 0.0))
    blue = int(np.sum(v < 0.0))
    neutral = n - red - blue
    return {
        "n": n,
        "red": red,
        "blue": blue,
        "neutral": neutral,
        "red_pct": 100.0 * red / n if n else 0.0,
        "blue_pct": 100.0 * blue / n if n else 0.0,
        "neutral_pct": 100.0 * neutral / n if n else 0.0,
    }


def _stance_counts_line(
    label: str,
    counts: Dict[str, Any],
    target_sign: Optional[float] = None,
) -> str:
    """One-line exact counts summary for one belief layer (+ optional target support)."""
    line = (f"{label}: {counts['red']} red ({counts['red_pct']:.1f}%) / "
            f"{counts['blue']} blue ({counts['blue_pct']:.1f}%)")
    if counts["neutral"]:
        line += f" / {counts['neutral']} neutral ({counts['neutral_pct']:.1f}%)"
    if target_sign in (-1.0, 1.0):
        if target_sign < 0:
            support, support_pct = counts["blue"], counts["blue_pct"]
        else:
            support, support_pct = counts["red"], counts["red_pct"]
        side = "blue" if target_sign < 0 else "red"
        line += f"; target({target_sign:+.0f}={side}) support: {support} ({support_pct:.1f}%)"
    return line


def _panel_title(
    label: str,
    frame: Dict[str, Any],
    mean_y: float,
    mean_x: float,
    n_active: int,
    target_sign: Optional[float] = None,
) -> str:
    """Panel heading with exact public/private stance counts from saved arrays.

    Target-direction support is deliberately NOT put here (long lines overlap
    adjacent panel titles); it is rendered on a dedicated figure-level line by
    _target_support_line.
    """
    public = _stance_counts(np.asarray(frame["y"], dtype=np.float64))
    private = _stance_counts(np.asarray(frame["x"], dtype=np.float64))
    return "\n".join([
        label,
        (f"$t={float(frame['t']):.1f}$ | $m_y={mean_y:+.2f}$ | $m_x={mean_x:+.2f}$ | "
         f"active field: {n_active} nodes"),
        _stance_counts_line("Public", public, None),
        _stance_counts_line("Private", private, None),
    ])


def _target_support_line(
    panel_counts: Sequence[Dict[str, Any]],
    target_sign: Optional[float],
) -> Optional[str]:
    """Array-exact target-direction support summary across shown frames.

    Returns None when no target direction can be derived from saved data.
    """
    if target_sign not in (-1.0, 1.0) or not panel_counts:
        return None
    pub_parts: List[str] = []
    priv_parts: List[str] = []
    for entry in panel_counts:
        for layer_name, parts in (("public", pub_parts), ("private", priv_parts)):
            layer = entry[layer_name]
            support = layer["blue"] if target_sign < 0 else layer["red"]
            parts.append(f"{support}/{layer['n']}")
    side = "blue (-1)" if target_sign < 0 else "red (+1)"
    return (
        f"Target direction s={target_sign:+.0f} ({side}, opposite of the saved initial "
        "public majority); target-side support at shown frames — "
        f"public: {', '.join(pub_parts)}; private: {', '.join(priv_parts)}"
    )


def _target_direction_sign(
    bundle: LoadedControlBundle,
    origin_frame: Dict[str, Any],
) -> Optional[float]:
    """Derive the recorded target direction from saved data, or return None.

    For the declared 'reversal' task the destination is the opposite of the
    initial public majority, so the sign is computed from the saved origin
    public layer. If the task is not a reversal or the origin mean is exactly
    tied, no direction is inferred and no support numbers are printed.
    """
    manifest = bundle.manifest or {}
    spec = manifest.get("spec", {}) if isinstance(manifest, dict) else {}
    if spec and spec.get("target_type") not in (None, "reversal"):
        return None
    y0 = np.asarray(origin_frame.get("y", []), dtype=np.float64).reshape(-1)
    if y0.size == 0:
        return None
    m0 = float(np.mean(y0))
    if m0 == 0.0:
        return None
    return -1.0 if m0 > 0.0 else 1.0


def _active_node_counts(frames: Sequence[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Initial, peak, and cumulative-distinct active-node counts from saved masks.

    Only frames that record a controlled_mask can contribute; counts are
    reported separately rather than merged into one number.
    """
    initial: Optional[int] = None
    peak = 0
    cumulative: set = set()
    for idx, frame in enumerate(frames):
        mask = np.asarray(frame.get("controlled_mask", []), dtype=bool).reshape(-1)
        if mask.size == 0:
            continue
        count = int(np.sum(mask))
        if idx == 0:
            initial = count
        peak = max(peak, count)
        cumulative.update(int(i) for i in np.flatnonzero(mask))
    return {
        "initial": initial,
        "peak": peak,
        "cumulative_distinct": len(cumulative) if frames else None,
    }


def _selection_reason_lines(
    bundle: LoadedControlBundle,
    rollout_key: str,
    has_recorded_actuation: bool,
) -> List[str]:
    """Explicit selection-reason lines from recorded manifest/work-unit metadata.

    Uses only fields that exist in the saved bundle. Missing metadata is
    reported as missing rather than invented.
    """
    manifest = bundle.manifest or {}
    sel = manifest.get("selection_summary", {})
    if not isinstance(sel, dict) or not sel:
        return ["Selection metadata: not recorded in this bundle (cannot state selection reason)."]

    lines: List[str] = []
    if has_recorded_actuation:
        lines.append("Shown arm = recorded active evaluation arm (nonzero actuation in saved frames).")
    else:
        lines.append("Shown arm = recorded evaluation arm (no nonzero actuation recorded).")

    selected = sel.get("selected_candidate_id")
    params = sel.get("selected_candidate_params") or {}
    amp = params.get("nominal_amplitude")
    dur = params.get("duration")
    is_zero = False
    if amp is not None and float(amp) == 0.0:
        is_zero = True
    if dur is not None and float(dur) == 0.0:
        is_zero = True
    if is_zero:
        lines.append(
            f"Overall selected policy = {selected} (static zero schedule: "
            f"nominal_amplitude={amp}, duration={dur}); its rollout is identical to baseline."
        )
    elif selected is not None:
        lines.append(f"Overall selected policy = {selected} (nonzero declared schedule).")
    else:
        lines.append("Overall selected policy = not recorded in this bundle.")

    cert = sel.get("certification_status") or sel.get("status")
    if cert:
        definition = (sel.get("status_definitions") or {}).get(cert)
        reason = f"reason: {cert}" + (f" — {definition}" if definition else "")
        lines.append(reason)

    unit = (getattr(bundle, "work_units", {}) or {}).get(rollout_key)
    if isinstance(unit, dict):
        effective = (unit.get("summaries") or {}).get("effective_settings") or unit.get("effective_settings") or {}
        exported = {
            k: effective.get(k)
            for k in ("direction_source", "declared_destination_sign",
                      "controller_inferred_target_sign", "inference_source",
                      "inference_report_mean", "inference_report_sample_time")
            if k in effective
        }
        if exported:
            lines.append(f"Controller exports for shown arm: {exported}")
        else:
            lines.append("Controller inference exports not recorded for this arm (pre-WP1 bundle).")
    else:
        lines.append("Controller inference exports not recorded for this arm.")
    return lines


def _top_directed_edges(
    W: np.ndarray,
    k: int = 12,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Top-k directed edges by row-normalized attention A_ij = W_ij / sum_k W_ik.

    The returned values are exactly the entries of the directed row-normalized
    attention matrix from src.analytics.influence (the definition the simulation
    uses for social influence), unlike the undirected W_ij+W_ji line widths.
    Edge filtering: only the k largest A_ij entries are listed for legibility;
    all other positive A_ij entries exist in A but are omitted from the view.
    """
    W = np.asarray(W, dtype=np.float64)
    A, _ = calculate_attention_matrix(W)
    n = A.shape[0]
    entries = [
        (float(A[i, j]), int(i), int(j))
        for i in range(n) for j in range(n)
        if i != j and A[i, j] > 0.0
    ]
    entries.sort(key=lambda t: (-t[0], t[1], t[2]))
    return A, [(i, j, v) for v, i, j in entries[:max(0, int(k))]]


def _network_scales(frames: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """Use shared attention and tie scales across every shown frame."""
    attention_max = 0.0
    tie_max = 0.0
    for frame in frames:
        W = np.asarray(frame["W"], dtype=np.float64)
        attention = calculate_influence_metrics(W)["r"]
        attention_max = max(attention_max, float(np.max(attention, initial=0.0)))
        projection = W + W.T
        tie_max = max(tie_max, float(np.max(projection, initial=0.0)))
    return max(attention_max, 1e-12), max(tie_max, 1e-12)


def _draw_control_network(
    ax: Any,
    frame: Dict[str, Any],
    pos: np.ndarray,
    cmap: Any,
    norm: Any,
    attention_max: float,
    tie_max: float,
) -> Tuple[float, float, int]:
    """Draw an undirected view of memory with public/private stance glyphs."""
    W = np.asarray(frame["W"], dtype=np.float64)
    n = W.shape[0]
    y = np.asarray(frame["y"], dtype=np.float64).reshape(n, -1)[:, 0]
    x = np.asarray(frame["x"], dtype=np.float64).reshape(n, -1)[:, 0]
    mask = np.asarray(frame.get("controlled_mask", np.zeros(n)), dtype=bool)
    attention = calculate_influence_metrics(W)["r"]
    sizes = 75.0 + 300.0 * attention / attention_max

    # The layout and displayed lines use W_ij + W_ji. Drawing each pair once
    # reduces clutter; this view does not encode the direction of either tie.
    projection = W + W.T
    for i, j in np.argwhere(np.triu(projection, 1) > 0):
        weight = projection[i, j] / tie_max
        ax.plot(
            [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
            color="#46546a", linewidth=0.35 + 1.6 * weight,
            alpha=0.05 + 0.38 * weight, zorder=1,
        )

    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=y, cmap=cmap, norm=norm,
               edgecolors="#202735", linewidths=0.8, zorder=3)
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes * 0.28, c=x, cmap=cmap, norm=norm,
               edgecolors="white", linewidths=0.6, zorder=4)
    if np.any(mask):
        ax.scatter(pos[mask, 0], pos[mask, 1], s=sizes[mask] * 1.55,
                   facecolors="none", edgecolors="#f9c74f",
                   linewidths=2.3, zorder=5)

    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return float(np.mean(y)), float(np.mean(x)), int(np.sum(mask))


def _add_network_key(fig: Any, cmap: Any, norm: Any, *, targeted_field: bool = False) -> None:
    """Use one legend and colorbar for every network panel and animation frame."""
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#bbbbbb",
               markeredgecolor="#202735", markersize=10,
               label=r"Outer disk: public stance $y_i$"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#bbbbbb",
               markeredgecolor="white", markersize=6,
               label=r"Center dot: private stance $x_i$"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
               markeredgecolor="#f9c74f", markeredgewidth=2.3, markersize=11,
               label="Yellow ring: receiving intervention"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#bbbbbb",
               markeredgecolor="#202735", markersize=8,
               label="Node area = 75 + 300·$r_j/r_{max}$ (monotonic, not proportional)"),
        Line2D([0], [0], color="#46546a", lw=1.6,
               label=r"Line width ∝ $W_{ij}+W_{ji}$: remembered interaction volume, not directed influence (direction hidden)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#c0392b",
               markeredgecolor="none", markersize=8,
               label="Red/blue here = stance colors; trajectory-plot curve colors do not encode stance"),
    ]
    if targeted_field:
        handles.append(Line2D([0], [0], marker="*", linestyle="",
                              markerfacecolor="#111111", markeredgecolor="white",
                              markersize=11, label="Star: additional targeted field while active"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, 0.015),
               ncol=3, fontsize=8.5, frameon=True)
    cax = fig.add_axes([0.935, 0.27, 0.012, 0.43])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label("Stance value", fontsize=9)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])


def plot_network_views_before_during_after(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    rollout_key: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    layout_seed: int = 42,
) -> Dict[str, Any]:
    """Show start, actuation, shutoff, and last saved network without false labels."""
    bundle, _ = _resolve_bundle(bundle_or_dir)
    if not bundle.network_frames:
        raise ValueError("No network frames recorded in bundle (record_network_frames was False).")

    rk = rollout_key
    if rk is None or rk not in bundle.network_frames:
        eval_keys = sorted((k for k in bundle.network_frames if k.startswith("evaluation__")),
                           key=lambda k: (not k.endswith("__adaptive"), k))
        active_eval = [k for k in eval_keys if _has_recorded_actuation(bundle, k)]
        rk = (active_eval or eval_keys)[0] if eval_keys else next(iter(bundle.network_frames))
    frames = bundle.network_frames[rk]
    if not frames:
        raise ValueError(f"No frames in network_frames[{rk}].")

    active = [i for i, frame in enumerate(frames)
              if np.any(frame.get("controlled_mask", []))]
    first_active = active[0] if active else None
    after_active = next((i for i in range(active[-1] + 1, len(frames))
                         if not np.any(frames[i].get("controlled_mask", []))), None) if active else None
    during_idx = first_active if first_active is not None else min(len(frames) - 1, max(1, len(frames) // 4))
    after_idx = after_active if after_active is not None else min(len(frames) - 1, max(during_idx, 3 * len(frames) // 4))
    last_time = float(frames[-1]["t"])
    horizon = _network_horizon(bundle, last_time)
    complete = abs(last_time - horizon) <= 1e-8
    labels = [
        "Start of rollout",
        "Active intervention" if active else "No active frame recorded",
        "After actuation stops" if after_active is not None else "Later recorded state",
        "At declared horizon" if complete else "Last saved (before horizon)",
    ]
    chosen = [frames[0], frames[during_idx], frames[after_idx], frames[-1]]
    W_origin = np.asarray(frames[0].get("W", []))
    if W_origin.ndim != 2 or W_origin.shape[0] == 0:
        raise ValueError("Invalid W matrix in origin frame.")
    pos = _seeded_spring_layout(W_origin, layout_seed=layout_seed)
    target_sign = _target_direction_sign(bundle, frames[0])
    active_counts = _active_node_counts(frames)
    panel_counts = []
    for frame in chosen:
        panel_counts.append({
            "t": float(frame["t"]),
            "public": _stance_counts(np.asarray(frame["y"], dtype=np.float64)),
            "private": _stance_counts(np.asarray(frame["x"], dtype=np.float64)),
        })
    plot_data = {
        "rollout_key": rk,
        "num_nodes": len(pos),
        "frame_indices": [0, during_idx, after_idx, len(frames) - 1],
        "timepoints": [float(f["t"]) for f in chosen],
        "panel_labels": labels,
        "horizon": horizon,
        "coverage_complete": complete,
        "active_frame_found": bool(active),
        "positions": pos,
        "stance_counts": panel_counts,
        "target_direction": target_sign,
        "active_node_counts": active_counts,
        "selection_reason": _selection_reason_lines(bundle, rk, bool(active)),
    }
    if not HAS_MATPLOTLIB:
        return plot_data

    fig, axes = plt.subplots(1, 5, figsize=(22, 6.2), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.70, bottom=0.22, wspace=0.08)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    attention_max, tie_max = _network_scales(chosen)
    for ax, label, frame in zip(axes, labels, chosen):
        mean_y, mean_x, n_active = _draw_control_network(
            ax, frame, pos, cmap, norm, attention_max, tie_max,
        )
        ax.set_title(
            _panel_title(label, frame, mean_y, mean_x, n_active, target_sign),
            fontsize=8.2, fontweight="bold", pad=8,
        )

    # Directed normalized-attention view (values equal row-normalized A).
    ax_dir = axes[-1]
    A_origin, top_edges = _top_directed_edges(W_origin, k=12)
    edge_labels = [f"{i}→{j}" for i, j, _ in top_edges]
    edge_vals = [v for _, _, v in top_edges]
    ypos = np.arange(len(edge_vals))
    ax_dir.barh(ypos, edge_vals, color="#46546a", alpha=0.85)
    ax_dir.set_yticks(ypos)
    ax_dir.set_yticklabels(edge_labels, fontsize=6.8)
    ax_dir.invert_yaxis()
    ax_dir.set_title(
        "Directed normalized attention (origin frame)\n"
        r"$A_{ij}=W_{ij}/\sum_k W_{ik}$ — top 12 edges shown",
        fontsize=8.2, fontweight="bold", pad=8,
    )
    ax_dir.set_xlabel(
        "Row-normalized attention (row sums to 1);\n"
        "all other positive A_ij edges omitted for legibility",
        fontsize=7.0,
    )
    ax_dir.tick_params(axis="x", labelsize=7)

    fig.suptitle("Network state across the controlled rollout", y=0.985,
                 fontsize=13, fontweight="bold")
    fig.text(0.47, 0.93,
             f"Arm: {_network_arm_label(rk)} | saved through $t={last_time:g}$ of declared $H={horizon:g}$",
             ha="center", va="center", fontsize=10)
    fig.text(0.47, 0.905,
             f"Fixed layout from origin memory (seed {layout_seed}); node positions are a "
             "nonphysical rendering choice — not geographic distance or empirical topology",
             ha="center", va="center", fontsize=7.6)
    fig.text(0.47, 0.882,
             (f"Active nodes — initial: {active_counts['initial']}, peak: {active_counts['peak']}, "
              f"cumulative distinct: {active_counts['cumulative_distinct']} "
              "(from recorded controlled masks; reported separately, not merged)"),
             ha="center", va="center", fontsize=7.6)
    fig.text(0.012, 0.862, "\n".join(plot_data["selection_reason"]),
             ha="left", va="top", fontsize=7.0, wrap=True)
    support_line = _target_support_line(panel_counts, target_sign)
    if support_line:
        fig.text(0.47, 0.845, support_line, ha="center", va="top", fontsize=7.2)
        plot_data["target_support_line"] = support_line
    _add_network_key(fig, cmap, norm)
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=120)
    plot_data["fig"] = fig
    plot_data["axes"] = axes
    plot_data["directed_attention_matrix"] = A_origin
    plot_data["directed_top_edges"] = top_edges
    return plot_data


def _origin_future_tag(key: str) -> Optional[str]:
    if "__origin_" not in key or "__fut" not in key:
        return None
    return key.split("__origin_", 1)[1]


def _select_animation_indices(frames: Sequence[Dict[str, Any]], max_frames: int) -> List[int]:
    """Retain start, active pulse, first off frame, and end when shortening a GIF."""
    n = len(frames)
    if n <= max_frames:
        return list(range(n))
    active = [i for i, frame in enumerate(frames)
              if np.any(frame.get("controlled_mask", []))]
    anchors = [0]
    if active:
        anchors.extend((active[0], active[-1]))
        off = next((i for i in range(active[-1] + 1, n)
                    if not np.any(frames[i].get("controlled_mask", []))), None)
        if off is not None:
            anchors.append(off)
    anchors.append(n - 1)
    chosen = list(dict.fromkeys(anchors))
    if len(chosen) > max_frames:
        chosen = [0, n - 1] + [i for i in chosen if i not in (0, n - 1)][:max(0, max_frames - 2)]
        return sorted(set(chosen))
    # Repeatedly split the largest physical-time gap. This distributes the
    # remaining frames over the whole run instead of filling only early times.
    times = np.array([float(frame["t"]) for frame in frames])
    while len(chosen) < max_frames:
        ordered = sorted(chosen)
        gaps = [(times[right] - times[left], left, right)
                for left, right in zip(ordered[:-1], ordered[1:])
                if right - left > 1]
        if not gaps:
            break
        _, left, right = max(gaps)
        candidates = np.arange(left + 1, right)
        target = (times[left] + times[right]) / 2.0
        chosen.append(int(candidates[np.argmin(np.abs(times[candidates] - target))]))
    return sorted(chosen)


def render_network_animation(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    fps: int = 5,
    interval: int = 200,
    max_frames: int = 40,
    layout_seed: int = 42,
) -> Dict[str, Any]:
    """Render a matched controlled/zero-control GIF and standalone HTML player."""
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2 to show start and end")
    bundle, _ = _resolve_bundle(bundle_or_dir)
    if not bundle.network_frames:
        raise ValueError("No network frames recorded in bundle for animation.")

    eval_keys = sorted((k for k in bundle.network_frames if k.startswith("evaluation__")),
                       key=lambda k: (not k.endswith("__adaptive"), k))
    active_eval = [k for k in eval_keys if _has_recorded_actuation(bundle, k)]
    eval_keys = active_eval + [k for k in eval_keys if k not in active_eval]
    baseline_keys = [k for k in bundle.network_frames if "baseline_zero" in k]
    ctrl_key = eval_keys[0] if eval_keys else next(iter(bundle.network_frames))
    unctrl_key = None
    for candidate in eval_keys:
        matched = next((b for b in baseline_keys
                        if _origin_future_tag(b) == _origin_future_tag(candidate)
                        and _origin_future_tag(b) is not None), None)
        if matched is not None:
            ctrl_key, unctrl_key = candidate, matched
            break
    ctrl_full = bundle.network_frames[ctrl_key]
    if not ctrl_full:
        raise ValueError(f"No frames in network_frames[{ctrl_key}].")
    unctrl_full = bundle.network_frames.get(unctrl_key, []) if unctrl_key else []
    if unctrl_full:
        ctrl_times = [float(f["t"]) for f in ctrl_full]
        unctrl_times = [float(f["t"]) for f in unctrl_full]
        if (len(ctrl_times) != len(unctrl_times)
                or not np.allclose(ctrl_times, unctrl_times, atol=1e-8, rtol=0.0)
                or not np.array_equal(ctrl_full[0]["W"], unctrl_full[0]["W"])):
            unctrl_full = []
            unctrl_key = None

    indices = _select_animation_indices(ctrl_full, max_frames)
    ctrl_frames = [ctrl_full[i] for i in indices]
    unctrl_frames = [unctrl_full[i] for i in indices] if unctrl_full else []
    pos = _seeded_spring_layout(np.asarray(ctrl_full[0]["W"]), layout_seed=layout_seed)
    target_sign = _target_direction_sign(bundle, ctrl_full[0])
    last_time = float(ctrl_full[-1]["t"])
    horizon = _network_horizon(bundle, last_time)
    matched = bool(unctrl_frames)
    anim_data = {
        "ctrl_key": ctrl_key,
        "unctrl_key": unctrl_key,
        "matched_side_by_side": matched,
        "num_frames": len(ctrl_frames),
        "frame_times": [float(f["t"]) for f in ctrl_frames],
        "horizon": horizon,
        "coverage_complete": abs(last_time - horizon) <= 1e-8,
        "fps": fps,
        "interval": interval,
    }
    if not HAS_MATPLOTLIB:
        return anim_data

    all_frames = ctrl_frames + unctrl_frames
    attention_max, tie_max = _network_scales(all_frames)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    if matched:
        fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=90)
    else:
        fig, ax_c = plt.subplots(figsize=(7.5, 6.5), dpi=90)
        ax_u = None
    fig.subplots_adjust(left=0.025, right=0.91, top=0.79, bottom=0.23, wspace=0.06)
    fig.suptitle("Controlled network and matched zero-control baseline" if matched
                 else "Controlled network", y=0.98, fontsize=13, fontweight="bold")
    fig.text(0.47, 0.91,
             f"Arm: {_network_arm_label(ctrl_key)} | fixed initial-memory layout (seed {layout_seed}) | saved through $t={last_time:g}$ of $H={horizon:g}$",
             ha="center", va="center", fontsize=9.0)
    fig.text(0.47, 0.888,
             "Node positions are a nonphysical rendering choice — not geographic distance or empirical topology",
             ha="center", va="center", fontsize=7.2)
    fig.text(0.47, 0.868,
             "\n".join(_selection_reason_lines(
                 bundle, ctrl_key, _has_recorded_actuation(bundle, ctrl_key))),
             ha="center", va="top", fontsize=6.6)
    anim_panel_counts = [
        {"public": _stance_counts(np.asarray(f["y"], dtype=np.float64)),
         "private": _stance_counts(np.asarray(f["x"], dtype=np.float64))}
        for f in ctrl_frames
    ]
    anim_support = _target_support_line(anim_panel_counts, target_sign)
    if anim_support:
        fig.text(0.47, 0.800, anim_support, ha="center", va="top", fontsize=6.6)
    _add_network_key(fig, cmap, norm)

    def draw_frame(ax: Any, frame: Dict[str, Any], label: str) -> None:
        ax.clear()
        mean_y, mean_x, n_active = _draw_control_network(
            ax, frame, pos, cmap, norm, attention_max, tie_max,
        )
        ax.set_title(
            _panel_title(label, frame, mean_y, mean_x, n_active, target_sign),
            fontsize=9, fontweight="bold", pad=8,
        )

    def update_frame(frame_idx: int) -> None:
        draw_frame(ax_c, ctrl_frames[frame_idx], "Controlled intervention")
        if ax_u is not None:
            draw_frame(ax_u, unctrl_frames[frame_idx], "Zero-control baseline")

    anim = animation.FuncAnimation(fig, update_frame, frames=len(ctrl_frames),
                                   interval=interval)
    anim_data["anim"] = anim
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".gif":
            try:
                anim.save(str(out_p), writer="pillow", fps=fps)
                anim_data["gif_path"] = str(out_p)
            except Exception as exc:
                anim_data["gif_error"] = str(exc)
        html_p = out_p.with_suffix(".html")
        try:
            # The stock player pulls Font Awesome from a CDN and labels its
            # buttons with icons only, so opened offline the controls are blank.
            # Fixing that up in place keeps the live figure and flash animation
            # untouched. See animation_player for the fully self-contained page.
            html_p.write_text(inline_player_fixups(anim.to_jshtml()), encoding="utf-8")
            anim_data["html_path"] = str(html_p)
        except Exception as exc:
            anim_data["html_error"] = str(exc)
        plt.close(fig)
    return anim_data


# -----------------------------------------------------------------------------
# 6. Bounded Exploratory Diagnostic Response Maps
# -----------------------------------------------------------------------------

def plot_diagnostic_response_maps(
    grid_data: Union[Any, Path, str, Dict[str, Any]],
    save_path: Optional[Union[Path, str]] = None,
    figsize: Tuple[float, float] = (16.0, 10.0),
    dpi: int = 150,
) -> Any:
    """Render 2D exploratory diagnostic response maps and trajectory failure accounting.

    Implementation:
    - Renders 2D response maps across declared axes (e.g. exposure cap vs candidate node fraction):
      1. Success probability P(success)
      2. Mean restricted completion time
      3. Peak target alignment
      4. Post-release retention
      5. Dominant trajectory mode
      6. Descriptive outcome patterns from recorded means and actions (no causal attribution)
    - Clearly marks unevaluated cells with distinct shading, hatching, and 'N/E' tags.
    - Applies strictly discrete cells with NO invented smoothing or interpolation over missing data.
    """
    if not HAS_MATPLOTLIB or plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    from src.control.grid import ExploratoryGridResult

    if isinstance(grid_data, ExploratoryGridResult):
        grid_res = grid_data
    elif isinstance(grid_data, (str, Path)):
        p = Path(grid_data)
        if p.is_dir():
            candidate = p / "diagnostic_grid_data.json"
            if candidate.exists():
                p = candidate
            else:
                raise FileNotFoundError(f"No diagnostic_grid_data.json found in {p}")
        grid_res = ExploratoryGridResult.load_json(p)
    elif isinstance(grid_data, dict):
        payload = grid_data.get("payload", grid_data)
        grid_res = ExploratoryGridResult.from_dict(payload)
    else:
        raise TypeError(f"Unsupported grid_data type: {type(grid_data)!r}")

    axis_x = grid_res.config.axis_x
    axis_y = grid_res.config.axis_y
    x_vals = axis_x.values
    y_vals = axis_y.values
    nx = len(x_vals)
    ny = len(y_vals)

    success_mat = np.full((ny, nx), np.nan, dtype=np.float64)
    time_mat = np.full((ny, nx), np.nan, dtype=np.float64)
    alignment_mat = np.full((ny, nx), np.nan, dtype=np.float64)
    retention_mat = np.full((ny, nx), np.nan, dtype=np.float64)
    evaluated_mask = np.zeros((ny, nx), dtype=bool)
    dominant_mode_mat: Dict[Tuple[int, int], str] = {}

    total_failures: Dict[str, int] = {
        "weak_joint_shift": 0,
        "partial_joint_shift": 0,
        "target_not_sustained": 0,
        "budget_limited": 0,
        "unclassified": 0,
        "legacy_unverified": 0,
        "success": 0,
    }

    mode_colors = {
        "weak_joint_shift": "#e66101",
        "partial_joint_shift": "#5e3c99",
        "target_not_sustained": "#b2abd2",
        "budget_limited": "#d7191c",
        "unclassified": "#636363",
        "legacy_unverified": "#969696",
        "success": "#2ca25f",
        "unevaluated": "#dcdcdc",
    }

    for cell in grid_res.cells:
        ix, iy = cell.coord_idx
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        if cell.is_evaluated:
            evaluated_mask[iy, ix] = True
            if cell.success_probability is not None:
                success_mat[iy, ix] = float(cell.success_probability)
            if cell.mean_restricted_time is not None:
                time_mat[iy, ix] = float(cell.mean_restricted_time)
            if cell.peak_joint_alignment is not None:
                alignment_mat[iy, ix] = float(cell.peak_joint_alignment)
            if cell.transformed_state_retained_to_horizon is not None:
                retention_mat[iy, ix] = float(cell.transformed_state_retained_to_horizon)

            legacy_keys = {"local_response", "propagation", "persistence", "budget_clipping"}
            breakdown = {key: int(count) for key, count in cell.failure_breakdown.items()
                         if key not in legacy_keys}
            legacy_count = sum(int(cell.failure_breakdown.get(key, 0)) for key in legacy_keys)
            if legacy_count:
                breakdown["legacy_unverified"] = legacy_count
            for key, count in breakdown.items():
                if key in total_failures:
                    total_failures[key] += count

            failures_only = {key: count for key, count in breakdown.items() if key != "success"}
            if cell.n_success == cell.n_rollouts and cell.n_rollouts > 0:
                dominant_mode_mat[(iy, ix)] = "success"
            elif failures_only and max(failures_only.values(), default=0) > 0:
                dominant_mode_mat[(iy, ix)] = max(failures_only.items(), key=lambda kv: kv[1])[0]
            elif cell.n_success > 0:
                dominant_mode_mat[(iy, ix)] = "success"
            else:
                dominant_mode_mat[(iy, ix)] = "unclassified"
        else:
            dominant_mode_mat[(iy, ix)] = "unevaluated"

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Success Probability Heatmap
    cmap_succ = plt.cm.YlGnBu.with_extremes(bad="#dcdcdc")
    im0 = axes[0, 0].imshow(success_mat, origin="lower", cmap=cmap_succ, vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_title("Certified successes / eligible trials (95% CI)", fontsize=11, fontweight="bold")

    # 2. Mean Restricted Time Heatmap
    cmap_time = plt.cm.magma_r.with_extremes(bad="#dcdcdc")
    im1 = axes[0, 1].imshow(time_mat, origin="lower", cmap=cmap_time,
                            vmin=0.0, vmax=float(grid_res.config.horizon), aspect="auto")
    plt.colorbar(im1, ax=axes[0, 1], label="Model time")
    axes[0, 1].set_title("Restricted time (failures assigned H)", fontsize=11, fontweight="bold")

    # 3. Peak Target Alignment Heatmap
    cmap_align = plt.cm.viridis.with_extremes(bad="#dcdcdc")
    im2 = axes[0, 2].imshow(alignment_mat, origin="lower", cmap=cmap_align,
                            vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im2, ax=axes[0, 2], label="Target alignment")
    axes[0, 2].set_title("Peak joint private/public target alignment", fontsize=11, fontweight="bold")
    if np.any(np.isfinite(alignment_mat)):
        low_align, high_align = np.nanmin(alignment_mat), np.nanmax(alignment_mat)
        if low_align < grid_res.config.target_threshold < high_align:
            axes[0, 2].contour(alignment_mat, levels=[grid_res.config.target_threshold],
                               colors=["white"], linewidths=0.7)

    # 4. Post-Release Retention Heatmap
    cmap_ret = plt.cm.Blues.with_extremes(bad="#dcdcdc")
    im3 = axes[1, 0].imshow(retention_mat, origin="lower", cmap=cmap_ret,
                            vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im3, ax=axes[1, 0], label="Retention fraction")
    axes[1, 0].set_title("Certified target retained through H", fontsize=11, fontweight="bold")

    # Annotate and hatch cells for panels 0..3
    map_axes = [
        (axes[0, 0], success_mat, lambda v: f"{v:.2f}", im0),
        (axes[0, 1], time_mat, lambda v: f"{v:.1f}", im1),
        (axes[0, 2], alignment_mat, lambda v: f"{v:.2f}", im2),
        (axes[1, 0], retention_mat, lambda v: f"{v:.2f}", im3),
    ]
    cell_by_index = {(c.coord_idx[1], c.coord_idx[0]): c for c in grid_res.cells}
    for ax, mat, fmt_fn, image in map_axes:
        ax.set_xticks(range(nx))
        ax.set_xticklabels([f"{v:.2g}" for v in x_vals], fontsize=9)
        ax.set_yticks(range(ny))
        ax.set_yticklabels([f"{v:.2g}" for v in y_vals], fontsize=9)
        ax.set_xlabel(axis_x.label, fontsize=9)
        ax.set_ylabel(axis_y.label, fontsize=9)

        for iy in range(ny):
            for ix in range(nx):
                if evaluated_mask[iy, ix] and not np.isnan(mat[iy, ix]):
                    value = mat[iy, ix]
                    val_str = fmt_fn(value)
                    if ax is axes[0, 0]:
                        cell = cell_by_index[(iy, ix)]
                        ci = cell.success_ci
                        val_str = (f"{cell.n_success}/{cell.n_eligible}\n"
                                   f"[{ci[0]:.2f},{ci[1]:.2f}]") if ci else "N/A"
                    rgb = image.cmap(image.norm(value))[:3]
                    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                    text_color = "white" if luminance < 0.48 else "black"
                    ax.text(ix, iy, val_str, ha="center", va="center",
                            color=text_color, fontsize=8 if ax is axes[0, 0] else 9, fontweight="bold")
                elif evaluated_mask[iy, ix]:
                    ax.text(ix, iy, "N/A", ha="center", va="center", color="#555555", fontsize=9, fontweight="bold")
                else:
                    rect = Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#888888", linewidth=0.5)
                    ax.add_patch(rect)
                    ax.text(ix, iy, "N/E", ha="center", va="center", color="#777777", fontsize=9, fontweight="bold")

    # 5. Dominant Trajectory Mode Panel
    ax_mode = axes[1, 1]
    ax_mode.set_title("Observed Trajectory Pattern", fontsize=11, fontweight="bold")
    ax_mode.set_xticks(range(nx))
    ax_mode.set_xticklabels([f"{v:.2g}" for v in x_vals], fontsize=9)
    ax_mode.set_yticks(range(ny))
    ax_mode.set_yticklabels([f"{v:.2g}" for v in y_vals], fontsize=9)
    ax_mode.set_xlabel(axis_x.label, fontsize=9)
    ax_mode.set_ylabel(axis_y.label, fontsize=9)
    ax_mode.set_xlim(-0.5, nx - 0.5)
    ax_mode.set_ylim(-0.5, ny - 0.5)

    mode_labels = {
        "weak_joint_shift": "Weak",
        "partial_joint_shift": "Partial",
        "target_not_sustained": "Hold",
        "budget_limited": "Budget",
        "unclassified": "Unknown",
        "legacy_unverified": "Legacy",
        "success": "Success",
        "unevaluated": "N/E",
    }
    for iy in range(ny):
        for ix in range(nx):
            mode = dominant_mode_mat.get((iy, ix), "unevaluated")
            color = mode_colors.get(mode, "#dcdcdc")
            is_un = (mode == "unevaluated")
            rect = Rectangle(
                (ix - 0.5, iy - 0.5), 1, 1,
                facecolor=color,
                edgecolor="#ffffff" if not is_un else "#888888",
                hatch="///" if is_un else None,
                linewidth=0.5,
            )
            ax_mode.add_patch(rect)
            txt_color = "black" if mode in ("target_not_sustained", "unevaluated", "legacy_unverified") else "white"
            ax_mode.text(ix, iy, mode_labels.get(mode, mode), ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    legend_patches = [
        Patch(facecolor=mode_colors["weak_joint_shift"], label="Weak joint shift"),
        Patch(facecolor=mode_colors["partial_joint_shift"], label="Partial joint shift"),
        Patch(facecolor=mode_colors["target_not_sustained"], label="Target not sustained"),
        Patch(facecolor=mode_colors["budget_limited"], label="Budget limit observed"),
        Patch(facecolor=mode_colors["unclassified"], label="No trajectory"),
        Patch(facecolor=mode_colors["legacy_unverified"], label="Legacy label (unverified)"),
        Patch(facecolor=mode_colors["success"], label="Certified"),
        Patch(facecolor=mode_colors["unevaluated"], hatch="///", edgecolor="#888888", label="Unevaluated (N/E)"),
    ]
    ax_mode.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7)

    # 6. Observed outcome-pattern counts
    ax_bar = axes[1, 2]
    ax_bar.set_title("Observed Outcome Patterns", fontsize=11, fontweight="bold")
    categories = ["weak_joint_shift", "partial_joint_shift", "target_not_sustained", "budget_limited", "unclassified", "legacy_unverified", "success"]
    counts = [total_failures[c] for c in categories]
    colors = [mode_colors[c] for c in categories]
    labels = ["Weak joint shift", "Partial joint shift", "Target not sustained", "Budget limit", "No trajectory", "Legacy unverified", "Target certified"]
    tot_rollouts = sum(counts)

    y_pos = np.arange(len(categories))
    bars = ax_bar.barh(y_pos, counts, color=colors, edgecolor="black", alpha=0.85)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Measured Rollout Count", fontsize=9)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

    if tot_rollouts > 0:
        for b, cnt in zip(bars, counts):
            width = b.get_width()
            pct = (cnt / tot_rollouts) * 100.0
            ax_bar.text(width + 0.2, b.get_y() + b.get_height() / 2.0, f"{cnt} ({pct:.0f}%)", va="center", fontsize=8, fontweight="bold")
        ax_bar.set_xlim(0, max(counts) * 1.35 + 1)
    else:
        ax_bar.text(0.5, 0.5, "No Evaluated Rollouts", ha="center", va="center", transform=ax_bar.transAxes, fontsize=10, color="#666666")

    budget_status = "Budget Cap Hit" if grid_res.compute_budget_exhausted else "Full Grid Evaluated"
    fig.suptitle(
        f"Diagnostic Response Grid: {axis_y.label} vs {axis_x.label} ({budget_status})\n"
        f"Total Rollouts: {grid_res.total_rollouts_executed}/{grid_res.config.compute_budget_cap} | "
        f"Evaluated Cells: {int(np.sum(evaluated_mask))}/{nx * ny}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path is not None:
        p_out = Path(save_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p_out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(p_out)

    return fig


# -----------------------------------------------------------------------------
# 7. Master Visualization Orchestrator
# -----------------------------------------------------------------------------

def render_bundle_visualizations(
    bundle_or_dir: Union[LoadedControlBundle, Path, str, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    render_animation: bool = True,
    save_html_animation: bool = True,
    max_animation_frames: int = 40,
) -> Dict[str, Any]:
    """Render all scientific plots and animations directly from saved bundle and update manifest.

    Consumes saved bundle data in a fresh process WITHOUT running any physical simulations.
    Registers all generated artifacts in manifest.json with updated checksums.
    """
    bundle, bundle_path = _resolve_bundle(bundle_or_dir)
    target_dir = Path(output_dir) if output_dir is not None else bundle_path
    if target_dir is None:
        raise ValueError("Cannot determine bundle output directory.")
    target_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}

    # 1. Trajectory diagnostics
    try:
        p_traj = target_dir / "trajectory_diagnostics.png"
        plot_scalar_trajectories(bundle, save_path=p_traj)
        artifacts["trajectory_diagnostics"] = p_traj.name
    except Exception as exc:
        artifacts["trajectory_diagnostics_error"] = str(exc)

    # 2. Budget and intervention schedules
    try:
        p_budget = target_dir / "budget_and_intervention.png"
        plot_intervention_and_budget(bundle, save_path=p_budget)
        artifacts["budget_and_intervention"] = p_budget.name
    except Exception as exc:
        artifacts["budget_and_intervention_error"] = str(exc)

    # 3. Policy reliability, completion time, and conditional relapse
    try:
        p_rel = target_dir / "policy_reliability_and_time.png"
        plot_policy_reliability_and_time(bundle, save_path=p_rel)
        artifacts["policy_reliability_and_time"] = p_rel.name
    except Exception as exc:
        artifacts["policy_reliability_and_time_error"] = str(exc)

    # 4. Paired adaptive vs frozen comparison
    try:
        p_paired = target_dir / "paired_adaptive_frozen.png"
        plot_paired_adaptive_frozen_comparison(bundle, save_path=p_paired)
        artifacts["paired_adaptive_frozen"] = p_paired.name
    except Exception as exc:
        artifacts["paired_adaptive_frozen_error"] = str(exc)

    # 5. Network views: before / during / after
    try:
        p_net = target_dir / "network_before_during_after.png"
        plot_network_views_before_during_after(bundle, save_path=p_net)
        artifacts["network_before_during_after"] = p_net.name
    except Exception as exc:
        artifacts["network_before_during_after_error"] = str(exc)

    # 6. Network animation (GIF + standalone HTML5 player)
    if render_animation and bundle.network_frames:
        try:
            p_anim_gif = target_dir / "network_animation.gif"
            anim_res = render_network_animation(
                bundle, output_path=p_anim_gif, max_frames=max_animation_frames
            )
            if "gif_path" in anim_res:
                artifacts["network_animation_gif"] = p_anim_gif.name
            if "html_path" in anim_res:
                artifacts["network_animation_html"] = Path(anim_res["html_path"]).name
        except Exception as exc:
            artifacts["network_animation_error"] = str(exc)

    # 7. Exploratory diagnostic response maps
    p_grid_json = target_dir / "diagnostic_grid_data.json"
    if not p_grid_json.exists() and hasattr(bundle, "bundle_dir") and bundle.bundle_dir:
        cand_p = Path(bundle.bundle_dir) / "diagnostic_grid_data.json"
        if cand_p.exists():
            p_grid_json = cand_p

    if p_grid_json.exists():
        try:
            p_grid_img = target_dir / "diagnostic_response_maps.png"
            plot_diagnostic_response_maps(p_grid_json, save_path=p_grid_img)
            artifacts["diagnostic_response_maps"] = p_grid_img.name
        except Exception as exc:
            artifacts["diagnostic_response_maps_error"] = str(exc)

    # Update manifest.json with registered visualizations and fresh file inventory
    manifest = update_bundle_manifest(target_dir, extra_fields={"visualizations": artifacts})

    return {
        "bundle_dir": str(target_dir),
        "visualizations": artifacts,
        "manifest": manifest,
    }
