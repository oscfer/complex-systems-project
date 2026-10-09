"""Plotting and visualization routines strictly adhering to Stage 4 measurement specifications.

Implements the core visualization requirements:
1. Intelligent Network Styling: Node size scaled with received attention r_j / degree,
   node color with continuous diverging colormap (RdBu) mapped to belief states in [-1, 1],
   and edge thickness/opacity scaled with memory count W_{ij}. Handles isolates explicitly without
   artificial thresholds on W.
2. Temporal Animations: Animated sequences (matplotlib.animation) showing temporal emergence.
3. Influence Plots: Normalized concentration C_N = (N*C - 1)/(N - 1), effective leaders 1/C,
   and top-fraction persistence overlap using explicit conservative tie policies.
4. Community & Fragmentation Plots: Weakly connected components (WCC), Leiden algorithm on
   undirected projection G = W + W^T with resolution gamma in {0.5, 1.0, 2.0}, handling isolates.
5. Null Model Comparisons: Side-by-side empirical network vs degree-preserving edge-swap null model.
6. Coupled Beliefs: Multi-component responses (D=3) plotted separately by component,
   distinguishing directly perturbed from unperturbed components.
7. Statistical Verification: Wilson score confidence intervals for success probabilities,
   strictly rejecting normal approximations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.analytics.communities import (
    connected_components_analysis,
    detect_communities,
    symmetrize_adjacency,
)
from src.analytics.influence import (
    calculate_influence_metrics,
    top_fraction_overlap,
)
from src.analytics.null_models import generate_degree_preserving_null
from src.experiment.baselines.reference import wilson_interval

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    animation = None


def _seeded_spring_layout(
    W: np.ndarray,
    layout_seed: int = 42,
    iterations: int = 100,
    k: Optional[float] = None,
) -> np.ndarray:
    """Deterministic seeded Fruchterman–Reingold spring layout (no networkx).

    Undirected projection G = W + W^T drives attraction; a weak repulsion keeps
    disconnected nodes apart. Initial positions on a seeded ring guarantee
    reproducibility across runs and platforms.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]
    rng = np.random.default_rng(layout_seed)
    G = W + W.T
    np.fill_diagonal(G, 0.0)
    m = float(np.max(G)) if np.max(G) > 0 else 1.0
    A = G / m

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = np.column_stack((np.cos(angles), np.sin(angles))) + rng.normal(0, 0.05, size=(N, 2))

    if k is None:
        k = 1.0 / max(np.sqrt(N), 1.0)

    # Row-normalize the projected adjacency: each node carries the same total
    # attraction weight regardless of degree, so dense graphs do not collapse
    # under self-attraction (classic FR failure mode on dense W).
    row_sums = A.sum(axis=1, keepdims=True)
    An = A / np.maximum(row_sums, 1e-12)

    t = 0.1
    dt = t / (iterations + 1)
    for it in range(iterations):
        disp = np.zeros_like(pos)
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1e-9)
        # Repulsive force ~ k^2 / d (Fruchterman-Reingold)
        rep = (k * k) / dist
        disp -= np.einsum("ijk,ij->ik", delta, rep)
        # Attractive force ~ d^2 / k along edges (row-normalized weights)
        attr = (dist * dist) / k * An
        disp += np.einsum("ijk,ij->ik", delta, attr)
        norm = np.sqrt((disp * disp).sum(axis=1))
        norm[norm < 1e-12] = 1.0
        displacement = np.clip(norm, 0.0, t)[:, np.newaxis] * disp / np.maximum(norm, 1e-12)[:, np.newaxis]
        pos += displacement
        # Keep points inside the unit disc: only project OUTWARD nodes back
        # onto the boundary; interior nodes must stay free so community
        # structure can contract toward the center.
        radius = np.sqrt((pos * pos).sum(axis=1))
        outside = radius > 1.0
        if np.any(outside):
            pos[outside] = pos[outside] / radius[outside][:, np.newaxis]
        t -= dt
    # Center and rescale
    pos -= pos.mean(axis=0, keepdims=True)
    ext = np.max(np.abs(pos)) or 1.0
    return pos / ext * 1.1


def plot_styled_network(
    W: np.ndarray,
    beliefs: np.ndarray,
    ax: Optional[Any] = None,
    component: int = 0,
    title: str = "Intelligent Network Topology & Belief State",
    layout_seed: int = 42,
    layout: str = "circular",
    iterations: int = 100,
    positions: Optional[np.ndarray] = None,
    base_node_size: float = 120.0,
    max_node_size: float = 800.0,
    colormap: str = "RdBu_r",
) -> Dict[str, Any]:
    """Render scientifically legible network styling with influence scaling and diverging beliefs.

    Specifications:
    - Node size: scales linearly with received attention r_j (in-degree/attention share).
    - Node color: continuous diverging gradient (RdBu) mapped to belief states in [-1, 1].
    - Edge thickness & alpha: scaled proportionally with interaction memory W_{ij}.
    - Isolates: handled explicitly without artificial disconnection thresholds on W.

    Returns:
        Dictionary containing extracted plot data and optional matplotlib Figure/Axes.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.ndim == 1:
        b_vals = beliefs
    else:
        b_vals = beliefs[:, component]

    # Compute node coordinates: explicit positions, circular, or seeded spring layout
    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64)
    else:
        rng = np.random.default_rng(layout_seed)
        if layout == "spring":
            pos = _seeded_spring_layout(W, layout_seed, iterations=iterations)
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            # Add subtle jitter to break perfect circle alignment
            radius = 1.0 + rng.uniform(-0.05, 0.05, size=N)
            pos = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

    # Influence metrics for node sizing
    influence = calculate_influence_metrics(W)
    r = influence["r"]
    r_max = float(np.max(r)) if np.max(r) > 0 else 1.0
    node_sizes = base_node_size + (max_node_size - base_node_size) * (r / r_max)

    # Edge extraction
    nonzero_edges = []
    edge_weights = []
    max_w = float(np.max(W)) if np.max(W) > 0 else 1.0

    for i in range(N):
        for j in range(N):
            if W[i, j] > 0 and i != j:
                nonzero_edges.append((i, j))
                edge_weights.append(float(W[i, j]))

    edge_widths = [0.5 + 3.5 * (w / max_w) for w in edge_weights]
    edge_alphas = [min(1.0, 0.15 + 0.85 * (w / max_w)) for w in edge_weights]

    plot_data = {
        "N": N,
        "positions": pos.tolist(),
        "beliefs": b_vals.tolist(),
        "received_attention": r.tolist(),
        "node_sizes": node_sizes.tolist(),
        "edge_count": len(nonzero_edges),
        "edges": nonzero_edges,
        "edge_weights": edge_weights,
        "edge_widths": edge_widths,
        "edge_alphas": edge_alphas,
        "f_active": influence["f_active"],
        "C_N": influence["C_N"],
    }

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        else:
            fig = ax.figure

        # Draw edges
        for (i, j), width, alpha in zip(nonzero_edges, edge_widths, edge_alphas):
            p_i, p_j = pos[i], pos[j]
            ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], color="#4A5568", linewidth=width, alpha=alpha, zorder=1)

        # Draw nodes
        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(norm(val)) for val in b_vals]

        scatter = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=node_sizes,
            c=colors,
            edgecolors="#1A202C",
            linewidths=1.5,
            zorder=3,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        plot_data["fig"] = fig
        plot_data["ax"] = ax

    return plot_data


def animate_network_temporal(
    trajectory_data: Union[Sequence[Dict[str, Any]], Any],
    filename: Optional[str] = None,
    fps: int = 5,
    interval: int = 200,
    max_frames: int = 48,
    layout: str = "circular",
    layout_seed: int = 42,
) -> Any:
    """Generate temporal animation of network topology emergence and belief transformations.

    Frames are the recorded network snapshots (see TrajectoryRunner
    `record_network_frames`); long runs are subsampled to `max_frames`.
    """
    # Extract snapshots; animation requires network copies. If the trajectory was
    # recorded without network frames, fall back to whatever frames carry W/y.
    if hasattr(trajectory_data, "timeseries"):
        snapshots = trajectory_data.timeseries
    elif isinstance(trajectory_data, dict) and "timeseries" in trajectory_data:
        snapshots = trajectory_data["timeseries"]
    elif isinstance(trajectory_data, (list, tuple)):
        snapshots = list(trajectory_data)
    else:
        snapshots = []
    network_frames = [s for s in snapshots if s.get("W") is not None and s.get("y") is not None]
    if not network_frames and snapshots:
        raise ValueError(
            "No animation frames available: the trajectory was recorded with "
            "record_network_frames=False. Re-run with record_network_frames=True."
        )
    snapshots = network_frames
    if len(snapshots) > max_frames:
        idx = np.linspace(0, len(snapshots) - 1, max_frames).astype(int)
        snapshots = [snapshots[k] for k in idx]

    anim_data = {
        "num_frames": len(snapshots),
        "fps": fps,
        "interval": interval,
        "filename": filename,
    }

    if not HAS_MATPLOTLIB:
        return anim_data

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    def update(frame_idx: int):
        ax.clear()
        if not snapshots:
            ax.set_title("No snapshot data available")
            return
        snap = snapshots[frame_idx]
        t = snap.get("t", frame_idx)
        W = snap.get("W")
        y = snap.get("y")
        if W is None or y is None:
            raise ValueError("Each animation snapshot must contain node-level 'W' and 'y' data")
        plot_styled_network(
            np.asarray(W),
            np.asarray(y),
            ax=ax,
            positions=anim_positions,
            title=f"Temporal Emergence (t={t:.2f}; frame {frame_idx + 1}/{len(snapshots)})",
        )
        m_y = snap.get("m_y", np.mean(np.asarray(y), axis=0).tolist())
        tot_mem = snap.get("total_memory", 0)
        ax.text(0.02, 0.98, f"Memory S = {tot_mem}\nMean stance = {m_y[0]:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Fixed node positions across all frames (computed once from the final
    # network) so the temporal animation is visually coherent.
    anim_positions = _seeded_spring_layout(
        np.asarray(snapshots[-1]["W"]), layout_seed
    ) if layout == "spring" else None

    anim = animation.FuncAnimation(fig, update, frames=max(1, len(snapshots)), interval=interval)
    anim_data["anim"] = anim

    if filename is not None:
        try:
            if filename.endswith(".gif"):
                anim.save(filename, writer="pillow", fps=fps)
            elif filename.endswith(".mp4"):
                anim.save(filename, writer="ffmpeg", fps=fps)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export animation to {filename}: {exc}. "
                "On Colab/VS Code install the GIF writer with: pip install pillow"
            ) from exc

    return anim_data


def plot_influence_metrics(
    timeseries: Sequence[Dict[str, Any]],
    ax: Optional[Any] = None,
    title: str = "Attention Inequality & Influence Persistence",
) -> Dict[str, Any]:
    """Plot normalized concentration C_N and effective number of leaders 1/C.

    Uses conservative tie policy for leadership tracking.
    """
    times = []
    C_N_vals = []
    N_eff_vals = []

    for entry in timeseries:
        times.append(float(entry.get("t", 0.0)))
        # Compute or retrieve influence metrics if W snapshot is stored
        if "W" in entry:
            metrics = calculate_influence_metrics(np.asarray(entry["W"]))
            C_N_vals.append(float(metrics["C_N"]))
            N_eff_vals.append(float(metrics.get("N_eff", metrics["effective_leaders"])))
        else:
            C_N_vals.append(float(entry.get("C_N", np.nan)))
            N_eff_vals.append(float(entry.get("N_eff", np.nan)))

    data = {
        "times": times,
        "C_N": C_N_vals,
        "N_eff": N_eff_vals,
    }

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(8, 4), dpi=100)
        else:
            ax1 = ax
            fig = ax1.figure

        color1 = "#2B6CB0"
        ax1.set_xlabel("Simulation Time t", fontsize=11)
        ax1.set_ylabel("Normalized Concentration C_N", color=color1, fontsize=11)
        ax1.plot(times, C_N_vals, color=color1, linewidth=2.0, label="Normalized Concentration C_N")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color2 = "#C53030"
        ax2.set_ylabel("Effective Leaders N_eff = 1/C", color=color2, fontsize=11)
        ax2.plot(times, N_eff_vals, color=color2, linewidth=2.0, linestyle="--", label="Effective Leaders 1/C")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_title(title, fontsize=12, fontweight="bold")
        data["fig"] = fig
        data["ax1"] = ax1
        data["ax2"] = ax2

    return data


def plot_community_fragmentation(
    W: np.ndarray,
    gammas: Sequence[float] = (0.5, 1.0, 2.0),
    ax: Optional[Any] = None,
    title: str = "Leiden Modularity & Component Fragmentation",
) -> Dict[str, Any]:
    """Analyze undirected projection G = W + W^T across resolutions gamma in {0.5, 1.0, 2.0}.

    Strictly complies with Traag et al. (2019) Leiden modularity optimization,
    explicitly handles isolates, and never imposes artificial thresholds on W.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    # 1. Undirected symmetrized projection G = W + W^T
    G = symmetrize_adjacency(W)

    # 2. Track components
    comp_info = connected_components_analysis(W)
    num_wcc = comp_info["num_wcc"]
    largest_cc = comp_info["largest_wcc_fraction"]
    f_iso = comp_info.get("isolated_fraction", comp_info.get("f_iso", 0.0))

    # 3. Leiden modularity across resolutions
    leiden_results = {}
    for gamma in gammas:
        res = detect_communities(W, gamma=float(gamma), restarts=5, seed=42)
        leiden_results[f"gamma_{gamma}"] = {
            "gamma": float(gamma),
            "modularity_Q": float(res["modularity_Q"]),
            "num_communities": int(res["num_communities"]),
            "partition": res["partition"].tolist(),
            "stability_ari": float(res["stability_ari"]),
            "non_isolate_communities": int(len(set(
                np.asarray(res["partition"])[np.sum(G, axis=1) > 0].tolist()
            ))),
        }

    data = {
        "N": N,
        "num_wcc": num_wcc,
        "largest_wcc_fraction": largest_cc,
        "isolated_node_fraction": f_iso,
        "leiden_resolutions": leiden_results,
    }

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        else:
            fig = ax.figure

        gamma_labels = [f"γ = {g}" for g in gammas]
        q_scores = [leiden_results[f"gamma_{g}"]["modularity_Q"] for g in gammas]
        num_comms = [leiden_results[f"gamma_{g}"]["num_communities"] for g in gammas]

        x_coords = np.arange(len(gammas))
        bars = ax.bar(x_coords, q_scores, width=0.4, color="#319795", label="Leiden Modularity Q")
        ax.set_xticks(x_coords)
        ax.set_xticklabels(gamma_labels, fontsize=11)
        ax.set_ylabel("Modularity Score Q", fontsize=11)
        ax.set_ylim(0, max(1.0, max(q_scores) * 1.2 if q_scores else 1.0))
        ax.grid(True, linestyle=":", alpha=0.6)

        # Annotate number of communities
        for idx, (rect, k) in enumerate(zip(bars, num_comms)):
            height = rect.get_height()
            ax.annotate(f"K={k}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{title}\n(WCC Components={num_wcc}, Largest CC={largest_cc:.1%}, Isolates={f_iso:.1%})",
                     fontsize=11, fontweight="bold")
        data["fig"] = fig
        data["ax"] = ax

    return data


def plot_null_model_comparison(
    W: np.ndarray,
    num_swaps: int = 50,
    seed: int = 42,
    ax: Optional[Any] = None,
) -> Dict[str, Any]:
    """Side-by-side comparison of empirical network vs degree-preserving edge-swap null model."""
    W = np.asarray(W, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # The declared null is defined on the simple undirected support of G=W+W^T.
    G = symmetrize_adjacency(W)
    emp_res = detect_communities(G, gamma=1.0, restarts=5, seed=seed)

    # 2. Degree-preserving null model
    W_null = generate_degree_preserving_null(G, num_swaps=num_swaps, rng=rng)
    null_res = detect_communities(W_null, gamma=1.0, restarts=5, seed=seed)

    data = {
        "empirical_Q": float(emp_res["modularity_Q"]),
        "empirical_K": int(emp_res["num_communities"]),
        "null_Q": float(null_res["modularity_Q"]),
        "null_K": int(null_res["num_communities"]),
        "modularity_excess": float(emp_res["modularity_Q"] - null_res["modularity_Q"]),
        "empirical_graph": G,
        "null_graph": W_null,
        "projected_degree_sequence": np.sum(G > 0, axis=1).astype(int).tolist(),
        "null_degree_sequence": np.sum(W_null > 0, axis=1).astype(int).tolist(),
    }

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        else:
            fig = ax.figure

        categories = ["Empirical Network", "Degree-Preserving Null"]
        q_vals = [data["empirical_Q"], data["null_Q"]]
        colors = ["#2B6CB0", "#A0AEC0"]

        bars = ax.bar(categories, q_vals, color=colors, width=0.5)
        ax.set_ylabel("Leiden Modularity Q (γ=1.0)", fontsize=11)
        ax.set_ylim(0, max(1.0, max(q_vals) * 1.25 if q_vals else 1.0))
        ax.grid(True, linestyle=":", alpha=0.5)

        for rect, val in zip(bars, q_vals):
            height = rect.get_height()
            ax.annotate(f"Q={val:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"Null Model Benchmark\n(Modularity Excess ΔQ = {data['modularity_excess']:+.3f})",
                     fontsize=11, fontweight="bold")
        data["fig"] = fig
        data["ax"] = ax

    return data


def plot_coupled_beliefs_response(
    trajectory_data: Union[Sequence[Dict[str, Any]], Any],
    perturbed_component: int = 0,
    control_trajectory: Optional[Union[Sequence[Dict[str, Any]], Any]] = None,
    ax: Optional[Any] = None,
    title: str = "Coupled Multi-Component Beliefs Response (D=3)",
) -> Dict[str, Any]:
    """Plot multi-component belief dynamics distinguishing perturbed from unperturbed components."""
    if hasattr(trajectory_data, "timeseries"):
        snapshots = trajectory_data.timeseries
    elif isinstance(trajectory_data, dict) and "timeseries" in trajectory_data:
        snapshots = trajectory_data["timeseries"]
    elif isinstance(trajectory_data, (list, tuple)):
        snapshots = list(trajectory_data)
    else:
        snapshots = []

    times = []
    comp_trajectories: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    for snap in snapshots:
        t = float(snap.get("t", 0.0))
        times.append(t)
        m_y = snap.get("m_y", [0.0, 0.0, 0.0])
        for alpha in range(min(3, len(m_y))):
            comp_trajectories[alpha].append(float(m_y[alpha]))

    data = {
        "times": times,
        "perturbed_component": perturbed_component,
        "components": comp_trajectories,
    }

    control_components: Optional[Dict[int, List[float]]] = None
    control_times: List[float] = []
    if control_trajectory is not None:
        if hasattr(control_trajectory, "timeseries"):
            control_snapshots = control_trajectory.timeseries
        elif isinstance(control_trajectory, dict) and "timeseries" in control_trajectory:
            control_snapshots = control_trajectory["timeseries"]
        else:
            control_snapshots = list(control_trajectory)
        control_components = {0: [], 1: [], 2: []}
        for snap in control_snapshots:
            control_times.append(float(snap.get("t", 0.0)))
            m_y = snap.get("m_y", [0.0, 0.0, 0.0])
            for alpha in range(min(3, len(m_y))):
                control_components[alpha].append(float(m_y[alpha]))
        data["control_times"] = control_times
        data["control_components"] = control_components

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        else:
            fig = ax.figure

        colors = ["#E53E3E", "#3182CE", "#38A169"]
        labels = [
            f"Component {alpha + 1} {'(DIRECTLY PERTURBED)' if alpha == perturbed_component else '(UNPERTURBED)'}"
            for alpha in range(3)
        ]
        styles = ["-", "--", "-."]

        for alpha in range(3):
            if alpha in comp_trajectories and comp_trajectories[alpha]:
                lw = 2.5 if alpha == perturbed_component else 1.8
                ax.plot(times, comp_trajectories[alpha], color=colors[alpha],
                        linestyle=styles[alpha], linewidth=lw, label=labels[alpha])
                if control_components is not None and control_components[alpha]:
                    ax.plot(control_times, control_components[alpha], color=colors[alpha],
                            linestyle=":", linewidth=1.2, alpha=0.75,
                            label=f"Component {alpha + 1} matched unperturbed control")

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Simulation Time t", fontsize=11)
        ax.set_ylabel("Mean Expressed Stance m^y", fontsize=11)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        data["fig"] = fig
        data["ax"] = ax

    return data


def plot_wilson_confidence_bands(
    trials_list: Sequence[int],
    successes_list: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    target_threshold: float = 0.8,
    ax: Optional[Any] = None,
    title: str = "Success Probabilities with Wilson Score Confidence Intervals",
) -> Dict[str, Any]:
    """Plot success probabilities with exact Wilson score intervals, strictly rejecting normal approximations."""
    k_items = len(trials_list)
    if labels is None:
        labels = [f"Condition {i + 1}" for i in range(k_items)]

    point_estimates = []
    lower_bounds = []
    upper_bounds = []

    for succ, n in zip(successes_list, trials_list):
        p_hat = float(succ / n) if n > 0 else 0.0
        low, high = wilson_interval(succ, n, confidence=confidence)
        point_estimates.append(p_hat)
        lower_bounds.append(low)
        upper_bounds.append(high)

    data = {
        "labels": list(labels),
        "point_estimates": point_estimates,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "confidence": confidence,
        "target_threshold": target_threshold,
    }

    if HAS_MATPLOTLIB:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        else:
            fig = ax.figure

        x_coords = np.arange(k_items)
        y_err_lower = [p - l for p, l in zip(point_estimates, lower_bounds)]
        y_err_upper = [u - p for p, u in zip(point_estimates, upper_bounds)]
        y_err = [y_err_lower, y_err_upper]

        ax.errorbar(
            x_coords,
            point_estimates,
            yerr=y_err,
            fmt="o",
            color="#2B6CB0",
            ecolor="#4A5568",
            elinewidth=2.0,
            capsize=5,
            capthick=1.5,
            markersize=7,
            label=f"Point Estimate & {int(confidence * 100)}% Wilson CI",
        )

        ax.axhline(target_threshold, color="#E53E3E", linestyle="--", linewidth=1.5,
                   label=f"Target Reliability (p0 = {target_threshold})")

        ax.set_xticks(x_coords)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Success Probability", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        data["fig"] = fig
        data["ax"] = ax

    return data
