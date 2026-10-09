"""2D parameter maps and 3D response surfaces.

parameter_map: sweep two model parameters on a grid, run short representative
origins per grid point, and render a heatmap of the chosen observable with a
colorbar. response_surface_3d: same data as a 3D surface. Grid cells reuse one
origin (representative check, not a precision estimate) unless n_replicates>1.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def sweep_grid(
    base_profile,
    x_param: str,
    x_values: Sequence[float],
    y_param: str,
    y_values: Sequence[float],
    observable_fn: Optional[Callable] = None,
    n_origins: int = 1,
    horizon_scale: float = 0.5,
) -> Dict[str, Any]:
    """Run a 2D parameter sweep; returns {'x', 'y', 'Z'} with the mean observable.

    Each grid point runs `n_origins` short origins from fresh realism states and
    records `observable_fn(traj_final_state) -> float` (default: final m_y).
    Horizons are scaled down (representative grids, not confirmatory runs).
    """
    from dataclasses import replace
    from src.experiment.ensembles import derive_origin_seed, run_single_origin

    if observable_fn is None:
        def observable_fn(state):
            return float(np.mean(state.y[:, 0]))

    Z = np.zeros((len(y_values), len(x_values)))
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            prof = replace(base_profile, **{x_param: xv, y_param: yv})
            prof = replace(prof, horizon=prof.horizon * horizon_scale,
                           establishment_time=prof.establishment_time * horizon_scale)
            vals = []
            for o in range(n_origins):
                seed = derive_origin_seed(prof.seed_base, 1000 + yi * len(x_values) + xi)
                from src.config.profiles import build_realism_runner
                runner = build_realism_runner(prof, seed, sample_interval=1e9,
                                              record_network_frames=False)
                traj = runner.run(horizon=prof.horizon)
                vals.append(observable_fn(traj.final_state))
            Z[yi, xi] = float(np.mean(vals))
    return {"x": list(x_values), "y": list(y_values), "Z": Z}


def plot_parameter_map(
    sweep: Dict[str, Any],
    ax=None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "Parameter map (representative origins)",
    cmap: str = "RdBu_r",
):
    if not HAS_MPL:
        return sweep
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    else:
        fig = ax.figure
    Z = sweep["Z"]
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=[min(sweep["x"]), max(sweep["x"]),
                           min(sweep["y"]), max(sweep["y"])])
    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return {"fig": fig, "ax": ax, "Z": Z}


def plot_response_surface_3d(
    sweep: Dict[str, Any],
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "observable",
    title: str = "Response surface",
):
    if not HAS_MPL:
        return sweep
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(sweep["x"], sweep["y"])
    ax.plot_surface(X, Y, sweep["Z"], cmap="viridis", linewidth=0.3, edgecolor="k", alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return {"fig": fig, "ax": ax, "Z": sweep["Z"]}
