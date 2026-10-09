"""Persistence, turnover, and time-to-event diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.analytics.influence import calculate_influence_metrics, top_fraction_overlap

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def rank_persistence_curve(W_snapshots, top_fraction: float = 0.2):
    """Jaccard overlap of the top-fraction attention set versus the final network.

    W_snapshots: list of (t, W) pairs or snapshots dicts containing 'W'.
    Returns (times, overlaps).
    """
    snaps = [s for s in W_snapshots if (s.get("W") is not None)]
    if not snaps:
        raise ValueError("No network snapshots recorded for persistence analysis")
    r_final = calculate_influence_metrics(np.asarray(snaps[-1]["W"]))["r"]
    times, overlaps = [], []
    for s in snaps[:-1]:
        r_t = calculate_influence_metrics(np.asarray(s["W"]))["r"]
        ov = top_fraction_overlap(r_t, r_final, fraction=top_fraction)
        times.append(float(s["t"]))
        overlaps.append(ov["jaccard_overlap"])
    return times, overlaps


def plot_rank_persistence(times, overlaps, ax=None, top_fraction: float = 0.2):
    if not HAS_MPL:
        return {"times": times, "overlaps": overlaps}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    else:
        fig = ax.figure
    ax.plot(times, overlaps, "o-", color="#2B6CB0")
    ax.set_xlabel("Snapshot time t")
    ax.set_ylabel(f"Top-{int(top_fraction * 100)}% overlap with final ranking (Jaccard)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Influence rank persistence", fontsize=11, fontweight="bold")
    return {"fig": ax.figure, "ax": ax, "times": times, "overlaps": overlaps}


def survival_curve(event_times: Sequence[float], horizon: float):
    """Kaplan–Meier-style cumulative 'no transition yet' curve with censoring at H."""
    t = np.sort(np.asarray([e for e in event_times if np.isfinite(e)]))
    H = float(np.max(event_times))
    grid = np.linspace(0.0, H, 200)
    surv = np.array([float(np.mean(np.asarray(event_times) > g)) for g in grid])
    return grid, surv


def plot_survival(grid, surv, ax=None, title: str = "P(target not yet completed)"):
    if not HAS_MPL:
        return {"grid": grid, "survival": surv}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    else:
        fig = ax.figure
    ax.step(grid, surv, where="post", color="#2B6CB0", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("P(event not yet occurred)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Time-to-event (administrative censoring at horizon)", fontsize=11, fontweight="bold")
    return {"fig": fig, "ax": ax, "grid": grid, "survival": surv}
