"""Scientific visualization and plotting suite for Emergent Order Simulations.

Provides:
- plot_styled_network: Intelligent network layout with influence-scaled sizes, diverging RdBu beliefs, and W memory edges.
- animate_network_temporal: Matplotlib animation of structural emergence and stance evolution.
- plot_influence_metrics: Normalized concentration C_N, effective leaders 1/C, and tie-aware top-fraction persistence.
- plot_community_fragmentation: Leiden modularity across resolutions gamma in {0.5, 1.0, 2.0}, isolates, and WCC.
- plot_null_model_comparison: Side-by-side comparison of empirical community structure against degree-preserving edge-swap nulls.
- plot_coupled_beliefs_response: D=3 component-by-component response curves distinguishing perturbed vs unperturbed stances.
- plot_wilson_confidence_bands: Wilson score confidence bands for success probabilities without normal approximations.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_modular3")

from src.visualization.plotting import (
    animate_network_temporal,
    plot_community_fragmentation,
    plot_coupled_beliefs_response,
    plot_influence_metrics,
    plot_null_model_comparison,
    plot_styled_network,
    plot_wilson_confidence_bands,
)
from src.visualization.control_plots import (
    plot_scalar_trajectories,
    plot_intervention_and_budget,
    plot_policy_reliability_and_time,
    plot_paired_adaptive_frozen_comparison,
    plot_network_views_before_during_after,
    plot_diagnostic_response_maps,
    render_network_animation,
    render_bundle_visualizations,
)

__all__ = [
    "plot_styled_network",
    "animate_network_temporal",
    "plot_influence_metrics",
    "plot_community_fragmentation",
    "plot_null_model_comparison",
    "plot_coupled_beliefs_response",
    "plot_wilson_confidence_bands",
    "plot_scalar_trajectories",
    "plot_intervention_and_budget",
    "plot_policy_reliability_and_time",
    "plot_paired_adaptive_frozen_comparison",
    "plot_network_views_before_during_after",
    "plot_diagnostic_response_maps",
    "render_network_animation",
    "render_bundle_visualizations",
]
