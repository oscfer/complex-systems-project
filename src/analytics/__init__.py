"""Analytics and Measurement Suite.

Submodules:
    - influence: Attention shares, concentration C_N, effective leaders, Spearman persistence.
    - discrepancy: Population discrepancy d_2, moments, response curves, reversal endpoints.
    - communities: Symmetrized projection G, Leiden modularity optimization, WCC/SCC tracking, ARI.
    - null_models: Degree-preserving edge swaps and weight multiset permutation.
"""

from src.analytics.influence import (
    calculate_attention_matrix,
    calculate_influence_metrics,
    spearman_rank_persistence,
    top_fraction_overlap,
)
from src.analytics.discrepancy import (
    calculate_population_discrepancy,
    calculate_second_moment,
    calculate_moments,
    track_response_curve,
    detect_reversal_endpoint,
    detect_crossing_endpoint,
)
from src.analytics.communities import (
    symmetrize_adjacency,
    detect_communities,
    calculate_partition_stability,
    connected_components_analysis,
    calculate_modularity,
)
from src.analytics.null_models import (
    extract_support_graph,
    generate_degree_preserving_null,
)

__all__ = [
    "calculate_attention_matrix",
    "calculate_influence_metrics",
    "spearman_rank_persistence",
    "top_fraction_overlap",
    "calculate_population_discrepancy",
    "calculate_second_moment",
    "calculate_moments",
    "track_response_curve",
    "detect_reversal_endpoint",
    "detect_crossing_endpoint",
    "symmetrize_adjacency",
    "detect_communities",
    "calculate_partition_stability",
    "connected_components_analysis",
    "calculate_modularity",
    "extract_support_graph",
    "generate_degree_preserving_null",
]
