"""Emergent & Collective Order Simulator (Modular3).

A modular simulation framework for emergent social order, belief dynamics,
interaction memory, and institutional transitions across five scientific research stages:

Simulation Stages Architecture:
------------------------------
Stage 0: Baseline Calibration & Cost Timing
    - Benchmarking event rates and memory footprints (N=256, H=10).
    - Separation of kernel execution time from compilation, setup, and I/O.
    - Modules: src.core.scheduler, src.core.heat_baths, src.config.schema.

Stage 1: Core Emergence & Pilot Exploration
    - Unperturbed trajectory exploration across discrete and continuous beliefs.
    - Stochastic Glauber/inverse-CDF heat baths and adaptive memory accumulation.
    - Modules: src.core.state, src.core.fields, src.core.memory, src.core.couplings.

Stage 2: Extension Campaigns & Controlled Perturbations
    - Systematic perturbations: field pulses, tie disruptions, multi-population contact.
    - Collective action participation games and institutional threshold holding.
    - Modules: src.experiment.interventions, src.experiment.generators, src.core.participation.

Stage 3: Adaptive Learning & Q-Learning Comparisons
    - Experience-accumulating Q-tables for public stance selection with TD updates.
    - Softmax decision policies compared against instantaneous thermal baselines.
    - Modules: src.core.q_learning, src.core.participation.

Stage 4: Measurement, Analytics & Neural Diagnostics
    - Network topology, Leiden community detection, degree-preserving null models.
    - Influence concentration, Spearman rank persistence, physical discrepancy d_2.
    - Modules: src.analytics.influence, src.analytics.discrepancy,
               src.analytics.communities, src.analytics.null_models.
"""

__version__ = "3.0.0"
