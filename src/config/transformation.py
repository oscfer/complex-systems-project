"""Versioned synthetic origins for the persistent-transformation study.

Initialization only; all subsequent dynamics remain in the shared simulator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.config.profiles import RunProfile
from src.core.state import State

CONTRACT_PATH = Path(__file__).resolve().parents[2] / 'research/experiments/persistent_transformation_v1.json'


def load_transformation_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def contract_digest() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def build_transformation_origin(profile: RunProfile, seed: int, cell_id: str) -> State:
    """Construct one origin without screening or simulating its outcome.

    Independent named RNG streams prevent attention/opportunity settings from
    silently changing initial stances, traits or core membership in contrasts.
    """
    if profile.D != 1 or profile.state_mode != 'binary' or profile.N < 8:
        raise ValueError('Transformation v1 requires binary D=1 and N>=8')
    if profile.memory_capacity_per_row < 20:
        raise ValueError('Transformation v1 needs capacity >=20 initial memory tokens')
    contract = load_transformation_contract()
    cells = {c['id']: c for c in contract['cells']}
    if cell_id not in cells:
        raise ValueError(f'Unknown transformation cell: {cell_id}')
    cell = cells[cell_id]
    streams = np.random.SeedSequence(seed).spawn(4)
    trait_rng, stance_rng, memory_rng, core_rng = [np.random.default_rng(s) for s in streams]
    state = State.create_realism_state(
        N=profile.N, D=1, state_mode='binary', J=profile.J,
        K_nominal=profile.K_nominal, tau_x_nominal=profile.tau_private_nominal,
        nu_nominal=profile.nu_nominal, row_cap=profile.memory_capacity_per_row,
        num_groups=4, between_eligibility=cell['between_affinity'], rng=trait_rng,
    )
    n = profile.N
    order = stance_rng.permutation(n)
    state.y.fill(-1.)
    state.y[order[:round(0.8*n)]] = 1.
    state.x = state.y.copy()
    state.x[stance_rng.permutation(n)[:round(0.1*n)]] *= -1.
    core = np.zeros(n, dtype=bool)
    core[core_rng.permutation(n)[:max(1,round(0.2*n))]] = True
    groups = state.opportunity_groups
    affinity = np.where(groups[:,None] == groups[None,:], 1., cell['between_affinity'])
    np.fill_diagonal(affinity, 0.)
    state.opportunity_matrix = affinity
    mass = cell['core_prior_mass']
    prior = np.where(core, mass/core.sum(), (1.-mass)/(n-core.sum()))
    for i in range(n):
        weights = prior * affinity[i]
        state.W[i] = memory_rng.multinomial(20, weights/weights.sum()).astype(np.uint32)
    state.scenario_metadata = {
        'source': 'synthetic_transformation_v1', 'contract_version': contract['contract_version'],
        'contract_sha256': contract_digest(), 'seed': int(seed), 'cell': dict(cell),
        'core_indices': np.flatnonzero(core).tolist(), 'initialization': 'constructed_not_equilibrated',
        'rng_stream_order': ['traits_groups','stances','memory','core'],
    }
    state.validate()
    return state


def resolve_resource_package(package_id: str, n: int) -> dict:
    if n < 8:
        raise ValueError('Transformation v1 requires N>=8')
    packages = {p['id']: p for p in load_transformation_contract()['packages']}
    if package_id not in packages:
        raise ValueError(f'Unknown resource package: {package_id}')
    p = packages[package_id]
    return {'id': package_id, 'accessible_count': max(1,round(p['accessible_fraction']*n)),
            'k': max(1,round(p['distinct_fraction']*n)),
            'exposure_cap': p['exposure_per_node']*n, 'amplitude_cap': p['amplitude_cap']}
