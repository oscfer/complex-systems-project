"""Batch1 development/search campaign: forecast gate, bounded observed-subset
search, and actual representative/selector continuations.

One stage-configured campaign with ordered phases:

A. Forecast-utility gate on existing development evidence (predeclared metric
   and threshold; failure selects the declared simple-selector fallback
   branch).
B. Bounded recipient-group search per fresh development origin, forecast-only,
   with an estimated Pareto frontier and predeclared representatives.
C. Actual representative trajectories plus simple-selector comparisons at
   matched settings, reusing compatible saved evidence where identity holds.

Completed forecast units are cached per origin, candidate signature and future
seed. Resume reuses completed work; an interrupted simulation repeats and is
not an independent replicate. Counters describe completed logical evaluations,
not an exact accounting of computation lost on abrupt process termination. This module implements no
scientific claim by itself; evidence quality is judged from returned bundles.

Controller-facing work uses only restricted observations. True initial
influence may be reported retrospectively by evaluators; it never enters
candidate proposals, pruning, ranking or per-network selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from src.control.checkpointing import serialize_state
from src.control.contracts import ControlProblem, TargetType, RolloutResult
from src.control.controllers import OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation
from src.control.evaluator import TargetEvaluator
from src.control.observation import SampledPublicObservationModel
from src.control.optimizers.observed_forecasts import ForecastAssumptions, ObservedInitialForecasts
from src.control.optimizers.observed_subsets import SubsetSearchConfig, search_observed_subsets
from src.control.optimizers.search import PolicyCandidate
from src.control.recording import RecorderConfig, RolloutRecorder
from src.config.transformation import resolve_resource_package
from src.control.transformation_study import _runtime_versions, _structure, source_identity
from src.config.transformation import contract_digest
from src.execution.runner import NumpyEncoder, _json_safe


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), cls=NumpyEncoder, sort_keys=True, indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload+'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


@dataclass(frozen=True)
class CampaignConfig:
    """Frozen before launch; every ceiling here is an absolute cap."""
    destination_sign: int = -1
    sizes: tuple[int, ...] = (4, 10)
    package_by_size: tuple[str, ...] = ('broad_diagnostic', 'broad_diagnostic')
    amplitude: float = 2.0
    duration: float = 19.0
    search_origins: tuple[str, ...] = ()
    candidate_limit: int = 16
    search_futures: int = 2
    validation_futures: int = 2
    gate_forecast_futures: int = 2
    forecast_gate_spearman: float = 0.3
    representative_limit: int = 2
    selector_size: int = 4
    wall_time_ceiling_s: float = 4.0*3600.0
    seed: int = 0
    gate_cases: tuple = ()
    gate_is_diagnostic: bool = True
    assumptions: ForecastAssumptions = field(default_factory=ForecastAssumptions)

    def __post_init__(self):
        if self.destination_sign not in (-1, 1):
            raise ValueError('Destination sign must be declared and binary')
        if len(self.sizes) != len(self.package_by_size) or not self.sizes:
            raise ValueError('Each recipient size needs a resource package')
        for size, package in zip(self.sizes, self.package_by_size):
            geometry = resolve_resource_package(package, 40)
            if size > geometry['k'] or size*self.amplitude*self.duration > geometry['exposure_cap'] + 1e-9:
                raise ValueError(f'Size {size} exceeds package {package} permissions')
        if not 0 < self.forecast_gate_spearman <= 1:
            raise ValueError('Forecast gate threshold must lie in (0, 1]')
        if len(set(self.search_origins)) != len(self.search_origins):
            raise ValueError('Search origins must be unique')

    def forecast_ceiling(self) -> int:
        return (len(self.gate_cases)*self.gate_forecast_futures
                + len(self.search_origins)*(self.candidate_limit*self.search_futures
                                            + self.representative_limit*self.validation_futures))


def _role_seed(seed: int, role: str) -> int:
    return int.from_bytes(hashlib.sha256(f'transformation-v1:{seed}:{role}'.encode()).digest()[:8], 'big')


def _origin_state(profile, seed: int):
    from src.config.scenarios import tiered_origin_provider
    return tiered_origin_provider(profile, seed)


def _roster(profile, seed: int, package_id: str) -> tuple[int, ...]:
    geometry = resolve_resource_package(package_id, profile.N)
    roster_rng = np.random.default_rng(_role_seed(seed, 'roster'))
    return tuple(sorted(int(i) for i in roster_rng.permutation(profile.N)[:geometry['accessible_count']]))


def _case_problem(profile, seed: int, package_id: str, destination_sign: int,
                  problem_id: str) -> ControlProblem:
    geometry = resolve_resource_package(package_id, profile.N)
    roster = _roster(profile, seed, package_id)
    return ControlProblem(
        problem_id=problem_id, target_type=TargetType.REVERSAL,
        target_params={'eligibility_mode': 'declared', 'declared_destination_sign': destination_sign,
                       'threshold': 0.5}, N=profile.N, D=1, k=geometry['k'],
        exposure_cap=geometry['exposure_cap'], amplitude_cap=geometry['amplitude_cap'],
        accessible_roster=roster, release_deadline=20.0, holding_duration=100.0,
        horizon=320.0, config_version='persistent-transformation-v1')


def reconstruct_t0_observation(profile, state, seed: int, accessible_roster: tuple[int, ...] | None = None):
    """Rebuild the restricted t0 report exactly as the engine's controller reset
    consumes it: a fresh sampled-public model with the origin observation seed,
    reset draw followed by the scheduled t0 draw. The observation hash is recorded for batch2 identity."""
    model = SampledPublicObservationModel(
        N=state.N, panel_fraction=profile.panel_fraction,
        binary_flip_prob=profile.binary_flip_prob,
        edge_reveal_prob=profile.edge_reveal_prob,
        delivery_delay=profile.delivery_delay_tau_public,
        accessible_roster=accessible_roster,
        rng=np.random.default_rng(_role_seed(seed, 'observation')))
    model.sample_observation(state=state, current_time=0.0)  # controller-reset draw
    return model.sample_observation(state=state, current_time=0.0)  # first delivered report


def observation_identity(observation) -> str:
    payload = {'panel': observation.panel_mask.tolist(),
               'public': observation.public_states.tolist(),
               'edges': sorted((int(i), int(j), int(c))
                               for (i, j), c in observation.observed_edges.items()),
               'roster': list(observation.accessible_roster)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_gate_cases(rows: list[dict], bundle_dirs) -> list[dict]:
    """Build gate cases from the recorded membership/outcome table.

    rows: dicts with origin_id, seed, policy, package_id, recipient_nodes,
    amplitude, duration, provenance (bundle key). Each row is cross-checked
    against the saved unit record: the bundle must contain a valid, eligible,
    non-reused physical unit with the same origin and policy, matching
    expenditure, and the saved success/relapse labels become the actual
    lasting-success ground truth. A row that fails verification is excluded
    with its reason; nothing is fabricated.
    """
    bundle_dirs = [Path(b) for b in bundle_dirs]
    cases = []
    for row in rows:
        excluded = {'origin_id': row['origin_id'], 'policy': row['policy'],
                    'package_id': row['package_id'], 'excluded': True,
                    'exclusion_reason': None}
        unit = None
        for bundle in bundle_dirs:
            manifest_path = bundle/'manifest.json'
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            for key, info in manifest.get('units', {}).items():
                if 'reused_from' in info:
                    continue
                if _sha(bundle/info['path']) != info['sha256']:
                    raise ValueError('Gate unit checksum mismatch')
                record = json.loads((bundle/info['path']).read_text())
                u = record.get('unit', {})
                origin = u.get('origin', {})
                if (u.get('policy') == row['policy']
                        and origin.get('origin_id') == row['origin_id']
                        and u.get('package') == row['package_id']):
                    unit = record
                    excluded['exclusion_reason'] = None
                    break
            if unit is not None:
                break
        if unit is None:
            excluded['exclusion_reason'] = 'no matching saved unit'
            cases.append(excluded)
            continue
        if not (unit.get('is_valid', True) and unit.get('is_eligible', True)):
            excluded['exclusion_reason'] = 'saved unit invalid or ineligible'
            cases.append(excluded)
            continue
        saved_nodes = unit.get('summaries', {}).get('effective_settings', {}).get('initial_nodes')
        if saved_nodes is None or sorted(saved_nodes) != sorted(row['recipient_nodes']):
            excluded['exclusion_reason'] = 'recipient membership mismatch'
            cases.append(excluded)
            continue
        expected = float(row['amplitude'])*len(row['recipient_nodes'])*float(row['duration'])
        if abs(float(unit.get('expenditure', expected)) - expected) > 1e-6:
            excluded['exclusion_reason'] = (
                f"expenditure mismatch: saved {unit.get('expenditure')} vs expected {expected}")
            cases.append(excluded)
            continue
        cases.append({
            'origin_id': row['origin_id'], 'seed': int(row['seed']), 'policy': row['policy'],
            'package_id': row['package_id'],
            'recipient_nodes': [int(n) for n in row['recipient_nodes']],
            'amplitude': float(row['amplitude']), 'duration': float(row['duration']),
            'actual_hold_success': bool(unit.get('success') and not unit.get('relapsed', False)),
            'provenance': f"{row.get('provenance', row['policy'])} [{unit.get('expenditure')} exposure]",
            'excluded': False})
    return cases


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j+1]] == v[order[i]]:
                j += 1
            mean = (i + j)/2 + 1
            for k in range(i, j+1):
                r[order[k]] = mean
            i = j + 1
        return r
    if len(a) < 2:
        return float('nan')
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den = math.sqrt(sum((ra[i]-ma)**2 for i in range(n))*sum((rb[i]-mb)**2 for i in range(n)))
    return num/den if den > 0 else float('nan')


class CampaignTimeLimit(Exception):
    pass


def _cached_forecasts(evaluator, cache_dir, deadline):
    if cache_dir is None:
        return evaluator
    cache_dir = Path(cache_dir)
    def evaluate(candidate, seeds):
        results = []
        for seed in seeds:
            signature = json.dumps([candidate.signature, int(seed)], sort_keys=True)
            path = cache_dir/(hashlib.sha256(signature.encode()).hexdigest()+'.json')
            if path.exists():
                result = RolloutResult(**json.loads(path.read_text()))
            else:
                if deadline is not None and time.monotonic() >= deadline:
                    raise CampaignTimeLimit()
                result = evaluator(candidate, (seed,))[0]
                payload = asdict(result)
                payload['recorder'] = None
                _atomic_json(path, payload)
            results.append(result)
        return results
    return evaluate


def run_forecast_gate(cases, profile, gate_futures: int, threshold: float,
                      destination_sign: int = -1, seed: int = 0, progress: bool = True,
                      origin_seeds: dict | None = None, cache_dir=None, deadline=None) -> dict:
    """Phase A. cases: list of dicts as produced by load_gate_cases.

    Predeclared metric: Spearman rank correlation between forecast estimated
    lasting-success rate and the saved actual lasting-success label across
    development cases. Constant and random-order baselines have expected
    correlation zero, so the positive threshold is the decision-relevant test.
    A failed gate selects the declared fallback; it never enlarges the search
    budget.
    """
    origin_seeds = origin_seeds or {}
    estimates = []
    started = time.monotonic()
    for case in cases:
        seed_value = origin_seeds.get(case['origin_id'], case.get('seed'))
        if seed_value is None:
            raise ValueError(f"No seed available for gate origin {case['origin_id']}")
        state = _origin_state(profile, seed_value)
        problem = _case_problem(profile, seed_value, case['package_id'], destination_sign,
                                problem_id=f"gate_{case['origin_id']}")
        observation = reconstruct_t0_observation(profile, state, seed_value, accessible_roster=problem.accessible_roster)
        evaluator = ObservedInitialForecasts(observation, problem, target_sign=destination_sign,
                                             assumptions=ForecastAssumptions())
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, _stable_int(case['origin_id']+case['policy'])]))
        future_seeds = tuple(int(s) for s in rng.choice(2**31, size=gate_futures, replace=False))
        candidate = PolicyCandidate(
            candidate_id=f"gate_{case['origin_id']}_{case['policy']}",
            target_nodes=tuple(sorted(int(n) for n in case['recipient_nodes'])),
            signs=tuple((int(n), float(destination_sign)) for n in case['recipient_nodes']),
            nominal_amplitude=case['amplitude'], duration=case['duration'], component=0,
            heuristic_name='gate_case', metadata={'information_level': 'restricted_forecast'})
        evaluator = _cached_forecasts(evaluator, None if cache_dir is None else Path(cache_dir)/case['origin_id']/case['policy'], deadline)
        try:
            rollouts = list(evaluator(candidate, future_seeds))
            p_hat = sum(bool(r.success and not r.relapsed) for r in rollouts)/len(rollouts)
            valid = all(r.is_valid and r.is_eligible for r in rollouts)
            error = None
        except CampaignTimeLimit:
            raise
        except Exception as exc:  # execution errors are recorded separately from scientific failure
            p_hat, valid, error = 0.0, False, f'{type(exc).__name__}: {exc}'
        estimates.append({
            'case_id': f"{case['origin_id']}__{case['policy']}",
            'provenance': case['provenance'],
            'actual_hold_success': bool(case['actual_hold_success']),
            'forecast_p_hat': p_hat, 'forecast_valid': valid, 'error': error,
            'future_seeds': [int(s) for s in future_seeds],
            'forecast_assumptions': asdict(ForecastAssumptions())})
        if progress:
            print(f"gate {estimates[-1]['case_id']}: p_hat={p_hat:.2f} "
                  f"actual={estimates[-1]['actual_hold_success']} "
                  f"({time.monotonic()-started:.0f}s elapsed)", flush=True)
    rho = _spearman([e['forecast_p_hat'] for e in estimates],
                    [1.0 if e['actual_hold_success'] else 0.0 for e in estimates])
    return {'kind': 'forecast_utility_gate', 'spearman_threshold': threshold,
            'spearman_rho': None if not np.isfinite(rho) else float(rho),
            'passed': bool(all(e['forecast_valid'] for e in estimates) and np.isfinite(rho) and rho >= threshold), 'cases': estimates,
            'forecast_simulations': len(cases)*gate_futures}


def run_search_phase(origin_specs, config: SubsetSearchConfig, destination_sign: int = -1,
                     progress: bool = True,
                     assumptions: ForecastAssumptions | None = None, cache_dir=None, deadline=None) -> list[dict]:
    """Phase B, forecast-only search per fresh origin.

    origin_specs: dicts with origin_id, seed, profile and package_id (the most
    permissive package of the frozen size set, so every size fits one problem).
    Returns per-origin search records; callers persist them durably.
    """
    outputs = []
    for spec in origin_specs:
        state = _origin_state(spec['profile'], spec['seed'])
        problem = _case_problem(spec['profile'], spec['seed'], spec['package_id'],
                                destination_sign, problem_id=f"search_{spec['origin_id']}")
        observation = reconstruct_t0_observation(spec['profile'], state, spec['seed'], accessible_roster=problem.accessible_roster)
        evaluator = ObservedInitialForecasts(observation, problem, target_sign=destination_sign,
                                             assumptions=assumptions or ForecastAssumptions())
        evaluator = _cached_forecasts(evaluator, None if cache_dir is None else Path(cache_dir)/spec['origin_id'], deadline)
        search_config = replace(config, seed=config.seed + _stable_int(spec['origin_id']) % (2**31))
        result = search_observed_subsets(observation, evaluator, target_sign=destination_sign,
                                         config=search_config)
        outputs.append({
            'origin_id': spec['origin_id'],
            'selected': _candidate_payload(result.selected, result.summaries),
            'representatives': [_candidate_payload(c, result.summaries) for c in result.representatives],
            'frontier': [_candidate_payload(c, result.summaries) for c in result.frontier],
            'candidates': [_candidate_payload(c, result.summaries) for c in result.candidates],
            'validation_summaries': result.validation_summaries,
            'forecast_calls': result.forecast_calls,
            'forecast_rollouts': result.forecast_rollouts,
            'budget_exhausted': result.budget_exhausted,
            'influence_ranks': {str(k): int(v) for k, v in result.influence_ranks.items()},
            'search_seeds': [int(s) for s in result.search_seeds],
            'validation_seeds': [int(s) for s in result.validation_seeds],
            'observation_hash': observation_identity(observation)})
        if progress:
            reps = [list(c.target_nodes) for c in result.representatives]
            print(f"search {spec['origin_id']}: {result.forecast_calls} forecast calls, "
                  f"representatives {reps}", flush=True)
    return outputs


def _candidate_payload(candidate, summaries) -> dict:
    payload = {'candidate_id': candidate.candidate_id,
               'target_nodes': list(candidate.target_nodes),
               'nominal_amplitude': candidate.nominal_amplitude,
               'duration': candidate.duration,
               'heuristic_name': candidate.heuristic_name}
    if summaries:
        for summary in summaries:
            if summary.get('candidate_id') == candidate.candidate_id:
                payload['summary'] = summary
                break
    return payload


def run_actual_unit(profile, seed: int, origin_id: str, contract: dict, memory_mode: str,
                    package_id: str, nodes: tuple[int, ...], amplitude: float,
                    duration: float, destination_sign: int = -1, checkpoint_interval: float = 20.0,
                    checkpoint_dir: Path | None = None, simulation_seed: int | None = None,
                    record_network_frames: bool = True, max_network_frames: int = 100,
                    recorder: RolloutRecorder | None = None) -> dict:
    """One actual controlled continuation for an explicit recipient group.

    Mirrors the shared study runner's identity/roster/seed construction; uses
    the shared engine and event-level evaluator. Empty `nodes` runs the matched
    zero control: same origin, roster, seeds, target and evaluator, no
    actuation. Returns the serialized unit payload (without recorder frames)
    suitable for both the campaign manifest and the companion study bundle."""
    sim_seed = _role_seed(seed, 'simulation') if simulation_seed is None else int(simulation_seed)
    if sim_seed < 0:
        raise ValueError('Simulation seed must be nonnegative')
    state = _origin_state(profile, seed)
    problem = _case_problem(profile, seed, package_id, destination_sign,
                            problem_id=f'{origin_id}_{len(nodes)}n')
    roster = problem.accessible_roster
    target_type = TargetType.REVERSAL
    evaluator = TargetEvaluator.create_for_state(
        target_type, state, 100.0, 320.0, 20.0, problem_id=problem.problem_id,
        policy_id='explicit_group', origin_id=origin_id, target_params=problem.target_params)
    observation_model = SampledPublicObservationModel(
        N=profile.N, accessible_roster=roster, panel_fraction=profile.panel_fraction,
        binary_flip_prob=profile.binary_flip_prob, edge_reveal_prob=profile.edge_reveal_prob,
        delivery_delay=profile.delivery_delay_tau_public,
        rng=np.random.default_rng(_role_seed(seed, 'observation')))
    if nodes:
        controller = OpenLoopPulseController(
            target_nodes=nodes, nominal_amplitude=amplitude, duration=duration,
            signs={int(n): float(destination_sign) for n in nodes}, component=0)
    else:
        controller = ZeroController()
    if recorder is None:
        recorder = RolloutRecorder(RecorderConfig(
            cadence=5.0, record_network_frames=record_network_frames, max_network_frames=max_network_frames,
            retention_window=(20.0, 320.0)))
    started = time.monotonic()
    result, final, ledger = run_controlled_simulation(
        record_target_transitions=True, adaptive_memory=memory_mode == 'adaptive',
        initial_state=state, problem=problem, controller=controller,
        observation_model=observation_model, target_evaluator=evaluator, recorder=recorder,
        sim_rng=np.random.default_rng(sim_seed),
        controller_rng=np.random.default_rng(_role_seed(seed, 'controller')),
        J=profile.J, T=profile.temperature, delta=profile.delta, epsilon=profile.exploration,
        closure=profile.closure, a=profile.prestige_global, c=profile.matching,
        tau_y=profile.tau_public, sample_interval=profile.sample_interval_tau_public)
    output = {
        'origin_id': origin_id,
        'policy': 'zero' if not nodes else 'explicit_group',
        'package': package_id,
        'recipient_nodes': [int(n) for n in nodes], 'amplitude': amplitude,
        'duration': duration, 'memory_mode': memory_mode,
        'is_valid': bool(result.is_valid), 'is_eligible': bool(result.is_eligible),
        'success': bool(result.success), 'relapsed': bool(result.relapsed),
        'completion_time': result.completion_time, 'restricted_time': result.restricted_time,
        'expenditure': result.expenditure, 'distinct_nodes_used': result.distinct_nodes_used,
        'success_through_horizon': bool(result.success and not result.relapsed),
        'terminal_target_violation': result.terminal_target_violation,
        'event_counts': dict(result.event_counts),
        'summaries': {k: v for k, v in result.summaries.items() if k != 'forecast_assumptions'},
        'target_definition': asdict(evaluator.defn),
        'final_state': serialize_state(final),
        'initial_structure': _structure(state), 'final_structure': _structure(final),
        'network_frames': [f.to_dict() for f in result.recorder.network_frames],
        'elapsed_seconds_this_invocation': time.monotonic()-started,
        'geometry': resolve_resource_package(package_id, profile.N),
        'accessible_roster': list(roster),
        'seeds': {role: _role_seed(seed, role) for role in
                  ('roster', 'simulation', 'observation', 'controller')}}
    output['seeds']['simulation'] = sim_seed
    return output


def run_batch1_campaign(out_dir, config: CampaignConfig, profile, contract: dict,
                        gate_cases: list[dict], origin_providers: dict,
                        memory_mode: str = 'adaptive', plan_only: bool = False,
                        resume_bundle: str | Path | None = None, progress: bool = True) -> dict | Path:
    """Single campaign entry point. plan_only resolves the full queue and all
    counts without creating files or simulating.

    origin_providers maps origin_id -> seed for every declared search origin.
    Returns the plan dict (plan_only) or the bundle Path."""
    ceiling = config.forecast_ceiling()
    search_plan = [{'origin_id': o, 'seed': origin_providers[o],
                    'package_id': config.package_by_size[-1]} for o in config.search_origins]
    selector_package = config.package_by_size[config.sizes.index(config.selector_size)]
    actual_plan = []
    for origin_id in config.search_origins:
        seed = origin_providers[origin_id]
        for size in config.sizes:  # representatives: at most one per frozen size
            actual_plan.append({'origin_id': origin_id, 'seed': seed, 'size': size,
                                'package_id': config.package_by_size[config.sizes.index(size)],
                                'kind': 'representative'})
        for size, package in zip(config.sizes, config.package_by_size):
            for name in ('attention', 'low_observed_attention', 'audience_coverage', 'random'):
                actual_plan.append({'origin_id': origin_id, 'seed': seed, 'size': size,
                                    'package_id': package, 'kind': f'selector_{name}'})
        for package in sorted(set(config.package_by_size)):
            actual_plan.append({'origin_id': origin_id, 'seed': seed, 'size': 0,
                                'package_id': package, 'kind': 'matched_zero'})
    plan = {
        'phases': ['forecast_gate', 'observed_search', 'actual_representatives_and_selectors'],
        'gate_cases': len(config.gate_cases),
        'gate_forecast_simulations': len(config.gate_cases)*config.gate_forecast_futures,
        'search_forecast_simulations': len(config.search_origins)*(
            config.candidate_limit*config.search_futures
            + config.representative_limit*config.validation_futures),
        'forecast_ceiling': ceiling,
        'planned_actual_units': len(actual_plan),
        'planned_actual_breakdown': {
            'representatives': len(config.search_origins)*config.representative_limit,
            'simple_selectors_ceiling': len(config.search_origins)*4*len(config.sizes),
            'matched_zeros': len(config.search_origins)*len(set(config.package_by_size))},
        'search_plan': search_plan, 'actual_plan': actual_plan,
        'memory_mode': memory_mode,
        'fallback_branch': 'simple_selector_comparison_only_with_documented_optimizer_negative'}
    if plan_only:
        return plan

    out_dir = Path(out_dir)
    identity = {'source_sha256': source_identity()['source_sha256'],
                'runtime': {'python': platform.python_version(), 'packages': _runtime_versions()},
                'profile': profile.model_dump(), 'contract': contract,
                'origins': origin_providers, 'memory_mode': memory_mode}
    identity = json.loads(json.dumps(identity, cls=NumpyEncoder))
    if resume_bundle is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        bundle = out_dir/f'campaign_v1_{stamp}_{uuid.uuid4().hex[:8]}'
        bundle.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': 2, 'identity': identity, 'status': 'running',
                    'config': json.loads(json.dumps(asdict(config), cls=NumpyEncoder)),
                    'plan': plan, 'started_utc': stamp,
                    'recomputed_tail_units': 0, 'wall_time_s': 0.0, 'forecast_simulations_completed': 0,
                    'actual_units_completed': 0, 'gate': None, 'search': {}, 'actual': {}}
        _atomic_json(bundle/'manifest.json', manifest)
    else:
        bundle = Path(resume_bundle)
        manifest = json.loads((bundle/'manifest.json').read_text())
        if manifest.get('identity') != identity:
            raise ValueError('Campaign source/runtime/origin identity differs; cannot resume')
        frozen = json.loads(json.dumps(asdict(config), cls=NumpyEncoder))
        if manifest['config'] != frozen:
            raise ValueError('Campaign resume configuration mismatch; start a new result series')

    invocation_started = time.monotonic()

    def time_expired():
        return time.monotonic() - invocation_started >= config.wall_time_ceiling_s

    def stop_for_time():
        manifest['cached_forecast_units'] = len(list((bundle/'forecast_cache').rglob('*.json')))
        manifest['status'] = 'incomplete_time_ceiling'
        _atomic_json(bundle/'manifest.json', manifest)
        return bundle

    # Phase A: forecast gate. Completed case records are reused; an interrupted
    # case is recomputed once and charged as recomputed tail.
    if manifest['gate'] is None:
        pending_cases = list(config.gate_cases)
    else:
        done = {c['case_id'] for c in manifest['gate'].get('cases', [])}
        pending_cases = [c for c in config.gate_cases
                         if f"{c['origin_id']}__{c['policy']}" not in done]
    for pending_case in pending_cases:
        if time_expired():
            return stop_for_time()
        started = time.monotonic()
        try:
            gate = run_forecast_gate([pending_case], profile, config.gate_forecast_futures,
                                     config.forecast_gate_spearman, config.destination_sign,
                                     config.seed, progress=progress,
                                     origin_seeds=origin_providers, cache_dir=bundle/'forecast_cache/gate',
                                     deadline=invocation_started+config.wall_time_ceiling_s)
        except CampaignTimeLimit:
            return stop_for_time()
        if manifest['gate'] is None:
            manifest['gate'] = gate
        else:
            manifest['gate']['cases'].extend(gate['cases'])
            _refresh_gate_verdict(manifest['gate'], config.forecast_gate_spearman)
        manifest['forecast_simulations_completed'] += config.gate_forecast_futures
        manifest['wall_time_s'] = _elapsed(manifest) + time.monotonic()-started
        manifest['status'] = 'incomplete'
        _atomic_json(bundle/'manifest.json', manifest)

    # Branch on the frozen gate verdict. A failure diverts to the declared
    # fallback: no search, but the simple-selector comparison and matched
    # zeros still run as the documented optimizer negative.
    if manifest['gate'] is None:
        manifest['gate'] = {'passed': False, 'spearman_rho': None, 'cases': [], 'forecast_simulations': 0}
    fallback = not config.gate_is_diagnostic and not manifest['gate']['passed']
    manifest['gate']['interpretation'] = 'Descriptive cross-coverage diagnostic; not validation of within-size ranking.'
    if fallback:
        manifest['decision'] = ('Forecast gate failed the predeclared threshold; executing the '
                                'declared simple-selector fallback branch. No recipient search '
                                'is commissioned on an unvalidated surrogate.')
        for item in search_plan:
            manifest['search'].setdefault(item['origin_id'], {'fallback': True})
        manifest['status'] = 'incomplete'
        _atomic_json(bundle/'manifest.json', manifest)
        if progress:
            print(f"Gate FAILED (rho={manifest['gate']['spearman_rho']}); "
                  'fallback branch: simple-selector comparison only.', flush=True)
    else:
        if progress:
            print(f"Gate diagnostic rho={manifest['gate']['spearman_rho']}; testing search experimentally.",
                  flush=True)

        # Phase B: bounded forecast-only search per origin.
        for item in search_plan:
            origin_id = item['origin_id']
            if origin_id in manifest['search']:
                continue
            if time_expired():
                return stop_for_time()
            started = time.monotonic()
            spec = {**item, 'profile': profile}
            try:
                record = run_search_phase([spec], SubsetSearchConfig(
                    sizes=config.sizes, amplitude=config.amplitude, duration=config.duration,
                    candidate_limit=config.candidate_limit, search_futures=config.search_futures,
                    validation_futures=config.validation_futures,
                    reliability_target=0.8, seed=config.seed,
                    representative_limit=config.representative_limit),
                    config.destination_sign, progress=progress,
                    assumptions=config.assumptions, cache_dir=bundle/'forecast_cache/search',
                    deadline=invocation_started+config.wall_time_ceiling_s)[0]
            except CampaignTimeLimit:
                return stop_for_time()
            manifest['search'][origin_id] = record
            manifest['forecast_simulations_completed'] += record['forecast_rollouts']
            manifest['wall_time_s'] += time.monotonic()-started
            manifest['status'] = 'incomplete'
            _atomic_json(bundle/'manifest.json', manifest)

    # Phase C: actual continuations for representatives and simple selectors.
    # Representatives come from the frozen search record; simple selectors are
    # the search's initial heuristic proposals at the predeclared selector
    # size, run as actual continuations at matched settings. In the declared
    # gate-failure fallback there is no search: the four heuristic selectors
    # and matched zeros still run, as the documented optimizer negative.
    # Repeated node sets deduplicate through the manifest keys.
    for item in search_plan:
        origin_id, seed = item['origin_id'], item['seed']
        record = manifest['search'][origin_id]
        queue = _actual_queue(config, record, profile, origin_id, seed)
        for nodes, amplitude, duration, kind in queue:
            if time_expired():
                manifest['status'] = 'incomplete_time_ceiling'
                _atomic_json(bundle/'manifest.json', manifest)
                if progress:
                    print('Wall-time ceiling reached; batch is resumable.', flush=True)
                return bundle
            nodes = tuple(sorted(int(n) for n in nodes))
            size = len(nodes)
            if size and size not in config.sizes:
                raise ValueError(f'Actual queue size {size} outside frozen sizes')
            package = (config.package_by_size[config.sizes.index(size)] if size
                       else kind.removeprefix('zero_'))
            key = f"{origin_id}__{package}__{'_'.join(map(str, nodes)) or 'zero'}"
            if key in manifest['actual']:
                roles = manifest['actual'][key].setdefault('roles', [manifest['actual'][key]['kind']])
                if kind not in roles:
                    roles.append(kind)
                    _atomic_json(bundle/'manifest.json', manifest)
                continue
            started = time.monotonic()
            unit = run_actual_unit(profile, seed, origin_id, contract, memory_mode, package,
                                   tuple(sorted(nodes)), amplitude, duration,
                                   config.destination_sign)
            path = Path('units')/(key+'.json')
            _atomic_json(bundle/path, unit)
            manifest['actual'][key] = {'path': str(path), 'sha256': _sha(bundle/path),
                                       'kind': kind, 'roles': [kind], 'recipient_nodes': unit['recipient_nodes'],
                                       'origin_id': origin_id, 'package_id': package,
                                       'seed': seed,
                                       'success': unit['success'], 'relapsed': unit['relapsed'],
                                       'success_through_horizon': unit['success_through_horizon']}
            manifest['actual_units_completed'] += 1
            manifest['wall_time_s'] = _elapsed(manifest) + time.monotonic()-started
            manifest['status'] = 'incomplete'
            _atomic_json(bundle/'manifest.json', manifest)
            _write_study_bundle(bundle, manifest, profile, contract, memory_mode)

    manifest['cached_forecast_units'] = len(list((bundle/'forecast_cache').rglob('*.json')))
    manifest['status'] = 'complete_fallback' if fallback else 'complete'
    _atomic_json(bundle/'manifest.json', manifest)
    _write_study_bundle(bundle, manifest, profile, contract, memory_mode)
    return bundle


def _actual_queue(config: CampaignConfig, record: dict, profile, origin_id: str,
                  seed: int) -> list[tuple]:
    """Phase C queue: search representatives plus the four heuristic selector
    proposals at matched settings, then matched zeros for every package in
    use. In the declared gate-failure fallback (no search), the four
    heuristic selectors are computed directly from the restricted t0
    observation — they need no surrogate."""
    from src.control.optimizers.observed_subsets import _proposal_orders
    queue = []
    queue.extend((rep['target_nodes'], config.amplitude, config.duration, 'representative')
                 for rep in record.get('representatives', []))
    for size, package in zip(config.sizes, config.package_by_size):
        state = _origin_state(profile, seed)
        roster = _roster(profile, seed, package)
        observation = reconstruct_t0_observation(profile, state, seed, accessible_roster=roster)
        rng = np.random.default_rng(config.seed + _stable_int(origin_id) % (2**31))
        for name, order in _proposal_orders(observation, rng)[3]:
            if len(order) < size:
                raise ValueError('Selector size exceeds accessible population')
            queue.append((tuple(sorted(order[:size])), config.amplitude, config.duration,
                          f'selector_{name}_{size}'))
    queue.extend(((), 0.0, 0.0, f'zero_{package}') for package in sorted(set(config.package_by_size)))
    return queue


def _write_study_bundle(campaign_bundle: Path, manifest: dict, profile,
                        contract: dict, memory_mode: str) -> Path:
    """Companion study-format bundle so the shared report/animation renderers
    consume campaign actuals unchanged. Origins are serialized exactly as the
    shared study runner would (same origin provider, seeds and profile), so
    matched zeros can be imported from a prior bundle under the established
    hash-bound review. Imported zeros must be provided by the caller; this
    writer only guarantees the format and checksums."""
    from src.config.scenarios import tiered_origin_provider
    study = campaign_bundle/'study'
    origins, units, queue = {}, {}, []
    for key, info in sorted(manifest['actual'].items()):
        unit = json.loads((campaign_bundle/info['path']).read_text())
        origin_id, package = info['origin_id'], info['package_id']
        nodes = tuple(info['recipient_nodes'])
        policy = 'zero' if not nodes else 'explicit_'+'_'.join(map(str, nodes))
        queue.append({'key': key,
                      'origin': {'origin_id': origin_id, 'seed': info.get('seed'),
                                 'cell_id': None},
                      'policy': policy, 'package': package})
        path = Path('units')/(key+'.json')
        record = {**unit,
                  'unit': {'key': key,
                           'origin': {'origin_id': origin_id, 'seed': info.get('seed'),
                                      'cell_id': None},
                           'policy': policy, 'package': package},
                  'policy_id': policy, 'origin_id': origin_id}
        _atomic_json(study/path, record)
        units[key] = {'path': str(path), 'sha256': _sha(study/path)}
        if origin_id not in origins:
            state = _origin_state(profile, info['seed'])
            origin_path = Path('origins')/(origin_id+'.json')
            _atomic_json(study/origin_path, serialize_state(state))
            origins[origin_id] = {'path': str(origin_path), 'sha256': _sha(study/origin_path)}
    config = {
        'runtime': {'python': platform.python_version(), 'architecture': platform.machine(),
                    'packages': _runtime_versions()},
        'contract_sha256': contract_digest(),
        'source_sha256': source_identity()['source_sha256'],
        'target_kind': 'state', 'memory_mode': memory_mode, 'stage': 'development',
        'smoke': False, 'N': profile.N, 'initial_conditions': 'tiered_reference',
        'notebook_code_digest': None, 'horizon': 320.0, 'release_deadline': 20.0,
        'holding_duration': 100.0, 'checkpoint_interval': 20.0,
        'profile': profile.model_dump(), 'queue': queue}
    study_manifest = {
        'schema_version': 1, 'status': ('complete' if manifest['status'] in ('complete','complete_fallback') else 'incomplete'), 'config': config,
        'contract': contract, 'source': source_identity(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform(), 'packages': _runtime_versions()},
        'label': 'development_not_confirmation', 'units': units, 'origins': origins,
        'physical_continuations_completed': len(units),
        'campaign_bundle': str(Path(campaign_bundle).resolve())}
    _atomic_json(study/'manifest.json', study_manifest)
    return study


def _elapsed(manifest) -> float:
    return float(manifest.get('wall_time_s') or 0.0)


def _refresh_gate_verdict(gate: dict, threshold: float) -> None:
    rho = _spearman([c['forecast_p_hat'] for c in gate['cases']],
                    [1.0 if c['actual_hold_success'] else 0.0 for c in gate['cases']])
    gate['spearman_rho'] = None if not np.isfinite(rho) else float(rho)
    gate['passed'] = bool(all(c['forecast_valid'] for c in gate['cases']) and np.isfinite(rho) and rho >= threshold)
