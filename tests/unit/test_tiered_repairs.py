"""Regression checks for the Tiered-population repair path."""
import numpy as np
import pytest

from src.config.scenarios import build_tiered_origin, load_tiered_spec
from src.control.campaign import _state_origin_hash, load_origin_snapshot, save_origin_snapshot
from src.io.results import make_campaign_bundle_dir


def test_fresh_bundle_is_distinct_and_resume_is_explicit(tmp_path):
    first = make_campaign_bundle_dir(tmp_path, "tiered", timestamp="20260924_120000", resume=False)
    second = make_campaign_bundle_dir(tmp_path, "tiered", timestamp="20260924_120000", resume=False)
    assert first != second
    assert make_campaign_bundle_dir(first, "tiered", resume=True) == first


def test_scenario_attention_mass_and_provenance():
    spec = load_tiered_spec()
    state = build_tiered_origin(spec, seed=42, N=500)
    core = np.isin(state.blocks, [1, 2, 3])
    assert 0.64 < state.W[:, core].sum() / state.W.sum() < 0.76
    assert state.scenario_metadata["jurisdictions"]
    assert len(state.scenario_metadata["roles"]) == state.N


def test_origin_identity_and_snapshot_include_scenario(tmp_path):
    spec = load_tiered_spec()
    state = build_tiered_origin(spec, seed=42, N=40)
    clone = state.copy()
    assert clone.scenario_metadata == state.scenario_metadata
    path = tmp_path / "origin.npz"
    save_origin_snapshot(state, str(path))
    restored = load_origin_snapshot(str(path))
    assert restored.scenario_metadata == state.scenario_metadata
    assert _state_origin_hash(restored) == _state_origin_hash(state)
    changed = state.copy()
    changed.K = np.ones(state.N)
    assert _state_origin_hash(changed) != _state_origin_hash(state)


def _rollout(origin: str, future: int, success: bool):
    from src.control.contracts import RolloutResult
    return RolloutResult(
        problem_id="p", policy_id="p", origin_id=origin, future_id=future,
        is_valid=True, is_eligible=True, success=success,
        completion_time=1.0 if success else None, restricted_time=1.0,
        expenditure=0.0, distinct_nodes_used=0, relapsed=False,
        terminal_target_violation=0.0, event_counts={},
    )


def test_nested_futures_do_not_certify_one_origin():
    from src.control.campaign import summarize_rollout_reliability
    result = summarize_rollout_reliability([_rollout("one", 0, True), _rollout("one", 1, True)], 0.8)
    assert result["status"] != "reliability_certified"
    assert result["sustained_status"] != "reliability_certified"


def test_nested_futures_estimate_is_origin_weighted():
    from src.control.campaign import summarize_rollout_reliability
    observations = [_rollout("one", i, True) for i in range(10)] + [_rollout("two", 0, False)]
    result = summarize_rollout_reliability(observations, 0.8)
    assert result["success_probability"] == pytest.approx(0.5)


def test_scenario_spec_contents_change_preparation_identity(tmp_path):
    import json
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.config.scenarios import tiered_origin_provider
    from src.control.campaign import _preparation_fingerprint
    spec_path = tmp_path / "scenario.json"
    spec = load_tiered_spec()
    spec_path.write_text(json.dumps(spec))
    profile = LIGHT_TIERED.model_copy(update={"scenario_spec_path": str(spec_path)})
    before = _preparation_fingerprint(profile, tiered_origin_provider)
    spec["core_attention_share"] = 0.6
    spec_path.write_text(json.dumps(spec))
    after = _preparation_fingerprint(profile, tiered_origin_provider)
    assert before != after


def test_attention_sensitivity_responds_to_configuration():
    spec = load_tiered_spec()
    observed = []
    for share in (0.6, 0.8):
        variant = {**spec, "core_attention_share": share}
        state = build_tiered_origin(variant, seed=42, N=500)
        observed.append(state.W[:, state.blocks <= 3].sum() / state.W.sum())
    assert 0.55 < observed[0] < 0.65
    assert 0.75 < observed[1] < 0.85
    assert observed[1] - observed[0] > 0.15


def test_scenario_preparation_preserves_tied_worlds(tmp_path):
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.config.scenarios import tiered_origin_provider
    from src.control.campaign import prepare_origins
    # The F2 joint role/jurisdiction redesign changed the scenario RNG stream,
    # so the historical seed 8800/N=12 no longer happens to contain a tied
    # public majority. The test's intent is preservation of tied worlds as
    # ineligible, not that one specific seed yields a tie: search a small
    # deterministic seed window for a configuration that still produces at
    # least one ineligible origin.
    prepared = None
    for seed_base in range(8800, 8824):
        profile = LIGHT_TIERED.model_copy(update={"N": 12, "n_origins": 8, "seed_base": seed_base})
        prep = prepare_origins(profile, snapshot_dir=str(tmp_path), origin_provider=tiered_origin_provider)
        if prep.summary["n_ineligible"] >= 1:
            prepared = (profile, prep)
            break
    assert prepared is not None, "seed window contained no tied public-majority origin"
    profile, prep = prepared
    assert prep.summary["n_prepared"] == 8
    assert prep.summary["n_ineligible"] >= 1
    assert prep.summary["n_attempted"] == 8
    assert all(origin.state.t == 0 for origin in prep.origins)
    assert len(list(tmp_path.glob("*.npz"))) == 8
    assert all("tier" in x and "jurisdiction" in x for x in prep.summary["origin_descriptives"].values())


def test_broad_diagnostic_uses_independent_matched_origin():
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.config.scenarios import tiered_origin_provider
    from src.control.transformation_diagnostics import run_tiered_positive_control
    profile = LIGHT_TIERED.model_copy(update={
        "N": 12, "horizon": 5.0, "release_deadline": 1.0,
        "holding_duration": 1.0, "diagnostic_origins": 1,
    })
    result = run_tiered_positive_control(profile, tiered_origin_provider)
    zero, pulse = result["rows"]
    assert [zero["arm"], pulse["arm"]] == ["zero", "broad_pulse"]
    assert zero["origin_id"] == pulse["origin_id"]
    assert zero["sim_seed"] == pulse["sim_seed"]
    assert zero["origin_seed"] == pulse["origin_seed"]
    assert zero["origin_id"].startswith("origin_diag_")


def test_full_encounter_matrix_survives_all_state_paths(tmp_path):
    from src.control.checkpointing import serialize_state, deserialize_state
    from src.core.memory import MemoryManager
    state = build_tiered_origin(load_tiered_spec(), seed=42, N=40)
    assert state.opportunity_matrix.shape == (40, 40)
    assert np.all(np.diag(state.opportunity_matrix) == 0)
    assert np.any(state.opportunity_matrix == 0.05)
    np.testing.assert_array_equal(state.copy().opportunity_matrix, state.opportunity_matrix)
    np.testing.assert_array_equal(deserialize_state(serialize_state(state)).opportunity_matrix, state.opportunity_matrix)
    checkpoint = tmp_path / "state.npz"
    state.save_checkpoint(checkpoint)
    from src.core.state import State
    np.testing.assert_array_equal(State.load_checkpoint(checkpoint).opportunity_matrix, state.opportunity_matrix)
    snap = tmp_path / "origin.npz"
    save_origin_snapshot(state, str(snap))
    np.testing.assert_array_equal(load_origin_snapshot(str(snap)).opportunity_matrix, state.opportunity_matrix)
    mm = MemoryManager(N=state.N, initial_W=state.W, C_matrix=state.opportunity_matrix,
                       opportunity_groups=state.opportunity_groups)
    np.testing.assert_array_equal(mm.C_matrix, state.opportunity_matrix)
    changed = state.copy()
    changed.opportunity_matrix[0, 1] = 0.2
    assert _state_origin_hash(changed) != _state_origin_hash(state)


def test_scenario_encounter_matrix_checkpoint_continuation():
    from src.control.contracts import ControlProblem, TargetType
    from src.control.controllers import ZeroController
    from src.control.engine import run_controlled_simulation
    from src.control.evaluator import TargetEvaluator
    from src.control.observation import SampledPublicObservationModel
    state = build_tiered_origin(load_tiered_spec(), seed=111, N=12)
    problem = ControlProblem(
        problem_id="scenario_c", target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.5, "eligibility_mode": "public_majority"},
        N=12, D=1, k=3, exposure_cap=10.0, amplitude_cap=1.0,
        release_deadline=2.0, holding_duration=1.0, horizon=6.0,
        accessible_roster=tuple(range(12)),
    )
    def fresh():
        return (
            ZeroController(),
            SampledPublicObservationModel(N=12, accessible_roster=problem.accessible_roster,
                                           rng=np.random.default_rng(1111)),
            TargetEvaluator.create_for_state(problem.target_type, state, 1.0, 6.0, 2.0,
                                             target_params=problem.target_params),
        )
    ctl, obs, evaluator = fresh()
    full, full_state, _ = run_controlled_simulation(
        initial_state=state.copy(), problem=problem, controller=ctl,
        observation_model=obs, target_evaluator=evaluator,
        sim_rng=np.random.default_rng(2222), controller_rng=np.random.default_rng(3333),
    )
    ctl, obs, evaluator = fresh()
    partial, _, _ = run_controlled_simulation(
        initial_state=state.copy(), problem=problem, controller=ctl,
        observation_model=obs, target_evaluator=evaluator,
        sim_rng=np.random.default_rng(2222), controller_rng=np.random.default_rng(3333),
        checkpoint_at=2.5, stop_at_checkpoint=True,
    )
    assert partial.checkpoint is not None
    np.testing.assert_array_equal(partial.checkpoint.memory_dict["C_matrix"], state.opportunity_matrix)
    resumed, resumed_state, _ = run_controlled_simulation(
        initial_state=state.copy(), problem=problem, controller=ZeroController(),
        resume_checkpoint=partial.checkpoint,
    )
    np.testing.assert_array_equal(full_state.W, resumed_state.W)
    np.testing.assert_array_equal(full_state.x, resumed_state.x)
    np.testing.assert_array_equal(full_state.y, resumed_state.y)
    assert full.event_counts == resumed.event_counts


def test_scenario_resume_requires_exact_bundle_path(tmp_path):
    from src.config.tiered_profiles import LIGHT_TIERED
    from src.config.scenarios import tiered_origin_provider
    from src.control.campaign import run_campaign
    with pytest.raises(ValueError, match="exact existing bundle"):
        run_campaign(LIGHT_TIERED, campaign_id="tiered", out_dir=str(tmp_path),
                     resume=True, origin_provider=tiered_origin_provider)
    assert not list(tmp_path.iterdir())


def test_tiered_profile_manifest_round_trip_accepts_time_integrated_budget():
    from src.config.tiered_profiles import TieredProfile, LIGHT_TIERED
    from src.config.control_profiles import preflight_control_profile
    parsed = TieredProfile.model_validate(LIGHT_TIERED.to_manifest())
    assert parsed.exposure_cap_normalized == 2.0
    assert preflight_control_profile(parsed)["exposure_cap"] == 2.0 * parsed.N * parsed.J * parsed.amplitude_cap_over_J
