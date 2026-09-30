"""Smoke check of the control engine through the full campaign pipeline.

Usage:
    python tools/run_control_smoke.py

Demonstrates the reconciled shared physics between preparation (TrajectoryRunner)
and control (run_controlled_simulation): explicit prestige a, matching c, public
timescale tau_y, and coupling J forwarded with no silent defaults. Prepares one
tiny established origin, then runs a zero-control rollout and one static-pulse
rollout through the control engine and prints the target-evaluator outcomes.

The script also exercises the declared ControlCampaignSpec and eligibility
verdict, the observable policy search and baselines, and the full
light-allocation campaign pipeline at tiny scale (preparation -> baselines ->
search -> selection -> fresh-origin adaptive/frozen evaluation) writing one
results bundle.

Control-clock convention: the control clock restarts at 0 on the established
snapshot.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.profiles import RunProfile, build_realism_runner
from src.control.contracts import ControlProblem, TargetType
from src.control.controllers import OpenLoopPulseController, ZeroController
from src.control.engine import run_controlled_simulation

PROFILE = RunProfile(
    topic="campaign", tier="light", N=12, D=1, horizon=24.0,
    establishment_time=8.0, n_origins=1, seed_base=8800,
)
SEED = 8801


def main() -> None:
    print("=== control engine on one established origin ===")
    print(
        f"profile: N={PROFILE.N}, horizon={PROFILE.horizon}, "
        f"establishment={PROFILE.establishment_time}, J={PROFILE.J}, "
        f"T={PROFILE.temperature}, delta={PROFILE.delta}"
    )
    print(
        "engine shared settings (explicit, matching the principal profile): "
        "a=0.0 (prestige), c=1.0 (matching), tau_y=1.0 (public), J=1.0"
    )

    runner = build_realism_runner(PROFILE, origin_seed=SEED)
    traj = runner.run(horizon=PROFILE.establishment_time)
    established = traj.final_state
    m_y0 = float(np.mean(established.y[:, 0]))
    m_x0 = float(np.mean(established.x[:, 0]))
    print(
        f"established origin: m_y={m_y0:+.3f}, m_x={m_x0:+.3f}, "
        f"total memory={int(established.W.sum())}"
    )

    rng = np.random.default_rng(4242)
    roster = tuple(sorted(rng.choice(PROFILE.N, size=max(1, PROFILE.N // 4), replace=False).tolist()))
    print(f"accessible roster (25%): {roster}")

    from src.control.campaign import ControlCampaignSpec
    from src.control.evaluator import TargetEvaluator

    spec = ControlCampaignSpec(
        target_type="reversal",
        target_params={"threshold": 0.5},
        measured_observables=["public_means", "private_means"],
        tolerances={"alignment": 0.5},
        holding_duration=4.0,
        horizon=18.0,
        release_deadline=8.0,
        reliability_target=0.8,
        task_information="reversal targeting opposite of initial orientation"
    )

    print("\n--- Declared Campaign Spec ---")
    print(f"{spec}")

    evaluator = TargetEvaluator.create_for_state(
        initial_state=established,
        target_type=TargetType.REVERSAL,
        target_params=spec.target_params,
        holding_duration=spec.holding_duration,
        release_deadline=spec.release_deadline,
        horizon=spec.horizon,
    )
    print(f"reversal-eligibility verdict: {evaluator.defn.is_eligible}")
    if not evaluator.defn.is_eligible:
        print(f"reason: {evaluator.defn.ineligibility_reason}")

    problem = ControlProblem(
        problem_id="m1_smoke",
        target_type=TargetType.REVERSAL,
        target_params=spec.target_params,
        N=PROFILE.N,
        D=1,
        k=3,
        exposure_cap=40.0,
        amplitude_cap=4.0,
        release_deadline=spec.release_deadline,
        holding_duration=spec.holding_duration,
        horizon=spec.horizon,
        accessible_roster=roster,
        proposal_seed=11,
        search_seed=10,
    )

    def run_one(label: str, controller, seed: int):
        res, final_state, ledger = run_controlled_simulation(
            initial_state=established,
            problem=problem,
            controller=controller,
            sim_rng=np.random.default_rng(seed),
            controller_rng=np.random.default_rng(seed + 1),
        )
        m_y_end = float(np.mean(final_state.y[:, 0]))
        m_x_end = float(np.mean(final_state.x[:, 0]))
        print(f"\n--- {label} ---")
        print(f"  eligible: {res.is_eligible}  success: {res.success}")
        print(
            f"  completion_time (T_G): {res.completion_time}  "
            f"restricted_time: {res.restricted_time:.2f}"
        )
        print(
            f"  expenditure: {res.expenditure:.2f} / {problem.exposure_cap}  "
            f"distinct nodes: {res.distinct_nodes_used} / {problem.k}  "
            f"relapsed: {res.relapsed}"
        )
        print(f"  endpoint: m_y={m_y_end:+.3f}, m_x={m_x_end:+.3f}")
        print(f"  event counts: {res.event_counts}")
        return res

    run_one("zero-control baseline", ZeroController(), seed=100)

    s_star = -1.0 if m_y0 >= 0.0 else 1.0
    pulse_nodes = roster[:2]
    pulse = OpenLoopPulseController(
        target_nodes=pulse_nodes,
        nominal_amplitude=2.0,
        duration=3.0,
        signs={n: s_star for n in pulse_nodes},
        component=0,
    )
    print(f"\npulse policy: nodes={pulse_nodes}, amplitude=2.0, "
          f"duration=3.0, target sign s*={s_star:+.0f}")
    run_one("static pulse on accessible roster", pulse, seed=200)

    from src.control.controllers import ObservableSelectorPolicy
    
    random_policy = ObservableSelectorPolicy(
        selector="random", nominal_amplitude=2.0, duration=3.0, quota=0.0, is_feedback=False
    )
    run_one("random targeting baseline (observable)", random_policy, seed=300)

    attention_policy = ObservableSelectorPolicy(
        selector="reported_attention", nominal_amplitude=2.0, duration=3.0, quota=0.0, is_feedback=False
    )
    run_one("observed-attention baseline (observable)", attention_policy, seed=400)

    # Run reduced search
    print("\n=== Observable Policy Search ===")
    from src.control.optimizers.search import ObservablePolicySearchOptimizer

    optimizer = ObservablePolicySearchOptimizer(
        problem=problem,
        candidate_cap=3,
        futures_per_candidate=1,
        selectors=("reported_attention", "random"),
        observed_slot_quotas=(0.0,),
        feedback_choices=(False,),
        nominal_amplitudes=(2.0,),
        durations=(3.0,),
    )
    
    print(f"Running mini search on {optimizer.candidate_cap} candidates, {optimizer.futures_per_candidate} future(s) each...")
    opt_result = optimizer.optimize(
        training_origins=[established],
        J=PROFILE.J, T=PROFILE.temperature, delta=PROFILE.delta, epsilon=0.1, closure=0.3,
    )
    
    print("\n--- Search Results ---")
    print(f"Total rollouts: {opt_result.total_rollouts}")
    print(f"Incumbent: {opt_result.incumbent.candidate_id} ({opt_result.incumbent.selector})")
    
    print("Candidate Ranking (top 3):")
    # sort candidate scores
    ranked = sorted(opt_result.candidate_scores.items(), key=lambda x: x[1])
    for i, (cid, score) in enumerate(ranked[:3]):
        summ = opt_result.candidate_summaries[cid]
        sel = summ.get('selector')
        sel_str = f"selector={sel} | " if sel else ""
        print(f"  {i+1}. {cid} | {sel_str}feasible={summ['is_feasible']}, p_hat={summ['p_hat']:.2f}, exp={summ['mean_exposure']:.2f}")

    evaluated_selectors = set(
        c.selector for c in opt_result.evaluated_candidates 
        if hasattr(c, "selector") and (c.selector != "random" or c.candidate_id != "observable_zero")
    )
    print(f"\nDistinct selectors evaluated: {sorted(list(evaluated_selectors))}")
    if "reported_attention" not in evaluated_selectors:
        print("WARNING: reported_attention selector was not covered!")

    # Run full campaign at tiny scale (light allocation, small physics)
    print("\n=== Full light-allocation campaign at tiny scale ===")
    from src.config.control_profiles import LIGHT_CONTROL_PROFILE

    tiny_profile = LIGHT_CONTROL_PROFILE.model_copy(update={
        "N": 12, "D": 1, "horizon": 24.0, "establishment_time": 8.0,
        "n_origins": 8, "seed_base": 8800,
        "holding_duration": 4.0, "release_deadline": 8.0,
        "k_distinct_nodes_fraction": 0.25, "exposure_cap_normalized": 0.5,
        "candidate_cap": 4,
        "selectors": ("reported_attention", "random"),
        "observed_slot_quotas": (0.0, 1.0),
        "feedback_choices": (False, True),
        "training_origins": 2, "training_futures_per_origin": 1,
        "selection_candidates": 2, "selection_origins": 2,
        "selection_futures_per_origin": 2,
        "evaluation_origins": 4, "evaluation_futures_per_origin": 1,
    })

    from src.control.campaign import run_campaign

    campaign_result = run_campaign(
        tiny_profile,
        campaign_id="m4_smoke",
        master_seed=99001,
        out_dir="results",
        snapshot_dir="results/origin_snapshots",
        progress=True,
    )

    print("\n--- Campaign Outcome ---")
    prep = campaign_result.preparation_summary
    print(f"preparation: accepted {prep['n_accepted']}/{prep['n_attempted']} attempts "
          f"({prep['n_rejected']} rejected; yield {prep['eligibility_yield']:.3f} "
          f"accepted/attempted)")
    print(f"declared spec: target={campaign_result.spec.target_type}, "
          f"threshold={campaign_result.spec.target_params.get('threshold')}, "
          f"holding={campaign_result.spec.holding_duration}, "
          f"horizon={campaign_result.spec.horizon}, "
          f"release_deadline={campaign_result.spec.release_deadline}")
    print(f"selected policy: {campaign_result.selection_summary['selected_candidate_id']} "
          f"params={campaign_result.selection_summary['selected_candidate_params']}")
    if campaign_result.selected_static_candidate:
        print(f"selected static winner: {campaign_result.selected_static_candidate.candidate_id}")
    if campaign_result.selected_feedback_candidate:
        print(f"selected feedback winner: {campaign_result.selected_feedback_candidate.candidate_id}")
    print("uncontrolled persistence check "
          f"(zero-control on fresh origins): {campaign_result.manifest['uncontrolled_persistence']}")
    print("reliability summaries (eligible trials only):")
    for key, s in campaign_result.reliability_summaries.items():
        print(f"  {key}: p_hat={s['success_probability']:.2f} "
              f"({s['successes']}/{s['eligible_trials']}) "
              f"mean_T_restricted={s['mean_restricted_time']}")
    n_pairs = len(campaign_result.adaptive_vs_frozen_pairs)
    print(f"adaptive/frozen pairs (matched origins+seeds): {n_pairs}")
    frozen_ok = all(
        p["frozen"].event_counts.get("encounters", 0) == 0
        and p["frozen"].event_counts.get("memory_decays", 0) == 0
        for p in campaign_result.adaptive_vs_frozen_pairs
    )
    print(f"frozen branch genuinely frozen (no encounter/decay events): {frozen_ok}")

    print(f"\nResults bundle written to: {campaign_result.bundle_path}")
    print("Done. The light-allocation campaign ran end-to-end at tiny scale "
          "and wrote one results bundle.")

if __name__ == "__main__":
    main()