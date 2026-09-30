"""Targeted tests: shared-physics reconciliation.

Validates that the control engine (``src/control/engine.py``) reproduces exactly
the declared pilot physics of the preparation runner (``src/execution/runner.py``):

1. Explicit ``a`` / ``c`` / ``tau_y`` forwarding (no silent MemoryManager defaults).
2. ``build_realism_runner`` forwards ``J`` explicitly.
3. A zero-control engine rollout matches the preparation runner trajectory
   bit-for-bit under the same state and seed (shared parameter contract).
4. The frozen branch is a genuine frozen W (encounters AND decay disabled).
"""

import inspect
import unittest

import numpy as np

from src.config.profiles import RunProfile, build_realism_runner
from src.control.contracts import ControlProblem, TargetType
from src.control.controllers import ZeroController
from src.control.engine import run_controlled_simulation
from src.experiment.ensembles import freeze_memory_dynamics

# Tiny representative profile: same physics constants as the principal design.
MINI_PROFILE = RunProfile(
    topic="campaign", tier="light", N=8, D=1, horizon=12.0,
    establishment_time=4.0, n_origins=1, seed_base=7700,
)


def _fresh_runner(seed: int = 101):
    return build_realism_runner(MINI_PROFILE, seed)


def _problem(N: int, horizon: float = 6.0, seed: int = 7) -> ControlProblem:
    return ControlProblem(
        problem_id="m1_test",
        target_type=TargetType.REVERSAL,
        target_params={"threshold": 0.5},
        N=N,
        D=1,
        k=2,
        exposure_cap=10.0,
        amplitude_cap=4.0,
        release_deadline=3.0,
        holding_duration=1.0,
        horizon=horizon,
        proposal_seed=seed + 1,
        search_seed=seed,
    )


class TestSharedPhysicsReconciliation(unittest.TestCase):
    """preparation and control paths must share explicit physics settings."""

    def test_control_engine_signature_defaults(self) -> None:
        sig = inspect.signature(run_controlled_simulation)
        self.assertEqual(sig.parameters["a"].default, 0.0)
        self.assertEqual(sig.parameters["c"].default, 1.0)
        self.assertEqual(sig.parameters["tau_y"].default, 1.0)
        self.assertEqual(sig.parameters["J"].default, 1.0)

    def test_build_realism_runner_forwards_J(self) -> None:
        profile = MINI_PROFILE.model_copy(update={"J": 2.5})
        runner = build_realism_runner(profile, origin_seed=11)
        self.assertEqual(runner.J, 2.5)

    def test_zero_control_engine_matches_runner_trajectory(self) -> None:
        """Same state, same seed, matched parameters -> identical trajectory.

        The control engine must reproduce the preparation runner's physics
        exactly when no intervention is applied (ZeroController releases at t=0).
        """
        runner_a = _fresh_runner(101)
        traj = runner_a.run(horizon=6.0)
        final_runner = traj.final_state

        runner_b = _fresh_runner(101)
        result, final_engine, _ledger = run_controlled_simulation(
            initial_state=runner_b.state,
            problem=_problem(N=MINI_PROFILE.N, horizon=6.0),
            controller=ZeroController(),
            J=MINI_PROFILE.J,
            T=MINI_PROFILE.temperature,
            delta=MINI_PROFILE.delta,
            epsilon=MINI_PROFILE.exploration,
            closure=MINI_PROFILE.closure,
            a=0.0,
            c=1.0,
            tau_y=1.0,
            sim_rng=np.random.default_rng(101),
            controller_rng=np.random.default_rng(999),
        )
        np.testing.assert_array_equal(final_engine.x, final_runner.x)
        np.testing.assert_array_equal(final_engine.y, final_runner.y)
        np.testing.assert_array_equal(final_engine.W, final_runner.W)

    def test_engine_forwards_tau_y_to_scheduler(self) -> None:
        """Public-channel rate R_y = N / tau_y must follow the tau_y argument.

        Note: encounters dominate the event mix (R_enc = sum nu_i ~ 16 > R_y),
        so the assertion targets public_updates specifically, not total events.
        """
        runner = _fresh_runner(202)
        publics = {}
        for tau_y in (0.25, 4.0):
            result, _, _ = run_controlled_simulation(
                initial_state=runner.state,
                problem=_problem(N=MINI_PROFILE.N, horizon=4.0),
                controller=ZeroController(),
                sim_rng=np.random.default_rng(202),
                controller_rng=np.random.default_rng(203),
                tau_y=tau_y,
            )
            publics[tau_y] = result.event_counts["public_updates"]
        # R_y ratio is 16x; observed ratio is diluted by the shared event mix.
        self.assertGreater(publics[0.25], 0)
        self.assertGreater(publics[4.0], 0)
        self.assertGreater(publics[0.25], 4.0 * publics[4.0])

    def test_engine_forwards_a_c_to_memory_manager(self) -> None:
        """Same seeds, different prestige/matching biases -> different memory."""
        runner = _fresh_runner(303)
        # Non-zero initial memory so the encounter kernel bias terms matter.
        runner.state.W[0, 1] = 3
        runner.state.W[1, 2] = 2
        runner.state.W[2, 3] = 2
        runner.state.W[3, 0] = 2
        _, final_default, _ = run_controlled_simulation(
            initial_state=runner.state,
            problem=_problem(N=MINI_PROFILE.N, horizon=6.0),
            controller=ZeroController(),
            sim_rng=np.random.default_rng(303),
            controller_rng=np.random.default_rng(304),
            a=0.0,
            c=1.0,
        )
        _, final_biased, _ = run_controlled_simulation(
            initial_state=runner.state,
            problem=_problem(N=MINI_PROFILE.N, horizon=6.0),
            controller=ZeroController(),
            sim_rng=np.random.default_rng(303),
            controller_rng=np.random.default_rng(304),
            a=5.0,
            c=0.0,
        )
        self.assertFalse(np.array_equal(final_default.W, final_biased.W))
        # Same parameters and seeds must reproduce exactly (determinism).
        _r, final_again, _ = run_controlled_simulation(
            initial_state=runner.state,
            problem=_problem(N=MINI_PROFILE.N, horizon=6.0),
            controller=ZeroController(),
            sim_rng=np.random.default_rng(303),
            controller_rng=np.random.default_rng(304),
            a=0.0,
            c=1.0,
        )
        np.testing.assert_array_equal(final_again.W, final_default.W)

    def test_frozen_branch_genuinely_freezes_W(self) -> None:
        """freeze_memory_dynamics disables encounters AND decay: W constant."""
        runner = _fresh_runner(404)
        runner.run(horizon=4.0)
        W_establishment = runner.memory_mgr.W.copy()
        self.assertGreater(int(W_establishment.sum()), 0)

        checkpoint = runner.get_checkpoint()
        cont = type(runner).from_checkpoint(checkpoint)
        freeze_memory_dynamics(cont)

        self.assertEqual(cont.scheduler.R_enc, 0.0)
        self.assertEqual(cont.scheduler.R_forget, 0.0)
        cont.run(horizon=MINI_PROFILE.horizon)
        np.testing.assert_array_equal(cont.memory_mgr.W, W_establishment)
        # NOTE: scheduler.stats counters are cumulative across the checkpoint
        # restore, so they cannot be asserted to be zero here; the exact W
        # equality plus zero propensities are the genuine frozen-W evidence.

    def test_rate_change_rescales_pending_clock(self) -> None:
        """Rate changes conserve the residual integrated hazard of the pending clock."""
        from src.core.scheduler import PoissonScheduler

        sched = PoissonScheduler(N=4, D=1, tau_x=2.0, tau_y=1.0, nu=0.0, delta=0.0,
                                 rng=np.random.default_rng(11))
        sched._draw_pending_clock()
        p0 = sched.pending_time
        r_old = sched.R_total
        self.assertGreater(p0, 0.0)

        sched.nu = 1.0  # R_enc: 0 -> 4, R_total: 2 -> 8
        sched._sync_pending_clock()
        p1 = sched.pending_time
        # Residual integrated hazard must be conserved across the rate change.
        self.assertAlmostEqual((p0 - sched.time) * r_old, (p1 - sched.time) * sched.R_total, places=9)

    def test_checkpoint_resume_preserves_pending_clock(self) -> None:
        """Checkpoint mid-run (pending clock drawn) resumes bit-for-bit."""
        runner = _fresh_runner(505)
        runner.run(horizon=3.0)
        self.assertIsNotNone(runner.scheduler.pending_time)
        checkpoint = runner.get_checkpoint()

        cont = type(runner).from_checkpoint(checkpoint)
        self.assertIsNotNone(cont.scheduler.pending_time)
        traj_cont = cont.run(horizon=6.0)

        ref = _fresh_runner(505)
        traj_ref = ref.run(horizon=6.0)
        np.testing.assert_array_equal(traj_cont.final_state.x, traj_ref.final_state.x)
        np.testing.assert_array_equal(traj_cont.final_state.y, traj_ref.final_state.y)
        np.testing.assert_array_equal(traj_cont.final_state.W, traj_ref.final_state.W)

    def test_stochastic_fires_before_deterministic_on_exact_tie(self) -> None:
        """A violating stochastic event exactly at a holding deadline fires first.

        Conservative holding convention: the state-changing event at the deadline
        timestamp must be processed before the deadline certification, so a
        violation exactly at the deadline prevents certification.
        """
        from src.core.scheduler import PoissonScheduler

        sched = PoissonScheduler(N=2, D=1, tau_x=1.0, tau_y=1.0, nu=0.0, delta=0.0,
                                 rng=np.random.default_rng(12))
        fired = []
        sched.register_callbacks(cb_private=lambda node: fired.append("stochastic"))
        sched.schedule_deterministic_event(
            time=5.0, priority=1, event_type="holding_deadline",
            callback=lambda s, e: fired.append("deterministic"),
        )
        # Force the pending stochastic event onto the exact deadline timestamp.
        sched._pending_time = 5.0
        sched._r_at_draw = sched.R_total

        self.assertTrue(sched.step())  # stochastic fires first at the tie
        self.assertEqual(fired, ["stochastic"])
        self.assertEqual(sched.time, 5.0)
        self.assertTrue(sched.step())  # then the deterministic deadline event
        self.assertEqual(fired, ["stochastic", "deterministic"])
        self.assertEqual(sched.time, 5.0)


class TestObservationBoundaries(unittest.TestCase):
    """Principal search/selection receives only restricted observations."""

    def test_observable_policy_candidate_label(self) -> None:
        from src.control.optimizers.search import ObservablePolicyCandidate
        cand = ObservablePolicyCandidate(
            candidate_id="obs", selector="random", observed_slot_quota=0.5,
            feedback=True, initial_recruitment_fraction=1.0, nominal_amplitude=1.0, duration=1.0
        )
        self.assertEqual(cand.metadata.get("information_level"), "observable")

    def test_oracle_path_label(self) -> None:
        from src.control.optimizers.search import PolicyCandidate
        cand = PolicyCandidate("id", (), (), 1.0, 1.0)
        self.assertEqual(cand.metadata.get("information_level"), "oracle")

    def test_mutation_of_hidden_states_does_not_change_action(self) -> None:
        from src.control.controllers import ObservableSelectorPolicy
        from src.control.observation import SampledPublicObservationModel
        from src.control.ledger import BudgetLedger

        runner = _fresh_runner(100)
        state1 = runner.state.copy(deep=True)
        state2 = runner.state.copy(deep=True)
        # Mutate hidden private state in state2
        state2.x[0, 0] = -state2.x[0, 0]
        
        obs_model1 = SampledPublicObservationModel(N=state1.N, accessible_roster=(0,1,2,3), rng=np.random.default_rng(42))
        obs1 = obs_model1.sample_observation(state1, 1.0)
        
        obs_model2 = SampledPublicObservationModel(N=state2.N, accessible_roster=(0,1,2,3), rng=np.random.default_rng(42))
        obs2 = obs_model2.sample_observation(state2, 1.0)
        
        problem = _problem(N=MINI_PROFILE.N)
        ledger1 = BudgetLedger(N=MINI_PROFILE.N, D=1, k=2, exposure_cap=10.0, amplitude_cap=4.0, release_deadline=3.0, accessible_roster=(0,1,2,3))
        ledger2 = BudgetLedger(N=MINI_PROFILE.N, D=1, k=2, exposure_cap=10.0, amplitude_cap=4.0, release_deadline=3.0, accessible_roster=(0,1,2,3))
        
        policy1 = ObservableSelectorPolicy(selector="random", seed=10)
        policy1.reset(problem, obs1)
        action1 = policy1.act(obs1, ledger1)
        
        policy2 = ObservableSelectorPolicy(selector="random", seed=10)
        policy2.reset(problem, obs2)
        action2 = policy2.act(obs2, ledger2)
        
        self.assertEqual(action1.entries, action2.entries)


if __name__ == "__main__":
    unittest.main()