"""Unit tests for src/core/scheduler.py."""

import math
import unittest
import numpy as np

from src.core.scheduler import (
    PRIORITY_HOLDING,
    PRIORITY_HORIZON,
    PRIORITY_INTERVENTION,
    PRIORITY_RESTORE,
    PRIORITY_SNAPSHOT,
    PoissonScheduler,
)


class TestCoreScheduler(unittest.TestCase):
    """Test suite for Poisson event scheduler, preemption engine, and event statistics."""

    def test_poisson_exponential_inter_event_times(self):
        """Verify empirical waiting times follow exponential distribution with mean 1 / R_total."""
        N = 10
        D = 1
        # Set parameters such that R_total is known
        # R_x = N * D / tau_x = 10 / 1.0 = 10
        # R_y = N * D / tau_y = 10 / 0.5 = 20
        # R_enc = N * nu = 10 * 2.0 = 20
        # R_forget = 0, R_n = 0
        # Total rate = 50.0. Expected mean dt = 1/50 = 0.02
        scheduler = PoissonScheduler(
            N=N, D=D, tau_x=1.0, tau_y=0.5, nu=2.0, delta=0.1, rng=np.random.default_rng(101)
        )
        self.assertEqual(scheduler.R_total, 50.0)

        n_samples = 40_000
        waiting_times = []
        for _ in range(n_samples):
            t_prev = scheduler.time
            scheduler.step()
            waiting_times.append(scheduler.time - t_prev)

        empirical_mean = float(np.mean(waiting_times))
        empirical_var = float(np.var(waiting_times))

        expected_mean = 1.0 / 50.0
        expected_var = 1.0 / (50.0 ** 2)

        std_err = math.sqrt(expected_var / n_samples)
        self.assertAlmostEqual(empirical_mean, expected_mean, delta=3.5 * std_err)
        self.assertAlmostEqual(empirical_var, expected_var, delta=0.15 * expected_var)

    def test_channel_selection_proportions(self):
        """Verify stochastic channel firing proportions match relative propensities."""
        # N=10, D=1
        # R_x = 10 / 1.0 = 10.0   (p = 0.10)
        # R_y = 10 / 0.5 = 20.0   (p = 0.20)
        # R_enc = 10 * 3.0 = 30.0 (p = 0.30)
        # S = 150, delta = 0.1 => R_forget = 15.0 (p = 0.15)
        # tau_n = 0.4 => R_n = 10 / 0.4 = 25.0 (p = 0.25)
        # R_total = 100.0
        scheduler = PoissonScheduler(
            N=10, D=1, tau_x=1.0, tau_y=0.5, nu=3.0, delta=0.1, tau_n=0.4, rng=np.random.default_rng(202)
        )
        scheduler.set_total_memory(150)
        self.assertEqual(scheduler.R_total, 100.0)

        n_steps = 40_000
        for _ in range(n_steps):
            scheduler.step()

        stats = scheduler.stats
        obs_x = stats.private_updates
        obs_y = stats.public_updates
        obs_enc = stats.encounters
        obs_dec = stats.memory_decays
        obs_n = stats.participation_updates

        observed = np.array([obs_x, obs_y, obs_enc, obs_dec, obs_n], dtype=np.float64)
        expected = np.array([0.10, 0.20, 0.30, 0.15, 0.25]) * n_steps

        # Chi-squared goodness of fit (4 degrees of freedom, critical value 13.28 at alpha=0.01)
        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        self.assertLess(chi2, 13.28)

    def test_deterministic_preemption_at_exact_time(self):
        """Verify deterministic event preempts stochastic event and fires at exact floating-point time."""
        # Low rate so mean stochastic step is 2.0
        scheduler = PoissonScheduler(
            N=1, D=1, tau_x=4.0, tau_y=4.0, nu=0.0, delta=0.0, rng=np.random.default_rng(303)
        )
        # R_total = 0.25 + 0.25 = 0.5

        exact_sched_time = 1.3456789
        callback_called = []

        def on_event(sched, ev):
            callback_called.append((sched.time, ev.event_type))

        scheduler.schedule_deterministic_event(
            time=exact_sched_time,
            priority=PRIORITY_INTERVENTION,
            event_type="test_pulse",
            callback=on_event,
        )

        # Step until preemption fires
        while scheduler.time < exact_sched_time:
            scheduler.step()

        # Clock must land at exact timestamp
        self.assertEqual(scheduler.time, exact_sched_time)
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0], (exact_sched_time, "test_pulse"))
        self.assertGreaterEqual(scheduler.stats.preemptions, 1)
        self.assertEqual(scheduler.stats.deterministic_events, 1)

    def test_simultaneous_event_priority_ordering(self):
        """Verify events scheduled at the identical timestamp execute in strict priority order."""
        scheduler = PoissonScheduler(
            N=1, D=1, tau_x=10.0, tau_y=10.0, nu=0.0, delta=0.0, rng=np.random.default_rng(404)
        )

        execution_order = []

        def make_cb(label):
            return lambda sched, ev: execution_order.append(label)

        t_simult = 2.0
        # Schedule in reverse priority order
        scheduler.schedule_deterministic_event(
            time=t_simult, priority=PRIORITY_SNAPSHOT, event_type="snap", callback=make_cb("snapshot")
        )
        scheduler.schedule_deterministic_event(
            time=t_simult, priority=PRIORITY_HOLDING, event_type="hold", callback=make_cb("holding")
        )
        scheduler.schedule_deterministic_event(
            time=t_simult, priority=PRIORITY_INTERVENTION, event_type="interv", callback=make_cb("intervention")
        )
        scheduler.schedule_deterministic_event(
            time=t_simult, priority=PRIORITY_RESTORE, event_type="restore", callback=make_cb("restore")
        )

        while scheduler.time < t_simult or scheduler.pq:
            scheduler.step()

        # Expected execution sequence strictly by priority: restore (0) -> intervention (1) -> holding (2) -> snapshot (3)
        self.assertEqual(execution_order, ["restore", "intervention", "holding", "snapshot"])

    def test_cumulative_event_count_formula_recovery(self):
        """Verify empirical total event counts recover theoretical formula E[K(H)] within 2% error.

        Theoretical formula for the expected cumulative event count:
        E[K(H)] = N * [ (tau_x^-1 + tau_y^-1 + 2*nu)*H - (nu/delta)*(1 - e^{-delta*H}) ]
        """
        N = 40
        D = 1
        tau_x = 5.0
        tau_y = 1.0
        nu = 0.4
        delta = 0.02
        horizon = 5.0

        # Exact expectation
        rate_fixed = (1.0 / tau_x) + (1.0 / tau_y) + 2.0 * nu  # 0.2 + 1.0 + 0.8 = 2.0
        mean_integrated_memory = (nu / delta) * (1.0 - math.exp(-delta * horizon))
        expected_total_events = N * (rate_fixed * horizon - mean_integrated_memory)

        n_runs = 30
        run_counts = []

        for run in range(n_runs):
            rng = np.random.default_rng(5000 + run)
            sched = PoissonScheduler(
                N=N, D=D, tau_x=tau_x, tau_y=tau_y, nu=nu, delta=delta, rng=rng
            )

            # Simulated memory token state
            sim_memory = 0

            def cb_enc():
                nonlocal sim_memory
                sim_memory += 1
                sched.set_total_memory(sim_memory)

            def cb_dec():
                nonlocal sim_memory
                if sim_memory > 0:
                    sim_memory -= 1
                sched.set_total_memory(sim_memory)

            sched.register_callbacks(cb_encounter=cb_enc, cb_decay=cb_dec)
            sched.run_until(horizon)
            run_counts.append(sched.stats.total_stochastic_events)

        empirical_mean = float(np.mean(run_counts))
        relative_error = abs(empirical_mean - expected_total_events) / expected_total_events

        # Relative error must be under 2%
        self.assertLess(relative_error, 0.02)

    def test_reciprocal_propensities_and_channel_selection(self):
        """Verify reciprocal mode halves encounter rate and decay propensity R_forget."""
        N = 10
        D = 1
        tau_x = 1.0
        tau_y = 0.5
        nu = 4.0
        delta = 0.2
        tau_n = 0.5

        # Directed baseline
        sched_dir = PoissonScheduler(
            N=N, D=D, tau_x=tau_x, tau_y=tau_y, nu=nu, delta=delta, tau_n=tau_n, reciprocal=False
        )
        self.assertEqual(sched_dir.R_enc, float(N) * nu)  # 40.0
        sched_dir.set_total_memory(200)
        self.assertEqual(sched_dir.R_forget, delta * 200.0)  # 40.0

        # Reciprocal scheduler
        sched_rec = PoissonScheduler(
            N=N, D=D, tau_x=tau_x, tau_y=tau_y, nu=nu, delta=delta, tau_n=tau_n, reciprocal=True
        )
        # R_enc must be halved to N * nu / 2.0 = 20.0
        self.assertEqual(sched_rec.R_enc, float(N) * nu / 2.0)  # 20.0

        # With S = 200 tokens (100 pairs), R_forget must be delta * (S / 2.0) = 20.0
        sched_rec.set_total_memory(200)
        self.assertEqual(sched_rec.R_forget, delta * (200.0 / 2.0))  # 20.0

        # Total jump propensity check
        # R_x = 10, R_y = 20, R_enc = 20, R_forget = 20, R_n = 20 => Total = 90.0
        expected_total = 10.0 + 20.0 + 20.0 + 20.0 + 20.0
        self.assertEqual(sched_rec.R_total, expected_total)

    def test_reciprocal_memory_stationary_expectation(self):
        """Verify PoissonScheduler in reciprocal mode produces stationary expectation E[S] = N*nu/delta.

        In reciprocal mode, each encounter creates 1 pair token (+2 tokens), and each decay
        event removes 1 pair token (-2 tokens). With R_enc = N*nu/2 and R_forget = delta*(S/2),
        the stationary mean is E[S] = N*nu/delta and stationary variance is Var(S) = 2*N*nu/delta.
        """
        N = 20
        D = 1
        tau_x = 50.0
        tau_y = 50.0
        nu = 2.0
        delta = 1.0
        reciprocal = True

        expected_mean = N * nu / delta  # 40.0

        sched = PoissonScheduler(
            N=N, D=D, tau_x=tau_x, tau_y=tau_y, nu=nu, delta=delta, reciprocal=reciprocal,
            rng=np.random.default_rng(12345)
        )

        sim_memory = 0
        sched.set_total_memory(sim_memory)

        def cb_enc():
            nonlocal sim_memory
            sim_memory += 2
            sched.set_total_memory(sim_memory)

        def cb_dec():
            nonlocal sim_memory
            if sim_memory >= 2:
                sim_memory -= 2
            sched.set_total_memory(sim_memory)

        sched.register_callbacks(cb_encounter=cb_enc, cb_decay=cb_dec)

        burn_in = 10.0
        horizon = 250.0
        integral_s = 0.0
        integral_s2 = 0.0
        t_sampled = 0.0

        while sched.time < horizon:
            t_prev = sched.time
            s_curr = sim_memory
            sched.step()
            if sched.time > burn_in:
                dt_eff = sched.time - max(t_prev, burn_in)
                integral_s += s_curr * dt_eff
                integral_s2 += (s_curr ** 2) * dt_eff
                t_sampled += dt_eff

        time_mean = integral_s / t_sampled
        time_var = (integral_s2 / t_sampled) - (time_mean ** 2)
        fano_factor = time_var / time_mean

        rel_err = abs(time_mean - expected_mean) / expected_mean
        self.assertLess(
            rel_err, 0.05,
            f"Reciprocal stationary mean {time_mean:.2f} diverged from {expected_mean:.2f} (rel_err={rel_err:.4f})"
        )

        self.assertTrue(
            1.5 <= fano_factor <= 2.5,
            f"Reciprocal Fano factor {fano_factor:.2f} outside expected theoretical range [1.5, 2.5]"
        )

    def test_scheduler_channel_fallthrough_guard_when_participation_disabled(self):
        """Verify Channel 5 never executes and boundary draw does not cause None return when R_n == 0."""
        class UpperBoundaryRNG:
            def random(self):
                return 1.0  # Exact upper boundary draw

        # Case 1: tau_n=None (R_n = 0.0), S = 0 (R_forget = 0.0)
        # Active channels: x, y, enc. Upper boundary must land in encounters.
        sched1 = PoissonScheduler(
            N=10, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.1, tau_n=None, rng=UpperBoundaryRNG()
        )
        res1 = sched1.step()
        self.assertTrue(res1)
        self.assertEqual(sched1.stats.participation_updates, 0)
        self.assertEqual(sched1.stats.encounters, 1)

        # Case 2: tau_n=None (R_n = 0.0), S > 0 (R_forget > 0.0)
        # Active channels: x, y, enc, forget. Upper boundary must land in memory decays.
        sched2 = PoissonScheduler(
            N=10, D=1, tau_x=1.0, tau_y=1.0, nu=1.0, delta=0.1, tau_n=None, rng=UpperBoundaryRNG()
        )
        sched2.set_total_memory(20)
        res2 = sched2.step()
        self.assertTrue(res2)
        self.assertEqual(sched2.stats.participation_updates, 0)
        self.assertEqual(sched2.stats.memory_decays, 1)

    def test_scheduler_participation_updates_strictly_zero_over_trajectory(self):
        """Verify participation_updates counter remains strictly zero when tau_n is None."""
        sched = PoissonScheduler(
            N=20, D=1, tau_x=2.0, tau_y=1.0, nu=1.5, delta=0.2, tau_n=None, rng=np.random.default_rng(999)
        )
        sched.set_total_memory(50)
        for _ in range(25_000):
            sched.step()

        self.assertEqual(sched.stats.participation_updates, 0)
        self.assertGreater(sched.stats.private_updates, 0)
        self.assertGreater(sched.stats.public_updates, 0)
        self.assertGreater(sched.stats.encounters, 0)
        self.assertGreater(sched.stats.memory_decays, 0)


if __name__ == "__main__":
    unittest.main()
