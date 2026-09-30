"""Controllers and policy search families for optimal transformation.

Implements:
1. Controller: Abstract base contract.
2. ZeroController: Baseline unactuated policy with immediate release at t=0.
3. OpenLoopPulseController: Static pulse with uniform effective amplitude clipping.
4. ObservableSelectorPolicy: Realism revision §6 principal policy family:
   - Observable selectors: random, reported_attention, reported_outgoing_degree,
     reported_cross_group, group_balanced.
   - Quota: slot allocation between observed accessible and unobserved accessible nodes.
   - Dynamic feedback vs static commitment.
   - Initial recruitment fraction (0.5 vs 1.0) with report-driven recruitment.
   - Strict observation isolation: uses only Observation dataclass and metadata.
5. OracleMPCController: Receding-horizon benchmark using counterfactual rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.control.contracts import Action, ActionEntry, ControlProblem, Observation, TargetType

# Explicit tie tolerance for direction inference (audit finding F3):
# an observed report mean with |mean| <= TIE_TOLERANCE is treated as ambiguous:
# no target direction is locked and the policy waits for a decisive report.
TIE_TOLERANCE = 1e-12
from src.control.ledger import BudgetLedger


class Controller(ABC):
    """Abstract base class for all intervention controllers."""

    @abstractmethod
    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize or reset controller state for a new episode."""
        pass

    @abstractmethod
    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        """Compute an action from the current observation and resource ledger."""
        pass


class ZeroController(Controller):
    """Zero-control baseline: immediately releases control at time 0."""

    def __init__(self) -> None:
        self.problem: Optional[ControlProblem] = None

    @property
    def effective_settings(self):
        return {"effective_node_count": 0}

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.problem = problem

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        return Action(entries=(), intended_end_time=0.0, release=True)


class OpenLoopPulseController(Controller):
    """Open-loop static pulse controller on predetermined target nodes."""

    def __init__(
        self,
        target_nodes: Sequence[int],
        nominal_amplitude: float = 1.0,
        duration: float = 5.0,
        signs: Optional[Dict[int, float]] = None,
        component: int = 0,
    ) -> None:
        self.target_nodes = tuple(sorted(set(target_nodes)))
        self.nominal_amplitude = float(nominal_amplitude)
        self.duration = float(duration)
        self.signs = dict(signs) if signs is not None else {}
        self.component = int(component)

        self.problem: Optional[ControlProblem] = None
        self.actuated: bool = False
        self.pulse_end_time: Optional[float] = None

    @property
    def effective_settings(self):
        return {"effective_node_count": len(self.target_nodes)}

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.problem = problem
        self.actuated = False
        self.pulse_end_time = None

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        if self.problem is None:
            raise RuntimeError("Controller must be reset before act()")

        t = observation.time
        if ledger.is_released:
            return Action(entries=(), release=True)

        if not self.actuated:
            # First actuation: compute effective amplitude clipped by budget
            n_s = len(self.target_nodes)
            if n_s == 0:
                self.actuated = True
                return Action(entries=(), release=True)

            # a_eff = min(a_nom, B / (n_s * d), u_max)
            rem_B = ledger.remaining_exposure
            max_amp_by_budget = rem_B / (n_s * self.duration) if (n_s * self.duration) > 0 else 0.0
            a_eff = min(self.nominal_amplitude, max_amp_by_budget, self.problem.amplitude_cap)

            if a_eff <= 1e-15:
                self.actuated = True
                return Action(entries=(), release=True)

            self.pulse_end_time = min(t + self.duration, self.problem.release_deadline)
            entries: List[ActionEntry] = []
            for node in self.target_nodes:
                sgn = self.signs.get(node, 1.0)
                entries.append(ActionEntry(node=node, component=self.component, field=float(sgn * a_eff)))

            self.actuated = True
            return Action(entries=tuple(entries), intended_end_time=self.pulse_end_time, release=False)

        # Pulse duration completed or expired
        if self.pulse_end_time is not None and t >= self.pulse_end_time - 1e-12:
            return Action(entries=(), intended_end_time=t, release=True)

        # Maintain existing action if still within pulse window
        if ledger.active_action is not None:
            return ledger.active_action
        return Action(entries=(), release=True)


def audit_policy_timing(
    delivery_delay: float,
    duration: float,
    sample_interval: float,
    release_deadline: float,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Audit report/actuation timing and classify policy feedback feasibility.

    Under the declared contract:
    - Initial report arrives at t = delivery_delay.
    - Intervention pulse runs from t to min(t + duration, release_deadline).
    - Periodic observation snapshots occur at np.arange(0, release_deadline, sample_interval).
    - Reports are delivered delivery_delay after their snapshot time.

    A policy is classified as genuinely adaptive only if at least one usable
    observation is delivered strictly after initial actuation and strictly before
    intervention end. Policies with no usable later report before intervention end
    are classified as effectively static.
    """
    first_delivery = float(delivery_delay)
    intervention_end = min(first_delivery + float(duration), float(release_deadline))

    snapshot_times = np.arange(0.0, float(release_deadline), float(sample_interval))
    delivery_times = [float(t + delivery_delay) for t in snapshot_times]

    usable_deliveries = [
        float(t) for t in delivery_times
        if (t > first_delivery + tolerance) and (t < intervention_end - tolerance)
    ]

    has_feedback_opportunity = len(usable_deliveries) > 0
    classification = "genuinely_adaptive" if has_feedback_opportunity else "effectively_static"

    return {
        "first_delivery_time": first_delivery,
        "intervention_end_time": intervention_end,
        "delivery_times": delivery_times,
        "usable_delivery_times": usable_deliveries,
        "feedback_opportunity_count": len(usable_deliveries),
        "has_feedback_opportunity": has_feedback_opportunity,
        "is_effectively_static": not has_feedback_opportunity,
        "classification": classification,
    }


class ObservableSelectorPolicy(Controller):
    """Principal realism policy: observable selection, slot quota, and feedback actuation.

    Selectors:
    - 'random': uniform random order among candidates.
    - 'reported_attention': ranked by received attention proxy from revealed edges.
    - 'reported_outgoing_degree': ranked by outgoing degree from revealed edges.
    - 'reported_cross_group': ranked by cross-group ties from revealed edges.
    - 'group_balanced': rounds through groups picking top-ranked candidate per group.

    Direction knowledge contract (audit finding F3, chosen option (b)):
    The transformation target is *not* a supplied objective sign for this policy.
    Instead, the policy performs a RESTRICTED INFERENCE of the initial public
    majority from its delayed, sampled, noisy public reports (never from private
    states x or hidden memory W), and acts to drive the majority to the opposite
    of the inferred sign. Two consequences, both by design:
    1. The immutable evaluation target remains ``TargetEvaluator.create_for_state``,
       which fixes s_* from the TRUE initial public majority. This policy's
       inference may be WRONG (small/late/tied report); an inference mistake is
       preserved as such and must be *counted*, never absorbed by evaluating
       against a different destination, and never corrected by leaking the true
       target or private state into the policy.
    2. The inference provenance is exported in ``effective_settings``:
       'controller_inferred_target_sign', 'inference_report_mean',
       'inference_report_sample_time' (snapshot time of the deciding report),
       and 'inference_source' ('focal_panel_mean' or 'full_panel_mean').
       Consumers can compute the direction mismatch flag as
       inferred_sign != TargetEvaluator.defn.s_star without any private leak.

    Missing/tie semantics (audit finding F3):
    - A report with no finite observed values is UNAVAILABLE: the policy takes
      no actuation, consumes no budget, and deploys no nodes while waiting.
    - Explicit tie rule: an observed panel mean with |mean| <= TIE_TOLERANCE is
      AMBIGUOUS for direction inference: no direction is locked, the policy
      waits (budget-free) for a decisive report. A tie never locks -1 by
      convention and never fabricates a majority of 0.
    """

    def __init__(
        self,
        selector: str = "reported_attention",
        nominal_amplitude: float = 1.0,
        duration: float = 5.0,
        quota: float = 0.5,
        is_feedback: bool = True,
        initial_recruitment_fraction: float = 0.5,
        component: int = 0,
        seed: int = 42,
        node_fraction: Optional[float] = None,
        roster_selection: bool = False,
        broadcast_amplitude: float = 0.0,
    ) -> None:
        valid_selectors = {
            "random",
            "reported_attention",
            "reported_low_attention",
            "reported_outgoing_degree",
            "reported_cross_group",
            "group_balanced",
        }
        if selector not in valid_selectors:
            raise ValueError(f"Unknown selector '{selector}'. Valid: {valid_selectors}")

        self.selector = selector
        self.nominal_amplitude = float(nominal_amplitude)
        self.duration = float(duration)
        self.quota = float(quota)
        self.is_feedback = bool(is_feedback)
        self.initial_recruitment_fraction = float(initial_recruitment_fraction)
        self.component = int(component)
        self.seed = int(seed)
        self.node_fraction = float(node_fraction) if node_fraction is not None else None
        self.roster_selection = bool(roster_selection)
        self.broadcast_amplitude = float(broadcast_amplitude)
        if not np.isfinite(self.broadcast_amplitude) or self.broadcast_amplitude < 0:
            raise ValueError("broadcast_amplitude must be finite and nonnegative")
        if self.broadcast_amplitude and (self.is_feedback or not np.isfinite(self.nominal_amplitude) or self.nominal_amplitude < 0):
            raise ValueError("Combined fields require static control and nonnegative finite boost")

        self.problem: Optional[ControlProblem] = None
        self.rng: Optional[np.random.Generator] = None
        self.committed_end_time: Optional[float] = None
        self.initial_actuation_done: bool = False
        self.used_nodes: Set[int] = set()
        self.s_star: Optional[float] = None

        # Tracking for R3 diagnostics
        self.fallback_records: List[Dict[str, Any]] = []
        self.feedback_opportunities_encountered: int = 0
        self.feedback_updates_performed: int = 0
        self.requested_settings: Dict[str, Any] = {}
        self.effective_settings: Dict[str, Any] = {}

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)
        self.s_star = None
        self.committed_end_time = None
        self.initial_actuation_done = False
        self.used_nodes = set()
        self.fallback_records = []
        self.feedback_opportunities_encountered = 0
        self.feedback_updates_performed = 0
        self.requested_settings = {}
        self.effective_settings = {}
        if self.broadcast_amplitude and (
            problem.target_params.get("eligibility_mode") != "declared"
            or len(problem.accessible_roster) > problem.k
        ):
            raise ValueError("Broadcast requires declared direction and distinct-node permission for its entire roster")
        if problem.target_params.get("eligibility_mode") == "declared":
            sign = problem.target_params.get("declared_destination_sign")
            if sign not in (-1.0, 1.0):
                raise ValueError("declared_destination_sign must be -1 or +1")
            self.s_star = float(sign)
            self.effective_settings.update({
                "direction_source": "declared_task",
                "declared_destination_sign": self.s_star,
            })

    def _has_directional_task(self) -> bool:
        return self.problem is not None and (
            self.problem.target_type == TargetType.REVERSAL
            or self.problem.target_params.get("eligibility_mode") == "declared"
        )

    def _observed_target_mean(self, observation: Observation) -> Optional[float]:
        """Infer the initial-public-majority mean from restricted reports only.

        Direction inference source, in priority order:
        1. Focal panel (``target_params['focal_indices']``): mean over focal
           nodes with finite observed values. Returns None if none observed.
        2. Full observed panel: ``Observation.observed_public_mean`` (returns
           None for an empty/fully-missing report — missing data is never
           fabricated into a majority of 0).

        Explicit tie rule (audit finding F3): a mean with
        |mean| <= TIE_TOLERANCE is AMBIGUOUS for direction inference and
        returns None (no direction lock; the policy waits for a decisive
        report without consuming actuation budget). This aligns non-focal
        missing-data behavior with the focal missing-data behavior.
        """
        focal = self.problem.target_params.get("focal_indices") if self.problem is not None else None
        if focal is not None:
            values = observation.public_states[list(focal), self.component]
            observed = values[np.isfinite(values)]
            if len(observed):
                mean = float(np.mean(observed))
                return None if abs(mean) <= TIE_TOLERANCE else mean
            return None
        mean = observation.observed_public_mean(self.component)
        if mean is None:
            return None
        return None if abs(mean) <= TIE_TOLERANCE else mean

    def _rank_nodes(
        self,
        candidates: Sequence[int],
        observation: Observation,
        return_scored: bool = False,
    ) -> Any:
        """Rank candidate nodes according to configured selector using ONLY observed data.

        Returns:
            ranked_nodes (List[int]) if not return_scored,
            or (ranked_nodes, scored_nodes_subset) if return_scored.
        """
        if len(candidates) == 0:
            return ([], []) if return_scored else []
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        cand_list = list(candidates)

        if self.selector == "random":
            perm = self.rng.permutation(len(cand_list))
            ranked = [cand_list[i] for i in perm]
            return (ranked, ranked) if return_scored else ranked

        # Compute proxy metrics strictly from revealed edges (never hidden W)
        N = observation.public_states.shape[0]
        rec_att = np.zeros(N, dtype=np.float64)
        out_deg = np.zeros(N, dtype=np.float64)
        cross_grp = np.zeros(N, dtype=np.float64)
        has_reported_edge = np.zeros(N, dtype=bool)

        groups = observation.group_labels if observation.group_labels is not None else np.zeros(N, dtype=int)

        for (i, j), count in observation.observed_edges.items():
            if count > 0:
                rec_att[j] += float(count)
                out_deg[i] += 1.0
                has_reported_edge[j] = True
                has_reported_edge[i] = True
                if groups[i] != groups[j]:
                    cross_grp[i] += float(count)

        if self.selector in ("reported_attention", "reported_low_attention"):
            # Scored nodes: positive reported incoming attention from revealed edges
            scored_candidates = [n for n in cand_list if rec_att[n] > 0.0]
            unscored_candidates = [n for n in cand_list if rec_att[n] <= 0.0]

            direction = -1.0 if self.selector == "reported_low_attention" else 1.0
            scored_pairs = [(direction * float(rec_att[n]), float(self.rng.random()), n) for n in scored_candidates]
            scored_pairs.sort(reverse=True)
            ranked_scored = [n for _, _, n in scored_pairs]

            unscored_pairs = [(float(self.rng.random()), n) for n in unscored_candidates]
            unscored_pairs.sort(reverse=True)
            ranked_unscored = [n for _, n in unscored_pairs]

            ranked_all = ranked_scored + ranked_unscored
            return (ranked_all, ranked_scored) if return_scored else ranked_all

        elif self.selector == "reported_outgoing_degree":
            scored_candidates = [n for n in cand_list if out_deg[n] > 0.0]
            unscored_candidates = [n for n in cand_list if out_deg[n] <= 0.0]

            scored_pairs = [(float(out_deg[n]), float(self.rng.random()), n) for n in scored_candidates]
            scored_pairs.sort(reverse=True)
            ranked_scored = [n for _, _, n in scored_pairs]

            unscored_pairs = [(float(self.rng.random()), n) for n in unscored_candidates]
            unscored_pairs.sort(reverse=True)
            ranked_unscored = [n for _, n in unscored_pairs]

            ranked_all = ranked_scored + ranked_unscored
            return (ranked_all, ranked_scored) if return_scored else ranked_all

        elif self.selector == "reported_cross_group":
            scored_candidates = [n for n in cand_list if cross_grp[n] > 0.0]
            unscored_candidates = [n for n in cand_list if cross_grp[n] <= 0.0]

            scored_pairs = [(float(cross_grp[n]), float(self.rng.random()), n) for n in scored_candidates]
            scored_pairs.sort(reverse=True)
            ranked_scored = [n for _, _, n in scored_pairs]

            unscored_pairs = [(float(self.rng.random()), n) for n in unscored_candidates]
            unscored_pairs.sort(reverse=True)
            ranked_unscored = [n for _, n in unscored_pairs]

            ranked_all = ranked_scored + ranked_unscored
            return (ranked_all, ranked_scored) if return_scored else ranked_all

        elif self.selector == "group_balanced":
            # Group nodes by group label, sort within group by reported attention, then cycle
            by_group: Dict[int, List[int]] = {}
            for node in cand_list:
                grp = int(groups[node])
                by_group.setdefault(grp, []).append(node)

            for grp, members in by_group.items():
                members.sort(key=lambda idx: (rec_att[idx], float(self.rng.random())), reverse=True)

            ordered: List[int] = []
            grp_keys = sorted(by_group.keys())
            while any(len(by_group[g]) > 0 for g in grp_keys):
                for g in grp_keys:
                    if by_group[g]:
                        ordered.append(by_group[g].pop(0))
            scored_set = [n for n in ordered if rec_att[n] > 0.0]
            return (ordered, scored_set) if return_scored else ordered

        return (cand_list, cand_list) if return_scored else cand_list

    def _select_initial_nodes(self, observation: Observation, k_target: int) -> List[int]:
        """Select initial nodes respecting quota between observed and unobserved accessible nodes."""
        roster = list(observation.accessible_roster)
        if len(roster) == 0:
            return []

        if self.roster_selection:
            # v1 fixed-roster comparison: no arbitrary panel quota. Random is
            # uniform over all accessible nodes; attention uses revealed edges.
            ranked, scored = self._rank_nodes(roster, observation, return_scored=True)
            if self.selector in ("reported_attention", "reported_low_attention") and len(scored) < k_target:
                self.fallback_records.append({
                    "step": "initial_selection", "pool": "accessible_roster",
                    "requested_slots": k_target, "scored_slots_available": len(scored),
                    "fallback_count": min(k_target, len(roster)) - len(scored),
                })
            return ranked[:k_target]

        observed_accessible = [n for n in roster if observation.is_node_observed(n)]
        unobserved_accessible = [n for n in roster if not observation.is_node_observed(n)]

        n_obs_slots = int(round(k_target * self.quota))
        n_unobs_slots = k_target - n_obs_slots

        # If either pool is smaller than assigned quota, redistribute remainder
        if len(observed_accessible) < n_obs_slots:
            rem = n_obs_slots - len(observed_accessible)
            n_obs_slots = len(observed_accessible)
            n_unobs_slots += rem
        if len(unobserved_accessible) < n_unobs_slots:
            rem = n_unobs_slots - len(unobserved_accessible)
            n_unobs_slots = len(unobserved_accessible)
            n_obs_slots = min(len(observed_accessible), n_obs_slots + rem)

        ranked_obs, scored_obs = self._rank_nodes(observed_accessible, observation, return_scored=True)
        selected_obs = ranked_obs[:n_obs_slots]

        # Record fallback if scored pool is smaller than requested slots
        if self.selector in ("reported_attention", "reported_low_attention") and n_obs_slots > 0:
            scored_available = len(scored_obs)
            if scored_available < n_obs_slots:
                fallback_count = n_obs_slots - scored_available
                self.fallback_records.append({
                    "step": "initial_selection",
                    "pool": "observed_accessible",
                    "requested_slots": n_obs_slots,
                    "scored_slots_available": scored_available,
                    "fallback_count": fallback_count,
                    "selected_scored": list(selected_obs[:scored_available]),
                    "selected_fallback": list(selected_obs[scored_available:]),
                })

        # For unobserved, rank or sample uniformly
        ranked_unobs = self._rank_nodes(unobserved_accessible, observation)
        selected_unobs = ranked_unobs[:n_unobs_slots]

        selected = list(dict.fromkeys(selected_obs + selected_unobs))
        return selected[:k_target]

    def _determine_target_signs(
        self,
        nodes: Sequence[int],
        observation: Observation,
    ) -> Optional[Dict[int, float]]:
        """Determine actuation sign for selected nodes.

        Direction knowledge contract (option (b), see class docstring): for a
        REVERSAL target the sign is RESTRICTED INFERENCE of the initial public
        majority, locked once at the first decisive report (the immutable
        evaluation target s_* from the true initial majority lives in
        TargetEvaluator and is never consulted here). Returns None if the
        target sign is missing/ambiguous (empty or tied report): the caller
        must WAIT without actuating, without spending budget, and without
        deploying nodes.

        On lock, the inference provenance is written to effective_settings:
        'controller_inferred_target_sign', 'inference_report_mean',
        'inference_report_sample_time', 'inference_source'.
        """
        signs: Dict[int, float] = {}
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # Target direction
        s_star = 1.0
        if self._has_directional_task():
            # Target sign s_* = -sign(m_y_obs) inferred from the restricted report
            if self.s_star is None:
                m_obs = self._observed_target_mean(observation)
                if m_obs is None:
                    return None  # Wait/no actuation until a decisive public report arrives
                self.s_star = -1.0 if m_obs >= 0.0 else 1.0
                focal = self.problem.target_params.get("focal_indices")
                self.effective_settings["controller_inferred_target_sign"] = float(self.s_star)
                self.effective_settings["inference_report_mean"] = float(m_obs)
                self.effective_settings["inference_report_sample_time"] = float(observation.sample_time)
                self.effective_settings["inference_source"] = (
                    "focal_panel_mean" if focal else "full_panel_mean"
                )
            s_star = self.s_star

        for n in nodes:
            if self._has_directional_task():
                signs[n] = s_star
            else:
                # Decentralization / structural: counter-alignment -sign(y_i)
                if observation.is_node_observed(n):
                    val = observation.public_states[n, self.component]
                    if np.isfinite(val) and abs(val) > 1e-6:
                        signs[n] = -1.0 if val > 0.0 else 1.0
                    else:
                        signs[n] = float(self.rng.choice([-1.0, 1.0]))
                else:
                    signs[n] = float(self.rng.choice([-1.0, 1.0]))
        return signs

    def _combined_initial_action(self, selected, t, ledger):
        """One actual field per node: broadcast plus an observed-targeted boost.

        Scale both components together if the combined amplitude or exposure
        would exceed permission. The shared ledger accounts for the actual sum.
        """
        roster = tuple(sorted(self.problem.accessible_roster))
        duration = self.committed_end_time - t
        background = self.broadcast_amplitude
        boost = self.nominal_amplitude
        rate = len(roster)*background + len(selected)*boost
        if duration <= 0 or rate <= 0:
            return Action(entries=(), intended_end_time=t, release=True)
        scale = min(1.0, self.problem.amplitude_cap/(background+boost),
                    ledger.remaining_exposure/(rate*duration))
        if scale <= 1e-15:
            return Action(entries=(), intended_end_time=t, release=True)
        selected = set(selected)
        entries = tuple(ActionEntry(node=n, component=self.component,
                            field=self.s_star*scale*(background + (boost if n in selected else 0.0)))
                        for n in roster)
        self.used_nodes.update(roster)
        self.requested_settings = {'broadcast_amplitude':background, 'targeted_increment':boost,
                                   'targeted_fraction':self.node_fraction, 'duration':self.duration,
                                   'selector':self.selector, 'is_feedback':False}
        self.effective_settings.update({
            'broadcast_amplitude':background*scale, 'targeted_increment':boost*scale,
            'targeted_nodes':sorted(selected), 'initial_nodes':list(roster),
            'effective_node_count':len(roster), 'effective_amplitude':(background+boost)*scale,
            'effective_end_time':self.committed_end_time, 'component_scale':scale,
            'spending_rate':rate*scale, 'broadcast_spending_rate':len(roster)*background*scale,
            'targeted_increment_spending_rate':len(selected)*boost*scale,
            'roster_selection':self.roster_selection,
            'amplitude_reduced_by_exposure_cap':ledger.remaining_exposure < rate*duration-1e-9,
        })
        return Action(entries=entries, intended_end_time=self.committed_end_time, release=False)

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        if self.problem is None:
            raise RuntimeError("Controller must be reset before act()")

        t = observation.time
        if ledger.is_released:
            return Action(entries=(), release=True)

        # Deadline or expiration
        if self.committed_end_time is not None and t >= self.committed_end_time - 1e-12:
            return Action(entries=(), intended_end_time=t, release=True)

        if not self.initial_actuation_done:
            # Check if target sign can be determined
            k_active = min(self.problem.k, len(observation.accessible_roster))
            if self.node_fraction is not None:
                n_req = max(1, int(round(self.problem.N * self.node_fraction)))
                k_active = min(n_req, k_active)

            if self.is_feedback and self.initial_recruitment_fraction < 1.0:
                initial_k = max(1, int(np.floor(k_active * self.initial_recruitment_fraction)))
            else:
                initial_k = k_active

            initial_nodes_temp = self._select_initial_nodes(observation, initial_k)
            signs = self._determine_target_signs(initial_nodes_temp, observation)
            
            if signs is None:
                # Wait for next observation
                return Action(entries=(), release=False)

            self.committed_end_time = min(t + self.duration, self.problem.release_deadline)
            self.initial_actuation_done = True
            initial_nodes = initial_nodes_temp
            for n in initial_nodes:
                self.used_nodes.add(n)

            if len(initial_nodes) == 0:
                return Action(entries=(), release=True)

            if self.broadcast_amplitude:
                return self._combined_initial_action(initial_nodes, t, ledger)

            # Compute effective amplitude
            rem_B = ledger.remaining_exposure
            dur = self.committed_end_time - t
            max_amp = rem_B / (len(initial_nodes) * dur) if (len(initial_nodes) * dur) > 0 else 0.0
            a_eff = min(self.nominal_amplitude, max_amp, self.problem.amplitude_cap)

            is_amp_reduced = bool(a_eff < min(self.nominal_amplitude, self.problem.amplitude_cap) - 1e-9)
            self.requested_settings = {
                "nominal_amplitude": self.nominal_amplitude,
                "duration": self.duration,
                "node_fraction": self.node_fraction,
                "quota": self.quota,
                "selector": self.selector,
                "is_feedback": self.is_feedback,
                "requested_active_nodes": int(round(self.problem.N * self.node_fraction)) if self.node_fraction is not None else self.problem.k,
            }
            self.effective_settings = {
                **self.effective_settings,  # preserve inference provenance keys (F3)
                "effective_amplitude": float(a_eff),
                "effective_node_count": len(initial_nodes),
                "initial_nodes": list(initial_nodes),
                "roster_selection": self.roster_selection,
                "effective_end_time": float(self.committed_end_time),
                "amplitude_reduced_by_exposure_cap": is_amp_reduced,
            }

            if a_eff <= 1e-15:
                return Action(entries=(), release=True)

            entries = [
                ActionEntry(node=n, component=self.component, field=float(signs[n] * a_eff))
                for n in initial_nodes
            ]
            return Action(entries=tuple(entries), intended_end_time=self.committed_end_time, release=False)

        # Subsequent observation deliveries
        if not self.is_feedback:
            # Static policy does not adapt between initial actuation and pulse end
            if ledger.active_action is not None:
                return ledger.active_action
            return Action(entries=(), release=True)

        # Feedback policy adaptation:
        # Record feedback opportunity encountered
        self.feedback_opportunities_encountered += 1

        rem_k = ledger.remaining_distinct_nodes
        rem_dur = self.committed_end_time - t
        if rem_dur <= 1e-12 or ledger.remaining_exposure <= 1e-12:
            return Action(entries=(), release=True)

        current_active = set(self.used_nodes)
        # Prioritize nodes that currently oppose target
        s_star = 1.0
        if self._has_directional_task():
            if self.s_star is None:
                m_obs = self._observed_target_mean(observation)
                if m_obs is None:
                    # Missing/ambiguous report: wait without actuation, budget, or deployment.
                    return Action(entries=(), release=False)
                self.s_star = -1.0 if m_obs >= 0.0 else 1.0
                focal = self.problem.target_params.get("focal_indices")
                self.effective_settings["controller_inferred_target_sign"] = float(self.s_star)
                self.effective_settings["inference_report_mean"] = float(m_obs)
                self.effective_settings["inference_report_sample_time"] = float(observation.sample_time)
                self.effective_settings["inference_source"] = (
                    "focal_panel_mean" if focal else "full_panel_mean"
                )
            s_star = self.s_star

        if self.node_fraction is not None:
            n_req = max(1, int(round(self.problem.N * self.node_fraction)))
            max_active = min(n_req, self.problem.k, len(observation.accessible_roster))
        else:
            max_active = min(self.problem.k, len(observation.accessible_roster))

        remaining_slots = max(0, max_active - len(self.used_nodes))
        recruit_allowance = max(1, int(np.floor(max_active * self.initial_recruitment_fraction)))
        n_to_recruit = min(rem_k, remaining_slots, recruit_allowance)

        nodes_added = False
        if n_to_recruit > 0:
            unused_roster = [n for n in observation.accessible_roster if n not in self.used_nodes]
            if unused_roster:
                ranked_unused = self._rank_nodes(unused_roster, observation)
                to_add = ranked_unused[:n_to_recruit]
                for n in to_add:
                    if len(self.used_nodes) < max_active:
                        self.used_nodes.add(n)
                        nodes_added = True

        # Active nodes for this interval
        active_list = sorted(self.used_nodes)
        if len(active_list) == 0:
            return Action(entries=(), release=True)

        signs = self._determine_target_signs(active_list, observation)

        # Zero out field on nodes that are already strongly aligned with target
        filtered_nodes: List[int] = []
        aligned_nodes_zeroed = False
        for n in active_list:
            if observation.is_node_observed(n):
                y_val = observation.public_states[n, self.component]
                if np.isfinite(y_val):
                    if self._has_directional_task():
                        # If already aligned (s_* * y >= 0.5), do not waste field
                        if s_star * y_val >= 0.5:
                            aligned_nodes_zeroed = True
                            continue
            filtered_nodes.append(n)

        if nodes_added or aligned_nodes_zeroed:
            self.feedback_updates_performed += 1

        if len(filtered_nodes) == 0:
            # All active nodes are observed and aligned with target.
            # Spec Section 9: "At aligned nodes set the field to zero, retaining their consumed slots."
            return Action(entries=(), intended_end_time=self.committed_end_time, release=False)

        rem_B = ledger.remaining_exposure
        max_amp = rem_B / (len(filtered_nodes) * rem_dur) if (len(filtered_nodes) * rem_dur) > 0 else 0.0
        a_eff = min(self.nominal_amplitude, max_amp, self.problem.amplitude_cap)

        if a_eff <= 1e-15:
            return Action(entries=(), release=True)

        entries = [
            ActionEntry(node=n, component=self.component, field=float(signs[n] * a_eff))
            for n in filtered_nodes
        ]
        return Action(entries=tuple(entries), intended_end_time=self.committed_end_time, release=False)


class OracleMPCController(Controller):
    """Reference oracle MPC controller using cloned rollouts and independent RNG streams."""

    def __init__(
        self,
        nominal_amplitude: float = 1.0,
        horizon_steps: int = 4,
        interval: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.nominal_amplitude = float(nominal_amplitude)
        self.horizon_steps = int(horizon_steps)
        self.interval = float(interval)
        self.seed = int(seed)

        self.problem: Optional[ControlProblem] = None
        self.rng: Optional[np.random.Generator] = None
        self.s_star: Optional[float] = None

    def reset(
        self,
        problem: ControlProblem,
        initial_observation: Observation,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)

    def act(self, observation: Observation, ledger: BudgetLedger) -> Action:
        if self.problem is None or not observation.is_oracle:
            # Fallback if non-oracle
            return Action(entries=(), release=True)

        if ledger.is_released or ledger.remaining_exposure <= 1e-12:
            return Action(entries=(), release=True)

        t = observation.time
        end_t = min(t + self.interval, self.problem.release_deadline)
        if end_t <= t:
            return Action(entries=(), release=True)

        # Select top nodes opposing target
        if self.s_star is None:
            m_y = float(np.mean(observation.public_states[:, 0]))
            self.s_star = -1.0 if m_y >= 0.0 else 1.0
        s_star = self.s_star

        scores = [observation.public_states[i, 0] * s_star for i in range(self.problem.N)]
        ranked_nodes = np.argsort(scores)  # Lowest score = most opposing

        # Select up to remaining k nodes from accessible roster
        roster_set = set(observation.accessible_roster)
        selected: List[int] = []
        for n in ranked_nodes:
            if n in roster_set:
                selected.append(int(n))
                if len(selected) >= min(self.problem.k, ledger.remaining_distinct_nodes):
                    break

        if not selected:
            return Action(entries=(), release=True)

        rem_B = ledger.remaining_exposure
        dur = end_t - t
        max_amp = rem_B / (len(selected) * dur) if (len(selected) * dur) > 0 else 0.0
        a_eff = min(self.nominal_amplitude, max_amp, self.problem.amplitude_cap)

        if a_eff <= 1e-15:
            return Action(entries=(), release=True)

        entries = [
            ActionEntry(node=n, component=0, field=float(s_star * a_eff))
            for n in selected
        ]
        return Action(entries=tuple(entries), intended_end_time=end_t, release=False)
