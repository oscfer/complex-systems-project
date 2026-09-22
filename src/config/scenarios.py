"""Constructed Tiered-population opinion-network origins.

All numeric stance, attention and opportunity inputs are declared synthetic assumptions,
not measured model parameters.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.heterogeneity import generate_quenched_heterogeneity
from src.core.state import State, StateMode


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_tiered_spec(path: str = "research/scenarios/tiered/scenario.json") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_tiered_spec(spec)
    return spec


def _allocate(shares: list[float], n: int) -> list[int]:
    """Largest remainders, with source order breaking exact ties."""
    exact = np.asarray(shares, dtype=float) * n
    counts = np.floor(exact).astype(int)
    order = sorted(range(len(shares)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:n - int(counts.sum())]:
        counts[i] += 1
    return counts.tolist()


def validate_tiered_spec(spec: dict) -> None:
    if not isinstance(spec, dict) or int(spec.get("schema_version", 1)) != 2:
        raise ValueError("Tiered-population specification requires schema_version=2 (joint role/jurisdiction allocation)")
    if not isinstance(spec, dict) or len(spec.get("tiers", [])) != 5:
        raise ValueError("Tiered-population specification needs five tiers")
    tiers = spec["tiers"]
    if [t.get("id") for t in tiers] != [1, 2, 3, 4, 5]:
        raise ValueError("Tiered-population tier IDs must be 1 through 5 in order")
    for name, entries, key in (("tier", tiers, "model_share"),
                               ("jurisdiction", spec.get("jurisdictions", []), "share")):
        if not entries or any(not 0 <= float(e[key]) <= 1 for e in entries):
            raise ValueError(f"Invalid {name} shares")
        if not np.isclose(sum(float(e[key]) for e in entries), 1.0, atol=1e-10):
            raise ValueError(f"{name} shares must sum to one")
    jurisdiction_defs = spec.get("jurisdictions", [])
    kinds = [j.get("kind") for j in jurisdiction_defs]
    if any(kind not in ("named", "transnational", "unspecified") for kind in kinds):
        raise ValueError("Every jurisdiction needs a declared kind: named, transnational or unspecified")
    if kinds.count("transnational") != 1 or kinds.count("unspecified") != 1:
        raise ValueError("Exactly one declared transnational and one declared unspecified/missing jurisdiction are required")
    for tier in tiers:
        if set(tier.get("contextual_attributes", {})) != {"material_power", "information_advantage", "coordination", "risk_exposure", "exit_power"}:
            raise ValueError("Every tier needs all five contextual attributes")
        if not tier.get("roles"):
            raise ValueError("Every tier needs at least one role")
        for role in tier["roles"]:
            if not isinstance(role, dict) or not role.get("name"):
                raise ValueError("Each role must be an object with a declared name")
            shares = role.get("jurisdiction_shares")
            if not isinstance(shares, list) or len(shares) != len(jurisdiction_defs):
                raise ValueError(f"Role '{role.get('name')}' must declare one jurisdiction share per declared jurisdiction")
            if any(not 0 <= float(s) <= 1 for s in shares):
                raise ValueError(f"Role '{role.get('name')}' has invalid jurisdiction shares")
            if not np.isclose(sum(float(s) for s in shares), 1.0, atol=1e-10):
                raise ValueError(f"Role '{role.get('name')}' jurisdiction shares must sum to one")
        for key in ("pilot_public_plus1_prob", "pilot_private_public_gap_prob"):
            if not 0 <= float(tier[key]) <= 1:
                raise ValueError(f"{key} must be a probability")
        if float(tier["attention_prior"]) < 0:
            raise ValueError("attention_prior must be nonnegative")
    if not 0 < float(spec.get("core_attention_share", 0.7)) < 1:
        raise ValueError("core_attention_share must be between zero and one")
    if int(spec.get("memory_draws_per_row", 20)) < 0:
        raise ValueError("memory_draws_per_row must be nonnegative")
    opportunity = spec.get("opportunity", {})
    for key in ("within_jurisdiction", "between_jurisdictions",
                "transnational_affinity_to_any", "unspecified_affinity_to_any"):
        default = 1.0 if key == "within_jurisdiction" else (0.2 if key == "transnational_affinity_to_any" else 0.15)
        if float(opportunity.get(key, default)) < 0:
            raise ValueError("opportunity affinity factors must be nonnegative")


def jurisdiction_pair_affinity(spec: dict) -> np.ndarray:
    """Label-level pairwise encounter affinity matrix from the declared opportunity rule.

    Same named jurisdiction: within_jurisdiction; different named: between_jurisdictions.
    Any pair involving a transnational actor (including transnational-transnational):
    transnational_affinity_to_any. Any pair involving an unspecified/missing actor
    (including unspecified-unspecified): unspecified_affinity_to_any. Missing or
    ambiguous jurisdiction therefore confers no within-community bonus, and declared
    transnational reach is a moderate open affinity, not a co-located community.
    """
    opportunity = spec.get("opportunity", {})
    within = float(opportunity.get("within_jurisdiction", 1.0))
    between = float(opportunity.get("between_jurisdictions", 0.05))
    trans = float(opportunity.get("transnational_affinity_to_any", 0.2))
    unspec = float(opportunity.get("unspecified_affinity_to_any", 0.15))
    kinds = [j.get("kind") for j in spec["jurisdictions"]]
    n = len(kinds)
    E = np.empty((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            if kinds[a] == "unspecified" or kinds[b] == "unspecified":
                E[a, b] = unspec
            elif kinds[a] == "transnational" or kinds[b] == "transnational":
                E[a, b] = trans
            elif a == b:
                E[a, b] = within
            else:
                E[a, b] = between
    return E


def allocate_joint_role_jurisdictions(spec: dict, tier_counts: list[int], rng: np.random.Generator):
    """Jointly allocate (role, jurisdiction) pairs consistent with declared memberships.

    Within each tier: role counts are largest-remainder on uniform role shares; each
    role's nodes are split across jurisdictions by largest remainders on that role's
    declared jurisdiction_shares, capped by the remaining global jurisdiction quota
    declared in spec["jurisdictions"][j]["share"]. Remaining column deviations after
    row allocation are removed by unit transfers between over- and under-quota
    jurisdictions within a single (tier, role) row, preferring donor rows whose
    declared share for the receiving jurisdiction is positive. Exact tier, role and
    jurisdiction marginals are therefore preserved, and the realized (tier, role,
    jurisdiction) cross-tab is deterministic and exactly checkable.
    Jurisdiction labels are shuffled within each tier so node index carries no
    systematic jurisdiction order. Returns (role_names_per_node, jurisdiction_ids,
    crosstab) with crosstab[(tier_index, role_index, jurisdiction_index)] = count.
    """
    jurisdiction_defs = spec["jurisdictions"]
    n_j = len(jurisdiction_defs)
    total = int(sum(tier_counts))
    targets = _allocate([float(j["share"]) for j in jurisdiction_defs], total)
    if sum(targets) != total:
        raise ValueError("Jurisdiction quota allocation failed")
    cells: dict[tuple[int, int, int], int] = {}
    col_sum = [0] * n_j
    for tier_index, (tier, count) in enumerate(zip(spec["tiers"], tier_counts)):
        role_names = [role["name"] for role in tier["roles"]]
        role_counts = _allocate([1 / len(role_names)] * len(role_names), count)
        for role_index, (role_def, rc) in enumerate(zip(tier["roles"], role_counts)):
            shares = [float(s) for s in role_def["jurisdiction_shares"]]
            ideal = [rc * s for s in shares]
            floors = [int(np.floor(v)) for v in ideal]
            for j in range(n_j):
                cells[(tier_index, role_index, j)] = floors[j]
                col_sum[j] += floors[j]
            deficit = rc - sum(floors)
            order = sorted(range(n_j), key=lambda j: (-(ideal[j] - floors[j]), j))
            # Admissibility: declared zero-share jurisdictions must never
            # receive nodes, even as remainders (audit F2: valid joint
            # allocation, exact allocated counts). Rows are allocated purely
            # by their declared shares; column quota deviations are repaired
            # afterwards by admissible transfers.
            eligible = [j for j in order if shares[j] > 0.0]
            while deficit > 0:
                progressed = False
                for j in eligible:
                    cells[(tier_index, role_index, j)] += 1
                    col_sum[j] += 1
                    deficit -= 1
                    progressed = True
                    if deficit == 0:
                        break
                if not progressed:
                    raise ValueError(
                        "Role row has no positive jurisdiction share to place nodes in"
                    )
    # Column reconciliation by admissible unit transfers. One transfer moves a
    # node between two columns inside a single (tier, role) row; the receiving
    # column must have positive declared share for that row (forbidden cells
    # are never filled), and multi-hop chains preserve every row sum.
    def _transfer(row: tuple[int, int], j_from: int, j_to: int) -> None:
        cells[(row[0], row[1], j_from)] -= 1
        cells[(row[0], row[1], j_to)] += 1
        col_sum[j_from] -= 1
        col_sum[j_to] += 1

    guard = 0
    while True:
        deficits = [targets[j] - col_sum[j] for j in range(n_j)]
        if all(d == 0 for d in deficits):
            break
        guard += 1
        if guard > 10 * total:
            raise ValueError("Joint role/jurisdiction allocation failed to reconcile marginals")
        j_under = max(range(n_j), key=lambda j: deficits[j])
        j_over = min(range(n_j), key=lambda j: deficits[j])
        # BFS for an admissible transfer chain j_over -> ... -> j_under. A row
        # can pass a unit from any column it currently holds to any other
        # column whose declared share for that row is positive.
        rows_by_col: dict[tuple[int, int], list[int]] = {}
        for (t, r, j), c in cells.items():
            if c > 0:
                rows_by_col.setdefault((t, r), []).append(j)
        visited: dict[int, tuple[int, tuple[int, int]] | None] = {j_over: None}
        queue = [j_over]
        found = None
        while queue and found is None:
            col = queue.pop(0)
            for (t, r), held in rows_by_col.items():
                if col not in held:
                    continue
                row_shares = spec["tiers"][t]["roles"][r]["jurisdiction_shares"]
                for to in range(n_j):
                    if to == col or to in visited or row_shares[to] <= 0.0:
                        continue
                    visited[to] = (col, (t, r))
                    if deficits[to] > 0:
                        found = to
                        break
                    queue.append(to)
                if found is not None:
                    break
        if found is None:
            raise ValueError(
                "No admissible transfer path to reconcile jurisdiction marginals"
            )
        to = found
        while visited[to] is not None:
            col, row = visited[to]
            _transfer(row, col, to)
            to = col
    # Materialize node lists per tier, shuffling jurisdictions within each
    # (tier, role) group so node index carries no systematic jurisdiction
    # order while preserving the exact role<->jurisdiction pairing.
    roles: list[str] = []
    jurisdiction_ids: list[int] = []
    crosstab = {key: count for key, count in cells.items() if count > 0}
    for tier_index, (tier, count) in enumerate(zip(spec["tiers"], tier_counts)):
        role_names = [role["name"] for role in tier["roles"]]
        role_counts = _allocate([1 / len(role_names)] * len(role_names), count)
        for role_index, rc in enumerate(role_counts):
            role_jurisdictions: list[int] = []
            for j in range(n_j):
                role_jurisdictions.extend([j] * cells[(tier_index, role_index, j)])
            if len(role_jurisdictions) != rc:
                raise ValueError("Role row materialization lost jurisdiction nodes")
            rng.shuffle(role_jurisdictions)
            roles.extend([role_names[role_index]] * rc)
            jurisdiction_ids.extend(role_jurisdictions)
    return roles, jurisdiction_ids, crosstab


def build_tiered_origin(
    spec: dict, seed: int, N: int = 40, *, profile: Any = None,
) -> State:
    validate_tiered_spec(spec)
    if N < 5:
        raise ValueError("Tiered-population origins require at least five model agents")
    if profile is not None and int(profile.D) != 1:
        raise ValueError("Tiered-population pilot scenario is declared for D=1")
    rng = np.random.default_rng(seed)
    tiers = spec["tiers"]
    tier_counts = _allocate([float(t["model_share"]) for t in tiers], N)
    tier_ids = np.concatenate([np.full(c, t["id"], dtype=np.int64) for t, c in zip(tiers, tier_counts)])
    if not np.any(tier_ids <= 3) or not np.any(tier_ids >= 4):
        raise ValueError("Both attention groups need at least one agent")

    roles, jurisdiction_ids, role_jurisdiction_crosstab = allocate_joint_role_jurisdictions(
        spec, tier_counts, rng
    )
    jurisdiction_ids = np.asarray(jurisdiction_ids, dtype=np.int64)
    jurisdiction_defs = spec["jurisdictions"]
    jurisdictions = [jurisdiction_defs[int(j)]["name"] for j in jurisdiction_ids]

    x = np.empty((N, 1), dtype=np.float64)
    y = np.empty((N, 1), dtype=np.float64)
    for i, tier_id in enumerate(tier_ids):
        tier = tiers[int(tier_id) - 1]
        y[i, 0] = 1.0 if rng.random() < tier["pilot_public_plus1_prob"] else -1.0
        x[i, 0] = -y[i, 0] if rng.random() < tier["pilot_private_public_gap_prob"] else y[i, 0]

    cap = int(spec.get("memory_capacity_per_row", getattr(profile, "memory_capacity_per_row", 40)))
    draws = int(spec.get("memory_draws_per_row", 20))
    if cap < draws:
        raise ValueError("memory_capacity_per_row must cover initial draws")
    core_mass = float(spec.get("core_attention_share", 0.7))
    core = tier_ids <= 3
    E_labels = jurisdiction_pair_affinity(spec)
    C = E_labels[np.ix_(jurisdiction_ids, jurisdiction_ids)].copy()
    np.fill_diagonal(C, 0.0)
    W = np.zeros((N, N), dtype=np.uint32)
    nodes = np.arange(N)
    priors = np.array([float(tiers[int(tid) - 1]["attention_prior"]) for tid in tier_ids])
    for i in range(N):
        weights = np.zeros(N, dtype=float)
        for group, mass in ((core, core_mass), (~core, 1.0 - core_mass)):
            eligible = group.copy()
            eligible[i] = False
            # Same affinity weighting as ongoing encounters (scenario.json
            # opportunity.initial_memory_uses_affinity): initial within-group
            # allocation is weighted by the declared pairwise jurisdiction
            # affinity C[i, j], so t=0 tie formation and later tie formation
            # share one opportunity structure.
            group_weights = np.where(eligible, priors, 0.0) * C[i]
            if group_weights.sum() <= 0:
                raise ValueError("Each attention group needs an eligible recipient")
            weights += mass * group_weights / group_weights.sum()
        W[i] = rng.multinomial(draws, weights).astype(np.uint32)

    J = float(getattr(profile, "J", 1.0))
    traits = generate_quenched_heterogeneity(
        N=N, D=1, J=J,
        K_nominal=float(getattr(profile, "K_nominal", 0.5)),
        tau_x_nominal=float(getattr(profile, "tau_private_nominal", 5.0)),
        nu_nominal=float(getattr(profile, "nu_nominal", 2.0)), rng=rng,
    )
    opportunity = spec.get("opportunity", {})
    metadata = {
        "schema_version": 2,
        "source": "tiered_constructed",
        "spec_sha256": _digest(spec),
        "seed": int(seed),
        "node_ids": [f"node_{i:04d}" for i in range(N)],
        "tier_assignments": tier_ids.tolist(),
        "roles": roles,
        "jurisdictions": jurisdictions,
        "jurisdiction_ids": jurisdiction_ids.tolist(),
        "jurisdiction_kinds": [jurisdiction_defs[int(j)]["kind"] for j in jurisdiction_ids],
        "role_jurisdiction_crosstab": {
            f"{tier_index + 1}|{spec['tiers'][tier_index]['roles'][role_index]['name']}|{jurisdiction_defs[j_idx]['name']}": count
            for (tier_index, role_index, j_idx), count in sorted(role_jurisdiction_crosstab.items())
        },
        "contextual_attributes": [copy.deepcopy(tiers[int(tid) - 1]["contextual_attributes"]) for tid in tier_ids],
        "focal_proposition": spec["focal_proposition"],
        "aggregation": spec["aggregation"],
        "attention_target_core_share": core_mass,
        "opportunity": {
            "grouping": opportunity.get("grouping", "declared_jurisdiction_pair_affinity"),
            "within_jurisdiction": float(opportunity.get("within_jurisdiction", 1.0)),
            "between_jurisdictions": float(opportunity.get("between_jurisdictions", 0.05)),
            "transnational_affinity_to_any": float(opportunity.get("transnational_affinity_to_any", 0.2)),
            "unspecified_affinity_to_any": float(opportunity.get("unspecified_affinity_to_any", 0.15)),
            "initial_memory_uses_affinity": bool(opportunity.get("initial_memory_uses_affinity", True)),
        },
        "persistent_structural_asymmetry": copy.deepcopy(spec.get("persistent_structural_asymmetry", {})),
        "attribute_effects": {
            key: "contextual_only" for key in (
                "material_power", "information_advantage", "coordination",
                "risk_exposure", "exit_power")
        },
        "assumptions": "Stance probabilities, attention mass, memory, opportunity and traits are pilot model assumptions, not measured public or institutional quantities",
    }
    state = State(
        N=N, D=1, state_mode=StateMode.BINARY, x=x, y=y, W=W,
        n=np.zeros(N, dtype=np.uint8), Q=np.zeros((N, 1, 2), dtype=np.float64),
        t=0.0, blocks=tier_ids, opportunity_groups=jurisdiction_ids,
        opportunity_matrix=C,
        row_caps=np.full(N, cap, dtype=np.uint32),
        K=traits.K, tau_x=traits.tau_x, nu=traits.nu, g=traits.g, b=traits.b,
        scenario_metadata=metadata,
    )
    state.validate()
    return state


def tiered_origin_provider(profile: Any, seed: int) -> State:
    """Resolve the versioned spec at call time, so changed files change origins."""
    spec = load_tiered_spec(profile.scenario_spec_path)
    return build_tiered_origin(spec, seed, int(profile.N), profile=profile)
