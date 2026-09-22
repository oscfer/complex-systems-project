# Tiered-population initial conditions

This document specifies the constructed initial states of the tiered-population opinion-transformation scenario. All numbers in `scenario.json` are synthetic modelling assumptions, not empirical calibrations.

## Focal opinion

The binary stance is **continued expansion of a contested activity (+1) versus slowdown/restriction (-1)**. The collective outcome is the equal-weight agent majority.

## Actor allocation

Model agents are split into five tiers:

1. **Gatekeepers and infrastructure**: 15% (e.g. states/regulators, component suppliers, utilities/energy)
2. **Leading developers**: 10% (e.g. lead developers, distributors)
3. **Capital and deployers**: 15% (e.g. investors, enterprise/public deployers)
4. **Institutional intermediaries**: 15% (e.g. researchers, standards, auditors, media, unions/civil society)
5. **Abundant periphery**: 45% (e.g. workers, users, educators, creators, local communities, other countries)

Jurisdiction labels (each carries an explicit `kind` attribute in `scenario.json`):

- Region A: 20% (kind: named)
- Region B: 12.5% (kind: named)
- Region C: 10% (kind: named)
- Other states/regions: 12.5% (kind: named)
- Transnational: 25% (kind: transnational), a declared global-reach attribute (e.g. standards bodies, multinationals, global platforms), not a catch-all.
- Unspecified/missing: 20% (kind: unspecified), a declared missing-jurisdiction attribute. Missing or ambiguous jurisdiction is an explicit label and is not a shared community.

Each role declares its own jurisdiction distribution (`roles[].jurisdiction_shares`).

## Initial beliefs and attention

- Attention prior: tiers 1-3 receive 70% of normalized incoming attention; tiers 4-5 receive 30%.
- Public +1 probability by tier: [0.80, 0.85, 0.75, 0.45, 0.55].
- Private/public gap probability by tier: [0.10, 0.15, 0.10, 0.10, 0.15].

## Executable construction

Tier counts are allocated by largest remainders, with source order breaking ties. Roles are divided within each tier by largest remainders on uniform role shares. Each role's nodes are jointly allocated across jurisdictions by largest remainders on that role's `jurisdiction_shares`, capped by the remaining global jurisdiction quota. Remaining column deviations are removed by unit transfers between over- and under-quota jurisdictions within a single (tier, role) row, preferring donor rows whose declared share for the receiving jurisdiction is positive. Tier, role and jurisdiction marginals are therefore exact at any N (6/4/6/6/18 tiers at N=40; 75/50/75/75/225 at N=500). The realized (tier, role, jurisdiction) cross-tab is deterministic and stored in each origin's metadata (`role_jurisdiction_crosstab`). Jurisdiction labels are shuffled within each tier so node index carries no systematic jurisdiction order.

Each row of the directed integer attention memory `W` receives 20 initial draws under a cap of 40. The configured `core_attention_share` divides each row's probability mass between tiers 1-3 and tiers 4-5 before within-group allocation, excluding self-attention. The 70% core share is therefore an expected value; realized finite samples vary. Setting it to 0.60 or 0.80 gives concentration sensitivities. Within each attention group the remaining mass is allocated with weights `priors[j] * C[i, j]`, where `C` is the same pairwise jurisdiction affinity used for ongoing encounters, so initial and later tie formation share one opportunity structure. `W` is remembered attention, not money, ownership, legal authority or power.

## Encounter opportunity structure

`opportunity.grouping = declared_jurisdiction_pair_affinity` materializes a full nonnegative N x N matrix `C` with zero diagonal, preserved in origin snapshots and controlled checkpoints. Pairwise affinities are declared in `scenario.json`:

| Pair | Affinity |
|---|---|
| Same named jurisdiction | 1.0 |
| Different named jurisdictions | 0.05 |
| Any pair involving a transnational actor | 0.2 |
| Any pair involving an unspecified/missing actor | 0.15 |

Missing jurisdiction confers no within-community bonus, and transnational reach gives a moderate open affinity in both directions. At N=40 the maximum exploration-only within-group probability is about 24.3% for a transnational node and 17.9% for an unspecified node, while same-jurisdiction nodes keep genuine jurisdictional communities (about 64-81% for the largest named group). Positive inter-jurisdiction affinities keep the encounter graph connected.

Use `python tools/inspect_tiered.py --validate-spec` for a deterministic preview of a seed's counts, means and realized attention.

## Persistent structural asymmetry

`scenario.json` records, under `persistent_structural_asymmetry`, that no persistent role-based mechanism is part of the model. The five ordinal attributes (material power, information advantage, coordination, risk exposure, exit power) are contextual descriptors with no conversion to dynamic parameters, and adding a persistent role-based mechanism without one would import an invented structural law. The decline of core received-attention share under adaptive memory is a model outcome to be measured, not a defect to be corrected by a clamp. Roles affect tier labeling, initial stance probabilities and the initial memory/affinity structure only. `W` is not money, the quenched susceptibility `g` is not power, and plot size encodes received attention, not leverage.

`tests/unit/test_f2_scenario_structure.py` runs a small uncontrolled rollout that records the core received-attention share as a diagnostic; it asserts only that the dynamics run and stay bounded.

## Reliability interpretation

The independent sampling unit is a constructed origin. With one future per origin, success uses the binomial point estimate, a Wilson descriptive interval and an exact one-sided binomial lower bound. With several futures, each origin contributes its own success fraction and the campaign estimates the equal-origin mean of those fractions. The origin bootstrap interval is descriptive only. A one-sided Hoeffding lower bound over independent bounded origin fractions is used for certification, assuming origins are independent draws from the declared scenario; it is conservative with few origins and cannot certify a single all-success origin. Unequal numbers of futures do not give an origin more weight. Certified-once and sustained-to-horizon probabilities are computed separately. All probabilities are model-run probabilities conditional on the stated assumptions, not empirical probabilities about the world.

## Persistence of relationships

Historical `W` represents recent attention, not a durable organizational relationship. Fixed opportunity affinities allow renewed contact but do not guarantee a baseline influence contribution, and the common per-token forgetting and full-row random replacement have no organization-specific calibration. Results are conditional synthetic outcomes.
