# Encounters between populations and mechanism controls

This extends the memory process of the core model to changing contact opportunities. It also makes the previously named closure and reciprocity variants concrete. These are controlled model choices, not equations copied unchanged from the cited reinforcement literature.

## Eligibility and encounter budgets

Let g_i be agent i's population of origin and let E_ab(t)≥0 specify relative encounter eligibility from population a to b. Set C_ij=E_(g_i,g_j) for i≠j and C_ii=0. Begin with E_aa=1 and off-diagonal entries zero for separated populations. Opening a specified interface changes the corresponding entries at scheduled times. Use symmetric E for the initial contact comparison; asymmetric exposure is a distinct mechanism.

For each row with c_i=Σ_j C_ij>0, define

$$
U_{ij}=C_{ij}/c_i,\qquad
V_{ij}=\frac{C_{ij}W_{ij}\exp(a r_j+c\,s_{ij})}{\sum_k C_{ik}W_{ik}\exp(a r_k+c\,s_{ik})},
\qquad s_{ij}=D^{-1}\sum_\alpha y_{i\alpha}y_{j\alpha}.
$$

If V's denominator vanishes, set V_i=U_i. Select the target using P_i=εU_i+(1−ε)V_i. Evaluate weighted exponentials with a stable normalization. With C_ij=1 for every j≠i, this exactly recovers the earlier unrestricted rule. Exploration can create ties; the preference biases in V apply only to previously remembered eligible contacts.

If c_i=0, the attempted encounter is null: it creates no contact. Do not replace its distribution with a global uniform distribution, which would violate the restriction. The attempt clock may still include rate ν for that agent. Empty persistent memory and zero eligibility are different conditions.

ν fixes attempted encounters per agent per unit time. Increasing off-diagonal E redistributes those opportunities; it does not automatically add more total encounters. Multiplying every entry of a row of C by the same positive factor leaves P unchanged. To study additional exposure at unchanged internal encounter frequency, introduce an explicitly separate external-encounter clock; that is a different comparison with a different memory budget.

## What a contact closure means

The first protocol applies C to encounters, including reinforcement encounters. A previously created tie can continue to transmit influence through A after its interface closes, until forgetting removes it. C does not automatically delete W or mask the state field. Immediate removal, or masking communication while retaining memory, are separate interventions with different consequences.

Global received attention r is still used in the preference term. This is an explicit information assumption even when encounter opportunities are restricted. A local estimate of prestige needs its own observation and updating rule. These restrictions therefore do not, by themselves, define complete informational isolation.

Scheduled contact events must occur before sampling any later stochastic event. Starting from separate populations and opening contacts does not increase N: the populations were already included in the total system. Adding new agents or migration is a separate demographic mechanism.

## Count identity and comparison fairness

With directed unit increments and every row eligible, total memory S=ΣW has birth rate Nν and death rate δS, so

$$
\frac{d\mathbb E[S]}{dt}=N\nu-\delta\mathbb E[S].
$$

If M(t) agents have at least one eligible target, the birth rate is νM(t), not Nν. For prescribed eligibility, replace the source term accordingly. Null encounter attempts do not contribute to births. This distinction is essential for testing fully closed or singleton populations.

Holding ν/δ fixed matches the memory-count scale where all agents remain eligible. It does not guarantee matched distinct degree, modularity, or influence concentration. Report those quantities rather than describing all networks as density-matched.

## Optional closure comparison

Keep the directed memory process. Construct an eligible path matrix M_ij=C_ij A_ij/Σ_k C_ik A_ik for nonzero rows and zero otherwise. Define a target weight

$$
H^*_{ij}=C_{ij}\,\mathbf1[j\ne i]\sum_k M_{ik}M_{kj}.
$$

Normalize H_i=H_i^*/Σ_j H_ij^* when possible; otherwise use H_i=U_i. Replace the target rule by

$$
P_i=\varepsilon U_i+(1-\varepsilon)[(1-\chi)V_i+\chi H_i],\qquad 0\leq\chi\leq1.
$$

χ=0 recovers the base rule. On a closure encounter increment only W_ij; do not reinforce path edges. This is a weighted two-step search conditioned on an eligible nonself target. It can reinforce an existing target as well as create a new one. The normalization means it is not identical to drawing one path and discarding failed paths. Weighted closure has precedent in [Kumpula et al. (2007)](https://arxiv.org/pdf/0708.0925), but this precise rule differs from their model.

## Optional reciprocal-memory comparison

For a separate reciprocal ensemble, start from empty or explicitly symmetric paired memory. Each successful encounter creates a single pair token that contributes one count to both W_ij and W_ji. Forget each pair token jointly at rate δ. Use symmetric eligibility and an encounter attempt rate ν/2 per agent so the mean total directed-count birth rate remains Nν when all agents are eligible. Do not silently reinterpret an arbitrary directed checkpoint as paired memory.

This preserves W symmetry but generally not A symmetry after row normalization. It also changes fluctuation structure: when every attempted encounter succeeds, S/2 has immigration rate Nν/2 and per-token death rate δ, so stationary mean S is Nν/δ but variance S is 2Nν/δ. Matching the mean does not make the processes identical. Individual row counts need not have the same mean as in the directed process because incoming encounters also add reciprocal memory.

Closure and reciprocal memory should first be varied separately, with the same state module. Their combination remains possible but is not an automatic extra axis in every experiment. A claim about the causal effect of reciprocity must acknowledge changes in partner memory and correlations, not attribute all differences solely to matrix symmetry.

## Required validation

Check normalized nonnegative target probabilities, forbidden-target exclusion, empty-memory fallback, no-eligible-target null attempts, recovery of the unrestricted and χ=0 limits, scheduled contact changes, and count identities appropriate to each ensemble. Verify that self-ties cannot be created. These are acceptance requirements for an implementation; no simulation or empirical validation is claimed by this specification.
