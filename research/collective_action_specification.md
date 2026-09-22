# Collective action and institutional transition

This specifies a generic participation-and-threshold abstraction. The payoff and timing details below are its explicit implementation; a distinct authority/resource network is outside it. It preserves the distinction between private states, public expression, participation, and an institutional outcome.

## Meaning and literature connection

Introduce participation n_i∈{0,1}, separate from x_i and y_i. A participant acts in support of replacing the incumbent institutional arrangement with an alternative. Nonparticipation does not imply private support for the incumbent. The incumbent orientation I∈{−1,+1} refers to the designated issue component, initially component 1; it does not identify a particular leader or encode all dimensions of a worldview.

[Takács, Janky and Flache (2008)](https://pure.rug.nl/ws/files/10343680/TakacsK-Collective-2008.pdf) connects participation incentives with changing relationships. It motivates separating action from preference and modeling their network dependence. The equations below are our proposed local coordination game, not their published payoff system. In particular, their strategic treatment of ties is richer than this first one-way extension from the existing opinion/adaptive-network dynamics to participation.

## Participation game

Define private opposition and locally observed participation by

$$
q_i=\frac{1-I x_{i1}}2,\qquad \rho_i=\sum_j A_{ij}n_j.
$$

Both lie in [0,1]; an empty attention row gives ρ_i=0. Observation of neighbors' current participation is an explicit information assumption. It is distinct from observing their private preferences and from estimating a probability of eventual institutional success.

The utility difference between participating and abstaining is

$$
\Delta u_i=b q_i+g\rho_i-c,
\qquad b,g\geq0,\quad c\geq0.
$$

Here b is the incentive associated with private opposition, g rewards participation alongside contacts, and c is the participation cost. These are identical across agents in the initial comparison. This is a game with strategic complementarity: participation becomes more attractive as neighbors participate. It is not a complete public-goods game, a model of forward-looking pivotality, or an estimate of material historical payoffs. High g can induce participation even by privately unsupportive agents; that is a stated implication of this payoff, not a coding error.

At independent rate τ_n^−1 per agent, redraw participation using

$$
P(n_i=1)=\frac{1}{1+\exp[-\Delta u_i/T_n]},\qquad T_n>0.
$$

The logistic exponent has no factor of two because actions are 0 and 1 and Δu is their utility difference. The original ±1 state heat bath uses a different field convention. Increasing c suppresses participation at fixed observations; g=0 removes direct coordination incentives. At positive T_n, zero participation is not absorbing. Large opposition alone does not guarantee mobilization when costs are large.

For fixed symmetric A and fixed q, the action game has potential

$$
\Phi(n)=\sum_i(bq_i-c)n_i+\frac g2\sum_{i,j}A_{ij}n_i n_j.
$$

Changing n_i from 0 to 1 changes Φ by Δu_i. The action heat bath then has reference distribution proportional to exp(Φ/T_n). This is a limited analytical control, not an equilibrium claim for the full adaptive system.

## Institutional outcome rule

Use equal action capacity, with participation fraction M_n=N^−1Σ_i n_i. A first institutional transition is recorded when M_n remains at or above θ throughout an uninterrupted interval of duration t_hold>0, where 0<θ≤1. Falling below θ resets the interval. Crossing once is not sufficient. The outcome changes from incumbent to alternative at completion of the interval.

θ and t_hold are declared parameters of this toy institution, not inferred universal constants. Study sensitivity to both and retain M_n(t) as a primary observable independent of the threshold. Do not import the tipping percentage from a naming-convention experiment as an institutional threshold. The outcome is termed a transition under the specified rule; it does not by itself demonstrate a coup, historical institutional collapse, or transfer of resources.

The first experiment measures time to the first transition and ends its institutional endpoint there. A repeated-transition model must additionally specify how the new institution affects preferences, rewards, resources, and the meaning of subsequent participation. That extension remains in the agenda and is not implicitly provided by flipping I without further rules.

Equal capacities distinguish this first comparison from an institution with unequal decision rights. A separate authority/resource network, coalition structure, or weighted capacity would be an additional specified mechanism. Received attention r_i is not automatically substituted for institutional capacity.

## Initial states, controls, and outputs

Start from independently established complete model states, with incumbent orientation defined explicitly before each comparison. Proposal: select orientation from a declared establishment criterion, or impose I=+1 and report its initial support; do not align I retrospectively with whichever branch survives. Begin with n_i=0 for spontaneous mobilization. Selected seed experiments compare clustered and dispersed initial participants at the same initial count, holding the private/public state and network fixed within each pair.

Keep the existing evolution of x,y,W. In this first layer participation does not itself alter rewards in the contact rule or public states. This permits measurement of how those dynamics drive action; reciprocal action-to-network feedback needs a separate contact term and ablation. The restriction is explicit so the model is not described as fully coevolving action and relationships.

Compare g=0 with g>0, frozen with adaptive social ties, and equal-size participation seeds with different placement. Report private opposition, public opposition, participation, transition probability by a declared horizon, and censored transition times. Also report influence turnover and community changes separately. Restrict numerical scans to selected established-state settings rather than duplicating all prior experiments.

## Scheduler and checks

Add N/τ_n to the total attempt rate. The holding-time deadline is a scheduled event that can occur between stochastic events: the simulator must process whichever comes first. A checkpoint must preserve participation, incumbent orientation, threshold status, and the start of an active holding interval. Restart must not reset the persistence clock.

Implementation checks: the payoff/potential identity in its symmetric frozen control, logistic normalization, cost monotonicity, no-neighbor behavior, deadline crossing and reset, and restart equivalence. No full simulator or completed action experiment is claimed here. Numerical parameter ranges, institutional sensitivity settings and observation horizons are set per study.

Specification check: enumeration of all eight action configurations on a three-node symmetric reference verified every single-agent potential increment against the stated payoff difference; maximum absolute error was 2.22e-16. This checks the algebraic control only.
