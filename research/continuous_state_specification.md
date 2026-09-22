# Continuous-state comparison

## Source

Continuous values in [−1,1] are updated with event-based heat-bath redraws. Smooth stochastic trajectories are not required for this comparison. The initialization and calibration comparisons below are modelling choices.

The bounded continuous Ising heat bath has a direct literature precedent: [Loison, Qin, Schotte and Jin, “Canonical local algorithms for spin systems: Heat Bath and Hasting's methods”](https://arxiv.org/pdf/cond-mat/0409422), particularly sections IV.A.3 and IV.B. Those sections give the bounded scalar Boltzmann weight and direct inverse-cumulative sampling. Our private/public variables and adaptive memory network are an extension, not a model validated by that paper.

## Shared conditional update

Keep the existing local fields F, update clocks, encounter selection, and forgetting. For each private or public component, let κ=F/T, with T>0. The binary conditional distribution is

$$
P(z=s\mid F)=\frac{e^{\kappa s}}{2\cosh\kappa},\qquad s\in\{-1,+1\}.
$$

The continuous conditional density, with respect to ds, is

$$
f(s\mid F)=\frac{e^{\kappa s}}{Z(\kappa)},\quad -1\leq s\leq1,
\qquad Z(\kappa)=\begin{cases}2\sinh\kappa/\kappa,&\kappa\ne0,\\2,&\kappa=0.\end{cases}
$$

At zero field the conditional density is uniform. This assumes a flat prior measure over the bounded interval; it does not encode an intrinsic preference for neutrality or for the endpoints. Zero represents the midpoint on the specified component, not absence of all beliefs. Each component is still a bipolar axis rather than an arbitrary set of ideologies.

For U uniform on (0,1), the exact inverse-CDF expression is

$$
z'=\frac{1}{\kappa}\log\left[e^{-\kappa}+U(e^\kappa-e^{-\kappa})\right],
$$

with z'=2U−1 at κ=0. Use stable exponential/logarithmic expressions in implementation; the displayed formula is not safe for direct floating-point evaluation at very large or very small |κ|. For κ>0 an equivalent form is 1+log[U+(1−U)e^(−2κ)]/κ; use symmetry for negative κ and a small-κ series where needed.

States remain constant between events. They can jump across the interval at an update, just as binary states can change sign. Continuous values do not imply continuous paths. No time-discretization step is introduced by this rule.

The shared Poisson scheduler retains

$$
R=ND(\tau_x^{-1}+\tau_y^{-1})+N\nu+\delta\sum_{i\ne j}W_{ij}.
$$

This is exact for the specified jump process with piecewise-constant external protocols, apart from numerical sampling error. It is a modeling choice for dynamics; an equilibrium sampling algorithm alone does not establish a physical or historical timescale.

## Equilibrium and utility checks

On frozen symmetric external couplings, symmetric internal belief couplings, common T, and fixed applied fields, reuse the Hamiltonian of the core model over the continuous domain. It is linear in each individual component when the others are held fixed. Consequently the density above is its exact single-component Gibbs conditional. The heat bath satisfies detailed balance with the Gibbs density relative to the product of uniform base measures. General directed or adaptive interactions still do not acquire an equilibrium Hamiltonian from this construction.

For the expressed state, the conditional density also retains the logit response to u(s)=sF over a bounded action space with uniform reference measure. The earlier two-action Q table cannot be substituted unchanged for a continuous-action learner. The initial learning comparison can remain on the binary branch; retaining that comparison does not imply a fully specified continuous-action reinforcement-learning module.

## What must be calibrated rather than assumed identical

Direct differentiation of Z gives

$$
\mathbb E[z\mid F]=\coth\kappa-\frac1\kappa
=\frac\kappa3+O(\kappa^3),
$$

whereas the binary mean is tanh κ=κ+O(κ³). Thus equal numerical temperatures do not give equal weak-field sensitivity. An isolated component with a prescribed weak field has susceptibility 1/(3T) in the continuous case and 1/T in the binary case. Setting T_cont=T_bin/3 matches that particular susceptibility; it does not match fluctuations, nonlinear response, coupling-induced correlations, or transition times throughout a network.

Propose reporting both equal-parameter comparisons and selected weak-response-calibrated comparisons. Use the same network mechanism and record any calibration explicitly. Do not attribute every observed difference to adaptivity when the state representation also changes local response.

The contact similarity remains D^−1 Σα y_iα y_jα. For continuous values, this rewards alignment weighted by amplitude. It is not simply a distance between opinions. Retaining it preserves the algebraic interaction rule; a distance-based alternative would be a separately declared mechanism comparison.

## Initialization and observables

Continuous emergence initialization: W=0, independent x_iα uniform on [−1,1], and y_iα=x_iα. This has no preferred orientation and introduces no private/public mismatch. Its amplitudes differ from the binary initial condition. Therefore include a selected endpoint-initialized continuous control using the same ±1 initial states as its paired binary comparison. This separates changing the initial distribution from changing the update rule. Independent private/public initialization remains a distinct control.

Keep component magnetizations, amplitude moments, network observables, and persistence measurements. For comparable private/public discrepancy, use

$$
d_2=\frac{1}{4ND}\sum_{i,\alpha}(x_{i\alpha}-y_{i\alpha})^2,\qquad 0\leq d_2\leq1.
$$

At binary values, (x−y)²/4=(1−xy)/2, so this is exactly the previous mismatch fraction. With continuous values it is a normalized squared distance, not a fraction of disagreeing agents. In particular x=y=0 gives zero discrepancy. The original expression (1−xy)/2 would incorrectly be 1/2 there if interpreted as mismatch.

If needed, report sign disagreement separately, with a declared treatment of near-zero states; do not silently call both quantities the same observable. A reversal of a small magnetization near zero is not necessarily a radical change. Set amplitude and persistence criteria before confirmatory response experiments.

The binary equilibrium calculation giving constant conditional mismatch q does not automatically transfer to this model. Integrating out a continuous internal state gives a factor Z(Ky/T) depending on the public amplitude y. Any continuous inference baseline must account for that dependence.

## Verification completed at specification stage

Single-site numerical checks used κ values spanning −1000 to 1000 for inverse-CDF checks, and −20 to 20 for density quadrature. Maximum CDF inversion error was 4.90×10^−14. Simpson quadrature with 20,000 intervals gave maximum normalization error 8.86×10^−14 and first-moment error 1.24×10^−13 on the tested fields. These validate the stated conditional formulas and stable evaluation strategy, not a full adaptive simulator, a benchmark, or a predicted phase diagram.
