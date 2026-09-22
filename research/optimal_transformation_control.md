# Optimal transformation of established adaptive networks

The [realism revision](realism_physics_control_revision.md) defines the principal design: heterogeneous agents, constrained local encounters, finite memory, incomplete observations, restricted intervention access, and observable feedback. The homogeneous/full-information methods below are reference comparisons.

This document specifies the optimal-transformation objective; it does not report an implemented controller or an experimentally established optimum. Numerical choices below are editable pilot values, not calibrated constants.

## Scientific question and position in the project

Operational details (contracts, timers, budgets, optimizer steps, statistical allocation and restart requirements) refine this research plan.

Given an established stochastic network, which nodes should receive an intervention, how many are necessary, and what strength and timing produce the fastest persistent change under a limited budget? Study both collective-state change and structural reorganization. Neither automatically implies the other.

The project now has three connected objectives: explain emergence, explain transformation, and optimize transformation. Control is an experimental layer on the same dynamics, primarily within the perturbation and transformation experiments. It is not a replacement for the six scenario families or a separately implemented physical model. The controlled and uncontrolled branches must reuse the same simulation foundation.

The working hypothesis is that useful intervention targets depend jointly on the current states, interaction memory, noise, and network adaptation; ranking nodes by connectivity alone will not generally identify the fastest persistent transformation. Whether feedback helps or inhibits control is a result to measure. The proposed combination is a research design, not a claim of priority over all existing literature.

## Literature and what transfers

| Primary source | Relevant result or method | Use and limitation here |
|---|---|---|
| [Lynn and Lee, 2016, Maximizing Influence in an Ising Network: A Mean-Field Optimal Solution](https://arxiv.org/abs/1608.06850) | Optimizes magnetization with a constrained external-field budget using mean-field susceptibility. Optimal allocation changes with temperature; high degree is not always preferable. | Motivates bounded node fields and response-based baselines. Its equilibrium/mean-field objective and conditional global guarantees do not solve our finite-time adaptive problem. |
| [Cornelius, Kath and Motter, 2013, Realistic control of network dynamics](https://dyn.phys.northwestern.edu/pdf/ncomms2939.pdf) | Uses constrained perturbations to redirect nonlinear trajectories into a desired basin of attraction. | Motivates testing autonomous persistence after release. Its deterministic trajectory method cannot be assumed valid unchanged for our stochastic jump process. |
| [Kempe, Kleinberg and Tardos, 2003, Maximizing the Spread of Influence through a Social Network](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf) | Greedy seed selection has a 1−1/e approximation guarantee for specified progressive cascade models with submodular expected spread. | Motivates subset-search algorithms and baselines. Reversible states, adaptive ties, and persistent-transition targets do not inherit that guarantee. |
| [Liu, Slotine and Barabási, 2011, Controllability of complex networks](https://www.nature.com/articles/nature10011) | Structural controllability identifies driver requirements in a linear-system framework. | A useful contrasting question. A structural driver count does not establish the minimum number needed for fast, bounded, stochastic transformation in this model. |

The cost functional, persistence endpoint, and proposed search protocol below are our specifications informed by these sources. They are not equations copied from a single paper. This is a focused methodological review, not an exhaustive systematic review.

## Define the transformation before optimizing it

Use complete established snapshots from the existing generated and constructed ensembles. Report those ensembles separately. For a state-reversal experiment, require a predeclared initial degree of order; report the fraction of generated snapshots eligible under that rule. Do not silently discard unsuccessful emergence trajectories.

Two primary target tracks are needed:

1. **Collective-state reversal.** Choose a target sign s* opposite the initial established orientation on a declared component. Let m_x and m_y be its private and expressed means. The target set G requires s*m_x ≥ m* and s*m_y ≥ m*. Private and public endpoints are also reported separately. For the bare Ising reference, only its single state mean exists.
2. **Structural reorganization.** Define the desired structural outcome explicitly. The first recommended contrast is decentralization: normalized attention concentration C_N must fall below a declared fraction of its initial value, while an activity floor and largest weak-component floor prevent trivial success by erasing interactions. A distinct turnover experiment measures replacement of initially influential nodes using the existing tie-aware rank protocol. Turnover, decentralization, community merger, and fragmentation are different targets and must have separate labels.

Pilot starting values can use m*=0.5 for reversal and a 50% reduction of C_N for decentralization. The latter is meaningful only for initially concentrated states. Calibrate persistence duration, activity/component floors, and eligible initial concentration using uncontrolled pilots, then freeze them before confirmation. Their values describe a particular target; none is a universal threshold. Include a joint state-and-structure target after the two separate targets have been characterized.

The worldview layer later applies a specified target vector to selected belief components, recording collateral changes in the others. Institutional success can use the existing participation threshold and holding rule. Preserve the distinction between that toy institutional endpoint and actual structural change.

## Admissible interventions and physical dynamics

The first actuator adds a bounded external field to selected expressed-state components:

$$
F^y_{i\alpha}(t)=F^{y,\mathrm{base}}_{i\alpha}(t)+u_{i\alpha}(t).
$$

Keep the existing heat-bath updates and endogenous memory dynamics unchanged. The controller cannot assign private beliefs, overwrite outcomes, or inspect future random draws. Start with scalar states and piecewise-constant fields on a fixed subset S; field amplitude, duration, and target set are decision variables. Reversal pilots use fields in the target direction. Structural pilots may compare signed allocations because a uniform directional pulse need not decentralize influence.

For each run impose

$$
\left|\bigcup_t\{i:\exists\alpha,\ u_{i\alpha}(t)\ne0\}\right|\le k,
\qquad |u_{i\alpha}(t)|\le u_{\max},
\qquad \int_0^{t_{\rm off}}\sum_{i,\alpha}|u_{i\alpha}(t)|\,dt\le B.
$$

Here k counts distinct nodes ever controlled, not merely simultaneous actuators. B is integrated field exposure, with field-times-time units; it is not automatically thermodynamic work. Require t_off ≤ T_c and zero intervention afterward. Compare methods under the same k, B, amplitude cap, target, and horizon; also report actual expenditure. Normalize k/N and B/(NJτ_y) for size comparisons, while retaining absolute values.

A subsequent actuator module can suppress selected encounter opportunities or impose bounded tie interventions using the already specified contact/disruption semantics. Record edge counts and memory mass changed, as well as duration. Do not equate an edge deletion with a unit of field exposure: compare separate budgets or explicitly declared multi-resource costs. Eligibility changes do not immediately erase existing ties. This stage tests direct structural control if field-only control cannot achieve the target; failure of one actuator class is an informative result.

## Speed, reliability, and persistence

Let Z(t) be the complete model state and Δ the required uninterrupted persistence duration after release. Define the completion time

$$
T_G=\inf\{t+\Delta:\ t\ge t_{\rm off},\ Z(s)\in G\ \text{for every }s\in[t,t+\Delta]\}.
$$

Thus time includes forcing, subsequent relaxation, and verification of persistence. A target reached only while the field is active is not persistent success. Monitor the target at state-changing events and schedule holding deadlines; sparse saved frames alone cannot verify uninterrupted residence. Finite noisy systems can reverse again later, so persistence is always stated over a finite duration, with selected longer follow-up checks.

For admissible policies π and horizon H, use

$$
\min_\pi\;\mathbb E_\pi[\min(T_G,H)]
\quad\text{subject to}\quad
\Pr_\pi(T_G\le H)\ge p_0,
\quad (k,B,u_{\max},T_c)\text{ constraints}.
$$

Failures contribute H to this restricted-time objective and remain explicit failures in the probability estimate. Do not rank policies solely by their average time among successful runs. If no policy meets the reliability target, report infeasibility among the policies searched and show the probability–time tradeoff. Do not label this proof that transformation is impossible.

Use p_0=0.8 as an editable initial reliability target, distinct from a confidence level. Confirm reliability with a predeclared one-sided lower confidence bound exceeding p_0, rather than a point estimate alone. A binomial bound applies to independent Bernoulli trials, such as independent futures conditional on one fixed snapshot, or one future per independently sampled origin under a fixed selection algorithm. For multiple futures per random origin, use an origin-clustered uncertainty procedure instead of treating all outcomes as independent. Results near the boundary may require substantially more runs than the existing approximate probability-precision targets.

Report a frontier of speed, probability, budget, and number of controlled nodes. The smallest successful k found is an empirical upper bound on the minimum required within the stated model and policy class; it is not a proven global minimum. Failure to find a smaller successful set is not a lower-bound proof.

## Search strategy and observation assumptions

Begin with static subsets and a small catalog of pulse amplitudes/durations, drawing amplitude ranges from the existing perturbation pilots. A fixed policy may adapt its selected subset to each observed initial snapshot using a frozen selection algorithm. Disallow adaptation to that run's future trajectory in this first stage.

Compare no intervention, random subsets, high received attention r_j, high outgoing influence strength under the actual j→i convention, community-distributed choices, bridge-based choices, and low-attention controls. Avoid counting equivalent rankings as independent baselines. Include model-based finite-perturbation response screening where affordable. A susceptibility surrogate is a proposal generator; evaluate its actual finite-time performance with the simulator.

Optimize subsets through greedy additions followed by multiple-start single-node swaps; jointly search a small pulse catalog. Use repeated simulation estimates, retain uncertainty, and reevaluate finalists on fresh realizations. Greedy search is a heuristic here, not a guaranteed approximation. Broaden the candidate pool with random and community-diverse nodes so centrality screening does not exclude every alternative. Record exclusions and search budgets.

For tiny fixed-network bare binary systems, enumerate all subsets and a finite pulse catalog and compute transition probabilities using the finite-state Markov generator. A particularly clean benchmark asks whether the state is in G at a fixed switch-off time and remains there for Δ; the post-release survival probability follows from the generator restricted to G with exits treated as killing. This validates search against the optimum of that finite catalog, not arbitrary time-dependent control. Compare Monte Carlo estimates with those generator calculations. This reference does not enumerate the unbounded adaptive-memory state space.

Then add feedback policies at scheduled observation times: reallocate remaining admissible fields using the current observed state, respecting cumulative distinct-node and exposure budgets. Compare fixed and feedback policies at equal physical budgets, and report controller computational cost separately. Model-predictive control can use short simulated rollouts; reinforcement learning is a later candidate if simpler searches cannot represent useful policies or repeated tasks amortize its training cost.

Separate full-state scientific control from public-observation control. Full-state optimization can access x,y,W and gives a labeled information-rich benchmark. Public-observation policies receive y,W and declared parameters, never actual hidden x. Their model-based rollouts require an explicit belief-state estimator or marginalization over private states. Target evaluation may use hidden states without passing them to the controller. An offline optimizer informed by actual hidden snapshots is an oracle even if its eventual intervention is static.

Controller learning is a third ML role, distinct from agents' reward learning and the existing diagnostic predictor. Do not replace the physical agent dynamics with the controller's learning algorithm.

## Experiments, inference, and compute

Start with established scalar binary networks, one generated ensemble and one constructed modular/concentrated ensemble, at a small selection of temperatures and adaptation rates identified by existing pilots. Include frozen and adaptive branches from matched snapshots. Expand to continuous states, coupled beliefs, multiple external populations, and institutional targets after the control pipeline is validated. Preserve these extensions in the agenda without requiring their full Cartesian product.

Use independent pools for optimizer search, policy selection, and final evaluation. Entire origin trajectories and their branches belong to one pool. For generalization, freeze the selection/controller algorithm before applying it to held-out initial networks and sizes. Report both per-snapshot optimized performance and transferable-policy performance when studied; they answer different questions. Search rollouts on a held-out snapshot are allowed only if explicitly part of the frozen deployable algorithm, use allowed information, and are charged to its computing budget; final futures stay independent.

Use matched initial snapshots across policies and independent, recorded future seeds. Common random numbers are optional variance reduction only when their coupling is implemented correctly across diverging event schedules. Repeated futures from one snapshot are nested observations, not additional independent network origins. Use paired differences and origin-level resampling for population comparisons. Single-snapshot probabilities must be labeled conditional on that snapshot.

The existing approximately ±10 percentage-point probability precision broadly, and ±5 centrally if affordable, remains applicable to selected final comparisons. Optimization can require many more exploratory rollouts; account for these separately. Report transition curves, uncertainty, spontaneous-transition controls, and longer-horizon relapse. Freeze confirmatory thresholds, controller settings, and allocation before examining confirmatory outcomes.

Small profile: tiny-system reference plus N=64–256 development runs and a few control budgets. Standard: selected existing N=1,000–2,000 cases with capped search evaluations and independent confirmation. Extended: selected larger sizes, feedback policies, additional actuators and model variants. These are staged scopes, not runtime promises. Benchmark rollout and optimization costs; preserve incomplete search state and checkpoints for interrupted free cloud sessions. A full campaign count requires the chosen number of candidates, seeds, targets, and settings.

## Interfaces and records

The interfaces are conceptual: ObservationAdapter(state), Controller(observation, budget_state), BudgetLedger(action, elapsed_time), TargetEvaluator(full_state, time), and OptimizationRunner(config, snapshot_pool). The scheduler applies control changes at scheduled boundaries using its intervention mechanism. Optimization must never silently alter the stochastic rules. Required properties are zero-control equivalence, exact cumulative-exposure and distinct-node accounting, exact switch-off, holding-timer reset, no private-state leakage in restricted observation mode, permutation consistency, and restart equivalence including controller state and holding timers. On a tiny finite catalog, heuristic solutions can be compared with exhaustive results; global optimality is not asserted for large systems.

Reusable Python modules implement these operations. Configuration records include target definition, observation mode, actuator constraints, node labels, actual field schedule, optimizer version, origin IDs, random streams, search cost, outcome, censoring, and post-release follow-up.
