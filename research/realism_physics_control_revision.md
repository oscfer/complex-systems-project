# Realism, physics, and computational design

This revision makes the model as realistic as reasonably supportable while retaining statistical mechanics, interpretable mechanisms, and practical computation. It defines the principal transformation/control experiments. The earlier homogeneous, fully observed model remains a reference, not the sole research model. Numerical values here are testable pilot assumptions, not empirical estimates. Greater complexity alone does not establish realism.

The earlier homogeneous, fully observed configuration is retained as reference material. No production simulator is claimed implemented here.

## 1. Three complementary experiments

1. **Unbiased emergence reference:** identical intrinsic parameters, empty persistent memory, symmetric state distributions, and no designated powerful agents. This preserves the original question of spontaneous hierarchy. Uniform encounter opportunities are the primary symmetry control; constrained opportunities are a separate comparison.
2. **Principal transformation/control experiment:** heterogeneous response and encounter rates, finite relationship memory, constrained encounters, evolving ties, private/public states, limited intervention access, and delayed incomplete observations. Feedback policies are part of this main design, not merely a distant optional feature.
3. **Empirical and structural robustness:** test selected results on multiple constructed/generated ensembles and, when a suitable dataset is selected, on empirical temporal network structure. Realistic topology alone does not validate the model's belief dynamics. Keep empirical calibration distinct from synthetic mechanism tests.

Binary states supply the simplest ordering reference. Continuous states and three coupled components remain substantive comparisons for belief/worldview transformation. They are not automatically more empirically accurate; compare their explanatory behavior under matched interventions. Preserve the participation game and its explicitly limited institutional interpretation. A network of attention is not a network of coercive authority or resources; historical institutional explanations would require separately specified variables and evidence.

## 2. Heterogeneity as quenched disorder

In established-network experiments, draw and save node parameters before network generation so states, relationships, and traits can coevolve before intervention. Do not add traits to a mature homogeneous snapshot without labeling that operation as a model change. Begin with independent traits to separate their effects; investigate correlations with group membership and network position only as declared alternatives or empirically fitted dependencies.

Keep the same event heat baths and use

$$
F^y_{i\alpha}=J\sum_j A_{ij}y_{j\alpha}+K_i x_{i\alpha}+h_{i\alpha}(t)+g_i u_{i\alpha}(t),
\qquad
F^x_{i\alpha}=K_i y_{i\alpha}+L\sum_{\beta\ne\alpha}B_{\alpha\beta}x_{i\beta}+b_{i\alpha}.
$$

Here $g_i$ is susceptibility to the applied intervention, not guaranteed compliance. State changes remain stochastic. Define the clipped lognormal multiplier $q_i(\sigma)=\operatorname{clip}(e^{\sigma Z_i-\sigma^2/2},0.5,2)$ with independent standard normal draws for different traits. Set $K_i=Kq_i(0.35)$, $\tau_{x,i}=\tau_xq_i(0.5)$, and $\nu_i=\nu q_i(0.5)$. Draw $g_i$ uniformly in $[0.5,1.5]$ and $b_{i\alpha}/J$ from a zero-mean normal of standard deviation 0.15, truncated by rejection to $[-0.45,0.45]$. No sample-wide centering or rescaling is implicit. Clipping means a multiplier's mean is not exactly one; record realized moments and compare matched-rate controls when isolating heterogeneity.

Initially keep a common noise scale $T$, common public attempt rate $1/\tau_y$, and common forgetting rate $\delta$. This avoids changing every mechanism simultaneously. Agent-dependent temperatures or memory lifetimes can be added only with their own nonequilibrium interpretation. At fixed symmetric couplings and common $T$, heterogeneous attempt rates alone do not invalidate detailed balance; the principal directed adaptive process does not generally have the reference Gibbs equilibrium.

The attempted event rate becomes

$$
R=D\sum_i(\tau_{x,i}^{-1}+\tau_y^{-1})+\sum_i\nu_i+\delta\sum_{ij}W_{ij},
$$

plus extension clocks. Weighted node selection must use these rates rather than selecting every node uniformly. Draw traits once per origin, retain them in all paired continuations, and record their arrays in checkpoints. The controller receives distributions or calibrated uncertainty, not hidden individual $b_i$, $g_i$, or private response parameters.

## 3. Local encounters and finite relationship memory

Use the existing eligibility and closure interfaces. For a first established-population setting, use four equal-sized opportunity groups, randomized labels, within-group eligibility 1, and cross-group eligibility 0.05. Group structure represents imposed opportunities; it is not evidence of emergent communities. Compare with unrestricted opportunities and with the initially separate multi-population/contact schedules. Opportunities and remembered relationships remain distinct objects.

Remove globally known prestige from the main local rule by setting $a=0$. Retain state-dependent reinforcement $c=1$ and exploration $\varepsilon=0.1$. Use the existing eligible two-step closure rule with $\chi=0.3$, compared against $\chi=0$. Local referral is a model of access through contacts, not a new claim that all agents know global centralities. Keep global-prestige dynamics as a named information-rich mechanism comparison.

Cap each directed memory row at $C_i=40$ units in the first main setting. On a successful encounter, sample its target from the pre-event state. If the row is full, remove one memory unit uniformly from that row's existing units, then increment the chosen target; perform both changes atomically. Otherwise simply increment. Per-unit forgetting continues at rate $\delta$. This bounds stored relationship memory, not the number of possible agents ever encountered. It also makes sparse storage scale at most as $O(NC)$ for equal capacities.

With $S=\sum W$, the original unconstrained Poisson count identity is replaced by

$$
\frac{d\mathbb E S}{dt}=\mathbb E\!\left[\sum_i\nu_i\mathbf1\{i\text{ has an eligible target},\ S_i<C_i\}\right]-\delta\mathbb E S.
$$

Do not apply the old $N\nu/\delta$ stationary mean to the capped model. Compare $C=20,40,80$ and an uncapped reference in selected sensitivity tests. Directed capacity is the first implementation; reciprocal pair memory with independent row caps needs a separate joint-capacity rule and is rejected in this configuration. Closure at empty memory falls back to eligible exploration, preserving possible emergence from no persistent ties.

## 4. What the controller can observe

The main controller uses `sampled_public`, not a clone of the full true snapshot. At times $0,5\tau_y,10\tau_y,\ldots$, the observation system samples the public states of a fixed random 50% panel. Each binary report flips with probability 0.05; continuous reports add independent normal noise of standard deviation 0.1 and clip to $[-1,1]$. For panel-node pairs, independently reveal each positive directed tie with probability 0.5 at each observation. A revealed count is exact in the first observation model; unrevealed pairs are marked unknown, not observed zero. Group labels and the intervention-access roster are known metadata.

Deliver each report after $\tau_y$. Thus a controller initially waits for the first report at $t=\tau_y$, and can react at $1,6,11,16$ in reference units before the $20\tau_y$ release deadline. Snapshot the physical quantities when sampled, not when delivered. Reports contain sampling and delivery timestamps, missingness masks, panel IDs and the noise-model version. Between deliveries use the last report with its age. Observation noise has its own RNG and does not alter physical states or event times.

No controller, policy feature, stop condition, or candidate search at deployment can inspect true $x$, unseen ties, the future RNG, or true target-evaluator feedback. Missing reports cannot be filled with simulator truth. Exact full-state target evaluation is allowed for the researcher after the fact. Report results conditional on any hidden-state eligibility screening, including the fraction excluded; additionally report unconditional public-eligibility deployment performance so the evaluated population does not silently require knowing private beliefs.

Compare complete-public observation, sampled-public observation, and full-state oracle on matched systems. The oracle is an information-rich comparator, not a proven bound on algorithmic performance. Include nonrandom observation/access selection in robustness checks because a random panel can be optimistic.

## 5. Feasible interventions

Sample an accessible roster of 25% of nodes once per origin independently of the observation panel. Cap the total distinct controlled nodes at the smaller of the configured $k$ and the roster size. A policy can choose roster members without observed states using group/random allocation, but cannot assign them fabricated centrality/state scores. Reject an action outside the roster rather than silently moving it to another node.

Retain the exposure, amplitude, distinct-node and switch-off constraints. Charge the commanded field $u$, not the unknown realized response $g_i u$. Allow updates only on report-delivery times; the mandatory deadline/budget exhaustion can shut off a field earlier. No intervention may directly set beliefs or participation. The first main actuator is a field; exact arbitrary edge deletion is a separate idealized actuator class, not an assumed capability. Later structural interventions require an explicit access mask over editable ties or interfaces.

The pilot amplitude cap remains $4J$, with sensitivity at $J$ and $2J$; none of these numbers is an empirically justified persuasive power. Demonstrate that findings are not driven solely by nearly saturating fields. Feedback reallocates remaining effort within the accessible roster and cumulative unique-node limit. A newly chosen node consumes a slot permanently. A policy cannot evade the cardinality constraint by rotating through the entire population.

Structural targets retain activity/connectivity floors and post-release persistence. Heterogeneity requires checking persistence against slow agents: use $\max_i\tau_{x,i}$ in the existing hold/horizon rule rather than the nominal $\tau_x$. Hold/horizon must also be checked against measured uncontrolled relaxation and network mixing times. If fragmented populations make a global endpoint inappropriate, predeclare separate per-population targets; do not relax a failed endpoint after looking at control outcomes.

## 6. Optimize observable policies across uncertain ensembles

The first principal optimizer searches a finite-dimensional *policy*, not an oracle node set customized using true hidden microstates. Its input is only the report history, known metadata and its resource ledger. Training can use fully simulated outcomes as labels, but the deployed policy must work on an untouched new origin without hidden-state search.

Use a policy family defined by: observable selector (random, reported attention, reported outgoing degree, reported cross-group connections, group-balanced selection), pulse amplitude/duration, and a static versus feedback switch. Reported scores are computed only on the incomplete graph and explicitly labeled proxies. For unseen candidates use an `unknown` status with seeded random selection among them rather than zero centrality. A configurable quota assigns half the initial slots to observed accessible candidates and half to all remaining accessible candidates; if either set is too small redistribute the remainder. Group-balanced policies cycle through known groups. For structural fields use reported counter-alignment; unknown signs use a seeded fair sign. State-reversal direction is a prescribed experiment goal based on available public information, not true private means.

At subsequent reports a feedback policy can replace selected nodes within remaining distinct-node slots or reactivate previously used ones; it cannot unspend slots. It prioritizes those currently opposing the target for reversal and updates structural counter-alignment from reports. When no action fits the remaining resources, use zero field. Pulse duration is measured from first actuation, bounded by the shared release deadline. A static policy commits its set/signs on the first delivered report. All algorithms have the same information latency and access constraints.

To permit later recruitment, the default feedback policy initially activates at most $\max(1,\lfloor k/2\rfloor)$ nodes, then may add up to that many previously unused accessible nodes per later report while respecting the cumulative cap. A static policy may use all $k$ immediately. Include a feedback variant that commits all $k$ initially to distinguish delayed recruitment from redistribution within a fixed set. These are alternative schedules under identical total constraints, not extra node allowances.

Search policy candidates using the existing deterministic batches, independent RNGs, bounded candidate cap and fresh selection data. Seed with all five selectors, mutate selector, amplitude, duration, quota (0, 0.5 or 1), and feedback flag; no other unspecified policy parameters enter this first family. Static baselines and feedback candidates must be compared at equal physical constraints. Retain failure-inclusive restricted time and $p_0=0.8$ as the proposed reliability constraint. Learning a richer policy is a subsequent comparison, not a substitute for this transparent optimizer.

Initial recruitment fraction is an additional explicit schedule switch, 0.5 or 1 for feedback and 1 for static policies. The first delivered observation fixes the requested pulse end time. Subsequent reports cannot reset that duration to extend forcing indefinitely. Mandatory release at the common deadline still takes precedence.

Each candidate is evaluated on the same training-origin *population*, with independent future and observation streams. Sample uncertain nuisance settings across training origins: response heterogeneity, cross-group exposure, observation coverage/error, and intervention susceptibility. Use nominal training plus four predeclared one-factor stress conditions for selected finalists (section 8). Optimize nominal performance while requiring acceptable stress reliability, or report a Pareto tradeoff if no candidate meets it. Never call a nominal optimum robust because it was averaged over a few favorable cases.

A small starting policy-search allocation is 24 candidates × 12 training origins × 2 futures = 576 continuations; three finalists × 12 fresh selection origins × 4 futures = 144; three frozen evaluation algorithms × 30 fresh origins × one future = 90. Total at most 810 continuations per selected model/target/budget condition, excluding establishment, stress tests and confirmation. These are exploratory caps, not a whole-project maximum. Thirty evaluation origins do not establish the final probability precision. Retain the existing 100/400-origin planning policy and paired-effect calculation for selected confirmation.

Public-observation model-based MPC is not permitted to initialize its rollouts with true hidden states. It requires a documented observation-conditioned state estimator and calibration tests; until that exists, the main feedback controller is the direct observable policy above. The earlier oracle MPC remains a labeled benchmark. This avoids an expensive and scientifically misleading pseudo-partially-observed implementation.

## 7. Physics and size remain central

The research observables remain order parameters, susceptibility/response, correlation and relaxation scales, metastability, first-passage/holding times, memory statistics, fragmentation, and influence concentration/persistence. Express thresholds and forcing in dimensionless units. Compare frozen/adaptive, homogeneous/heterogeneous and local/global-information limits. The local choice rule keeps its logit-utility interpretation; participation retains its coordination-game role. Do not claim equilibrium energy minimization for the full adaptive process.

Tiny enumerated systems are validation tools. They do not determine the scale of the principal research. Benchmark selected cases at $N=1{,}000$, $5{,}000$, and $10{,}000$; add a $50{,}000$ probe if memory/event costs allow and extend further when scientifically informative. These are probe sizes, not a hard ceiling or evidence of realism. Freeze actual scientific sizes after measuring cost and checking whether important observables still change with $N$. Compare at matched per-node budgets and opportunity structure, with selected fixed-community-size versus fixed-number-of-communities controls.

With bounded sparse memory, prioritize exact row updates, weighted event sampling, cached scalar observables, and independent-realization parallelism. Avoid drawing large graphs during simulation. Use the existing exact two-step law with sparse support or an equivalent correctly weighted sampler; do not replace its conditioned target distribution by a different failed-walk process for speed. Communities and fine visualizations run on selected saved snapshots. Compute limits may reduce parameter coverage before removing a mechanism essential to the question. Explicitly label exploratory uncertainty if replication must be reduced.

For block-constant eligibility, sample an opportunity group in proportion to its eligibility times its number of nonself agents, then sample a node uniformly in that group; this implements the exact exploration distribution without a dense matrix. The closure branch can enumerate at most $C^2$ two-step candidates for the selected row, accumulating duplicate destinations before normalizing. Its cost must be timed separately. Compute the full capped-memory replacement as one physical event so observables do not register a nonexistent temporary disconnection. Legacy checkpoints lacking traits/capacity/report queues require an explicit conversion experiment, not an implicit resume into the new model.

## 8. Ablations, calibration, and evidence

For each principal finding compare the full proposed bundle with one change at a time: homogeneous traits; unlimited memory; unrestricted opportunity mixing; complete-public reports; full intervention access; static rather than feedback policy. Use matched origins where mathematically valid; where generation laws differ, use independently generated matched ensembles and do not pretend their snapshots are identical. Separate policy effects from pulse/access/budget changes.

The four initial stress conditions are: lower observation panel fraction 0.25; lower access fraction 0.10; cross-group eligibility 0.01; and doubled heterogeneity widths with the same clipping bounds. Change one factor per stress test. Their joint extreme is an optional additional condition, not covered by the separate tests. Keep all failures and report uncertainty. Larger parameter sweeps follow evidence of sensitivity, not a full Cartesian product.

Empirical work must declare a domain and observable dataset before fitting. Fit measurable quantities such as encounter frequencies, tie lifetimes, degree/strength patterns, group mixing and public-state autocorrelation using a training time window; test predictions on held-out time periods or independent networks. Private beliefs and intervention efficacy are generally not identifiable from topology alone. Use sensitivity ranges for them and disclose observational equivalences. Compare competing mechanisms rather than assuming our selected combination is uniquely correct. No historical reconstruction is justified without historical data and a defensible observation model.

Report principal control results from a small, preselected set of physically motivated comparisons, with detailed manifests and auxiliary sensitivity results stored alongside.

## 9. Literature basis and limits

- [Gross and Blasius, 2008](https://arxiv.org/abs/0709.1858) motivates studying feedback between node states and topology as a statistical-physics problem. It does not validate this particular social interpretation.
- [Garrod and Jones, 2021](https://arxiv.org/abs/2011.05774) studies Ising influence with coarse network information and ambient fields, supporting the relevance of limited-information control. Its assumptions and objectives differ from our adaptive post-release target.
- [Li and Porter, 2023](https://arxiv.org/abs/2206.09490) investigates heterogeneous activity in a bounded-confidence model. It supports testing activity heterogeneity; it does not supply our heat-bath dynamics or numerical trait distributions.

The trait laws, observation process, capacities and budgets above are our explicit modeling choices. The defensible claim at this stage is a more realistic and testable mechanistic design, not an empirically validated model of society.


## Persistence of relationships

Historical W represents recent attention, not a durable organizational relationship. Fixed opportunity affinities allow renewed contact but do not guarantee a baseline influence contribution. Common per-token forgetting and full-row random replacement have no organization-specific calibration; delta=0 still permits replacement. Results are conditional synthetic outcomes, and a durable-relationship mechanism would have to be added as a separately versioned model before claiming organizational robustness.
