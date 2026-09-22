# Measurement and analysis

These definitions apply across resource profiles. Numerical observation intervals, event thresholds, and analysis budgets must be declared in the campaign configuration. They are not fixed parameter values merely because they appear here.

## Distinct physical outcomes

Use separate measurements for state alignment, persistent concentration of influence, community organization, fragmentation, participation, and the declared institutional transition. No one of these is a proxy for all the others.

For each belief component record m_x and m_y, second moments, and d_2=mean[(x−y)^2]/4. The second moments distinguish weak continuous states from balanced strongly opposed states. Also retain per-population means in contact experiments. A zero aggregate mean can conceal strongly aligned opposing populations.

## Influence inequality and persistence

Use r_j=Σ_i A_ij, normalized shares p_j=r_j/Σ_k r_k, concentration C=Σ_j p_j², effective number of influential agents 1/C, and normalized concentration

$$
C_N=\frac{NC-1}{N-1},\qquad N>1.
$$

C_N equals zero for equal shares and one for all attention on one recipient. Its normalization does not guarantee every extreme is accessible under all network constraints. Report the nonempty-row fraction separately. All attention-share measures are undefined when total attention is zero; do not enter zero concentration for an empty graph.

Measure rank persistence using Spearman correlation of received-attention ranks across a declared lag, with average ranks for ties. If either vector is constant, correlation is undefined. Use top-fraction membership overlap as a complementary diagnostic; record the fraction and use an explicit tie policy, such as including every node tied at the cutoff and reporting the resulting set size. Do not silently choose tied leaders by node identifier.

Persistent unequal received attention supports an influence-ranking interpretation. It does not establish a nested hierarchy of communities or institutional authority. Claims of nested structure require a separate structural analysis, not just a high concentration score.

## Communities and fragmentation

Analyze nonnegative weighted adjacency G=W+Wᵀ for undirected communities, while retaining directed W and A for influence measurements. This projection discards direction and must not be described as measuring directed control.

Propose Leiden optimization of weighted modularity at reference resolution γ=1, with selected γ=0.5 and γ=2 sensitivity checks. These resolution values are diagnostic choices, not physical parameters. Run multiple declared random starts at each setting, retain the best objective value, and report partition variability. [Traag, Waltman and van Eck (2019)](https://arxiv.org/abs/1810.08473) motivates choosing Leiden because of connectivity problems that can arise with Louvain. It does not make any algorithmically detected partition a true social grouping.

Handle isolates separately and report them. With no edges, weighted modularity is undefined. For persistence, compare partitions on the same declared node set using label-invariant co-membership or adjusted Rand index. Track active-set changes separately; a partition can appear stable because many nodes remain isolated. Changing resolution does not automatically produce a nested hierarchy.

For a first null comparison, randomize the simple undirected support through degree-preserving edge swaps without loops or parallel edges, then randomly reassign the observed positive edge weights. This preserves the support degree sequence and the edge-weight multiset, not each node's strength, directed structure, or spatial constraints. Use the same partition-optimization effort for observed and null graphs. Check swap mixing through independent chains and observable stability; highly constrained supports may not admit a useful ensemble. If node-strength heterogeneity explains the result, a strength-constrained null is needed before a stronger community claim. Do not describe the initial null as preserving all network properties.

Measure actual fragmentation through the number and sizes of weakly connected components of positive W, including largest-component fraction and isolated-node fraction. Strongly connected components are an additional directed diagnostic. Do not impose a plotting threshold on W to create disconnection. A modular but connected graph is not fragmented. With ongoing global exploration, disconnection may be transient rather than absorbing.

## Major changes and response

Record full response curves before converting them into event labels. For a single-component reversal, require a declared initial alignment magnitude, a crossing to the opposite orientation beyond a declared amplitude, and persistence for a declared interval. If initial alignment is below the criterion, classify that realization as ineligible for this reversal endpoint rather than counting failure. Report eligibility frequency, and retain other outcomes for those runs.

For coupled beliefs, measure responses separately by component, distinguishing directly perturbed from unperturbed components. Compare with unperturbed branches from the same origin. Do not count ordinary frame-to-frame flips as worldview reorganization. Sign reversals, amplitude changes, and changes in cross-component association are separate observables.

Hierarchy turnover, network fragmentation, and the participation threshold have their own endpoint definitions. In particular, high participation does not imply a change in received-attention rankings. Report transitions by a fixed horizon and censored waiting times. Thresholds and persistence durations are fixed after pilots and before confirmation; sensitivity checks disclose dependence on those definitions.

## Cost and statistical units

Update inexpensive state summaries frequently; compute community partitions on a declared, coarser snapshot schedule. Store enough information to reproduce analysis. Choosing that schedule is an experimental recording decision because brief structural events can be missed. Verify selected results at finer observation resolution rather than letting faster hardware silently determine what is observed.

Independent simulation origins supply replication. Time points, optimization restarts, null graphs, and multiple continuation branches serve different purposes and are not interchangeable independent realizations of the physical process. Compute uncertainty at the appropriate level and retain paired-origin comparisons for interventions.
