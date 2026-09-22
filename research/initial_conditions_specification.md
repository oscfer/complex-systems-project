# Initial conditions and sampling allocation

This separates initial graph structure, state distribution, contact opportunities, and independent realizations. These are not interchangeable meanings of “initial condition.” No full Cartesian product of choices is required.

## Emergence

The main graph is W(0)=0. All agents share intrinsic parameters, with zero applied and private biases. Encounters remain possible. Binary components are independent equiprobable ±1 values; continuous components are independent uniform values on [−1,1]. Initialize each public component equal to its private counterpart. This is a symmetric distribution over realizations, not a constraint forcing exactly zero sample magnetization.

For a selected binary/continuous comparison, initialize both from identical random endpoint states. Otherwise their principal initial distributions differ in amplitude as well as representation. An initially independent private/public state is a control for imposed discrepancy, not the default starting condition.

## Transformation from generated states

Save complete independently generated trajectories at a prescribed establishment time determined using pilots. Save every realization, including those that did not form a hierarchy or collective order. Report the frequency of those outcomes. Any analysis conditioned on a particular regime must identify that conditioning and the number of eligible independent origins.

Within each origin, branch into unperturbed, perturbed, and frozen-network controls. Each branch starts with identical W, states, and any learning or action memory. Use separately recorded future random streams. This creates paired initial conditions, not independent origins. Additional futures from the same origin remain nested observations.

Do not overwrite the private states of a generated snapshot to manufacture hidden opposition and then describe it as spontaneously generated. Such an operation is a declared state intervention.

## Constructed structural controls

A common generator fixes an integer memory budget m per row. For each agent i, draw m targets independently from a declared distribution π_i over j≠i, then set W_ij to the number of times j was drawn. Thus row counts equal m and there are no self-ties. Actual distinct degrees vary; matching memory count does not match degree. These controls impose initial structure, not intrinsic differences in agent parameters.

Three distributions define the structural controls:

1. **Decentralized random:** π_ij=1/(N−1).
2. **Modular:** π_i=(1−η)u_i+ηv_i, where u_i is uniform over every other agent and v_i is uniform over other members of i's assigned block. Require each block to have at least two agents and 0≤η<1. This raises within-block probability while retaining cross-block opportunities. Assigned blocks are reference labels, not a substitute for detecting communities.
3. **Concentrated attention:** π_ij=(1−η)/(N−1)+η 1[j=h] for i≠h, with a uniformly chosen hub h; its own row is uniform over other agents. This makes h a likely recipient of attention. It does not give h different intrinsic update parameters or institutional authority.

Choose m and η after memory-scale pilots and record them explicitly. η=0 recovers the decentralized generator. The hub and block assignments are randomized across realizations. A hub is deliberately present in this transformation control; it must never be used as evidence of spontaneous hierarchy emergence.

Sparse constructed controls are primary. Complete uniform networks remain a separate dense reference, particularly for analytical checks. With integer memories, W_ij=1 on a complete graph already gives m=N−1, so it cannot be treated as a fixed-small-memory control as N grows.

## State distributions on constructed graphs

Select state conditions relevant to each comparison: unbiased random states, initial collective alignment, opposing group orientations, and public alignment with imposed private discrepancy. For binary component 1, an explicit generator draws public signs with probability p_g of +1 in group g, then flips the private sign relative to it independently with probability q. q=0 gives equality; q=1/2 makes the private sign independent of the public sign. Record p_g and q; do not prescribe a hidden discrepancy simply by calling a graph “hierarchical.”

For continuous comparisons, one candidate multiplies each public sign by an independent uniform amplitude on [0,1] and retains that magnitude when imposing private sign flips. Then q is a sign-flip probability, not squared discrepancy: E[d_2]=q/3 under this construction, versus q for binary states. The initial signed mean is also halved. Use endpoint-initialized comparisons where an exactly matched initial state is required. Additional components begin independently unless their initial correlation is the declared factor under study.

## Multiple populations

Generate each population independently before opening contact, or declare a constructed population setup. For the first contact comparison, use 2, 3, and 5 approximately equal groups at fixed total N; allocate any remainder deterministically and record actual sizes. Then repeat selected cases at fixed per-group size to separate increasing group count from shrinking each group.

Contact eligibility is a separate matrix from W. Initially prevent cross-population encounters; later open the prescribed interfaces. In the basic protocol, this matrix restricts encounters only. An existing tie continues to transmit influence if an interface subsequently closes; immediate removal or suppression of those ties is a distinct intervention. Uniform per-agent encounter opportunities imply larger groups receive more total encounters, which differs from choosing destination groups uniformly.

## Numerical allocation

Use N=64 and 256 for development; N=1,000 and 2,000 for exploration; N=5,000 and 10,000 for selected size checks. Use 30 independent origins per pilot setting. This number estimates variance and runtime; it does not certify precise event probabilities or critical behavior.

Select confirmatory sample sizes from pilot variance and a declared precision target, using fresh origins. For context, the worst-case probability standard error with 30 independent realizations is about 0.091; approximate 95% half-width is about 0.18, not 0.05. Branches and saved frames do not multiply the independent-origin count.

Maintain a manifest with one row per actual trajectory: family, representation, D, N, graph generator or source snapshot, initial-state distribution, parameters, intervention, seed, horizon, and stage. Count shared establishment trajectories once and each continuation once. The final number of trajectories follows this manifest and the selected comparisons; “six families” does not imply a fixed number of initializations or a fixed total run count.
