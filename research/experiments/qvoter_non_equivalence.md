# NDlib reference: explicit non-equivalence finding

This satisfies the M2 alternative of documenting a mathematical mismatch; it is **not** an executed NDlib parity benchmark or independent validation of production persistence.

Inspected artifact: NDlib 6.0.1 wheel, SHA-256 `3e1fe7426cfb00a1dc0b96c54aadace49744c91259736db9eab56f0e6c171346`, verified against [PyPI](https://pypi.org/project/ndlib/6.0.1/). Read `ndlib/models/opinions/QVoterModel.py` directly from that wheel without installing or modifying the simulation environment. Its q=1 rule copies a sampled neighbor into a uniformly sampled listener. The [official documentation](https://ndlib.readthedocs.io/en/latest/reference/models/opinion/QVoter.html) specifies zero disagreement noise for this implementation.

Use two nodes joined by one undirected unit edge. Map NDlib status 0→−1, 1→+1. In our W, row i attends to column j; on a directed NDlib graph that relationship would be the reversed edge j→i because it reads predecessors. Symmetric edges remove this orientation ambiguity. Strip away private coupling, biases, intervention, heterogeneous clocks and adaptive memory. Compare kernels conditional on one public update, not full simulator event counts. One NDlib update corresponds to one public attempt; for two identical rate-one public clocks the mean waiting time is 1/2, not one model time unit.

Even this reduced positive-temperature case is not equivalent. At T=0.3 and J=1 our heat bath retains a nonzero probability of opposing the sole neighbor. Q-voter q=1 copies it with probability one. In consensus, Q-voter cannot leave; the heat bath can. Equal RNG seeds cannot fix unequal transition probabilities.

`tools/check_qvoter_reference.py` enumerates the four-state analytical reference matrix from that inspected rule and evaluates our heat-bath probabilities. [Computed matrices](qvoter_non_equivalence.json) show row total variation ≈0.001271. The formal zero-temperature limit agrees on this two-node public kernel, but that is not the noisy two-layer model and cannot validate its post-release persistence. No noise parameter was changed to manufacture parity.

Scope of evidence: source inspection plus an exact mathematical comparison. NDlib was not installed or executed. A later independent stochastic-engine comparison must identify a truly equivalent nontrivial model first; no such equivalence is claimed here.
