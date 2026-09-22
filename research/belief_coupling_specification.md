# Coupled-belief specification

The continuous and binary heat baths can both use this module. The specification takes D=3 components with fixed shared couplings and compares uncoupled, reinforcing and frustrated internal matrices. Evolving internal couplings are outside this specification. Numerical scans and intervention settings below are illustrative choices.

## Physical question and literature grounding

Does coupling components within an agent make collective configurations more persistent, and can a perturbation applied to one component reorganize the others? The internal field remains

$$
F^x_{i\alpha}=K y_{i\alpha}+L\sum_{\beta\ne\alpha}B_{\alpha\beta}x_{i\beta}+b_{i\alpha}.
$$

[Dalege et al. (2016)](https://dennyborsboom.com/wp-content/uploads/2017/01/dalegecan2016.pdf) supplies a precedent for interacting evaluative components. This motivates within-agent couplings; it does not empirically identify the three-component matrices below. These are controlled physical examples, not reconstructions of a named ideology or a general representation of logical implication.

## Three comparisons

Use D=3 and compare

$$
B_0=0,\qquad
B_+=\frac12\begin{pmatrix}0&1&1\\1&0&1\\1&1&0\end{pmatrix},\qquad
B_f=\frac12\begin{pmatrix}0&1&1\\1&0&-1\\1&-1&0\end{pmatrix}.
$$

Every agent receives the same B within an experiment. Nonzero matrices are symmetric, have zero diagonal, and have absolute row sums equal to one. Thus changing signs does not also change total absolute internal coupling strength. L controls that strength relative to J and K; its numerical scan remains pilot-dependent.

- B_0 removes direct internal cross-component coupling. The shared adaptive social network can still generate dependencies between components, so this does not imply full statistical independence.
- B_+ favors all three components aligning. It does not select the all-positive configuration over the all-negative one.
- B_f has an incompatible cycle: component 1 favors matching both others, while components 2 and 3 favor opposing each other. Binary states cannot satisfy all three preferences simultaneously. This is frustration in the physical sense, not a judgment that a person or belief is irrational.

Three components are the smallest symmetric pairwise system that can contain a frustrated cycle. Two components can have a dependence but cannot form such a cycle. D=3 is therefore justified as a minimal mechanism test, not as sufficient dimensionality for actual worldviews. A selected larger-D comparison remains in the agenda; it must specify topology and normalization rather than merely adding components.

Negative couplings alone are not evidence of frustration. For example, a triangle with two negative edges and one positive edge has compatible preferences; a change of component sign conventions maps it to the reinforcing case when fields and states transform consistently. The product of edge signs around the triangle distinguishes these cases.

## Initial states and intervention

Propose initially independent components within and across agents, with no preferred sign, and public states equal to their corresponding private states. Use equiprobable endpoints for binary states; use the continuous initialization and endpoint control described in `continuous_state_specification.md`. Do not initialize all components equal and then claim the model generated their initial alignment.

After a declared establishment period, branch complete independent snapshots into an unperturbed control and a field-pulse experiment. Apply the field to public component 1, retaining the private/public coupling K. Measure responses in both directly targeted and untargeted components, alongside network changes. Repeat selected pulses on components 2 and 3: for B_f the positive-edge junction and the negative-edge endpoints are distinct intervention locations under fixed sign conventions. Pulse amplitude, duration, target fraction, and observation horizon must be fixed after pilots and before confirmation.

Use the same initial distribution and parameters across B cases. Use matched snapshots within each case for intervention comparisons. Different B matrices generally generate different established states; that difference is part of the total mechanism effect and should be reported. If isolating the immediate effect of changing B itself, branch one common snapshot and label this as a separate parameter-change experiment.

## Measurements and exact control

Record component magnetizations, within-agent products averaged across agents, squared private/public discrepancy, and the response of untargeted components. Compare perturbed trajectories with controls; zero population magnetization can conceal strongly organized but opposing groups.

The isolated binary internal energy is

$$
E_B(x)=-L\sum_{\alpha<\beta}B_{\alpha\beta}x_\alpha x_\beta.
$$

Enumeration of all eight states at L=1 gives B_+ two states at −1.5 and six at +0.5; B_f has six at −0.5 and two at +1.5. The two-negative-edge balanced control has the same energy multiplicities as B_+. These are verified isolated-motif results, not predictions of whole-network equilibrium or adaptive transitions. Continuous states admit intermediate amplitudes, so the binary state counts do not describe their energy landscape or probability distribution.

## Scope preserved

This layer studies formation and transformation of configurations on a prescribed internal coupling structure. Formation of the internal structure itself requires a rule for B(t), including what is learned, from which observations, and with what constraints. Keep that extension explicit. Fixed shared B is a controlled comparison, not a claim that real people have identical or immutable belief relationships.
