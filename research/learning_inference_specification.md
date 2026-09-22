# Learning and inference comparisons

This specifies learning and inference comparisons. Numerical training settings below are starting choices, to be revised using validation data and recorded before held-out evaluation. They are not scientific predictions or claims of completed experiments.

## Agent learning: a change to dynamics

The first comparison uses binary expressed actions. Retain the same private-state, encounter, and forgetting rules, and the same public attempt rate. At an attempt of component α, draw s∈{−1,+1} from softmax[Q_iα(s)/T_Q], compute reward sF_iα^y using the pre-action private state and neighbors, and then update only the selected value:

$$
Q_{i\alpha}(s)\leftarrow Q_{i\alpha}(s)+\eta[sF^y_{i\alpha}-Q_{i\alpha}(s)].
$$

Initialize both values to zero for every agent/component, with T_Q>0. Then assign the selected public action. No reward is silently supplied for the unchosen action. Save Q in checkpoints and use it unchanged when branching a learned established state. This is sampled immediate-reward learning, not a discounted Bellman update or a claim that agents solve a forward-looking game.

Begin with η∈{0.01,0.1,0.5} at selected scalar-core settings, using T_Q=T_y as the equal-numerical-temperature comparison. Retain the instantaneous heat bath as the primary control. With constant fields and values Q(±1)=±F, softmax reproduces that heat bath, but finite-step learned values generally do not. Changing η changes memory of experienced rewards; an action selected rarely learns on a different effective timescale. Do not assume τ_y/η is an exact global learning time.

Compare ordering, discrepancy, influence persistence, and perturbation response at matched initial states and physical horizons. A selected D=3 extension uses a separate two-action table per component. Continuous private states with binary actions are possible under this interface, but a fully continuous expressed-action learner needs a different representation and is not supplied by this table. The fully continuous heat-bath comparison is separate.

## External inference: an observation task

The first target is instantaneous population discrepancy d_2 from the public graph and public states. Training labels can use simulator-private x; prediction inputs cannot. Inputs are A, y, and declared known experimental parameters. Features derived from A, such as received attention and nonempty-row indicators, introduce no new private information. Exclude node identifiers, initial random seeds, private states, realized discrepancy, and labels encoding the outcome. Institution or participation variables are excluded from this first observation set; adding them defines a different task.

Evaluate the following baselines before a neural model:

1. The training-set mean, conditioned on known experimental settings where data permit.
2. An analytical conditional expectation where available, particularly the scalar binary frozen symmetric equilibrium.
3. Ridge regression on public component means and second moments, active-row fraction, degree summaries, and attention concentration, together with the same known parameters supplied to the neural model.

For a first lightweight graph predictor use two message-passing layers of width 32, combining each node's public features with A-weighted neighbor features, then global mean pooling and a scalar sigmoid output. Supply graph-level experimental parameters after pooling. This architecture is a starting choice, not an assertion that two layers are optimal. It is invariant to relabeling when implemented correctly. Use the same definition of input information for all competing predictors.

## Data splitting and fitting

Partition complete independent simulation origins into 60% training, 20% validation, and 20% test groups, with deterministic rounding and recorded assignments. Every saved frame and continuation from an origin stays in its group. Balance sampling or loss weights across origins so long trajectories do not dominate. If there are too few independent origins for a meaningful split, report the task as exploratory and obtain additional independent origins before strong generalization claims.

In addition to within-setting test origins, reserve complete parameter settings and larger N values for separate extrapolation tests; exclude these from fitting and hyperparameter selection. Do not call an unseen frame from a familiar trajectory out-of-distribution evidence. Record which representations and scenario families are included in training. Cross-representation evaluation needs compatible inputs and the same d_2 target; it is not assumed to succeed.

Use mean squared error for fitting, reporting both MSE and mean absolute error per origin. Initial neural fitting budget: at most 200 epochs, early stopping after 20 epochs without improved validation MSE, and three optimizer seeds. Use validation only for model selection, and evaluate the held-out test after the selection rule is frozen. Optimizer seeds measure fitting variability; they do not replace independent simulation origins. If architecture or optimization is changed after viewing test outcomes, designate a fresh test set for the revised claim.

## Identifiability and interpretation

In the scalar binary frozen symmetric equilibrium with b=0 and known fixed parameters, the conditional expectation is q=[1−tanh(K/T)]/2 and conditional variance q(1−q)/N. No public-snapshot predictor can beat that expectation in population squared error under those assumptions. Apparent improvement on a finite test set requires uncertainty assessment and a leakage check, not a claim that the bound was overcome.

That identity does not automatically apply to adaptive transients, continuous values, or coupled beliefs. A history-based predictor is a distinct observation experiment and must compare baselines using the same history. Prediction of discrepancy does not recover unique individual beliefs, demonstrate deliberate concealment, or establish a cause of institutional change.

## Allocation and reporting

Reuse eligible simulation outputs, with group splits fixed before training. Do not run the full parameter Cartesian product merely to populate a training dataset. Benchmark training separately from simulation and use a capped pilot fitting budget first. The free/limited profile may therefore contain an exploratory inference check, while larger profiles support additional independent test origins and extrapolation cases. Report actual data coverage and uncertainty; absence of predictive improvement is a valid result.
