# 06b-p follow-up research synthesis

Status: post-result hypothesis generation; not a preregistration and not an
authorization for full training.

## What the completed experiment establishes

- Scaling and objective choice alone do not make the effective source
  decision-grade learnable.
- Net effective source and the residual after removing the known synaptic
  term are almost identical (`r = 0.998807`), so cancellation between those
  two targets is not the current bottleneck.
- The apparent exact-event main effect is not causal evidence: the nonzero
  exact-event fraction was `5.79e-5` in fit, `5.79e-7` in calibration and zero
  in development.
- Every adaptive 8 ms rollout was worse than passive overall. Improvements
  on active/regenerative coordinates were purchased by large errors on quiet
  and moderate coordinates.
- The 0.25/0.5/1 ms voltage-only audit does not identify the 1 ms macro-step as
  the primary cause. It cannot close the question because complete internal
  state was not available at the intermediate boundaries.

## Newly identified implementation confound

The structured input normalizer takes a p99 over all transition-segment
coordinates. Because the event tensor is overwhelmingly zero, the reported
scale has minimum and median `1e-6` and maximum only `2.07e-6`. An event-count
feature of one can therefore be presented to the MLP at order `1e6`; a
boundary tail of `0.002103` can be presented at order `1e3`. Thus the 06b-p
comparison confounds information content with pathological sparse-feature
conditioning. This is a direct code-and-artifact diagnosis, but its effect on
performance still requires a controlled intervention.

The object named `exact_events` is also not a raw event operator. The encoder
collapses events per segment and receptor into 10 statistics (counts,
moments, extrema and propagated conductances). It retains more causal detail
than the compact baseline but can still map different timestamped event
sequences to the same fixed vector.

## Literature-derived architectural hypotheses

1. **Sparse support and normalization, before capacity.** The optimizer may
   be unable to use rare event channels because their support is absent from
   selection roles and their normalization explodes the rare nonzero values.
   This must be tested before attributing failure to model capacity.

2. **Synaptic input is a marked jump process.** A timestamped synaptic event
   should update a local state at its real intra-ms time, while the membrane
   evolves continuously between jumps. Neural jump differential equations
   provide the matching flow-plus-jump abstraction. A Deep-Sets encoder is a
   useful control for a variable-size collection of timestamped events, but a
   chronological shared jump cell is the stronger biological hypothesis.

3. **The remaining source may be learnable only after mechanism
   factorization.** Removing the known synaptic term does not test
   cancellation among Na/K/Ca/leak and calcium-dependent currents. Predicting
   constrained current or conductance families and summing them through the
   known cable operator is closer to a universal differential equation than
   predicting one residual scalar.

4. **Quiet-state safety needs an explicit causal abstention path.** The model
   repeatedly helps active coordinates and damages quiet ones. A causal gate
   that defaults exactly to the passive operator can test whether this is a
   zero-inflated/hurdle problem. A future-state activity oracle may be used
   only as a nonselectable upper bound.

5. **Gate-like STATE should be updated in rate form.** Hodgkin-Huxley gates
   obey quasi-linear relaxation equations. A learned `x_inf` and positive
   `tau`, integrated exponentially, is a more stable atomic parameterization
   than an unconstrained state delta. This should be tested as a separate
   fragment before composition.

6. **Rollout robustness is downstream, not the first repair.** Dataset
   aggregation, scheduled sampling, state noise and pushforward training
   address train-rollout distribution shift. They cannot rescue an atomic
   source that is not identifiable or numerically learnable, so they become
   eligible only after an atomic arm passes.

## Proposed single-run follow-up

Working title: `06b_q_event_supported_jump_and_mechanism_playground.ipynb`.
Use only train-derived fit/calibration/development roles; do not read
validation, test or fresh-test state.

### Phase A — support and synthetic learnability preflight

- Build roles stratified by causal `U_realized` support, receptor, event
  multiplicity and intra-ms timing, with matched no-event controls from the
  same state/regime families.
- Require nonzero event support in fit, calibration and development before
  training.
- Audit raw and normalized nonzero quantiles and maximum values per feature.
- On a synthetic teacher with a known event-to-source map, cross event
  sparsity/multiplicity with legacy versus nonzero-robust normalization. This
  is the Allen-Zhu-style atomic learnability check.

### Phase B — aligned 3 x 2 real-data matrix

Event representation:

1. existing moment encoder;
2. receptor-resolved Deep Sets over individual timestamped events;
3. chronological shared jump cell over the same events.

Normalization:

1. legacy all-coordinate p99, retained as a negative control;
2. train-fit-only nonzero-robust scaling with declared clipping/log transform.

All six arms use the same causal event list, parameter budget, seeds,
initialization, minibatches and checkpoints. Record post-warmup (not only
initial) gradient cosines and event-feature ablation sensitivity. Timestamp
shuffle, receptor-label shuffle and event deletion are frozen causal controls.

### Phase C — adaptive 2 x 2 fragment matrix

Only if Phase B shows an event-positive development signal, cross:

- monolithic source versus mechanism-factored current heads whose outputs are
  summed by the known operator;
- ungated residual versus a causal passive-default safety gate.

The same run should additionally execute the cheap gate-rate fragment
(`direct delta` versus `x_inf/tau` exponential relaxation) on the shared
batches. It is diagnostic and cannot select the voltage candidate by itself.

### Decision gates

- An event representation is informative only if the true event stream beats
  its paired timestamp/receptor shuffles on event-positive development data
  across seeds.
- Added complexity is used only if post-warmup probes and frozen ablations
  show dependence on the added event/mechanism variables.
- A candidate must improve event-positive source and one-step voltage error
  without worsening quiet/moderate coordinates, and its 8 ms rollout must be
  no worse than the passive prior.
- Calibration/development arm rankings and per-seed signs must be reported;
  a median driven by one seed is not a pass.
- Failure of the synthetic preflight closes optimization/normalization before
  any biological conclusion. Passing synthetic but failing real data points
  to missing state/target structure rather than generic trainability.

## Primary sources used to generate the hypotheses

- Jia and Benson, *Neural Jump Stochastic Differential Equations*, NeurIPS
  2019: https://proceedings.neurips.cc/paper/2019/hash/59b1deff341edb0b76ace57820cef237-Abstract.html
- Kidger et al., *Neural Controlled Differential Equations for Irregular Time
  Series*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html
- Zaheer et al., *Deep Sets*, NeurIPS 2017:
  https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Lee et al., *Set Transformer*, ICML 2019:
  https://proceedings.mlr.press/v97/lee19d.html
- Rackauckas et al., *Universal Differential Equations for Scientific Machine
  Learning*: https://arxiv.org/abs/2001.04385
- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph
  Networks*, ICML 2020: https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011:
  https://proceedings.mlr.press/v15/ross11a.html
- Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent
  Neural Networks*, NeurIPS 2015:
  https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html
- Qiu et al., *Exponential Time Differencing Algorithm for Pulse-Coupled
  Hodgkin-Huxley Neural Networks*, 2020:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7227390/

