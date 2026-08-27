# HayFlow preliminary report source brief

Status: source-controlled writing brief prepared before the Overleaf template
is connected. This file is not the final paper and must not be cited as an
experimental artifact.

## Editorial inputs

The brief incorporates:

1. the 2026-08-27 call with Alessio Ragno;
2. Alessio's paper-structure mind map;
3. the registered experiment and checkpoint map in
   `experiments/hayflow/CHECKPOINT_MAP.md`;
4. the completed information-matched Branch-ELM sidecar;
5. the registered 05j-n/05j-o and 06b-r artifacts.

The mind map requires the usual progression:

- abstract;
- introduction: problem, importance, standard solutions, proposal,
  hypotheses and advantages;
- related work: pioneers, representative methods and precise differences;
- method: prerequisites, workflow, components and hypotheses;
- results: setup, tuning, metrics, competitors and ablations;
- conclusion: advantages, one-sentence result summary, limitations and future
  work.

## Report type and current claim level

This should initially be written as a **technical progress report**, not as a
finished full-scale paper. The experiments are deliberately diagnostic and
causal, use bounded support and have not yet produced a validated autonomous
surrogate or a mass-scale training run.

The report can claim:

- a richer, replay-validated Hay teacher dataset and causal state contract;
- a fresh-tested one-step checkpoint family;
- a fair, information-matched small-model comparison with Branch-ELM;
- a systematic experimental decomposition of the autoregressive failure;
- causal evidence that pushforward boundary exposure improves recursive
  stability;
- a precise list of remaining blockers.

It cannot claim:

- a finished replacement for NEURON;
- a validated autonomous/free-running model;
- that 0.3308 mV and the published Branch-ELM 0.6376 mV form a direct
  leaderboard;
- that the most recent 06b-r checkpoint passed its registered promotion gate;
- that rapid-firing, spike-timing or full biological coverage has already been
  validated;
- that the current experiments have the scale of the original NeuronIO study.

## Problem formalization

Given the complete causal state of the Hay L5 pyramidal-cell teacher at a
millisecond boundary and the realized inputs during the next interval, learn a
flow map

`(S_t, U_[t,t+1ms]) -> (S_t+1, V_micro, events)`

that preserves the complete 642-segment morphology, authentic receptor-aware
synaptic input and the teacher's relevant mechanism state. The central
challenge is not only one-step expressivity: the learned map must remain
identifiable, optimizable and stable when repeatedly applied to its own
predicted state.

## Recommended report thesis

The core message is not that one final architecture has already solved the
neuron. It is:

> A richer causal state contract makes the one-millisecond Hay flow map highly
> predictable, but one-step accuracy alone does not imply a stable surrogate.
> Controlled component-level experiments isolate the resulting closed-loop
> distribution shift and identify pushforward boundary exposure as the first
> intervention that materially repairs learned recursive behavior without
> physical-voltage violations.

This thesis naturally accommodates both required models:

1. **05j-n / 05j-o:** the strongest decision-grade one-step model;
2. **06b-r:** the most balanced learned recursive diagnostic and current
   architecture frontier.

## Model A: canonical one-step checkpoint family

Use the exact identity and metrics from `checkpoint_registry.json`.

- Architecture: frozen HayFlow-Hines H2 plus registered direct-tree residual
  decoder.
- Morphology: 642 segments.
- Decoder: 226 tree features, 16-dimensional segment embedding and shared
  `242 -> 96 -> 96 -> 1` SiLU network with bounded residual output.
- Trainable decoder parameters: 43,009; the frozen H2 core is additional.
- Selection: internal calibration only.
- Evaluation: preregistered 05j-o fresh test, with no checkpoint selection or
  retraining after opening it.
- Fresh-test RMSE: 0.4037, 0.3774 and 0.3775 mV for seeds 17, 29 and 43.
- Canonical ensemble RMSE: **0.3308 mV**.
- Scope: one-step candidate only.
- Failure: the same frozen family reaches 62.72--95.47 mV at 8 ms in closed
  loop and is not an autonomous simulator.

Seed 29 may be used as a single-file representative because it already had the
lowest internal-calibration RMSE. The fresh test must not be described as the
seed-selection mechanism.

## Model B: recursive architecture frontier

- Experiment: 06b-r recursive event exposure playground.
- Arm: `auxiliary_off|pushforward_4ms|stability_off`.
- Architecture: passive Hines physical prior plus a 12,972-parameter recursive
  effective-source cell and compact causal `U` input.
- Development median 8 ms RMSE: **14.720 mV**.
- Support-matched passive reference: 18.162 mV.
- Gain over passive: 18.95%.
- Error reduction relative to the frozen 06b-q learned arm: 68.64%.
- Physical-voltage violations: zero.
- Development: all three seeds non-regressive.
- Registered veto: seed 61043 regresses by 10.08% on calibration half B.
- Deployment limitation: teacher mechanism state is supplied during rollout.
- Status: learned recursive diagnostic, not promoted candidate.

The ordered event auxiliary and directional stability penalty should appear as
negative ablations, not as parts of the proposed successful arm.

## Professor-requested Branch-ELM comparison

The oral call correctly challenged the original 0.4-versus-0.6 comparison:
the numbers came from different data distributions, target scopes and metrics.
That comparison must be retracted in the report and replaced by the completed
information-matched 06b-c result.

Corrective contract:

- same 76-value per-segment causal tensor;
- same paired transitions and sample order;
- same optimizer, loss and checkpoint-selection role;
- same unclipped target `V_(t+1)-V_t`;
- no teacher endpoint input;
- no rollout or spike claim;
- 8,002-parameter Branch-ELM core versus 8,985-parameter HayFlow voltage path;
- the frozen 7,212-parameter state updater cannot influence the voltage score.

Development results:

| Scope | Branch-ELM median RMSE | HayFlow median RMSE | HayFlow median reduction |
| --- | ---: | ---: | ---: |
| All 642 segments | 6.380 mV | 5.995 mV | 6.44% |
| Soma | 14.824 mV | 9.577 mV | 34.94% |
| Active examples | 21.012 mV | 18.867 mV | 8.92% |

HayFlow wins every paired seed globally. This is evidence for the compact
causal voltage representation under equal information, not evidence that the
complete surrogate is solved. It is train-derived development support, not a
new fresh test.

## Proposed section structure

### 1. Abstract

Write last. Include the problem, richer state contract, component-level
methodology, one-step result, recursive finding and explicit limitation in
approximately 150--200 words.

### 2. Introduction

1. Biophysical neuron simulations are accurate but computationally costly.
2. Voltage-only surrogates discard state needed for causality and branching.
3. One-step prediction is insufficient when a model is deployed
   autoregressively or inside a network.
4. Proposal: learn a causal one-millisecond flow map over a richer teacher
   state, using physical morphology and controlled learnability experiments.
5. Contributions:
   - canonical teacher/state/data contract;
   - decision-grade one-step candidate;
   - information-matched Branch-ELM comparison;
   - causal decomposition of recursive failure;
   - pushforward exposure as a supported recursive intervention.

### 3. Related work

Organize by question, not chronology:

- detailed multicompartment neuron models and the Hay teacher;
- voltage-only neural surrogates and NeuronIO/Branch-ELM;
- graph/morphology-aware neural dynamics;
- learned physical simulators and autoregressive distribution shift;
- pushforward/noise/curriculum approaches for stable rollout.

Every comparison must state differences in assumptions, information, target,
loss, morphology and deployment regime. Primary literature must be collected
before final prose is frozen.

### 4. Problem and teacher contract

- Hay morphology and 642 instantiated segments;
- causal `S_t`, `U_realized` and `S_t+1` boundary contract;
- voltage, mechanism STATE, ions/calcium, synaptic state and metadata;
- snapshot/restore and deterministic replay;
- event-aware diagnostic support and train/calibration/development/fresh-test
  separation.

### 5. Methodology

- Allen-Zhu-inspired atomic learnability workflow;
- distinguish capacity, representation, optimization and recursive exposure;
- paired seeds, aligned minibatches, factorial interventions and negative
  controls;
- Model A architecture and checkpoint selection;
- Model B physical prior, effective source and pushforward exposure;
- evaluation and promotion gates.

### 6. Experimental setup

- diagnostic dataset scale and why it is deliberately bounded;
- input/target definitions;
- split hygiene and replay validation;
- hyperparameters and model sizes;
- baselines: persistence, frozen H2, Branch-ELM and passive Hines;
- metrics: global/region/activity RMSE, maximum segment error, drift,
  branching retention and physical violations.

### 7. Results

1. Information-matched Branch-ELM comparison.
2. One-step fresh-test result for 05j-n/05j-o.
3. Failure of naive autoregressive composition in 05k.
4. Recursive progression to 06b-r.
5. 06b-r factorial ablation: pushforward supported; event auxiliary and
   directional penalty inert.
6. Remaining regime and cross-seed failures.

### 8. Discussion and limitations

- new inputs do not reset internal state error;
- slow variables, voltage-dependent NMDA and threshold events propagate drift;
- teacher mechanism-state access prevents deployment claims for 06b-r;
- bounded experiments are scientifically diagnostic but not mass-scale
  evidence;
- differences in support prevent collapsing every result into one scalar
  leaderboard.

### 9. Conclusion and future work

One-sentence summary:

> HayFlow currently combines a fresh-tested 0.3308 mV one-step checkpoint
> family with a separate pushforward-trained recursive prototype that improves
> 8 ms error by 18.95% over its physical baseline, while autonomous state
> closure and cross-seed robustness remain unresolved.

Future work should prioritize autonomous mechanism-state prediction,
cross-seed nested calibration, event-positive evaluation, longer rollout and
only then full-scale training.

## Required figures and tables

1. Teacher/data-contract diagram: `S_t + U_realized -> S_t+1`.
2. Architecture diagram showing Model A and Model B as separate stages/results.
3. Table of 05j-o fresh-test metrics and baselines.
4. Information-matched Branch-ELM comparison table.
5. Horizon plot comparing 05j-n frozen rollout, passive reference and 06b-r.
6. 06b-r factorial ablation plot.
7. Timeline/decision tree of successful and failed atomic experiments.
8. Limitations table separating one-step, recursive and deployment claims.

## Items to resolve when Overleaf is connected

- exact template and venue style currently installed by Alessio;
- title, author order, affiliations and corresponding author;
- report length and deadline;
- whether the first version should be English or Italian;
- which plots are already suitable and which must be regenerated for paper
  quality;
- bibliography style and required related-work depth.
