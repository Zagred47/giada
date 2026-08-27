# Report structure compliance

This audit maps every branch of Alessio Ragno's paper-structure mind map to
the revised `main.tex`. It is an editorial aid, not part of the paper.

## Abstract and Introduction

| Requirement | Location | Evidence |
| --- | --- | --- |
| Abstract | `abstract` | Problem, approach, one-step result, matched baseline, rollout result, and limitation |
| Problem attacked | `Introduction / Problem and Importance` | Detailed-neuron acceleration without discarding causal state |
| Importance | Same subsection | Network scale, intervention, optimization, and mechanistic use cases |
| Standard/SOTA solutions | `Standard Solutions and the Remaining Gap` | Exact acceleration, morphology reduction, and learned surrogates |
| Proposal | `Proposal and Design Rationale` | Full-state 1 ms morphology-aware flow map |
| Hypotheses and reasoning | `Proposal...` plus `Hypotheses` | Four design arguments and H1--H6 |
| Why preferable | Introduction plus `Discussion / When HayFlow...` | Conditional comparison rather than universal superiority |

## Related Work

| Requirement | Location | Evidence |
| --- | --- | --- |
| Pioneers | `Detailed Multicompartment Models` | Cable theory, NEURON, Hay model |
| Recent works | All Related Work subsections | DeepDendrite, Neuron_Reduce, TCN, CNN--LSTM, ELM, Graph/PDE solvers |
| What they do | Each subsection | State, objective, and contribution described |
| What they solve | Each subsection and positioning table | Appropriate target/use case stated |
| Improvement over previous work | Each subsection | Hardware, reduction, compact memory, graph and pushforward advances |
| Difference from HayFlow | `Comparative Positioning` | Assumptions, state, objective, and use conditions compared explicitly |
| When to prefer each method | Positioning table and Discussion | Exact, reduced, output-only, and full-state cases distinguished |

## Method

| Requirement | Location | Evidence |
| --- | --- | --- |
| Prerequisites/background | `Background and Problem Formulation` | Compartment equation, hidden-state kinetics, flow map, rollout equation |
| Prior elements identified | Background and Related Work | Hines/cable solve, causal state, residual learning, pushforward exposure |
| Full workflow | `Evidence-Gated Workflow`, Figure 1 | Audit-to-fresh-test sequence and six experimental rules |
| Formulas | Equations 1--6 | Cable dynamics, flow map, recursion, both models |
| Potential benefits independent of results | Model A and Model B subsections | Tree context, sharing, residual baseline, physical transport/source split |
| How/why hypotheses are tested | `Hypothesis-to-Experiment Traceability` | Each H1--H6 tied to a contrast, observable, and decision meaning |

## Results

| Requirement | Location | Evidence |
| --- | --- | --- |
| Setup | `Experimental Setup` | Teacher, stores, splits, roles, implementation |
| Hyperparameters | Table 3 | Seeds, parameters, optimizer, LR, decay, clipping, budgets, losses |
| Parameter tuning | `Parameter Tuning and Selection` | Internal-only selection and nested calibration protocol |
| Competitors (at least 2--3) | `Baselines and Metrics` and result tables | Persistence, H2, Branch-ELM, passive Hines, prior 06b-q |
| Multiple metrics | Setup and results | Global/soma/active RMSE, max error, retention, drift, violations, marginal effects |
| Metric caveats | Setup and each result subsection | One-step, development, recursive, and speed scopes kept separate |
| Ablation study | `Ablation and Hyperparameter Impact` | 2x2x2 factorial and causal path deletions |
| Hyperparameter impact | Same subsection | Per-seed selected budget 240/120/40 and cross-seed sensitivity |
| Hypothesis decisions | Consolidated hypothesis table | Supported/partial/rejected with qualifications |

## Conclusion

| Requirement | Location | Evidence |
| --- | --- | --- |
| Advantages summary | `Scientific Progress and Method Advantages` | Three independent gains and matched ELM interpretation |
| One-sentence main result | First paragraph of `Conclusion and Future Work` | One-step, matched baseline, and recursive result summarized |
| Main result, not every result | Same section | Causal-state learnability plus need for exposure alignment |
| Future work | Final paragraph | State closure, nested calibration, long rollout, scale, speed, morphology, networks |

## Honest unresolved presentation work

The textual structure now covers the full mind map. Publication-quality plots
and a controlled common-hardware runtime benchmark remain future artifacts;
the paper says so explicitly and makes no unsupported speed claim.
