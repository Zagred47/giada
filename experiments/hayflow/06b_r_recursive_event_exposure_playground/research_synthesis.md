# Research synthesis for 06b-r

This note records the external evidence consulted before preregistering 06b-r.
Only primary papers and the official NEURON mathematical documentation are
used to motivate interventions.  None of these sources is treated as evidence
that an intervention will work on HayFlow; each intervention remains an
independent, falsifiable factor.

## Evidence translated into atomic hypotheses

1. **Training-distribution mismatch.**  Graph Network Simulators identify
   training-input corruption as one of the main determinants of long-rollout
   performance, while Message Passing Neural PDE Solvers introduce the
   pushforward trick specifically to encourage zero-stability.  Therefore
   06b-r compares teacher-boundary training with a four-step pushforward arm,
   using identical windows, targets, seeds and optimizer streams.

2. **Local error amplification.**  Recent work on neural differential
   equations reports that directional-Jacobian regularization can improve
   long integration while avoiding the full cost of long unrolls.  Therefore
   06b-r tests a finite-difference, passive-relative directional penalty.  The
   penalty does not demand global contraction: it only penalizes sensitivity
   above the authentic passive Hines step on the same perturbation.

3. **Hybrid physical/data-driven decomposition.**  Universal Differential
   Equations formalize the combination of known mechanistic operators with a
   learned missing component.  NEURON's official mathematical basis states
   that the compartmental update is a current-balance equation coupling
   capacitive, ionic and axial currents.  HayFlow consequently retains the
   exact passive Hines solve and asks the network only for the effective
   residual source.  A causal auxiliary target is computed from `S_t` and
   `U_realized`; it never uses `S_t+1` as an input.

4. **Distribution induced by the learner.**  DAgger establishes the general
   problem that supervised learners fail when inference visits states absent
   from the expert-generated training distribution.  06b-r does not implement
   online DAgger (which would require new teacher queries), but the pushforward
   arm is the bounded offline analogue testable with the existing snapshots.

## Primary sources

- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph
  Networks*, ICML 2020: https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
- Brandstetter et al., *Message Passing Neural PDE Solvers*, ICLR 2022:
  https://openreview.net/forum?id=vSix3HPYKSU
- Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning*, AISTATS 2011:
  https://proceedings.mlr.press/v15/ross11a.html
- Rackauckas et al., *Universal Differential Equations for Scientific Machine
  Learning*: https://arxiv.org/abs/2001.04385
- Janvier, Salomon and Meunier, *Jacobian Regularization Stabilizes Long-Term
  Integration of Neural Differential Equations*:
  https://arxiv.org/abs/2602.04608
- Hines and Carnevale, *The NEURON Simulation Environment — Mathematical
  Basis*: https://www.neuron.yale.edu/neuron/static/papers/nc97/nc3p1.htm

## Allen-Zhu / fragment-based translation

The experiment is not an architecture sweep.  It follows the registered
workflow: a known-answer auxiliary probe first; a small orthogonal factorial
matrix second; mini scaling checkpoints along the same trajectories; internal
utilization and sensitivity probes; and a hard support-matched non-regression
gate.  Each factor has a negative control and a preregistered interpretation.

