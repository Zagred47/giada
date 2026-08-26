# 06b-q — Event-supported jump and mechanism playground

This experiment is the causal follow-up to the registered 06b-p result.  It
does not treat the old 42 pooled synaptic statistics as an exact event list.
Instead it constructs the ordered, receptor-resolved `U_realized` sequence at
every segment and compares three matched encoders under two normalization
contracts.

The notebook contains three stages in one run:

1. a known-answer sparse-event playground with mini scaling checkpoints;
2. a paired 3x2 biological matrix with train-only support-balanced roles,
   causal ablations and post-warm-up gradient probes;
3. an adaptive ungated versus passive-default safety test using only the arm
   selected on the second calibration half.

Per-mechanism factorization is deliberately conditional.  Boundary currents
are not valid substitutes for the current integral over a 1 ms macro-step.  If
the exact integrated targets are absent, the experiment records the missing
teacher-logger contract and trains no factorized-current candidate.

The development role is used only after checkpoint and arm selection.  No
validation, test or fresh sealed-test state is accessed.
