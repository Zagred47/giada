# 06b-q — Event-supported jump and mechanism playground

This experiment is the causal follow-up to the registered 06b-p result.  It
does not treat the old 42 pooled synaptic statistics as an exact event list.
Instead it constructs the ordered, receptor-resolved `U_realized` sequence at
every segment and compares three matched encoders under two normalization
contracts.

The train-only split is built from seed/snapshot-disjoint components that each
contain both event and no-event transitions according to `U_realized`.  The
50/50 balance is then enforced at window and minibatch level; it is not imposed
by incorrectly requiring entire biological episodes with no events.

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

## Registered result

The completed archive is registered in `result.json`; all 30 indexed members
pass independent size and SHA-256 verification.  The support repair worked:
fit/calibration/development contain respectively 66/22/22 event transitions,
so the zero-support confound from 06b-p is gone.

The selected chronological-jump plus nonzero-robust arm trained to step 300 in
all three seeds and improved median one-step RMSE from the passive prior's
7.17 mV to 6.84 mV, a real 4.66% gain.  That gain is not yet attributable to
the event content: deletion adds only 0.0081 mV (0.12%), while timestamp and
receptor perturbations are smaller still.  These signs pass the implementation
boolean but remain far below the registered 2% materiality scale.

Recursive behavior remains the blocker.  The trained arm reaches a median
46.94 mV RMSE at 8 ms.  The later safety stage selects checkpoint zero for
every seed and both gate arms, yielding identical 18.16 mV passive-prior
rollouts.  Thus the passive-default gate did not learn a protective policy;
the run simply returned to the safer zero residual.

The qualified diagnosis is
`ONE_STEP_SOURCE_LEARNABILITY_IMPROVED_BUT_EVENT_UTILIZATION_IS_SUBMATERIAL_AND_RECURSIVE_BOUNDARY_INSTABILITY_PERSISTS`.
No candidate is promoted.  The authorized follow-up is a paired train-only
playground crossing a causal event-utilization auxiliary with explicit
recursive-boundary exposure, keeping the passive Hines solve as the fixed
safety baseline.
