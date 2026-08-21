# HayFlow 06a - atomic state-dynamics playground

Status: implemented and preregistered; awaiting Kaggle execution.

Phase 05 established that semantically aligned teacher state improves an
8 ms voltage rollout, but the fixed boundary state does not remain valid at
16--32 ms.  Phase 06 therefore closes the fixed-state GraphGRU branch and
opens the `HayFlow-ESI` explicit-state architecture family.

The first experiment does **not** train a full neuron surrogate.  It asks the
atomic question that must be answered first:

```text
(mechanism STATE_t, local ions_t, V_t, U_realized[t,t+1])
    -> mechanism STATE_t+1
```

Two parameter-identical pilot arms are paired with the same initialization,
data order and optimizer:

- `causal_start_voltage` receives only the causal boundary voltage `V_t`;
- `teacher_interval_voltage` additionally receives `V_t+1 - V_t` as a
  diagnostic oracle for the current interval.

The second arm is never a deployment proposal.  Its purpose is to distinguish
an unlearnable state representation from missing voltage/state coupling.

Only original `train` episodes are partitioned into seed/snapshot-disjoint
fit, calibration and development roles.  Validation, test and fresh-test
states or outcomes are not read.  A single paired seed is a technical pilot,
not model selection and not decision-grade evidence.

The updater shares embeddings by biological identity
`(mechanism, variable, kind)` across segments.  Mechanism gates are represented
in logit space and the learned update is a bounded, zero-initialized residual,
so the untrained model starts exactly at persistence.  Weak per-coordinate
delta scales are repaired only from semantically identical train coordinates.
Local ion concentrations at `t` are log-transformed and normalized from fit
data, then supplied at their authentic owner segment. This is necessary for
calcium-dependent gates and remains fully causal.

The experiment reports one-step normalized-delta error and recursive
mechanism-state error at 1, 2, 4 and 8 ms while membrane voltage remains
teacher-forced.  If the diagnostic arm cannot improve over persistence by 2%,
the next action is to inspect the state contract, normalization or missing
local context.  A positive causal arm authorizes only a multi-seed 06b explicit
state-updater canary.  No validation selection, full training, mass data or
fresh sealed test is authorized by 06a.

Expected artifact: `hayflow_atomic_state_dynamics_playground.zip`.
