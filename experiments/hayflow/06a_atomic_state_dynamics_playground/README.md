# HayFlow 06a - atomic state-dynamics playground

Status: completed; artifact independently verified; technical gate failed.

Verified artifact:

- archive SHA-256: `b2aaa071925c0eea4c34f7e116d491faeea52f9fc166a8b6370e085c6532983d`;
- artifact-index SHA-256: `ad28ed4666e8bd99fb0be5f5d2230e7b731868e20740e94fdd7537aa56e96cb5`;
- final-report SHA-256: `fa72141a9ca50ceb2582d6eec1852dcd5af7d83e6e34b7e408750a48bf518fa2`;
- all ten indexed members passed size and digest verification.

The run was valid and leakage-free but registered
`ATOMIC_STATE_UPDATE_NOT_YET_LEARNABLE`. The causal arm improved one-step
normalized-delta RMSE over persistence by only `0.89%`; the diagnostic teacher
interval-voltage arm improved it by `1.41%`. Both remained below the
preregistered `2%` technical gate. Active-coordinate gains were similarly
small (`1.13%` and `1.66%`). No candidate, full model, validation access or
fresh test is authorized.

The failure is informative but does not reject explicit state evolution. The
teacher endpoint voltage adds only about half a percentage point over causal
start voltage, so endpoint information is not the missing primitive. The
calibration curves were still improving at step 300, and the current updater
does not observe the intra-ms voltage path that actually drives gate kinetics.
Moreover, horizon-specific rollout windows were selected independently; the
reported `20--22%` gain at 2 ms and `6--7%` at 8 ms are positive secondary
signals, not a valid monotonic horizon curve.

The registered follow-up is one bounded train-only
`06a_b_atomic_voltage_path_identifiability` experiment. It must use nested
common rollout windows and a paired factorial separating optimization budget
from fixed intra-ms voltage-path features. It must not increase model capacity,
use validation, compose the full neuron or reinterpret the failed 2% gate.

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
