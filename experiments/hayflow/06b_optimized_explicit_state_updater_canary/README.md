# HayFlow 06b - optimized explicit-state updater canary

Status: completed; artifact independently verified; explicit voltage coupling required.

Verified artifact:

- archive SHA-256: `0d44d0f6aeb90c7df67a65cd2f92ffbdad9c9163acc11af92ab90f1c52d785ec`;
- artifact-index SHA-256: `0fe4985566f7276c333bd288280b3756751e82f02353d2e43c791374f126a612`;
- final-report SHA-256: `89512fc5cd37a06c21d59d9d5d74d6418f40afa0d09a3bbaf7a8bf2ff1e4ccc7`;
- all 19 indexed members passed size and digest verification.

The component-decision-grade result is valid and registered as
`ATOMIC_STATE_REQUIRES_EXPLICIT_VOLTAGE_COUPLING`. Every causal seed learned a
real signal: one-step gains were `7.85%`, `7.64%` and `7.93%`; the median
semantic-macro gain was `5.10%`; `94.4%` of semantic groups improved; all
seed/horizon rollout gains were positive; and median 8 ms gain was `14.94%`.
There were no non-finite values or state-domain violations.

The full causal gate nevertheless failed three preregistered requirements. Its
median one-step gain was `7.85%` rather than `10%`, median active-coordinate
gain was `8.20%` rather than `10%`, and it retained only `44.3%` of the endpoint
reference gain rather than `70%`. The endpoint arm was robust across all three
seeds, with a median one-step gain of `17.54%`. Thus the missing endpoint is a
material causal variable, not seed noise or an aggregate-only artifact.

All six calibration curves were still improving at step 1200, but the causal
and endpoint curves retained a large, consistent separation across seeds.
Longer optimization alone is therefore not the registered next action. The
authorized `06b_b_causal_voltage_state_coupling_forensic` must test a bounded
causal co-evolution mechanism for voltage and STATE, using paired/frozen
controls so a gain cannot be attributed merely to more parameters or training.
It must remain train-only and cannot authorize the full neuron.
