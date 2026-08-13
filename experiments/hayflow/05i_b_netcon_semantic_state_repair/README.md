# HayFlow 05i-b - class-aware NetCon semantic state repair

Status: completed on Kaggle; semantic mapping passed, full input contract did
not pass. Full training remains prohibited.

This experiment repairs the semantic schema defect isolated by 05i. Raw
NetCon slots are decoded according to the owning point-process class, so
`weight[4]` is no longer allowed to pool excitatory `Pr` with inhibitory
`tsyn`. The complete expected mapping covers 6,390 NetCon state coordinates:
3,834 probability states, 1,278 AMPA/NMDA amplitude weights, and 1,278 last
event timestamps across 639 synapses of each class.

The model-facing timestamp representation is causal last-event age at the
current boundary. It is derived from stored `start_time_ms`, is reversible to
the raw timestamp, and uses no future information. The raw teacher snapshot is
preserved for replay and authentic causal-input construction.

05i-b retains all numerical thresholds from 05i. It performs a semantic
mapping/round-trip audit, a coordinate support audit, and frozen-H2 audits with
authentic and zeroed causal input. Held-out future targets remain sealed. No
candidate head is trained or evaluated, and no rollout or full-training path
exists.

Expected artifact: `hayflow_hines_netcon_semantic_state_repair.zip`.

## Registered result

The completed run used code revision `59563704889d081f82d4e69947ed8d3a5aa2db3e`.
The downloaded archive and all 19 indexed members passed independent SHA-256
and size verification. The class-aware mapping covered all 6,390 NetCon
coordinates with zero unmapped slots. The causal `tsyn` age transform had
exact round-trip error `0.0`, so the semantic correction itself passed.

The frozen-H2 audit also passed with finite values and bounded voltages. The
input contract nevertheless failed because the held-out teacher state reached
`|z| = 223.622915`, above the unchanged limit of 100. Seven coordinates
remain above the limit: four recent-event `tsyn` ages and three authentic
AMPA/NMDA trace states. In the selected fit-train sample, the affected trace
states are identically zero and the affected `tsyn` ages are approximately
1061--1140 ms; the input-only held-out examples contain active traces and
recent ages of 2.35--3.75 ms. This is a support/representation mismatch, not a
NetCon slot-label error and not evidence for relaxing the registered gate.

The next authorized experiment is a separate 05i-c representation revision.
It must use a preregistered bounded recency/recovery representation for `tsyn`
and domain-calibrated scales for nonnegative synaptic trace states. It must
retain the raw teacher state, the current thresholds, the train-only fitting
rule and the sealed held-out future targets. No candidate-head training or
rollout is authorized.
