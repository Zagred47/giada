# HayFlow 05i-b - class-aware NetCon semantic state repair

Status: implementation ready; execution pending on Kaggle.

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
