# HayFlow 05i-c - bounded recency and synaptic-domain repair

Status: completed on Kaggle; all registered input-contract gates passed. A
separate 05j train/development representation recheck is authorized, while
full training remains prohibited.

05i-b proved that all NetCon slots can be decoded class-wise and that the
frozen H2 path is numerically supported, but seven teacher-state coordinates
remain outside the fixed input gate. Four are recent-event `tsyn` ages and
three are authentic dynamic AMPA/NMDA traces absent from the selected
fit-train support.

05i-c replaces model-facing age by the causal reversible coordinate
`recency = tau / (tau + age)`. `tau` comes from positive authentic `Dep/Fac`
parameters, with a preregistered 1 ms minimum for teacher synapses whose
short-term-plasticity constants are zero. The complete domain `(0, 1]` gets a
fixed scale floor of `1/50`, leaving a factor-two margin below the unchanged
`|z| <= 100` gate.

Dynamic point-process traces `A_AMPA`, `B_AMPA`, `A_NMDA`, `B_NMDA`, `A`, and
`B` retain their nonnegative `log1p` representation. Their fixed transformed
scale floor is `log1p(1)/35`, defined from a unit release increment and not
from development or held-out observations. All other hierarchical train-only
scale rules remain those of 05i.

The raw teacher snapshot, dataset, thresholds and frozen H2 weights are
unchanged. Held-out future voltages and event labels remain sealed. No
candidate head is trained or evaluated and no rollout is performed.

Expected artifact: `hayflow_hines_synaptic_domain_repair.zip`.

## Registered result

The completed run used code revision `2f38632d9d0e0abebe649f790118f78868d2784d`.
The downloaded archive and all 20 indexed members passed independent SHA-256
and size verification. Bounded recency was exactly reversible with maximum
round-trip error `0.0`; all 1,278 recency coordinates and all 3,834 dynamic
synaptic traces satisfied their preregistered domain floors.

The complete 17,220-coordinate input contract passed. The held-out maximum
fell from the 05i-b value of `223.622915` to `39.813518`, below the unchanged
limit of 100. The recency family reached at most `14.879954` and the dynamic
trace family at most `39.813518`. Only `0.0813%` of the held-out standardized
values exceeded the diagnostic threshold of 8, below the fixed 1% limit. No
raw, transformed, standardized or H2 feature value was nonfinite.

The frozen-H2 authentic and zero-causal audits also passed. The held-out/train
maximum-norm ratios were `0.331327` and `0.698655`, while maximum standardized
H2 values were `18.821954` and `14.344117`. Physical boundary voltage remained
bounded at `83.379829 mV` absolute maximum.

This result authorizes only the separate
`05j_repaired_representation_train_development_recheck`. Held-out future
targets remain sealed and no candidate-head training, rollout or full training
is authorized directly by 05i-c.
