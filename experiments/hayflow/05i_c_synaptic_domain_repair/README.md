# HayFlow 05i-c - bounded recency and synaptic-domain repair

Status: implementation ready; execution pending on Kaggle.

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
