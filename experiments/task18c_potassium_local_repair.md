# 🧬 GIADA Task18c — K_Pst local repair

## 🔎 Evidence, not a changed threshold

Task18b completed technically valid (native audit, frozen checkpoint hashes,
clean pinned source and vectorization verified). Three seeds pass uniform
gate RMSE and all negative-tail gates. Uniform maximum error remains
0.0141–0.0158 against the original0.01 limit; Task19 is not authorized.

Read-only consumed-fresh probe localizes all three maxima to m at−60.284mV,
dt25ms. Replacing predicted tau alone by the exact tau reduces error below
0.0008; replacing inf alone does not. Other broad voltage bands have maximum
below0.01 on these consumed samples. This implicates local mTau approximation,
not a proven universal inability of the architecture. The canonical branch
change at−60mV includes a cusp and a small jump; neither is edited.

## ⚗️ Paired factorial

| Axis | Control | Treatment |
|---|---|---|
| Coverage |25%tail[-135,-115]mV |same tail +25%[-65,-55]mV |
| Transition loss |mean squared error |mean +mean worst5% squared errors |
| Learning rate |0.003 |0.0003 |

Eight arms ×three independent seeds. Width32, same frozen parent per seed,
same tuplecount24k, states, dt choices, minibatch indices, rate labels,
independent Adam moments and clipping. All arms reset optimizer state equally:
not an exact continuation of the old Adam trajectory. Checkpoints0/5k/15k/30k
provide a budget axis without duplicate training. The parent is a minimal,
hash-verified fixture derived from the complete archived18b result.

## 🔒 Freeze and confirmation

New fit/development/fresh seeds are registered in the JSON. Development uniform,
negative-tail and kink strata select one common arm/checkpoint across all
three seeds. Original gate thresholds unchanged. Fresh uniform, negative-tail
and kink must all pass for all three seeds; negative OOD remains diagnostic.
The frozen parent is evaluated on identical new fresh tuples only after freeze
to provide a genuinely aligned performance comparison. No fresh selection.

All fixed-factor development contrasts and scalar metrics are preserved.
Consumed18b fresh is diagnosis, not independent confirmation. Ca/Na bounded
Task18 successes retained; no embedded substitution or hardware speedup claim.

## ⚙️ Execution

Native NEURON oracle runs before CUDA in a separate worker. GPU uses one
leading-axis ensemble with independent parameters and optimizer moments;
per-model learning-rate scaling is preflighted against standalone Adam for
both rates. Sparse checkpoint logs, no large state dumps. Worker errors produce
a technical failure report; scientific NO-GO is separate. Pin notebook, code,
config and parent fixture to the same published revision.
