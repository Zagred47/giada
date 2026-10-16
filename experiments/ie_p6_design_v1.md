# 🔬 IE-P6 — residual repair and local learned operator

## 🎯 Decision

Separate coverage, temporal accumulation and execution representation in one
notebook. Keep dt0.1ms, explicit18gates and all other physical blocks fixed.
This is an IE extension, not a replacement or renumbering of the original roadmap.

## 🧪 Paired matrix

Three seeds in each of A uniform/standard, B targeted/standard,
C uniform/short-recurrence, D targeted/short-recurrence. All start from their
own P5standard checkpoint. Same data pools, optimizer reset, LR and update count.
Half-uniform targeting weights come from training-only parent error probes.
Checkpoints0/1k/3k/10k/30k expose both matched-budget and selected outcomes.
Temporal evaluation is shared; only C/D use its gradient. Extra compute is not
claimed to be equal FLOPs. No post-hoc weight tuning or failed balanced package.

## ⚡ Representation sidecar

Learn81/161knot local coefficient operators, initialized from parent predictions.
They are numerical piecewise-linear learned models, not MLPs, and do not
necessarily have fewer parameters. Both train on A's minibatches/objective.
Measure initial projection error, model bytes, bound violations and actual cost.
All12MLPs also get a frozen .25mV deployment table; conversion error is separate
from teacher error. Formula and analytic LUT remain controls. Expected35maps,
210timing cells across3batches x2backends if every graph succeeds.

## 🔍 Internal probes

Repeat18oracle interventions on the NEW P5parents, never on the old P4 models.
Save voltage/gate error profiles and gradient scale/cosine diagnostics for each
old balanced extra term. Conflicting gradients are an association, not a causal
proof. All oracle outputs remain diagnosis-only, including worsening cases.

## 🔒 Evaluation and decisions

Freeze all selections before new fresh transitions and24new400step paths.
Historical Task35 stays already-opened diagnostic. Quality thresholds unchanged;
no selecting the best fresh seed. Report seed-paired contrasts and interaction,
quality AND >=10%resident-core savings with positive paired interval for speed.
Training is member-vectorized on one GPU; timing starts only after training ends.
No end-to-end or new native confirmation is claimed. Durable reports/checkpoints,
compact progress and subprocess failure report prevent output flooding.

Full numeric protocol: `ie_p6_preregistration_v1.json`.
