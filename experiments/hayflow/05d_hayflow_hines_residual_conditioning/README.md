# HayFlow 05d experimental record

This directory records the completed residual-conditioning ladder. Generated
checkpoints remain external; `result.json` binds the ZIP and decision-grade
members through SHA-256 values.

05d passed the free 642-value residual controls at numerical precision for
both the worst transition and the authentic counterfactual pair. The target,
loss, metric, and update plumbing are therefore sound. All nine frozen shared
decoders failed, however, and neither local-feature nor base-dynamics
unfreezing reached the preregistered absolute gates. The run consequently
returned `SHARED_REPRESENTATION_BOTTLENECK` and retained the full-training
prohibition.

This diagnosis is scoped to the tested shared 97-parameter boundary decoder,
the current 05b features, and the preregistered optimization budget. It is not
evidence that every morphology-aware HayFlow architecture is incapable of
learning the teacher. The next diagnostic should remove the remaining
optimizer ambiguity with closed-form linear probes and add explicit
segment-conditioned capacity before another neural canary is considered.
