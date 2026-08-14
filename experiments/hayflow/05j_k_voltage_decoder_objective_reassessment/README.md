# HayFlow 05j-k - voltage-decoder objective reassessment

Status: implemented; Kaggle execution pending.

05j-j formally rejected the registered regenerative-state hypothesis on 24
independent pairs, but its aligned diagnostic oracle diverged to 695.67 mV RMSE
while the intercept and spatial-shift control remained near 24 mV. This
notebook does not overwrite that registered result. It asks whether the result
is scientifically interpretable or is dominated by feature transport and
support mismatch.

The complete frozen pipeline is reconstructed without candidate training.
Fit-only state-surface normalizers are applied to the independent support and
their standardized tails, fit-envelope violations and oracle conditioning are
measured. The voltage side is then decomposed into persistence, frozen H2 and
frozen direct-tree errors, target-transition shift, regenerative activation
coverage, regional error, branching amplification and residual-target range.

This is explicitly a post-result audit. It cannot retroactively confirm the
state-transition hypothesis and it does not tune a model on the 05j-i support.
If near-regenerative support is outside the original training regime, the next
step must acquire a train-only shard plus a fresh untouched test shard before
another canary is evaluated.

Expected artifact: `hayflow_hines_voltage_objective_reassessment.zip`.
