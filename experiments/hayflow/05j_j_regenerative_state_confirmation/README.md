# HayFlow 05j-j - independent regenerative-state confirmation

Status: completed on Kaggle; valid execution with a non-interpretable oracle
transport failure.

The exact artifact is registered in `result.json`; all 41 indexed members
passed independent size and SHA-256 verification. The frozen 05j-h result was
reproduced before the new support was touched (specificity error
`3.83e-7`). All 24 preregistered 05j-i pairs were then evaluated without
selection or retraining.

The formal preregistered decision rejected both the causal-boundary and
future-state-delta explanations. However, the aligned oracle extrapolated to
`695.67 mV` RMSE and `22,949.78 mV` maximum error, while the intercept and
spatial-shift control remained near `24 mV`. This is a numerical transport
failure, not an ordinary negative result. Therefore 05j-k first audits
feature-support and voltage-objective mismatch before the project accepts the
registered route to a decoder reassessment.

05j-i acquired 24 outcome-blind, snapshot-matched causal pairs, of which 23
landed in the missing near-regenerative voltage band. This notebook applies the
unchanged 05j-h diagnostic to that independent validation-only shard.

The original 05j-h train fit pairs remain the only data used to fit feature
normalizers, probe normalizers, ridge hyperparameters and coefficients. The
05j-d topology transform and three direct-tree checkpoints are reconstructed
and frozen. Before the new shard is inspected, the registered 05j-h
aligned-versus-shifted result must be reproduced within tolerance.

The primary decision uses all 24 preregistered pairs; the 23-pair
near-regenerative subset is descriptive only. Four families are fixed in
advance: intercept, causal boundary state, aligned future-state-delta oracle
and its equal-width spatial-shift control. The future-state delta is diagnostic
only and can never become a deployable input.

No candidate is trained, no held-out input is opened and no rollout is run.
The result determines whether the next canary should introduce an explicit
causal regenerative state, jointly predict the regenerative transition, use a
regime-conditioned objective, or return to the voltage-decoder objective.

Expected artifact: `hayflow_hines_regenerative_confirmation.zip`.
