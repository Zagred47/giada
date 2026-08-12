# HayFlow 05c experimental record

This directory records the completed causal-isolation experiment. Generated
checkpoints remain external; `result.json` binds the ZIP and the decision-grade
members through SHA-256 values.

05c completed successfully as a diagnostic and returned
`DIAGNOSTIC_ONLY_NO_FULL_TRAINING`. The original timed event correction is not
the cause of the 55.35 mV boundary-peak failure: the worst transition contains
no teacher event, and disabling the complete event jump leaves the maximum
peak error unchanged while marginally improving mean RMSE.

Neither fresh H2 path could overfit one transition. The direct residual control
was substantially worse and exhibited very large pre-clipping gradients. This
supports the registered `ENCODER_OR_OPTIMIZATION_BOTTLENECK` classification,
but does not distinguish those two causes because the direct head itself was
randomly initialized, bounded by a tanh, and trained jointly with the complete
random model. The next experiment must isolate numerical conditioning with a
zero-initialized free residual before making another architectural decision.
