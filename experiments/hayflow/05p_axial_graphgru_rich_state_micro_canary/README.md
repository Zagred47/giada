# HayFlow 05p - axial GraphGRU and rich-state micro-canary

Status: implemented; independent Kaggle execution pending.

05o found strong and seed-consistent nonlinear one-step information in both
the authentic axial-voltage block and the 64-dimensional semantic initial
teacher-state sketch.  05p asks whether those signals survive a genuinely
autoregressive voltage rollout rather than another boundary-reset probe.

Four parameter-identical GraphGRUs are trained on the same paired windows and
initializations:

- voltage, causal drive and static segment features only;
- the same contract plus authentic axial parent/child voltage features;
- the base contract plus the verified rich initial-state sketch;
- the joint axial and rich-state contract.

All four retain the same authentic hidden-state graph mixer.  Contract blocks
are zero-filled when absent, so architecture width and parameter count remain
identical.  Axial features are recomputed from the model's predicted voltage
at every step.  The rich teacher state is used only to initialize the hidden
state at the beginning of each window; it is never refreshed during rollout.

Training is closed-loop at 2/4/8 ms with causal `U_realized`, no teacher
forcing and no teacher state after the initial boundary.  Only the expanded
train-derived fit/calibration/development roles are read.  Fit supplies
gradients, calibration selects checkpoints, and development is evaluated once
after freezing.  Validation, test and sealed fresh-test data are excluded.

Each information signal must improve both paired recurrent comparisons by at
least 5% in median RMSE, win at least two seeds and remain non-inferior on
regenerative endpoint coordinates.  The joint candidate must also beat
persistence by at least 10% in two seeds without non-finite or non-physical
voltages.

This micro-canary cannot authorize full training, a fresh test or mass dataset
generation.  It can authorize only one bounded 05q rollout expansion.

Expected artifact: `hayflow_axial_rich_state_recurrent_canary.zip`.
