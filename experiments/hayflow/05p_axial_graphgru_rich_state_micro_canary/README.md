# HayFlow 05p - axial GraphGRU and rich-state micro-canary

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `f133ed92009464a6bc120a13860669230605245b245d6adcb2da946c58f3020e`;
- artifact-index SHA-256: `8e5e81a6af86fef15e07da589a4ca2d03c23c8d8caeb37b0545992e4d27b63f6`;
- final-report SHA-256: `feb189283547a2143069ac8f74de54284b2f9383e7778e66fa1e8479a4bc3f3a`;
- all fifteen indexed members passed size and digest verification.

All four families beat 8 ms persistence in all three seeds without physical
violations.  The joint model achieved 16.16, 16.92 and 17.31 mV RMSE versus
17.96, 18.29 and 18.49 mV for voltage-only.  Axial or state information alone
added only 0.63% and 0.99% median gain, but each added 6.57% and 6.91% when
conditioned on the other block.  The strict independent-signal gate therefore
failed and registered `NONLINEAR_ONE_STEP_SIGNAL_DID_NOT_TRANSFER_TO_ROLLOUT`.
The interaction pattern authorizes only a frozen 05q temporal-observability
reassessment; it does not justify retraining or an architecture claim.

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
