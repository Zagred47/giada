# HayFlow 05q - temporal-observability contract reassessment

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `73ed6fc3e244f49fe5172e5e917b253519e0bd7b614ab88584f1f8a06b41e0ec`;
- artifact-index SHA-256: `26ed1c29cded7a53f2d2dfce0519b19f6dfa5b9f4945a2d0867f046a51ac5d9b`;
- final-report SHA-256: `51a29770690b9fd52306d5946bb5611649da03a702f06c72be27b392166cd94e`;
- all four indexed members passed size and digest verification.

Frozen checkpoints reproduced with exactly 0.0 mV error.  The joint model
beat voltage-only by 7.49% median closed-loop RMSE and 9.09% with teacher
voltage boundaries.  Correct state was material: zeroing it degraded RMSE by
5.97%, while within-regime permutation degraded it by 11.18%.  Removing the
axial block degraded RMSE by only 2.63%, below the registered 5% gate.  The
result therefore rejects symmetric graph/state synergy and registers
`FROZEN_COUNTERFACTUALS_DO_NOT_SUPPORT_TEMPORAL_SYNERGY`, authorizing only a
frozen 05r temporal-representation reassessment.

05p found a robust joint recurrent candidate but failed its deliberately
strict independent-signal gate: axial and rich-state blocks were weak alone,
yet each was material when conditioned on the other.  05q distinguishes a
real temporal interaction from optimization coincidence or exposure error.

No model is trained.  The twelve exact 05p checkpoints and the same
train-derived development windows are reconstructed byte-for-byte.  Stored
05p metrics must reproduce within 0.0001 mV before interpretation.  The
following frozen counterfactuals are then evaluated for every seed:

- authentic closed-loop rollout;
- correct versus zeroed or deterministically permuted initial state;
- authentic versus disabled axial-voltage block;
- both blocks disabled;
- teacher-voltage-boundary rollout with the recurrent hidden path preserved.

Metrics are reported at every horizon from one through eight milliseconds,
including regenerative coordinates.  Temporal synergy requires at least 5%
joint gain over voltage-only, at least 5% degradation after removing either
block, at least 2% degradation after permuting state identity, regenerative
non-inferiority and positive evidence in two of three seeds.

Only original train-derived development data are evaluated.  Validation,
test and sealed fresh-test inputs remain excluded.  05q cannot authorize
training, architecture selection, fresh-test generation or mass data; at
most it authorizes one bounded 05r rollout expansion.

Expected artifact: `hayflow_temporal_observability_reassessment.zip`.
