# HayFlow 05q - temporal-observability contract reassessment

Status: implemented; independent Kaggle execution pending.

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
