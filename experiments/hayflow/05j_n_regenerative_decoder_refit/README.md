# HayFlow 05j-n - regenerative decoder refit

Status: completed and fully verified.

05j-m supplied 96 complete train-only boundary pairs from the previously
missing near-regenerative domain while keeping a disjoint 32-pair fresh test
sealed. This experiment now tests the narrowest registered repair: keep the H2
core frozen, keep the direct-tree decoder architecture unchanged, and refit
only its representation statistics and trainable weights.

The original 54 fit pairs and the 96 new train pairs are partitioned by a
deterministic, outcome-blind hash. The combined internal fit contains 120 pairs
and is the only support used for normalization, PCA and gradient updates. The
combined internal calibration contains 30 pairs and is the only support used
for checkpoint selection. The already observed 24-pair 05j-i shard is used
strictly as development evidence after each selected checkpoint is frozen.

Three fixed seeds are trained. At least two must improve internal calibration
over frozen H2 by 10%, improve development over the better of H2 and
persistence by 5%, retain the branch contrast within [0.5, 2.0], and not exceed
the H2 development maximum segment error. Only that robust gate authorizes the
next notebook to generate and evaluate the preregistered fresh test.

This notebook performs no architecture search, no rollout and no full model
training. It neither extracts nor generates fresh-test outcomes.

Expected artifact: `hayflow_hines_regenerative_decoder_refit.zip`.

## Result

All 45 indexed members passed size and SHA-256 verification. The registered
120/30 internal fit/calibration split remained episode-disjoint, and the
24-pair 05j-i support was evaluated only after checkpoint freezing. All three
seeds passed the robust development gate (the preregistered minimum was two).

Development RMSE was 0.3681, 0.3347 and 0.3543 mV for seeds 17, 29 and 43,
respectively, versus 2.7364 mV for frozen H2 and 2.4807 mV for persistence.
Median branching retention stayed close to one (1.1304, 1.0343 and 1.1213).
No fresh-test input or outcome was accessed during fitting.

The result authorizes exactly one next operation: generate all 32 pairs from
the previously sealed fresh-test plan and evaluate the three frozen
checkpoints without selection, tuning or retraining. It does not yet authorize
a candidate, rollout or full training.
