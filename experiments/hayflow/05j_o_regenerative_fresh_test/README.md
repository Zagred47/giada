# HayFlow 05j-o - preregistered regenerative fresh test

Status: implemented and locally verified; Kaggle execution pending.

05j-n passed its development gate for all three registered seeds and therefore
authorizes the first and only opening of the 32-pair fresh-test plan sealed in
05j-m. This experiment generates all 64 teacher trajectories exactly as frozen,
retains every outcome, performs exhaustive replay, and evaluates the three
unchanged 05j-n checkpoints.

No fresh-test result may alter the representation, normalization, architecture,
checkpoint or acceptance thresholds. Each seed is compared with frozen H2 and
persistence. At least two of three seeds must improve RMSE over the better
baseline by five percent, retain paired-future distance within [0.5, 2.0], and
not exceed the H2 maximum segment error. A pass authorizes only a separate
micro-rollout experiment; it does not authorize full training.

Expected artifact: `hayflow_hines_regenerative_fresh_test.zip`.
