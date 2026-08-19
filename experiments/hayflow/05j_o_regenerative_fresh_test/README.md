# HayFlow 05j-o - preregistered regenerative fresh test

Status: completed and fully verified.

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

## Result

All 243 indexed members passed size and SHA-256 verification. The teacher
generated all 32 frozen pairs (64 episodes, 768 transitions), and exhaustive
replay reproduced every transition with maximum error 3.815e-6, below the
registered 1e-5 tolerance.

The three unchanged decoder checkpoints all passed the fresh-test gate. Their
aggregate voltage RMSE values were 0.4037, 0.3774 and 0.3775 mV for seeds 17,
29 and 43, versus 2.5763 mV for the better of frozen H2 and persistence.
Median branching retention remained close to one (1.1133, 1.0976 and 1.1389),
and maximum segment errors remained well below H2. No checkpoint selection,
architecture search or retraining was performed after opening the outcomes.

The result authorizes only a frozen 2/4/8 ms micro-rollout of all three
checkpoints. It does not authorize full training.
