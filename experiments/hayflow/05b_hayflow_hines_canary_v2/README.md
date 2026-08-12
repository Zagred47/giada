# HayFlow 05b experimental record

This directory is the version-controlled, data-light record of the completed
05b corrected canary. Generated model artifacts remain external by project
policy. Their identity is bound to `result.json` through archive and member
SHA-256 values.

The experiment completed successfully as a diagnostic but returned
`NO_GO_FULL_TRAINING`: neither HayFlow-Hines H2 nor the fixed-order ConvGRU met
all four preregistered overfit thresholds. H2 retained an advantage in voltage
RMSE and counterfactual branching, while ConvGRU alone exceeded the event-F1
threshold. Neither model reproduced boundary peaks adequately.

The record also preserves the reason not to continue directly to full
training. The H2 maximum peak error remained 55.35 mV, effectively unchanged
from notebook 05, and the selected biological-delta target contained a
normalized outlier above 23,000. The next authorized activity is the small 05c
causal-isolation experiment described in `result.json`, not another full
300-epoch canary or the full curriculum.
