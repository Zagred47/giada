# HayFlow 06b-i - voltage-objective recalibration playground

The sealed 06b-h audit established that multiplying the source scalar bridge
by 0.75 restores overall voltage and STATE utility, but also exposed severe
quiescent, moderate and somatic failures. This train-only playground asks
whether that heterogeneity is learnable through a small causal gain or a
balanced voltage objective before changing the bridge architecture.

Five interventions start from the same source checkpoint, use the same
minibatches and freeze the mechanism-STATE updater. They compare the original
active-weighted bridge update, activity-balanced bridge training,
activity-by-region-balanced bridge training, a single global gain and a small
causal gain conditioned only on predicted delta, current voltage and region.
The frozen 0.75 candidate is the common reference.

All fit, calibration and development components were used historically. This
is therefore a component learnability playground, not a new independent
confirmation. A pass can authorize only confirmation on fresh train support.
It cannot authorize 06c, validation/test access or full training.

Notebook: `notebooks/06b_i_voltage_objective_recalibration_playground.ipynb`.
