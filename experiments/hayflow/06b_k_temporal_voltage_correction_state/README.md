# 06b-k — atomic temporal voltage-correction state

This optimizer-free component playground follows the 06b-j diagnosis that a
static heterogeneous gain is identifiable but fails low-activity recursive
composition. All correction models are fit on recursive exposures generated
by the frozen 06b-j static lookup.

The aligned matrix compares an instantaneous exposure-matched ridge control,
fast and slow EMA states, predicted displacement, their causal combination,
and a prohibited teacher-current-error oracle. Models are region-specific
closed-form ridge regressions with the same correction bound and calibration
protocol. The primary temporal arm must also beat the instantaneous exposure
control, not merely the original static lookup.

Fit, calibration and development are historically reused train roles. No
validation/test state is accessed, no independent confirmation is claimed and
the experiment cannot authorize 06c or full training.
