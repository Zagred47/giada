# HayFlow 05j-e - architecture reassessment

Status: implemented; Kaggle execution pending.

05j-d established two simultaneous facts: the direct nonlinear tree decoder
improved RMSE relative to its same-split ridge reference, but no seed met the
absolute segment-level gates. 05j-e is a frozen-checkpoint forensic experiment
designed to localize that remaining blocker before any further architecture is
trained.

The exact 05j-d archive and all indexed members are verified. The registered
36 fit, 12 calibration and one development pairs and their 226-dimensional
multiscale-tree features are reconstructed deterministically. All six saved
checkpoints are loaded and their metrics must reproduce the registered values
within `1e-4`. No checkpoint is updated and no new candidate is trained.

The analysis has four independent parts:

1. ensemble error and seed disagreement, separating systematic bias from
   optimizer variance;
2. error maps by segment, morphology region, protocol and regenerative regime,
   top-10% energy concentration, and a one-step axial low/high-frequency
   decomposition;
3. a fit-only per-segment affine diagnostic oracle, testing whether static
   scale or offset errors can explain the failure;
4. paired-future identifiability, measuring whether large teacher branch
   differences occur in at least 5% of the bottom 10% of multiscale-feature
   distances, on both calibration and development.

The affine probe is diagnostic only and cannot authorize a model. Development
is never used to fit or select anything. The decision routes the next notebook
to calibration/optimization repair, state/input representation revision,
region/mechanism experts, decoder-objective revision, or a controlled
capacity/optimization grid according to the observed failure anatomy.

Held-out inputs and targets, rollout and full training remain prohibited.
Expected artifact: `hayflow_hines_architecture_reassessment.zip`.
