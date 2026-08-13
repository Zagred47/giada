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

## Registered result

The run completed validly at revision
`20423ba709d29b784e5e88d7eb789bef660adf06`. The archive and all 31 indexed
members passed SHA-256 and size verification. All six 05j-d checkpoints
reproduced their registered metrics within the fixed `1e-4` tolerance; no
retraining, candidate selection, held-out access or rollout occurred.

The diagnosis is `LOCALIZED_MORPHOLOGY_REGIME_ERROR_DOMINATES`. On the
development pair, the direct-tree ensemble retained the future separation
almost exactly (`0.9943`) but had `2.9953 mV` RMSE and `18.7010 mV` maximum
segment error. The top 10% of segments contained `73.40%` of its squared error.
The largest contributors were segments 274, 273, 272, 305, 560, 306, 292 and
561. Regionally, `other` and `apical_trunk` contributed about `65.49%` and
`18.56%` of development error energy; tuft RMSE was only `0.5728 mV`.

The failing development example was a regenerative mixed BAP/calcium pair.
This localization repeats on fit and calibration and is therefore not solely
an idiosyncrasy of development. The error is spatially smooth rather than
high-frequency: the registered high-frequency fraction was only `1.18%` on
development.

The alternative controls were negative. A fit-only segment-affine oracle
improved calibration/development RMSE by only `1.03%/0.90%` and worsened the
maximum errors. No large teacher branch difference occurred in the
near-collision feature tail on fit, calibration or development. Predicted and
teacher branch amplitudes correlated above `0.984` on both diagnostic roles.
Thus the current features identify the paired-future direction, but the shared
decoder lacks the region/mechanism specialization needed to reconstruct
absolute regenerative voltages.

The authorized next experiment is
`05j_f_region_mechanism_expert_revision`. It remains a scoped train-only
diagnostic; 05k and full training are still prohibited.
