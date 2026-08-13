# HayFlow 05j-f - region/mechanism expert revision

Status: implemented; Kaggle execution pending.

05j-e showed that the direct-tree decoder preserves paired-future separation
but concentrates absolute-voltage error in regenerative trunk and adjacent
compartments. This experiment tests whether explicit biological specialization
repairs that localized error, while controlling exactly for added capacity.

For each seed, the registered 05j-d direct-tree checkpoint is reconstructed and
frozen. A zero-initialized correction is trained around that prediction. Both
candidate families contain the same eight expert networks and therefore the
same trainable parameter count:

1. `uniform_expert_control` averages all experts uniformly at every segment;
2. `region_mechanism_experts` gates the same networks using fixed metadata-only
   membership in general, apical trunk, basal, tuft, soma/axon, calcium
   regenerative, sodium regenerative, and repolarization/Ih groups.

The masks use only the canonical segment regions and mechanism names. They do
not use the 05j-e error ranking, fit targets, calibration targets or development
values. The general expert covers every segment and overlapping masks are
normalized to sum to one.

The original 36 fit, 12 calibration and one development pair contract is
unchanged. The loss and bounded residual parameterization remain aligned with
05j-d. Calibration alone selects checkpoints. Development is evaluated after
freeze. A robust absolute pass requires at least two of three seeds to meet all
original gates. A weaker expert-specific signal requires at least two seeds to
improve both calibration and development RMSE by 15% and maximum error by 10%
relative to their own frozen baseline, plus at least 10% median RMSE advantage
over the capacity-matched uniform control on both roles.

Only an absolute robust pass can authorize 05k. Other outcomes route to a
scoped regenerative expert, objective/capacity, or state-target decomposition
experiment. Full training is never authorized here. Held-out data and rollout
remain sealed.

Expected artifact: `hayflow_hines_region_mechanism_expert_revision.zip`.
