# HayFlow 05j-d - trainable topology decoder micro-canary

Status: implemented; Kaggle execution pending.

05j-c established that expanded train support and multiscale morphology context
are both necessary, role-consistent improvements, but its fixed ridge decoder
did not meet the original voltage and branching gates. This experiment tests
only the next authorized hypothesis: whether a small nonlinear topology-aware
decoder can turn that verified signal into accurate counterfactual transitions.

The exact 05j-c archive and every indexed member are verified first. Its 48
episode-disjoint train pairs are rebuilt deterministically and partitioned by
protocol family into 36 fit pairs and 12 internal-calibration pairs. Every one
of the six families is represented on both sides and episodes cannot overlap.
The already verified 05i-c synaptic normalizer remains frozen; every new
channel sketch, PCA component and topology-design normalizer is fit on the 36
fit pairs only. Development cannot select transforms, ridge strength, model
family, seed or checkpoint.

The fixed multiscale axial-tree representation from 05j-c is retained. A
fit-only grouped-CV ridge model is frozen as the reference. Two small shared
decoders are then trained with seeds 17, 29 and 43:

1. `direct_tree`, which predicts a bounded voltage residual directly from the
   tree features and a learned segment embedding;
2. `ridge_corrected_tree`, which starts exactly at the frozen ridge prediction
   and learns a bounded nonlinear correction in logit space.

The loss combines pointwise voltage error, paired-future difference error and
a worst-segment tail term. Checkpoints are selected only on the 12 internal
calibration pairs. Development is evaluated once after the chosen checkpoint
has been frozen. A family passes robustly only if at least two of three seeds
meet the unchanged per-pair RMSE, maximum-error and branching-retention gates
on fit, calibration and development.

A robust pass can authorize only the separate 05k micro-rollout. A material
but incomplete improvement over the fixed ridge reference routes to a decoder
refinement; otherwise the architecture must be reassessed. Held-out inputs,
rollout and full training remain prohibited in every outcome.

Expected artifact:
`hayflow_hines_trainable_topology_decoder_micro_canary.zip`.
