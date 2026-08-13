# HayFlow 05j-c - spatial context and support revision

Status: implementation ready; execution pending on Kaggle.

05j-b ruled out simple feature-tail saturation and ordinary numerical
instability. Stable segment-local ridge decoders still failed, while paired
counterfactual local feature distances were extremely small relative to the
teacher response. 05j-c tests whether the remaining blocker is lack of
non-local morphology context, insufficient train support, or both.

The exact 05j-b artifact and all indexed members are verified first. The
original 12 train pairs are preserved verbatim. The teacher dataset contains
776 eligible train counterfactual candidates across six protocol families, so
the support is expanded deterministically to 48 episode-disjoint train pairs:
the original 12 plus 36 round-robin additions. Development episodes are
excluded. No held-out input or target is read.

H2 and the teacher encoder remain frozen. H2 and causal channels are compressed
to deterministic train-only 16-dimensional PCA sketches. Three fixed contexts
are compared:

1. local H2 plus causal sketches;
2. the same features after symmetric axial-neighbour diffusion sampled at
   0, 1, 2, 4, 8, 16 and 32 tree steps;
3. multiscale tree features plus causal summaries pooled by morphology region
   and broadcast to every segment.

These three contexts are crossed with the original and expanded support,
yielding six candidates. Each uses the same segment-specific bounded ridge
decoder and branching-weighted fit as 05j-b. Nine ridge values are selected by
six-fold grouped pair cross-validation on train. Development is evaluated only
after selection. The original pair RMSE, maximum-error, branching-retention,
condition-number and coefficient-norm gates remain unchanged.

The factorial design distinguishes topology and support effects with a
predeclared 20% material-improvement threshold that must be reached on both
train cross-validation and development; a one-role gain is inconclusive.
Passing can authorize only a
separate 05k micro-rollout. A partial topology benefit can authorize only a
trainable topology-decoder micro-canary. Full training is never authorized by
this notebook.

Expected artifact: `hayflow_hines_spatial_support_revision.zip`.
