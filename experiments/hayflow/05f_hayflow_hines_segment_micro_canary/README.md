# HayFlow 05f experimental record

This directory records the completed segment-conditioned neural micro-canary.
Generated checkpoints remain external; `result.json` binds the downloaded ZIP
and decision-grade members through SHA-256 values.

The pair plan was contract-valid and leakage-free: eight training pairs, two
held-out pairs from two dedicated test splits, no episode overlap, and complete
exclusion of the 05e development episodes. Both rank-64 and rank-96 heads began
with an exactly zero residual and ran for the complete 1,200-epoch budget
without non-finite values.

Neither head fitted the training pairs. Train RMSE remained approximately
15.79 mV and branching retention approximately 0.236. Held-out absolute
predictions diverged catastrophically into the millions of millivolts. This is
not evidence against the entire segment-conditioned architecture family. All
eight training pairs came from the same targeted-BAP protocol, while the local
train-only feature designs had rank only 13--15 for 96 features. Their
unregularized coefficient matrix was full rank but had singular values of
order 1e8--1e9, making the spectral initialization severely underdetermined
and numerically unsafe outside the training regime.

The registered diagnosis remains
`SEGMENT_CONDITIONED_MICRO_CANARY_OPTIMIZATION_FAILURE`. Full training is
prohibited. The next experiment must audit conditioning and pair diversity
before another generalization claim is attempted.
