# HayFlow 05j-c - spatial context and support revision

Status: completed on Kaggle. Expanded support and non-local morphology context
both produced large, consistent improvements, but no candidate met every
original pairwise gate. The result authorizes only
`05j_d_trainable_topology_decoder_micro_canary`.

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

## Registered result

The completed run used code revision
`f42f366bfda5f9ba2614c1cb14349d8055a68228`. The downloaded archive and all
22 indexed members passed independent SHA-256 and size verification. The exact
05i-c normalizer fingerprint was retained. The expanded support contained the
original 12 pairs plus 36 deterministic additions from 776 eligible train
candidates, covered all six protocol families, remained episode-disjoint and
had no development overlap. Held-out data remained sealed.

Support expansion and topology were both material on train cross-validation
and development. Relative to the preregistered factorial controls, expanded
support improved RMSE by `58.87%` on cross-validation and `64.46%` on
development; non-local context improved it by `58.57%` and `84.89%`. Both
effects exceed the fixed 20% threshold on both roles.

The strongest balanced candidate was expanded support with multiscale tree
context. It reached cross-validation RMSE `2.3932 mV`, train RMSE `1.4017 mV`
and development RMSE `2.6936 mV`. Development branching retention was
`0.9739`, now inside the required `0.9--1.1` interval. However, development
maximum segment error remained `16.9684 mV`, and only 14/48 cross-validation
pairs and 23/48 train-fit pairs passed all gates jointly. Thus the result is
not a representation pass.

Adding global region broadcasts slightly improved development RMSE and maximum
error (`2.6397 mV`, `14.3959 mV`) but degraded cross-validation branching and
overall robustness. Fixed multiscale tree context is therefore the cleaner
candidate for the next test.

The result establishes that local features were missing a necessary spatial
information path and that 12 pairs were insufficient. It does not yet establish
the required voltage accuracy. The next experiment may train only a small
topology-aware decoder micro-canary on the 48-pair support; rollout and full
training remain prohibited.
