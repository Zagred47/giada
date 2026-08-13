# HayFlow 05j-b - repaired representation revision

Status: completed on Kaggle; all six regularized segment-local candidates
failed the unchanged train/development gates. The next scoped experiment is
`05j_c_support_and_decoder_revision`; rollout and full training remain
prohibited.

05j validly established that the repaired 05i-c numerical input contract is
not sufficient for any of the three registered compact shared heads: all nine
runs failed already on train and development branching retention collapsed.
At the same time, an unrestricted train-only local projection interpolated the
small support with very low error but extreme condition numbers. 05j-b is a
controlled attempt to resolve that apparent contradiction.

The exact 12 train pairs and one episode-disjoint development pair are reused.
The exact downloaded 05j artifact and every indexed member must pass SHA-256
verification before any diagnostic is run. The 05i-c normalizer is reconstructed
again from train-only rules and must retain the verified fingerprint. No
held-out input, voltage target or event label is read.

Two hypotheses are separated:

1. the registered `tanh(z/4)` feature map saturates the repaired but relatively
   large synaptic/H2 tails and destroys distance information;
2. the shared compact MLP and eight-dimensional segment embedding impose the
   wrong decoder sharing, even if the frozen representation contains the needed
   information.

For the first hypothesis, 05j-b compares the unchanged `tanh` control with
`asinh(z)/asinh(8)`, a finite monotone transform that compresses tails without
collapsing their order. For the second, it uses an affine decoder independently
for each of the 642 segments. This decoder is deliberately diagnostic rather
than a proposed final architecture.

The voltage residual is represented through an invertible 120 mV bounded
target coordinate. Ridge regularization is selected separately for every input
family/transform candidate using 12-fold leave-one-pair-out on train. The
development pair does not select the ridge value and is evaluated only after
selection. The ridge fit contains both pointwise rows and paired-future
difference rows with the same unit branching weight used by the 05j head loss.
The same pair RMSE, maximum-error and branching-retention gates are
retained, together with explicit condition-number and coefficient-norm gates.

A pass can authorize only the already scoped 05k repaired-representation
micro-rollout. It does not authorize full training. A failure routes to a
separate 05j-c support/decoder revision and does not invalidate the teacher,
dataset or successful 05i-c input-domain repair.

Expected artifact: `hayflow_hines_repaired_representation_revision.zip`.

## Registered result

The completed run used code revision
`e8d0a43d73beb9c0379bd85e9297723db5775014`. The downloaded archive and all
21 indexed artifacts passed independent SHA-256 and size verification. The
05i-c normalizer fingerprint matched exactly, the 12 train pairs and single
development pair remained episode-disjoint, and leave-one-pair-out selected
regularization without using development. Held-out data remained sealed.

No candidate passed even the complete train-pair gate. The best candidate was
H2 with the tail-preserving `asinh` transform: train RMSE `9.4567 mV`,
development RMSE `12.9842 mV`, development maximum segment error `52.6778 mV`
and branching retention `0.4971`. The corresponding `tanh` control reached
`9.4938/13.0637 mV` train/development and retention `0.4954`. The difference
is too small to identify saturation as the primary blocker. Causal-only and
combined candidates were worse, with development RMSE `16.90--17.27 mV` and
retention `0.237--0.250`.

All selected ridge systems passed the registered numerical-stability gates.
The failure is therefore not optimizer divergence or an exploding affine
solution. Geometry is instead the important clue: the median local H2 distance
between paired counterfactual futures was only `2.17e-7` under `tanh` and
`2.80e-7` under `asinh`; the causal-only local distance was `7.63e-8`, and its
median per-segment design rank was only three. A segment-local decoder cannot
recover the teacher's large non-local change from such weak local differences.

This localizes the next question to support and spatial/topological decoder
structure. It does not invalidate the teacher, the dataset or the repaired
05i-c numerical input domain. 05k is not authorized.
