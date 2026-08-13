# HayFlow 05j-b - repaired representation revision

Status: implementation ready; execution pending on Kaggle.

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
