# HayFlow 05j - repaired representation train/development recheck

Status: implementation ready; execution pending on Kaggle.

05i-c established a complete input-contract pass. 05j now asks the narrower
learning question that was previously confounded by the broken teacher-state
representation: can a compact bounded residual head learn the registered
counterfactual train support and generalize to the separate development pair?

The exact 12-pair train support and one development pair from 05g are reused.
Their episodes must be disjoint. The 05i-c normalizer is reconstructed from
train-only quantities and must reproduce the exact verified 05i-c fingerprint.
No held-out input is extracted anywhere in 05j, and no held-out voltage or
event target is materialized.

H2 remains frozen. Three compact diagnostic head families consume H2, causal
input or H2 plus causal input, using the original three seeds and original
pairwise gates. Heads train only on train; checkpoint selection and evaluation
use development. A family passes the robust gate only if at least two of its
three seeds pass both train and development. Projection remains a train-only
diagnostic oracle.

Even a pass cannot authorize full training. It can authorize only a separate
05k repaired-representation micro-rollout. A failure remains a scoped head or
representation result and does not invalidate the successfully repaired 05i-c
input contract.

Expected artifact: `hayflow_hines_repaired_representation_recheck.zip`.
