# HayFlow 05j - repaired representation train/development recheck

Status: completed on Kaggle; the repaired compact-head representation recheck
failed its robust train/development gate. Full training and rollout remain
prohibited. The next scoped experiment is
`05j_b_repaired_representation_revision`.

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

## Registered result

The completed run used code revision
`4cd8ff005647ca1c4159c030cbd69b16bc0287d0`. The downloaded archive and all
38 indexed members passed independent SHA-256 verification. The exact repaired
normalizer fingerprint matched 05i-c, the 12 train pairs and one development
pair were episode-disjoint, and normalization remained train-only. No held-out
input or target was accessed.

None of the nine runs passed the pairwise gates on train or development. The
median train/development RMSE values were respectively `10.8126/15.6795 mV`
for frozen H2, `11.3745/16.8328 mV` for causal-only, and
`10.8483/15.6233 mV` for H2 plus causal input. Development branching retention
was only `0.2960--0.4049`, far below the required `0.9--1.1`, while maximum
segment errors remained `69.20--76.89 mV`. Every family therefore obtained
zero joint passing seeds instead of the required two of three.

The unrestricted train-only linear projection reached at most `0.0636 mV`
RMSE per segment, showing that the tiny train support is algebraically
interpolable from frozen-H2 features. Its design matrices were nevertheless
severely ill-conditioned, reaching condition number `4.60e9`, with very large
coefficient norms. It is therefore a diagnostic clue, not evidence of a
stable or deployable representation, and it cannot override the failed
pre-registered head gates.

This failure does not revoke the valid 05i-c numerical input contract. It says
more narrowly that the current bounded compact heads cannot learn that repaired
surface robustly, even on the registered train pairs. The next authorized path
is the separate `05j_b_repaired_representation_revision`; 05k micro-rollout,
held-out evaluation and full training are not authorized.
