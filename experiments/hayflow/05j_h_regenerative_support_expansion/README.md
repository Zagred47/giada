# HayFlow 05j-h - regenerative support expansion

Status: implemented; Kaggle execution pending.

05j-g found a large calcium/sodium state-transition oracle signal, but the
aligned oracle did not beat its spatially shifted equal-width control by the
registered 15% margin. This notebook repeats the exact diagnostic on a larger,
independent train-only support instead of changing the model or relaxing the
gate.

Up to 72 episode-disjoint branching pairs are selected from the existing train
split by biological target-peak regime alone: regenerative, near-regenerative
and subthreshold. Model error is not used in support selection. Eighteen pairs
from independent episodes form a reserved confirmation role; the remainder are
used only for grouped-pair ridge cross-validation. At least 56 well-stratified
pairs are required to call the expansion scientifically sufficient. If the
immutable dataset cannot supply them, the notebook completes with an explicit
data-generation route instead of crashing or weakening the gate. The original development
pair stays descriptive, and all held-out inputs remain sealed.

The exact 05j-d topology transformation and frozen three-seed direct-tree
checkpoints are reconstructed and verified before inference on the expanded
support. The causal boundary-state, aligned future-state oracle, spatial-shift
null, intercept control and individual state-group probes are unchanged from
05j-g. Confirmation requires both the aggregate 15% spatial-specificity margin
and aligned wins on at least two thirds of confirmation pairs.

This remains a diagnostic experiment. Future internal state is never a
candidate input, and 05j-h cannot authorize candidate training, rollout or full
training. It only determines whether the next canary should learn an explicit
causal state input, a joint regenerative state transition, a regime-conditioned
objective, or return to voltage-decoder reassessment.

Expected artifact: `hayflow_hines_regenerative_support_expansion.zip`.
