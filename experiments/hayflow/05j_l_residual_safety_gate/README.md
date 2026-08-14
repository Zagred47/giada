# HayFlow 05j-l - frozen residual safety gate

Status: implemented; Kaggle execution pending.

05j-k showed that the frozen H2 core remains accurate on the independent
near-regenerative support, while the direct-tree residual correction causes a
roughly tenfold degradation. This notebook asks whether a conservative,
non-trainable safety layer can retain the direct-tree benefit in-distribution
and fall back toward H2 when its correction or ensemble disagreement leaves
the original fit envelope.

Segment clipping, ensemble-uncertainty fallback, sample-energy scaling and
their combinations are compared. Every threshold and the family/quantile
choice use grouped-pair cross-validation exclusively on the original 05j-h fit
pairs. Candidates within two percent of the best fit-only score are resolved
by a preregistered conservative priority. The already observed 05j-i support is
evaluated only afterward and is explicitly descriptive.

This is a post-result canary and cannot authorize a model. Even a large rescue
must be replicated on a newly generated, untouched near-regenerative test
shard before rollout or candidate authorization.

Expected artifact: `hayflow_hines_residual_safety_gate.zip`.
