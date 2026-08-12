# HayFlow 05e experimental record

This directory records the completed closed-form segment-capacity probe.
Generated binary artifacts remain external; `result.json` binds the downloaded
ZIP and all decision-grade members through SHA-256 values.

The shared 97-parameter linear probe failed both the worst transition and the
authentic counterfactual pair. A free bias per segment memorized the single
transition exactly but failed the branch pair, demonstrating that static
segment identity alone cannot explain the causal difference between futures.
Adding shared features to the segment biases also failed.

The segment-conditioned SVD path improved monotonically. Rank 64 nearly met the
pair gates but missed the 5 mV maximum-error threshold; only the maximum tested
rank 96 passed, at numerical precision and with branching retention 1.0. This
shows that the frozen features contain enough information for the registered
pair when coupled to segment-specific coefficients, but it does not establish
a compact representation or out-of-sample generalization. The passing fit has
71,490 parameters and was estimated on only two counterfactual transitions.

The result therefore authorizes only a fresh zero-initialized
segment-conditioned neural micro-canary. Full training remains prohibited.
