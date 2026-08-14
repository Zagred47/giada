# HayFlow 05j-i - regenerative confirmation support

Status: completed on Kaggle; valid independent-support acquisition.

The exact artifact is registered in `result.json`. All 166 indexed members
passed independent size and SHA-256 verification. The exhaustive teacher replay
reproduced all 576 transitions with a maximum error of
`3.8146719560927522e-06`, below the registered `1e-5` tolerance.

All 24 preregistered snapshot-matched pairs and all 48 episodes were retained.
Twenty-three pairs realized the missing near-regenerative band and one remained
subthreshold, comfortably exceeding the preregistered minimum of 18. The
acquisition therefore resolves the support blocker from 05j-h and routes to
05j-j, where the unchanged frozen diagnostic is evaluated on this shard.

05j-h found 48 regenerative and 24 subthreshold train branching pairs but no
pair whose one-step target peak lies in the pre-registered intermediate band
`[-45, -20) mV`. Therefore its 8.35% aggregate aligned-oracle advantage cannot
decide the regenerative-state hypothesis against the 15% specificity gate.

This notebook creates a separate, validation-only teacher shard rather than
changing the gate or training another decoder. Candidate dendritic schedules
come only from the already completed biological pilot in the immutable base
dataset. A short pilot with a seed namespace disjoint from both base and new
acquisition evaluates every historical canonical dendritic schedule and
identifies schedules that put a causal one-step
branch inside the missing voltage band. No NetCon weight is rescaled: low and
high arms differ only by a deterministic subset of authentic synaptic events.

The acquisition plan is persisted before any new support outcome is observed.
It contains 24 snapshot-matched causal pairs (48 episodes), and every registered
episode is retained regardless of its realized voltage stratum. Eighteen
near-regenerative pairs are required for scientific sufficiency. Falling below
that number is a valid, downloadable acquisition result and routes to a second
adaptive acquisition; it never permits post-hoc cherry-picking.

Every transition stores the full v1.1 teacher state and causal release views.
The complete shard is structurally checked and exhaustively replayed. This
notebook performs no model training, no rollout and no held-out reveal.

Expected artifact: `hayflow_regenerative_confirmation_support.zip`.

Required Kaggle inputs are the complete targeted v1.1 base dataset, the 01b
dataset `alessandrobelli/hayflow-dendritic-protocol-calibration`, and the exact
05j-h artifact. The notebook accepts the 01b calibration either as an extracted
Kaggle directory or as a ZIP containing `selected_dendritic_protocols.json`.
