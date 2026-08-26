# 06b-r — Recursive event-exposure playground

This is the preregistered follow-up to 06b-q.  It preserves the exact passive
Hines solve and the selected chronological, nonzero-robust event
representation.  It does not change morphology, reveal validation/test state,
or generate new biological outcomes.

The notebook first verifies that a causal synaptic auxiliary target is
learnable relative to a parameter-matched permuted-target control.  It then
trains one paired 2x2x2 matrix crossing causal auxiliary supervision,
four-millisecond pushforward exposure, and passive-relative directional
stability.  All arms share initialization, windows, minibatches and budgets.

Checkpoint selection and arm selection use disjoint halves of the calibration
role.  The primary gate is support-matched 8 ms recursive RMSE: an arm cannot
be promoted if it regresses against the passive Hines prior in any seed.  The
development role is opened only after selection.  The old 06b-q checkpoint is
replayed on the same support as a frozen historical reference.

See `research_synthesis.md` for the primary-source evidence and its mapping to
the factorial design.

