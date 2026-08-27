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

## Registered result

The completed archive is registered in `result.json`; all 36 indexed members
pass independent size and SHA-256 verification.  The decisive result is not
the automatic failure label but the factorial contrast: four-millisecond
pushforward exposure reduces median development 8 ms RMSE from the
support-matched passive value of 18.16 mV to 14.72 mV.  The frozen 06b-q model
was at 46.94 mV on exactly this support.  All three 06b-r development seeds are
non-regressive and produce zero physical-voltage violations.

Promotion nevertheless remains blocked under the preregistration.  Calibration
half B improves for seeds 61017 and 61029 but regresses by 10.08% for seed
61043, so the no-seed-regression gate fails before development is opened.
Development cannot be used retrospectively to overturn that veto.

The other two factors are inert at the registered scale.  The auxiliary adds
only 0.0058% marginal gain and the directional penalty about 0.00004%.
Deleting the extra ordered-event path changes 8 ms error by only 0.0145%, while
deleting all current causal `U` information changes it by 2.17%.  The learned
model therefore uses the compact causal drive materially but not the ordered
event representation.

The auxiliary negative result is qualified by a post-hoc contract issue: its
target included boundary synaptic tails derived from `S_t`, whereas the
auxiliary head saw only the event embedding.  It cannot establish that the
event encoder itself is intrinsically unlearnable.

The registered diagnosis is
`PUSHFORWARD_CAUSALLY_REPAIRS_ROLLOUT_BUT_ORDERED_EVENT_AND_DIRECTIONAL_PATHS_ARE_INERT_AND_CROSS_SEED_CALIBRATION_REMAINS_FRAGILE`.
No candidate is promoted.  The next train-only experiment should keep the
compact causal input and pushforward exposure, remove the inert factors, and
isolate horizon, budget and cross-seed calibration robustness before a sealed
test.
