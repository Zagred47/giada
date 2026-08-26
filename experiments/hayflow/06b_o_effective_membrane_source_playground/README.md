# 06b-o — effective membrane-source playground

Status: preregistered; not yet executed.

The terminal 06b-n result rejected gate bias, persistence-regret weighting,
endpoint composition, a generic relaxation updater and missing microtrace
resolution as sufficient repairs.  It authorized a revision of the voltage
expert family or causal STATE contract.  This notebook tests both possibilities
in one aligned train-only run.

The primary architectural intervention predicts a dimensionless effective
membrane source and applies it through the fixed canonical one-millisecond
Hines cable solve.  Its parameter-matched control predicts voltage delta
directly from exactly the same numeric input tensor.  The 2x2x2 matrix crosses:

- direct voltage delta versus effective Hines RHS source;
- frozen initial STATE versus recursively predicted mechanism STATE;
- instantaneous local operator versus local recurrent memory.

All eight arms share seeds, initialization, minibatches, endpoint/native loss
weights and parameter allocation.  Progressive calibration checkpoints produce
mini scaling trajectories.  Development is read once after checkpoint
selection.

Before training, an algebraic teacher-source oracle must reconstruct the true
endpoint through Hines within `1e-4 mV`.  After training, every frozen arm is
also evaluated with teacher STATE refresh, relabelled topology, axial coupling
removed and a deterministic spatial-output shuffle.  These are diagnostic
counterfactuals and are not selection-eligible.

The blocking algebraic identity is evaluated in float64 to avoid cancellation
between the large cable-equation mass/axial terms and the small one-millisecond
voltage increment.  The corresponding float32 reconstruction error is still
recorded separately as an operational numerical diagnostic; it is not hidden
and does not become a model-selection signal.

Metric strata with zero supporting coordinates are encoded explicitly with
`coordinate_count: 0` and JSON `null` values.  They are never represented as
NaN, imputed as performance measurements or allowed to pass a safety gate.

Only historical train-derived roles are read.  Validation, test and sealed
fresh-test states remain inaccessible.  The experiment cannot authorize full
training, mass generation or 06c.  A positive result authorizes only an
independent train-support confirmation of the exact source-operator contract.

Kaggle entry point:
`notebooks/06b_o_effective_membrane_source_playground.ipynb`.

The bootstrap defaults to code revision
`abe09f40a737f5df183bd2c3801c3beefe02c323`, verifies that the 06b-o module is
physically present in the checkout and evicts stale `src.*` modules before the
first project import. `HAYFLOW_ELM_REF` remains an explicit expert override.

Expected artifact: `hayflow_effective_membrane_source_playground.zip`.

## Registered result

The independently verified archive is structurally complete: all 31 indexed
members match their recorded sizes and SHA-256 digests, and every strict JSON
report parses.  The exact cable-equation source identity reconstructs the
teacher endpoint to `2.05e-12 mV` in float64.  The corresponding float32 error
is `1.27e-3 mV`, confirming why numerical precision had to be separated from
the algebraic audit.

The run does **not** establish a learned effective-source operator.  Eleven of
the twelve Hines seed-arm selections chose checkpoint zero.  All four Hines
median mini-scaling curves became worse between steps 0 and 400, by 31% to
206%.  Consequently, the reported 9.88% source-parameterization main effect is
mostly the advantage of the zero-output passive Hines prior, not an effect of
neural learning.  The four median Hines summaries are identical for the same
reason, and their spatial-output shuffle has no effect because there is no
learned output to shuffle.

That passive physical prior is nevertheless informative.  It reaches
`7.43 mV` median RMSE at 8 ms, improves active and regenerative coordinates,
has no physical-voltage violations and degrades by 31.9% under relabelled
morphology.  It is worse at 1 ms than the learned direct-voltage control and
remains severely harmful in quiescent and moderate regimes.  Authentic
morphology therefore helps long-rollout regularization, but the source learner
does not yet exploit it safely.

The formal registered diagnosis is
`PASSIVE_HINES_PRIOR_HELPS_LONG_ROLLOUT_BUT_EFFECTIVE_SOURCE_IS_NOT_LEARNED`.
No candidate, independent confirmation, validation/test access, 06c run, full
training or mass generation is authorized.  The next experiment must first
isolate one-step source learnability, target scaling, loss alignment and the
teacher-boundary versus recursive-boundary gap before coupling this operator
to another STATE or memory revision.
