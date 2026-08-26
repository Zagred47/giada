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
