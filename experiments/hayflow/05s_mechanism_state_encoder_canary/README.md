# HayFlow 05s - mechanism-state encoder canary

Status: implemented; independent Kaggle execution pending.

05r localized the useful temporal boundary signal to canonical mechanism
states.  This bounded canary tests one representation hypothesis with three
parameter-identical paired arms:

- the verified legacy coordinate-specific signed sketch of all non-voltage
  state;
- a fixed semantic sketch of all non-voltage state, where equal
  `(category, mechanism, variable, kind)` identities use equal signed vectors
  across segments;
- the same semantic sketch restricted to `mechanism_states`.

Every arm uses the same axial rich-state GraphGRU, seeds, initialization,
training order, 30-epoch curriculum and fit/calibration/development roles.
The full semantic arm separates semantic alignment from category restriction;
it is an internal control, not another experiment.  Fit selects gradients,
calibration selects the checkpoint and train-derived development is read once
after freezing.  Validation, tests and sealed fresh-test data remain excluded.

A representation can advance only if it is robust in at least two seeds,
beats the legacy control by at least 2% median 8 ms RMSE, is regeneratively
non-inferior within 1%, wins at least two paired seeds and produces no
non-finite or physically invalid voltages.  A pass authorizes only one
consolidated autoregressive go/no-go; a failure stops the current state-encoder
branch.  Full training, fresh-test generation and mass data remain forbidden.

Expected artifact: `hayflow_mechanism_state_encoder_canary.zip`.
