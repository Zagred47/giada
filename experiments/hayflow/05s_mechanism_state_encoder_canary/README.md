# HayFlow 05s - mechanism-state encoder canary

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `cd879a8cf8fe25b42e9b139544424d7674144b983d3de126a34a882c4a1fd090`;
- artifact-index SHA-256: `eac72ae71568dec45f3e454d6729733b8ebb0afbc38d3b6d447d980b2fb5ccda`;
- final-report SHA-256: `500cdd8a7920d47412b727af915bc44d2392a5f674397aed3f06037a546068bb`;
- all fifteen indexed members passed size and digest verification.

All arms passed in all three seeds.  Relative to the legacy sketch, semantic
full state improved median 8 ms RMSE by 13.04% and regenerative RMSE by
23.00%; semantic mechanism state improved them by 13.79% and 15.03%.
Mechanism-only did not improve ordinary RMSE over semantic full state
(`-0.24%` median; one of three wins), although regenerative RMSE improved by
3.36%.  The scientific conclusion is therefore that semantic alignment is
robust while category restriction remains unresolved.  05t must carry both
frozen semantic candidates into one validation-only autoregressive go/no-go;
the automatic `selected_representation` field is not treated as final model
selection.

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
