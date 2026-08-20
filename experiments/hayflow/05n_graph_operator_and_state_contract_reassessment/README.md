# HayFlow 05n - graph-operator and state-contract reassessment

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `495c4419d2205c9a3e4c58fb0cf67da534d9ada38e9e28b393ba8dbc97b80316`;
- artifact-index SHA-256: `975c8b0741d8639d19095447d1b06c06795020e478b21a036e5e3aadc60bd627`;
- final-report SHA-256: `2460f61579d972464159dfe60f7d73e02e9ce0c9245138dc750253655db9ab30`;
- all eight indexed members passed size and digest verification.

The best development RMSE was 2.201 mV for voltage/input plus axial features,
versus 2.214 mV for the local contract.  Median axial gain was 0.63% overall
and 3.97% on regenerative rows, below the registered 5% gate.  The rich-state
sketch changed RMSE by -0.07% overall and -0.06% on regenerative rows.  Every
lambda ladder was essentially flat and selected the smallest registered
lambda, so the negative result is not a regularization accident.  Diagnosis:
`NO_LINEAR_GRAPH_OR_STATE_CONTRACT_SIGNAL`; only 05o nonlinear observability
reassessment is authorized.

05m replicated the rollout-aware recurrent signal but rejected a morphology
claim: authentic topology lost to the isomorphic relabelled-tree control in
all three seeds.  This does not distinguish a weak graph operator from partial
observability caused by initialising recurrence from voltage alone.

05n is therefore an information audit, not another architecture search.  Four
fixed ridge probes predict the one-millisecond per-segment voltage delta:

- local voltage, causal `U_realized` and static segment features;
- the same features plus authentic parent/child voltage differences and axial
  conductance-weighted differences;
- local features plus a deterministic 16-dimensional semantic sketch of all
  non-voltage teacher state available at the initial boundary;
- both the axial graph features and rich initial-state sketch.

The class-aware NetCon view, bounded last-event recency, semantic state
transform and robust normalization are fitted only on the fit role.  The
fixed signed projection preserves mechanism/variable identity
without learning an encoder.  Calibration chooses one lambda from a sealed
ladder; development is evaluated once after selection.  The four probes share
the same rows and target.  Regenerative rows are reported separately using
fixed voltage/delta thresholds.

Only the reconstructed 05m train-derived fit/calibration/development roles are
read.  State at `t+1` supplies the prediction target but is never a feature;
validation, test and the fresh sealed test are not loaded.  A 5% incremental
RMSE gain plus regenerative non-inferiority is required to call either axial
graph information or initial-state information material.

This linear one-step diagnostic cannot authorize an architecture, full
training, fresh-test generation or mass dataset generation.  It selects only
the next small recurrent canary or a nonlinear observability reassessment.

Expected artifact: `hayflow_graph_state_contract_reassessment.zip`.
