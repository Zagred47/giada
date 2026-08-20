# HayFlow 05n - graph-operator and state-contract reassessment

Status: implemented; execution pending.

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
