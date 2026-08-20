# HayFlow 05o - nonlinear observability reassessment

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `a51fbdab2b7d14c33c112c17c7cc2f50a26cb86e67f6aa2c655cc52d22e29477`;
- artifact-index SHA-256: `e8f83a57ba51ef6db5a6fdd4a10c393a934fda493df071d898e0fe299d622ea3`;
- final-report SHA-256: `80842fe8e56f25a6c8735da06f5cec860cb22d924f4798ca9a973ced058fa984`;
- all sixteen indexed members passed size and digest verification.

Both nonlinear information signals passed in all three paired seeds.  The
axial block improved median development RMSE by 18.04% given voltage and
30.68% given rich state; its regenerative gains were 7.17% and 18.96%.
The rich initial-state block improved RMSE by 30.11% given local features and
39.37% given the axial block, with regenerative gains of 29.33% and 37.23%.
The registered diagnosis is `NONLINEAR_JOINT_GRAPH_AND_STATE_SIGNAL`, which
authorizes only the 05p paired closed-loop micro-canary.

05n found no material *linear* information gain from either authentic axial
features or a 16-dimensional sketch of the complete initial teacher state.
That result does not establish that the information is absent: the useful
coordinates may interact nonlinearly.  05o is the registered bounded test of
that remaining explanation.

The experiment fits four paired one-step probes on the same train-derived
fit/calibration/development roles:

- local voltage, causal `U_realized` and static segment features;
- the local contract plus authentic axial parent/child features;
- the local contract plus a 64-dimensional semantic initial-state sketch;
- the joint axial-graph and rich-state contract.

Every probe has the same fixed input width, hidden width, residual blocks and
parameter count.  Missing graph or state blocks are zero-filled, and each
contract receives the same initialization for a given seed.  Three registered
seeds are trained with regenerative rows upweighted and ordinary rows sampled
at a fixed ratio.  Calibration selects the checkpoint; development is read
once after selection.  Each matched comparison supporting a signal requires
at least 5% median RMSE gain, regenerative non-inferiority and positive wins
in at least two seeds.

This is a nonlinear one-step information audit, not a recurrent architecture
test.  It does not load validation, test or fresh-test inputs, and cannot
authorize full training, fresh-test generation or mass data generation.  Its
only output is which minimal 05p diagnostic is justified next.

Expected artifact: `hayflow_nonlinear_observability_reassessment.zip`.
