# HayFlow 05o - nonlinear observability reassessment

Status: pending independent Kaggle execution.

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
