# HayFlow 06b-f - synchronized recursive joint repair

The frozen 06b-e matrix identified two coupled failures: predicted mechanism
STATE is out of its training exposure, while recursively predicted voltage is
unstable and can compensate STATE error with biologically invalid trajectories.
This experiment repairs both boundaries together; it does not promote a new
full architecture.

Six arms restore the same 06b-d `joint_cosine` bridge and the same 06b
`linear_endpoint_path` STATE updater within each seed. They receive identical
train-derived windows, realized inputs, targets, optimizer settings and fixed
budgets. Teacher/predicted STATE and voltage boundaries are crossed, together
with a full-feedback scalar objective, a preregistered voltage-protected
gradient route, and a shuffled causal control.

The voltage-protected arm is primary. Its STATE loss trains the STATE updater
through the four-ms unroll, but gradients from that loss are removed from the
voltage bridge; the bridge is updated only by voltage accuracy, physical-range
and drift terms. This directly tests whether the compensation pathology found
in 06b-e can be prevented without removing recurrent STATE learning.

Checkpoints at 0, 200, 400 and 600 steps form a mini scaling law from each
single training trajectory. Final evaluation uses common nested 1/2/4/8 ms
train-derived development windows. A GO requires joint STATE and voltage
improvement, retained one-step quality, continued scaling, authentic-over-
shuffled specificity and physical safety in every seed. Even a pass authorizes
only a small 06c coupled canary.

Notebook: `notebooks/06b_f_recursive_joint_repair_matrix.ipynb`.
