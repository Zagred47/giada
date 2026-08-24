# HayFlow 06b-g - STATE scheduled-sampling confirmation

The preregistered 06b-f primary arm failed, but its scalar full-feedback arm
passed every gate except fixed-budget scaling. This experiment treats that arm
as a new hypothesis rather than retroactively declaring it the winner.

First, the exact scalar and teacher-forced 600-step checkpoints are evaluated
frozen on train components that were not used in any previous fit, calibration
or development role. The confirmation role is selected structurally as the
next connected component in every regime, not by its outcomes.

Second, five synchronized continuations start from the same scalar checkpoint.
They compare plain full feedback, an eight-ms STATE curriculum, a joint
STATE/voltage curriculum, fixed 25% STATE mixing and a shuffled causal control.
All arms share minibatches, stochastic teacher-forcing draws, optimizer, loss,
parameter count and checkpoints at 0/100/200/400 steps.

The 06b-f artifact did not store optimizer state. Therefore these are weight
continuations with an explicitly registered, identical AdamW restart in every
arm, not uninterrupted optimizer trajectories.

The STATE-linear curriculum is primary. Plain scalar continuation is a
preregistered hierarchical fallback, not a candidate selected after viewing
the results. A pass can authorize only a small 06c canary. Validation/test
access, fresh-test generation, full training and mass data remain prohibited.

Notebook: `notebooks/06b_g_state_scheduled_sampling_confirmation.ipynb`.
