# HayFlow 06b-d - nested coupling and optimization scaling forensic

This train-only component experiment follows the registered 06b-c diagnosis
that the existing local voltage bridge retains an optimization signal but its
global voltage improvement is seed-dependent. Rather than splitting the next
questions across several notebooks, 06b-d runs one synchronized factorial
matrix.

Five arms cross voltage-only versus downstream-STATE-aware objectives with
constant versus cosine learning-rate schedules, plus a shuffled voltage-path
negative control. Every arm starts from the same frozen 06b-b bridge within a
seed and receives the same minibatches in the same order. Checkpoints at 0,
250, 500, 1000 and 1500 steps produce a mini scaling law without retraining
the same trajectory for each budget.

The 7,212-parameter mechanism-STATE updater remains frozen. In joint arms its
loss is differentiable with respect to the predicted voltage path, so the
gradient updates only the 8,985-parameter bridge. A fit-only initial gradient
norm ratio fixes the relative STATE-loss scale before training. Gradient
cosines are probed to determine whether the voltage and STATE objectives are
aligned or conflicting.

Each joint minibatch samples 1,024 mechanism coordinates. This preserves a
broad stochastic STATE objective while avoiding the cost of evaluating all
4,074 coordinates at every optimizer step; fixed-budget evaluations still use
the complete diagnostic state surface.

The experiment reports voltage and STATE metrics at every fixed budget, plus
nested mechanism-STATE recurrence at 1, 2, 4 and 8 ms for the final models.
Teacher voltage remains the boundary condition at each millisecond, so this
is not an autonomous voltage rollout. Development is a disjoint train-derived
diagnostic role and cannot select a deployable candidate.

Notebook:
`notebooks/06b_d_nested_coupling_optimization_scaling_forensic.ipynb`.
