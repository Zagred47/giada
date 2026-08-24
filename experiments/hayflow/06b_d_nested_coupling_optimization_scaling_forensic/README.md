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

## Registered result

The returned archive from revision `723953a` passed independent ZIP CRC and
all 88 indexed member-size and SHA-256 checks. The synchronized matrix answers
several questions cleanly. Extending the joint-cosine trajectory from 500 to
1500 steps adds 2.51 percentage points of median voltage gain, so the previous
500-step bridge was still optimization-limited. At 1500 steps, adding the
frozen downstream-STATE objective improves STATE gain by 1.20 points over the
voltage-only arm without degrading voltage; the true causal alignment also
beats the shuffled-path control by 1.59 STATE points. These three registered
gates pass.

Cosine decay itself is not identified: both registered schedule effects remain
below one percentage point. Gradient alignment is also not stationary. Its
cosine changes sign across seeds, budgets and arms, and the initial STATE-loss
scale for seed 61017 hits the preregistered lower clip. A fixed scalarized loss
is therefore an incomplete description of the optimization geometry.

The decisive recursive gate fails. At 8 ms, the preregistered joint-cosine arm
beats voltage-cosine by 1.66 percentage points rather than the required 2%, and
seed 61017 is slightly negative. A constant-schedule contrast has a promising
3.93-point median but reverses sign in seed 61017; because it was not the
registered recursive contrast, it is recorded only as a new hypothesis and
not promoted to a success.

The registered diagnosis is `ONE_STEP_COUPLING_OBJECTIVE_ONLY`. More bridge
optimization and causally aligned downstream supervision improve one-step
learnability, but that gain is not yet robustly compositional. Teacher voltage
and ion context were still supplied at every millisecond, so this result does
not test an autonomous neuron rollout. No full training, fresh-test generation
or mass-data generation is authorized. The next experiment must be a bounded
train-only recursive voltage/STATE contract matrix separating exposure bias,
missing recurrent variables and optimizer-trajectory effects.
