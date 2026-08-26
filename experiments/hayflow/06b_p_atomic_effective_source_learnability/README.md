# 06b-p — atomic effective-source learnability

Status: preregistered; not yet executed.

The 06b-o physical decomposition is exact and its zero-source passive Hines
prior improves long rollout, but eleven of twelve Hines seed-arm selections
retained checkpoint zero.  This experiment therefore does not add STATE,
memory, width or a new full-neuron architecture.  It first asks whether the
effective membrane source is learnable at a single teacher boundary.

One synchronized 3x3 matrix crosses raw source units, a global train-fit p99
scale and fixed region-specific train-fit p99 scales with native-source-only,
endpoint-voltage-only and hybrid objectives.  All nine arms use the same
causal numeric input tensor, initialization, seeds and minibatch stream.
Checkpoints at 0/50/100/200/300 steps yield mini scaling laws, while initial
gradient norms distinguish target-scale starvation from ordinary capacity
failure.

Checkpoint selection uses only one-step calibration endpoint RMSE.  Frozen
models are then evaluated on reused train development in two modes: independent
teacher voltage boundaries and an eight-millisecond recursive voltage rollout.
Teacher mechanism STATE remains available in both modes so this experiment
isolates voltage-boundary distribution shift; it does not claim to solve the
STATE recursion problem.

The teacher endpoint is a target, never a model input.  Teacher source is an
unselectable algebraic oracle.  No validation, test or fresh-test state is
read.  A passing arm can authorize only an independent train-support
confirmation of the exact scaling/objective contract.  It cannot authorize
06c, full training or mass dataset generation.

Kaggle entry point:
`notebooks/06b_p_atomic_effective_source_learnability.ipynb`.

The notebook bootstrap pins implementation revision
`8bc229b37583d39b9671df3aa96b8c563906540b` and evicts stale `src.*` modules
before the first project import.

Expected artifact: `hayflow_atomic_effective_source_learnability.zip`.
