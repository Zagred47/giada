# 06b-p — adaptive atomic effective-source forensics

Status: preregistered; not yet executed.

Historical note: the original stage-1-only revision (`8bc229b`) has now been
executed and immutably registered in `result_v1.json`. It found calibration
movement but no development source identification or recursive benefit, so
scaling/objective repair alone is closed as an explanation. The adaptive v2
protocol described below has not yet been executed.

The 06b-o physical decomposition is exact and its zero-source passive Hines
prior improves long rollout, but eleven of twelve Hines seed-arm selections
retained checkpoint zero.  This experiment therefore does not add STATE,
memory, width or a new full-neuron architecture.  It first asks whether the
effective membrane source is learnable at a single teacher boundary.

Stage 1 preserves the synchronized 3x3 matrix crossing raw source units, a global train-fit p99
scale and fixed region-specific train-fit p99 scales with native-source-only,
endpoint-voltage-only and hybrid objectives.  All nine arms use the same
causal numeric input tensor, initialization, seeds and minibatch stream.
Checkpoints at 0/50/100/200/300 steps yield mini scaling laws, while initial
gradient norms distinguish target-scale starvation from ordinary capacity
failure.  It selects only scaling and objective on deterministic calibration
half A.

Stage 2 then runs a preregistered adaptive 3x2 fragment matrix.  Three nested,
fixed-width causal input contracts compare compact event moments, exact
receptor-resolved events, and exact events plus authentic synaptic A/B state
already present at the boundary.  Two physical targets compare the net cable
source with an intrinsic residual after authentic synaptic conductance and
source have been moved into the fixed Hines operator.  Every arm within this
stage has the same parameter count, initialization and minibatch stream;
checkpoint selection uses the disjoint calibration half B.

Before training, nonselective audits quantify boundary-tail materiality, the
known-synaptic Hines control, source cancellation, and algebraic source
conditioning at 1/0.5/0.25 ms.  The substep audit cannot select a candidate:
stored microtraces lack complete intermediate mechanism and synapse state.
Initialization gradient cosines across quiet/moderate/active regimes and
frozen voltage-sensitivity probes make optimization conflicts observable.

Checkpoint selection uses only the assigned one-step calibration partition. Frozen
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

The notebook bootstrap pins the implementation revision recorded in its setup
cell and evicts stale `src.*` modules before the first project import.  Code,
configuration and notebook must always be used from that same revision.

Expected artifact: `hayflow_atomic_effective_source_learnability.zip`.
