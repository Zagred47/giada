# 06b-n — structure-preserving coupling forensic

This train-only notebook is the registered continuation of the 06b-m
diagnosis `MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS`.

It replaces a sequence of redundant binary notebooks with one aligned run.
Frozen counterfactuals first test whether the learned 06b-m controller becomes
safer when shifted toward carry and whether evolving complete state futures
before mixing differs materially from updating STATE after voltage mixing.

A separate bounded relaxation updater then tests the structure

`x_next = x_inf + (x_t - x_inf) * exp(-rate)`

using causal inputs only.  The canonical teacher `.mod` files are audited for
`METHOD cnexp`, but exact Rush-Larsen replay is not claimed unless all kinetic
parameters and rate functions are independently executable from the dataset.
The authentic teacher microtrace is retained only as a non-selectable upper
bound.

Finally, a synchronized 2x2x2x2 matrix crosses gate initialization, rollout
objective, state-flow composition and STATE updater.  All sixteen arms share
the same seeds, frozen voltage expert and minibatch stream.  Progressive
checkpoints provide optimization trajectories without duplicated runs.

No validation or test state is read.  The notebook cannot authorize 06c,
full training or mass data generation.  Its only possible positive decision
is a fresh train-support confirmation of an exact arm that passes every
absolute safety and rollout gate.

Kaggle entry point:
`notebooks/06b_n_structure_preserving_coupling_forensic.ipynb`.
