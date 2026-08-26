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

## Registered result

The independently verified run completed all sixteen aligned arms and selected
no candidate.  The best arm was the neutral-prior, standard-objective,
pre-mixed, generic-STATE control; it remained 2.63% worse than the frozen
static expert at eight milliseconds.  Active and dendritic gains survived,
but quiescent, AIS and axonal coordinates remained strongly harmful.

Carry-first initialization greatly reduced quiescent intervention but removed
too much useful active correction.  Persistence regret modestly improved the
same protected regimes but did not improve the global paired score.  Endpoint
mixing had a near-zero effect.  The learned relaxation updater used semantic
mechanism identities, as shown by its shuffled-label control, yet supplied
almost no recursive STATE gain and degraded the coupled rollout.

The teacher-microtrace upper bound did not outperform the causal linear path,
so missing intra-millisecond voltage resolution is not identified.  Exact
`cnexp` replay remains ineligible because the current transition contract does
not expose every kinetic parameter and there is no independently verified
Python implementation of all rate functions.

The formal diagnosis is
`OBJECTIVE_COUPLING_AND_RELAXATION_DO_NOT_CLOSE_ROLLOUT_GAP`.  No fresh
confirmation, 06c run, full training, validation/test access or mass data
generation is authorized.  The next admissible direction is a revision of the
voltage expert family or the causal STATE contract, not another larger gate or
generic recurrent controller.
