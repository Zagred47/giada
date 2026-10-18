# Task39e — Adam lifetime failure and numerical evaluator repair

## Preserved evidence

All30 members have finite weights at0/3k/10k/30k. At60k,17/30 are nonfinite:
long_000 seeds17/29, all12 short split-head members, all3 long_111 members.
Every parameter of each affected member is nonfinite. Remaining finite weight
maxima stay below10, not a global growing-weight blow-up. Complete per-parameter
counts and per-member/step records: results/task39e_instability_v1/checkpoint_audit.json.
The preserved failure report identifies the evaluator dereferencing `gate=None`.

## Reproduced implementation cause

`task39d_current_timing.capture_training` allocated Adam moments during eager
warmup on a side stream, then returned only the CUDA Graph. The optimizer list
was local. Captured kernels retain device addresses, not Python ownership of
the eager-allocated state tensors. On function return the states could be freed
and reused during subsequent graph construction/probes/evaluation.

On the same T4, a paired causal probe changes ONLY lifetime ownership:

- Without ownership: weak references to all3 Adam state tensors are dead;
  allocator stress on the allocation stream reuses a state allocation; replay
  produces nonfinite parameters.
- With ownership: all state tensors remain alive; no state allocation is reused;
  replay remains finite under the identical stress.

Fix: attach optimizer owners to the returned graph for its full lifetime.
No LR/loss/clip/Adam hyperparameter changes. This reproduces an implementation
corruption mechanism consistent with the run; the exact first failing old
minibatch cannot be reconstructed because old Adam/RNG states were not saved.
The shared capture helper was also used in39d: preserve that provenance caveat;
do not retroactively claim its optimizer trajectory was immune, or erase its
observed evaluation results. Reassess optimization-specific interpretations.

## Evaluation and recovery

Nonfinite predictions are failed cases, never eligible ranks. Healthy members
continue. JSON diagnostic missing values are null, never fake finite scores.
Earlier finite checkpoints remain selectable under the unchanged original rule;
any arm with observed numerical invalidity is explicitly blocked from promotion.
The failure is disclosed, not hidden by selecting an earlier checkpoint.

Checkpoints now also retain Adam state dictionaries, RNG state, step and clipping
counts for forensics. Existing PC-side SHA256/CRC-verified backups remain active.
Retry2 restarts the exact registered initialization and data stream: no claim of
exact resume from the old incomplete state. Fixed180k budget and all scientific
thresholds, arms, data, seeds, selection and abstraction remain unchanged.

## Required validation

CPU regression injects NaN into one member and checks failed/nonselectable status
with other29 unaffected. Full original CPU tests and CUDA optimizer/graph parity
still required. Crossing60k and completing the run is empirical validation still
pending at launch, not guaranteed by the toy allocator test.
