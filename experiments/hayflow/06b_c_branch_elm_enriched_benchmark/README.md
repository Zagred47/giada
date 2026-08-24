# HayFlow 06b-c supplement - Branch-ELM enriched-data benchmark

This professor-requested benchmark is deliberately separate from the primary
06b-c causal decision.  It evaluates the exact published `num_memory_20`
Branch-ELM architecture with 8,002 trainable parameters in two ways:

1. the published checkpoint is transferred zero-shot;
2. the same architecture is retrained from scratch at small scale on
   train-only enriched HayFlow episodes with three seeds.

Both `U_scheduled` and causal `U_realized` inputs are reported.  The original
1,278-channel contract is preserved as 639 dendritic excitatory channels plus
639 dendritic inhibitory channels.  Episodes containing injected somatic
current are excluded and counted because that input cannot be represented by
the published architecture without changing the comparison.

The initial preflight showed that exactly 48 train episodes remain compatible
after this exclusion, rather than the 80 initially budgeted. Before any
training or fresh-outcome evaluation, the role allocation was amended to 28
fit, 10 calibration, and 10 development episodes. All compatible episodes are
therefore retained in disjoint roles without altering the model contract.

The first completed training run encountered a report-only interruption:
spike AUC is undefined when an evaluated subset contains only one class, and
the strict JSON writer correctly rejected the resulting `NaN`. The metric is
now serialized as JSON `null` with an explicit `spike_auc_defined` flag. A
checkpoint-resume path reloads the six already completed models and recomputes
calibration, development, and fresh metrics without retraining.

The retrained checkpoints are selected only on a train-derived calibration
role and are frozen before fresh-test outcomes are read.  The 05j-o fresh set
has already been opened historically in this project, so this is a
retrospective matched-dataset benchmark rather than a new pristine test.
The recurrent state is reset at every independent teacher episode and only a
short burn-in is possible.  This differs from the original 500 ms windows and
150 ms burn-in, and is reported as a limitation rather than hidden.

The metric caveat is essential: Branch-ELM's published 0.6376 mV result is
clipped soma-only RMSE, whereas the HayFlow value around 0.40 mV is an
aggregate over all 642 segment voltages.  The report must not rank those two
numbers directly, even when both models see the same fresh transitions.

Notebook: `notebooks/06b_c_supplement_branch_elm_enriched_benchmark.ipynb`.

## Registered result

The recovered archive is valid and contains all six completed checkpoints; no
model was retrained during recovery.  On the 64 compatible fresh episodes, the
published checkpoint transfers at 8.239 mV with `U_scheduled` and 4.660 mV
with `U_realized`.  Small-scale retraining of the unchanged 8,002-parameter
architecture improves the median fresh RMSE to 3.301 mV and 2.710 mV,
respectively.

This is a real improvement over zero-shot transfer, but not a successful
replication of the published 0.638 mV result.  The median development errors
are 0.113 mV (`U_scheduled`) and 0.417 mV (`U_realized`), producing fresh to
development gaps of 29.2x and 6.50x.  Together with the seed variability,
this identifies a strong support/generalization shift rather than a simple
failure to optimize the compatible training episodes.

Every reported subset contains zero positive somatic spikes.  AUC is therefore
undefined and F1=0 is not interpretable as a discrimination failure.  A
spike-positive evaluation set is required for that comparison.

## Retraction of the first matched addendum

The first matched addendum is retracted as an answer to the
professor-requested comparison.  Although the archive and every indexed hash
are valid, it evaluated the retired 05j-n H2 plus refit stack (320,829
parameters), not the current compact 06b system composed of the 7,212-parameter
mechanism-STATE updater and the 8,985-parameter voltage bridge.  It also gave
the old H2 stack the complete teacher boundary state `S_t` plus
`U_realized`, while Branch-ELM received event history only.

Consequently, the observed 1.349 mV versus 2.710 mV result and the nominal
50.2% reduction are retained only as historical diagnostics and must not be
reported as the requested HayFlow-versus-Branch-ELM conclusion.  Equal target,
transitions, burn-in, and metric are insufficient when the compared model and
causal information differ.

The corrective comparison must use the current compact system and an
explicitly identical information contract: the same causal external inputs,
initialization, history, transitions, target, burn-in, and metric.  No teacher
boundary state may be injected into only one arm.  The sidecar remains open
only for this correction; it must not branch into additional ELM experiments.
The retraction and raw historical numbers are recorded in `result.json`.

## Corrective execution prepared

The replacement notebook now implements a bounded teacher-boundary one-step
comparison.  Both arms receive the exact same numeric per-segment tensor made
from current voltage and axial differences, normalized mechanism STATE,
mechanism presence, local ions, causal `U_realized`, static morphology and
region identity.  The target is the unclipped authentic NEURON transition
`V_(t+1) - V_t`; neither arm receives the teacher endpoint and no rollout is
performed.

The matched active voltage paths are the 8,002-parameter Branch-ELM core
(7,981 parameters influence its voltage output) and the current
8,985-parameter HayFlow voltage bridge.  The 7,212-parameter STATE updater is
reported as part of the 16,197-parameter complete compact transition system,
but it is frozen (zero trainable parameters in this comparison), downstream
and cannot affect the voltage score.  Both arms use the
same paired samples, optimizer hyperparameters, loss, checkpoint-selection
role and development role over seeds 61017, 61029 and 61043.

The published checkpoint cannot be reused under this strict common-input
contract; the ELM core is retrained while retaining its published branch and
memory hyperparameters.  The original event-only ELM result remains secondary
context, not the matched ranking.  The executable contract is recorded in
`information_matched_transition_amendment.json`.  This is the sole corrective
run and is now registered below.

## Corrective information-matched result

The corrective archive from code revision `a26ee9e` passed independent ZIP,
member-size, member-SHA-256 and CRC verification. Both models received the
same 76-value per-segment causal tensor, paired examples and ordering, loss,
optimizer, and unclipped teacher target `V_(t+1) - V_t`. Neither model
received the teacher endpoint, and no rollout or spike metric was part of this
comparison.

HayFlow obtained a lower development RMSE in every paired seed. Across all
segments, Branch-ELM scored 6.190, 6.450 and 6.380 mV while HayFlow scored
5.995, 6.034 and 5.745 mV. The corresponding HayFlow error reductions are
3.15%, 6.44% and 9.96%, with a median of 6.44%.

The soma result is stronger: median RMSE falls from 14.824 mV for Branch-ELM
to 9.577 mV for HayFlow, a paired median reduction of 34.94%. On active
examples, the median reduction is 8.92%. HayFlow also beats the persistence
baseline globally and at the soma for all three seeds; Branch-ELM is worse
than soma persistence in two seeds.

This is a valid ranking for the bounded authentic one-step voltage question,
not evidence that the complete HayFlow surrogate is solved. The absolute
errors remain large, no autoregressive behavior was tested, and the active
voltage paths are close but not exactly parameter matched (8,985 versus 8,002
trainable parameters). The frozen 7,212-parameter STATE updater is downstream
and cannot influence the scored voltage. The professor sidecar is closed
without further ELM experiments, and work returns to the primary HayFlow
causal program.
