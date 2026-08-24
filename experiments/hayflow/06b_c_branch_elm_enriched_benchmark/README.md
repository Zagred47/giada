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

The amended archive completes the voltage comparison inside this same sidecar.
The frozen HayFlow candidate obtains a median clipped soma RMSE of 1.349 mV
(seeds: 1.555, 1.156, and 1.349 mV), compared with 2.710 mV for retrained
Branch-ELM with `U_realized` (2.660, 2.710, and 5.319 mV).  HayFlow therefore
reduces the paired median error by 50.2%; its ensemble prediction is 1.350 mV.
No retraining, checkpoint selection, architecture search, or fresh-outcome
selection was performed for HayFlow.

This is a valid matched system-level voltage comparison: both models use the
same 64 episodes, 512 transitions, 4 ms burn-in, clipped somatic target, and
pooled RMSE.  It is not a capacity- or information-matched architecture
ablation: Branch-ELM has 8,002 parameters and consumes event history, whereas
the frozen HayFlow stack has 320,829 parameters and consumes the complete
17,220-variable boundary state plus `U_realized`.  The earlier HayFlow value
near 0.40 mV remains a different all-segment/boundary metric and is not used in
the ranking.  Spike comparison is unavailable because the shared support has
zero positive somatic spikes.

The professor-requested sidecar is now closed.  No additional comparison
experiment is planned; work returns to the primary 06b-c causal trajectory.
The registered result is `result.json`, and the frozen metric-alignment
contract is `matched_comparison_amendment.json`.
