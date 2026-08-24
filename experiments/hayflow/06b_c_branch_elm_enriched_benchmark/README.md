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
