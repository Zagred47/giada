# HayFlow 05m - topology-controlled recurrence expansion

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `155d7234e6ce27e8bd7eaa4378f6b90eb240d350cc6befa454cf8a5d3b8eccc6`;
- artifact-index SHA-256: `d7e0d88a9e90fdf97f00041157beba54fd948c2b6d146a46fdda58182aae91ba`;
- final-report SHA-256: `6c100e4fe6983dfb0477afe2938567590cd523b640e9864dbb5940bbaaf5bd98`;
- all twelve indexed members passed size and digest verification.

Every family passed all three recurrent-signal seeds.  Median development
8 ms RMSE was 18.361 mV for authentic morphology, 18.277 mV for ordered
ConvGRU and 17.565 mV for the relabelled-tree control.  Authentic morphology
lost to the relabelled control in every paired seed and its median gain was
-2.84%; its median gain over ConvGRU was only +0.37%.  The registered diagnosis
is `RECURRENCE_SIGNAL_REPLICATED_TOPOLOGY_STILL_UNRESOLVED`, authorizing only
the 05n graph-operator/state-contract reassessment.

05l established a robust rollout-aware recurrent signal in both the
morphology GraphGRU and a parameter-matched ordered ConvGRU, but the median
GraphGRU advantage was only 1.63%, below the registered 5% topology threshold.
05m asks the narrower causal question: does the authentic assignment of
segments to the morphology tree matter once recurrent training support is
expanded?

Three parameter-identical candidates are trained with seeds 17, 29 and 43:

- `authentic_morphology_graph_gru` uses the canonical parent/child tree;
- `relabelled_morphology_graph_control` uses exactly the same tree shape,
  depths and degree distribution, but deterministically relabels non-root
  nodes while leaving voltages, causal inputs and static segment features at
  their authentic segment identifiers;
- `ordered_convgru_control` retains the arbitrary segment-order convolution.

Within a seed the authentic and relabelled graph models have identical tensor
shapes and random initialization.  All three candidates have exactly matched
parameter counts.  The original 05l fit/calibration/development components
are reconstructed deterministically; previously unused train-only connected
components are added only to fit, up to eight per regime.  Calibration and development remain isolated
from the expanded fit role by seed, snapshot and trajectory.

Training remains closed-loop at 2/4/8 ms with only the initial teacher voltage
and strictly causal `U_realized` input.  There is no teacher forcing within a
window.  Existing validation/test splits and the sealed fresh test are not
loaded.  The reused development role is explicitly limited to a development
architecture decision and cannot authorize a paper-level claim.

Authentic topology is supported only if its recurrent signal remains robust,
it beats both controls in at least two of three paired seeds, and its median
gain over each control is at least 5%.  Regardless of outcome, 05m cannot
authorize full training, fresh-test generation or mass dataset generation.

Expected artifact: `hayflow_topology_controlled_recurrence_expansion.zip`.
