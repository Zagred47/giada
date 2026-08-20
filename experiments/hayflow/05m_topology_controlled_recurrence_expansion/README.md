# HayFlow 05m - topology-controlled recurrence expansion

Status: implemented; execution pending.

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
