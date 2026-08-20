# HayFlow 05l - rollout-aware GraphGRU versus ConvGRU canary

Status: completed; artifact independently verified.

Verified artifact:

- archive SHA-256: `9a5c423aba80da51830a57c6f97808a3366069c6e42807193143d49de13be634`;
- artifact-index SHA-256: `03e77d8a7104a25b2636f05f406031b01b81604c9e15d1dcb4d12a859ac0d757`;
- final-report SHA-256: `a43227c545088b7bc1f9dbf2bafbac9f73534504033cbd365b9e6fe805cca930`;
- all nine indexed members passed size and digest verification.

Both families passed all three registered seeds.  At 8 ms the median
development endpoint RMSE was 18.296 mV for the morphology GraphGRU and
18.599 mV for the ordered ConvGRU, against 21.021 mV for persistence.  The
GraphGRU gain over ConvGRU was only 1.63%, below the preregistered 5%
morphology-materiality threshold.  The resulting diagnosis is therefore
`ROLLOUT_AWARE_RECURRENCE_SIGNAL_TOPOLOGY_UNRESOLVED`, which authorizes only
the train-only topology-controlled 05m expansion.

05k-d retired the free-running H2 latent recurrence while preserving evidence
that the local one-millisecond map is useful under teacher boundary resets.
05l tests the next atomic question: can a small recurrent architecture trained
through its own predicted voltages remain useful for 2, 4 and 8 ms, and does
the authentic morphology graph add signal beyond a parameter-matched
fixed-order convolution?

The two candidates share the same voltage/causal-drive encoder, GRU cell,
hidden width, bounded voltage head, optimizer, seeds, windows and losses.  They
differ only in their spatial mixer:

- `morphology_graph_gru` receives self, parent and mean-child hidden states;
- `ordered_convgru_control` receives a kernel-3 Conv1D over the arbitrary
  segment identifier order.

The graph linear mixer and kernel-3 convolution have the same `3 H^2 + H`
parameter count.  A preflight rejects the experiment if total parameter counts
differ by more than 5%.

Only episodes from the original `train` split are used.  Seed- and
snapshot-connected episode components are partitioned into fit, calibration
and development roles.  Fit supplies gradients, calibration selects the
checkpoint, and development is evaluated once after freezing.  Existing
validation/test splits and the sealed 05j-o fresh test are never loaded.

Every training window starts from one authentic boundary voltage.  There is no
teacher forcing inside the window: all later membrane voltages are the
model's own predictions.  Inputs contain only the ordered causal
`U_realized` records reduced to receptor increments, release counts/timing and
somatic current.  No teacher boundary state after the initial voltage is a
model input.

The canary passes a seed only when the 8 ms endpoint RMSE improves over
persistence by at least 10%, every requested horizon is finite, and there are
no voltages outside the fixed physical interval.  At least two of three seeds
are required for a robust family signal.  A morphology claim additionally
requires at least 5% median GraphGRU improvement over ConvGRU.

This experiment cannot authorize full training, a new fresh test, or mass
dataset generation.  It can authorize only a separate expanded development
canary or a more specific architecture reassessment.

Expected artifact: `hayflow_rollout_aware_graphgru_vs_convgru_canary.zip`.
