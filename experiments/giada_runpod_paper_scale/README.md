# GIADA paper-scale validation track

This experiment family is isolated from the sequential Kaggle architecture
research under `experiments/hayflow/`. Its purpose is confirmation by data
scale, not architecture selection.

The starting observation is the valid 06b-c information-matched comparison:
the 8,985-parameter GIADA voltage bridge reduced authentic one-step soma RMSE
from a Branch-ELM median of 14.824 mV to 9.577 mV on the small train-derived
development support, a paired median reduction of 34.94%. The result ranked
the two compact voltage paths but did not establish a complete neuron
surrogate and was not directly comparable to the 64 biological hours of the
published NeuronIO train/validation corpus.

RunPod S0--S4 asks whether that ranking survives increasing amounts of teacher
time while keeping the numerical input tensor, target, sample order, loss,
optimizer family, and seeds aligned. No result in this track authorizes a new
architecture. Architecture hypotheses remain in the Kaggle/Allen-Zhu track.

S1 integrity validation succeeded for all 600,000 transitions, but the later
distribution audit exposed a missing prerequisite check: all 18,000 validation
transitions were subthreshold, with no `|delta V| >= 1 mV` example and no
somatic spike. The matched S1 comparison is retained as a genuine subthreshold
result (median RMSE 0.246 mV for GIADA versus 0.425 mV for Branch-ELM), while
active/spiking conclusions and S2 advancement are blocked.

The correction is prospective. S1 is not resplit or regenerated. A fresh,
development-only 2x2x2 input-support pilot tests conditional slices of the
original NeuronIO distribution before an independent event-support corpus is
preregistered. From this point onward, schema/integrity validation and
per-split dynamical-support validation are both mandatory before GPU training.

The first S1b attempt was aborted before analysis because its proposed
low-inhibition range was marginally canonical but jointly infeasible with
near-zero excitation.  The corrected v2 pilot used a fresh output root and an
explicit joint-support guard.  All 96,000 transitions and 16 shards validated.
The aggregate train/validation splits contained respectively 212/225
transitions with `|delta V| >= 5 mV` and 77/78 somatic upcrossings.  The
high-excitation, low-inhibition, fast-temporal cell was strongest in both
independent trajectories (109/62 large transitions and 41/20 upcrossings),
while its broad-temporal counterpart was the most balanced secondary cell
(52/45 and 18/15).  This identifies prospective input regimes; it does not
convert the pilot into training or final validation data.

Operational source, configs, and instructions live in `runpod_scale/` and
`src/giada_runpod/`. Generated HDF5 shards and checkpoints never enter Git.
