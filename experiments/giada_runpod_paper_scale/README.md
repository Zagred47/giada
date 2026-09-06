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

The S1b result is now explicitly classified as a NeuronIO-distribution control,
not the primary GIADA generator. The next development stage is S1c, a separate
hybrid pilot that restores the evolved methodology: stochastic background,
targeted canonical-weight dendritic events, hard negatives, and matched
counterfactual futures from the same state. Its immutable preregistration is
`s1c_hybrid_pilot_preregistration.json`.

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

S1c and its prospective S1d repair are now complete. S1c established robust
NMDA and calcium contrasts but exposed two transcription errors in its somatic
and BAP arms. S1d repaired only those arms and passed every preregistered gate
independently in train and validation: one 3 nA pulse stayed negative, one
6 nA pulse spiked, the selected n12/b3/w400 dendritic assist stayed
subthreshold, p2-factor3 soma-only stayed below the registered trunk BAP
threshold, and p3-factor3 soma-only crossed it. Paired combined arms further
confirmed that the dendritic assist changes the trunk response causally.
Neither pilot is training data. The next step is a prospective specification
of the final hybrid corpus; S2 remains blocked until that specification and
its per-split support gates are frozen.

That specification is now frozen as S1e. It contains exactly 600,000
transitions at the same scale as S1: 360,000 from long 6-second stochastic
background trajectories and 240,000 from 80-ms targeted episodes using only
the S1c/S1d-confirmed arms. Both physical components use an 80/20
trajectory-level split and are exposed to training through a validated logical
manifest, without duplicating the HDF5 files. S1e remains development data;
it is not a fresh paper test. Its preregistration is
`s1e_hybrid_production_preregistration.json`.

S1e generation subsequently passed every structural and support gate. The
immutable observed outcome is recorded in `s1e_hybrid_production_result.json`.
The paired GPU comparison is preregistered separately in
`s1e_matched_training_preregistration.json`: it reuses the frozen S1
architectures and optimization contract, adds stratified diagnostics without
feeding them back into training, and selects the final 3,000-step checkpoint
rather than selecting on validation.

Operational source, configs, and instructions live in `runpod_scale/` and
`src/giada_runpod/`. Generated HDF5 shards and checkpoints never enter Git.
