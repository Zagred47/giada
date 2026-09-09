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

The S1e GPU comparison then passed all three preregistered gates. GIADA reduced
median overall soma RMSE by 20.7%, active-transition RMSE by 25.2%, and
somatic-upcrossing RMSE by 31.0% relative to the information-matched
8,002-parameter Branch-ELM core. It won all three seeds and every per-protocol
median comparison. The immutable result and artifact hashes are recorded in
`s1e_matched_training_result.json`. This authorizes S2 under a new frozen
contract; it is not an autoregressive-rollout or fresh paper-test claim.

That S2 contract is now frozen in `s2_hybrid_scale_preregistration.json`.
S2 is an exact sixfold expansion of S1e: 2.16 million background and 1.44
million targeted transitions, with the same 60/40 mixture, 80/20 complete-
trajectory split, protocol registry, canonical weights and causal release
semantics. It deliberately supersedes the old single-protocol `s2_soma.yml`
for this paper claim. Training uses 18,000 fixed steps so that each model sees
the same 25.6 effective corpus passes as at S1e. Five paired seeds report mean,
sample standard deviation and paired uncertainty; the three original seeds
remain identifiable for the direct S1e-to-S2 scaling comparison.

The S2 corpus subsequently passed every registered structural, protocol and
distribution-fidelity gate. Its six contract-file hashes and aggregate
fingerprint over all 1,080 shard completion markers are frozen in
`s2_hybrid_production_result.json` and in the matched-training configuration.
Before allocating the GPU, training re-hashes every physical HDF5 and rejects
any mismatch between a shard, its completion marker, or the frozen corpus.

The preregistered S2 GPU comparison is also complete. Across five paired
training seeds, the 8,985-parameter GIADA voltage-transition component reduced
median overall soma RMSE from 2.121 to 1.529 mV (27.9%), active-transition RMSE
from 9.927 to 8.434 mV (15.0%), and somatic-upcrossing RMSE from 11.048 to
9.560 mV (13.5%) relative to the information-matched 8,002-parameter
Branch-ELM core. GIADA won five of five seeds, all five protocol families and
thirteen of fourteen protocols. The paired mean RMSE difference was 0.803 mV
with a deterministic seed-bootstrap 95% interval of [0.408, 1.444] mV and a
one-sided paired t-test p-value of 0.0331. All registered gates passed and S3
is authorized under a separately frozen contract. This remains a one-step
soma-voltage development-validation result, not an autoregressive, full-state,
or fresh sealed paper-test claim. Exact metrics and artifact hashes are in
`s2_matched_training_result.json`.

S3 is now prospectively frozen in `s3_hybrid_scale_preregistration.json`.
It multiplies every S2 component, protocol and split count by exactly eight:
17.28 million long-background and 11.52 million targeted transitions, for
28.8 million total. The teacher, 60/40 mixture, 80/20 complete-trajectory
split, canonical weights, causal-release semantics, model definitions, loss,
optimizer and five paired seeds remain unchanged. New root seeds keep S3
teacher trajectories independent of S2.

The S3 CPU corpus must pass exact structural, cryptographic, per-protocol and
distribution-fidelity gates before a GPU configuration is created. Training
is preregistered for 144,000 steps, preserving 25.6 effective train-corpus
passes. Unlike the earlier one-million-row cap, every registered S3
checkpoint will evaluate the complete 5.76-million-transition development
split; this prevents physical component ordering from silently biasing the
comparison. Because the somatic-upcrossing margin narrowed from 31.0% at S1e
to 13.5% at S2, a positive upcrossing advantage is now an explicit safety gate
for S4 rather than a descriptive metric. This addition was frozen before S3
generation and does not modify either architecture.

The S3 corpus has now passed the structural and distribution audit plus a
second physical fingerprint pass: all 8,640 shards, totalling 5,852,939,725
bytes, match their completion markers and there are zero mismatches. The seven
corpus identity hashes are frozen in the S3 preregistration and in
`runpod_scale/configs/s3_matched_training.yml`; GPU execution is now allowed
under those settings, while post-hoc scientific changes remain forbidden.

The preregistered S3 GPU comparison is complete but does not authorize S4.
At the final 144,000-step checkpoint GIADA reduced median overall soma RMSE
from 1.045 to 0.882 mV (15.5%), active-transition RMSE from 6.132 to 5.461 mV
(10.9%), and somatic-upcrossing RMSE from 7.651 to 7.284 mV (4.8%). It won all
five protocol families and all fourteen protocols, so the biological breadth
and all three median-error gates survived. It nevertheless won only three of
five paired seeds, below the preregistered four-of-five requirement. The paired
mean advantage was only 0.032 mV, its deterministic bootstrap interval crossed
zero, and the one-sided paired t-test gave p=0.396.

The registered checkpoint trajectory localizes the failure to late
optimization rather than to a general inability to learn S3. GIADA won five of
five seeds at every checkpoint through 72,000 steps, where its median advantage
was 32.6%. During the final 72,000 constant-learning-rate updates, seeds 61017
and 61103 deteriorated enough to lose even though the GIADA median continued to
improve. The favorable 72,000-step result is retrospective and cannot replace
the frozen final-checkpoint rule. A separate preregistered optimization
forensic is required before any new scale claim. Exact metrics and artifact
hashes are in `s3_matched_training_result.json`.

Operational source, configs, and instructions live in `runpod_scale/` and
`src/giada_runpod/`. Generated HDF5 shards and checkpoints never enter Git.
