# HayFlow development boundary

HayFlow is developed in this repository, but it is kept separate from the
existing ELM/NeuronIO pipeline. The original
`SelfishGene/neuron_as_deep_net` checkout is a pinned teacher reference and is
not modified by HayFlow code.

## Package boundaries

- `src/hayflow_schema`: dependency-light contracts shared by generation and
  training. It must not import NEURON, PyTorch, or JAX.
- `src/hayflow_teacher`: adapters around the instantiated NEURON teacher,
  including manifest extraction, logging, snapshot/restore, and event
  extraction.
- `src/hayflow_data`: storage readers, window sampling, batching, and format
  validation for HayFlow datasets.
- `src/hayflow_model`: full-state flow-map baselines and, later, the latent
  HayFlow architecture.
- `src/hayflow_eval`: event, voltage, rollout, and restore-fidelity metrics.
- `src/neuronio`: the existing ELM baseline and infrastructure. It remains
  independently usable.

Teacher generation and model training intentionally use separate runtime
environments. Dataset files, snapshots, compiled NEURON mechanisms, and model
artifacts are generated locally and are not committed.

## First implementation milestone

The first milestone is an observability and Markov-state experiment, not the
complete latent architecture:

1. inspect the fully instantiated NEURON morphology and write a versioned
   `TeacherManifest`;
2. log complete 1 ms boundary states and native-precision restart snapshots;
3. extract configurable axonal, somatic, bAP, calcium, NMDA-spike, and
   NMDA-plateau events;
4. prove snapshot/restore fidelity by replaying the same interval;
5. generate a small diagnostic dataset with continuous 0.025 ms microtraces;
6. train and roll out a one-millisecond full-state flow map before introducing
   state compression.

The teacher's morphology, mechanisms, input-generation distributions, and
CVODE configuration must remain unchanged while instrumentation is added. The
upstream point processes use an unowned process-global random stream; HayFlow
partitions the same `negexp(1)` distribution into deterministic per-synapse
`Random123` streams and records this instrumentation choice in provenance.

## Source provenance

The initial teacher reference is:

- repository: `https://github.com/SelfishGene/neuron_as_deep_net`
- branch: `master`
- commit: `074c4666300a8ad246601dab179a97a6942f0f29`
- local development path: `../neuron_as_deep_net`

The local path is configurable; the commit is part of the dataset provenance
and must be recorded in every generated manifest.

## Current implementation status

`scripts/hayflow/build_teacher_manifest.py` now performs the first structural
pass against an instantiated NEURON cell. The pass is intentionally incomplete
in two explicit ways:

- synapses are inventoried only after the generator creates its point
  processes and supplies `NeuronSynapseBinding` objects;
- the structural pass keeps apical labels broad; the runtime audit validates
  nexus, trunk, and tuft against upstream conventions and adds hot zone as an
  overlapping segment tag.

The manifest records these limitations in metadata instead of presenting an
incomplete inventory as a complete teacher contract.

The reproducible Linux entry point is
`notebooks/hayflow_teacher_manifest.ipynb`. It pins NEURON 8.2.7, checks out
the exact teacher commit, compiles the original mechanisms, runs the contract
tests, builds the manifest, and rejects a dendrite-only or topologically
invalid result.

After that setup, `notebooks/00_teacher_audit_and_smoke_test.ipynb` performs
the canonical runtime audit: full morphology, mechanisms, instantiated
synapses, no-input and active smoke tests, representative internal-state
recording, and a deterministic snapshot/restore diagnostic. Its small outputs
are written under `artifacts/teacher_audit/` and remain excluded from Git.
The audit notebook is also standalone: when no mounted checkout is provided,
its first executable cell clones the owned repository before fetching or
loading any teacher code.

The runtime manifest also includes the `NET_RECEIVE` state stored in each
indexed `NetCon.weight` vector. These values are part of the Markov state even
though `MechanismStandard` does not expose them as ordinary point-process
STATE variables.

## Diagnostic transition dataset

`notebooks/01_burnin_snapshots_and_transition_dataset.ipynb` is the second
experiment. It does not train HayFlow. It establishes a replayable data
contract for one-millisecond teacher transitions.

The notebook:

- measures rest convergence from `v_init = -76 mV` and rejects an arbitrary
  burn-in cutoff;
- writes a canonical equilibrium `SaveState` plus the externally owned
  Random123 key, distribution, and sequence positions;
- generates 36 short diagnostic trajectories covering rest/subthreshold,
  local excitation/inhibition, somatic-event candidates, and dendritic-event
  candidates, with seed/protocol isolation across train, validation, and test;
- stores rich boundary state and RNG arrays in compressed HDF5, while keeping
  native NEURON snapshots for transition-level replay;
- stores ordered intra-millisecond inputs rather than count aggregation;
- samples representative internal variables and all 642 voltages at 0.025 ms
  for this small dataset only, with per-segment minima, maxima, and integrals;
- writes provisional, versioned event definitions and plots representative
  trajectories for mandatory visual review;
- verifies sampled transition replay, a complete test trajectory, branching,
  finite values, time grids, split isolation, boundary/microtrace consistency,
  and canonical teacher hashes.

Because `SaveState` does not contain CVODE's adaptive-history internals, both
generation and replay call `CVode.re_init()` at each 1 ms boundary. This leaves
the teacher equations and tolerances unchanged and defines a symmetric,
replayable numerical flow-map contract. The instrumentation choice is recorded
in every dataset manifest and must be reconsidered explicitly before massive
generation.

The default output is `artifacts/transition_dataset_diagnostic/` locally, or
`/kaggle/working/hayflow_transition_dataset_diagnostic/` in the notebook. It
contains `burnin_report.json`, the equilibrium snapshot, `state_schema.json`,
Parquet morphology/synapse tables, `transition_dataset.h5`, provisional event
configuration, example figures, `validation_report.json`, and a hashed artifact
index. Only after this report is green should
`02_full_state_flowmap_baseline.ipynb` be implemented.

## Dendritic protocol calibration

`notebooks/01b_dendritic_protocol_calibration.ipynb` is the gate between the
replayable diagnostic contract and the first full-state model. The original
diagnostic dataset proved boundary-state replay and somatic spiking, but its
dendritic candidates remained subthreshold. Starting a flow-map baseline from
that data alone would not test the nonlinear calcium/NMDA regimes that motivate
the Hay teacher.

Notebook `01b` therefore performs a separate, staged stimulus search. It:

- preserves every canonical synaptic weight and all teacher mechanisms;
- selects local excitatory synapses along one ancestor/descendant path through
  the instantiated morphology tree, rather than mixing nearby branchlets;
- places a candidate-specific voltage/calcium probe at the actual cluster
  center instead of assuming one fixed nexus segment represents every event;
- varies recruited synapse count, burst count, repetitions, and an explicit
  intra-millisecond synchrony window;
- separates strict isolated-NMDA searches from event-rich plateau protocols;
- treats the reproducible soma-spike-paired calcium response as the canonical
  required BAC-like calcium family, while retaining the unpaired search as an
  optional diagnostic;
- repeats each candidate across multiple Random123 trajectory keys;
- records representative voltages, `cai`, aggregate and mechanism-specific
  calcium currents, and summed AMPA/NMDA conductance/current;
- accepts the least intense configured level that crosses the provisional
  event definition on the required fraction of seeds;
- checks both required protocol families and robust coverage of NMDA spike,
  NMDA plateau, and calcium-spike labels;
- marks events that remain above their reset threshold at the trace boundary as
  `right_censored`, so a recording cutoff is never reported as a biological
  offset;
- replays only the selected protocols for 160 ms, requires every selected
  event to recover below its reset threshold on all three seeds, and proves
  that the complete first 35 ms are numerically identical to the search trace;
- writes every trace, the complete input schedule, selected synapse/segment
  identifiers, plots for selected and best-rejected candidates, a hashed
  artifact index, and
  `selected_dendritic_protocols.json`.

The calibration is intentionally allowed to finish with `valid: false`. In
that case the archive must still be inspected before extending the stimulus
grid; weights and event thresholds must not be changed merely to force a
positive result. Once the required event-rich tuft plateau and paired
hot-zone calcium families pass, all three dendritic event labels have robust
coverage, the long-horizon confirmation is uncensored and exact on the shared
prefix, and the traces pass visual review, their selected schedules are used
by `01c` to build the diagnostic transition dataset v1 before implementing
`02_full_state_flowmap_baseline.ipynb`.

## Diagnostic dataset v1

`notebooks/01c_build_diagnostic_transition_dataset.ipynb` consumes the complete
hashed artifact bundle produced by `01b` and generates
`artifacts/diagnostic_dataset_v1/`. It is the final data-contract gate before
the full-state baseline and does not train a neural model.

The v1 dataset:

- recomputes derived currents and conductances immediately after every native
  restore, before exposing `S_t`, so branching does not depend on which
  trajectory ran previously;
- routes calibration, generation, and replay through one canonical 1 ms driver
  with a fixed IClamp/CVODE/NetCon ordering;
- calibrates paired somatic stimulation and a genuine single-pulse spike
  protocol separately rather than extrapolating one threshold from the other;
- runs a mandatory six-prefix/single-spike/branching preflight before creating
  the expensive HDF5 store; the six confirmed schedules are replayed once
  through the corrected calibration path and once through the storage path,
  with exact identity required between those current-runtime traces, and the
  green preflight is bound to an exact hash of the subsequent trajectory plan;
- reuses the two selected schedules exactly on seeds 310001--310003;
- includes rest, subthreshold, somatic spiking, confirmed tuft NMDA plateau,
  confirmed paired hot-zone calcium spike, true near-threshold negative
  controls, the wider-window NMDA timing perturbation as an explicitly
  positive boundary case, and five futures from a common branching state;
- keeps all 17,220 core state variables at every boundary and stores the 9,182
  current/conductance observables as a separate privileged category;
- stores all 642 voltage microtraces plus canonical probes, the confirmed tuft
  cluster center, an alternate-branch control center, local calcium currents,
  and summed AMPA/NMDA observables at 0.025 ms;
- uses periodic native NEURON checkpoints and replays the ordered prefix from
  the nearest checkpoint, avoiding one large `SaveState` file per millisecond
  while preserving transition-level reproducibility;
- writes HDF5 data and Parquet indices for transitions, protocols, events,
  splits, and branching, together with a storage cost report;
- exhaustively replays every transition and requires uncensored positive
  events, clean whole-trajectory splits, stable segment/state schemas, exact
  overlap with the corrected-runtime preflight references, and valid hashes
  for all 88 historical `01b` calibration
  artifacts;
- requires every trajectory declared as a negative control to suppress its
  target event; one successful negative per family is no longer sufficient;
- produces stimulus-relative figures with separate axes for voltage, calcium,
  calcium current, NMDA conductance, and NMDA current.

The `01b` ZIP or its extracted directory is a required input. This keeps the
provenance check real: `01c` does not silently trust copied protocol names when
the reference traces and their hashes are unavailable.

## Full-state flow-map baseline

`notebooks/02_full_state_flowmap_baseline.ipynb` consumes only the green
diagnostic dataset schema `1.0.1`. It refuses schema `1.0.0`, an unexpected
teacher commit, a non-green validation report, or any artifact whose size or
SHA-256 differs from the hashed index. This notebook is a feasibility gate for
the 1 ms macro-step; it is not the final HayFlow architecture.

The experiment keeps the 17,220 dynamic variables semantically separated into
voltage, mechanism state, calcium/ions, and synaptic state. Static morphology is
an input, while the 9,182 currents/conductances and intra-step observations are
training-only privileged targets. Random123 is retained for teacher replay and
is never regressed. Because schema `1.0.1` does not expose release outcomes,
both input encodings use the ordered scheduled events and record that limitation
explicitly:

- `U1` aggregates counts, weights, conductance, timing moments, and somatic
  current per segment;
- `U2` retains the ordered event list with time, segment, type, weight, and
  release-availability mask.

The notebook compares persistence (`B0`), dual ridge (`B1`), a deliberately
small flat residual MLP sanity check (`B2`), and a shared structured residual
model (`B3`). It runs voltage-only/full-state and `U1`/`U2` ablations, plus
matched `P0` and `P1` versions where `P1` adds privileged decoders. All
normalization statistics come only from the train split. Whole trajectories
remain isolated across train, validation, deterministic, event-boundary, and
branching tests.

Outputs are written below `artifacts/full_state_flowmap_baseline/` and include
resumable checkpoints, one-step/event/rollout/ablation Parquet tables,
representative predictions, figures, and a conservative `final_report.json`
with a `GO`, `CONDITIONAL_GO`, or `NO_GO` decision. The report must be read with
the stated limitation: 1,224 deliberately enriched transitions can reveal an
apparent learnable signal and immediate rollout instability, but cannot
establish high-dimensional generalization.

## Reconditioned full-state flow-map baseline

`notebooks/02b_reconditioned_full_state_flowmap_baseline.ipynb` repeats the
diagnostic B0/B1/B3 comparison on the exact same dataset `1.0.1`, trajectories,
splits, U1/U2 inputs, and B3 backbone. It is a controlled correction of the
notebook 02 objective, not a new architecture experiment. The original 02
artifact is a required read-only input and its dataset-manifest hash must match.

The train split alone defines a variable-level distribution audit and the new
normalization contract. Sparse transformed deltas use a documented numerical
activity threshold, an activity head, and active-only value scaling; dense
variables use a hybrid MAD/standard-deviation/RMS scale with an explicit floor.
Gate logit transforms are compared with an identity ablation, positive
quantities use `log1p`, and privileged quantities are normalized per variable
while non-applicable entries are masked. Synaptic states are tested both as an
early-stopping-independent metric-only block (S0) and with a hurdle objective
(S1). The absent release outcome remains an explicit identifiability limit and
is never inferred from `S_(t+1)` as a model input.

The main objective stratifies subthreshold, somatic/axonal, dendritic, and
near-threshold boundary windows. It reports both raw and effectively weighted
shared-representation gradient norms, and refuses methodological validity if a
single component still dominates. Learnable event classes use train-only
support, dendritic oversampling and class weights; absent classes are marked
`not_learnable_from_current_split`, and thresholds are calibrated on validation
only. Privileged supervision is compared as P0, a small normalized fixed weight
(P1a), and a gradient-capped weight (P1b).

Early stopping uses operational validation quantities rather than the raw mean
of state losses. Separate best one-step, event-fidelity, composite-selection,
and 8 ms rollout checkpoints are saved. Every checkpoint is bound to dataset,
schema, split, normalization, loss, model, code-commit, and seed fingerprints;
stale checkpoints are rejected. Three common B3 seeds are aggregated as mean
and standard deviation.

The evaluation covers 2/4/8/16 ms rollouts, common and rare regimes, per-region
drift, peak error/attenuation, lost and added event labels, recovery endpoints,
physical-domain violations, and branching divergence retention. The final
report separately states artifact validity, methodological validity, modeling
result, identifiability limits, and answers the six predeclared comparisons
against B1, U1, privileged supervision, the original negative drift, and
branching collapse. It explicitly does not test Hines coupling, persistent
latents, morphology reduction, Mamba, or S4.

## Targeted dataset v1.1 and BAP validation supplement

`notebooks/03_build_targeted_transition_dataset_v1_1.ipynb` extends the data
contract with causal release outcomes and targeted event-boundary protocols.
Its first complete shard contains 29,240 transitions whose exhaustive replay
passed with zero failures.  The realized biological support missed only one
hard acceptance floor: validation contained three independent positive
backpropagating-AP episodes instead of the preregistered minimum of four.

`notebooks/03b_bap_validation_support_topup.ipynb` corrects that finite-sample
shortfall without regenerating, editing, or physically merging the base shard.
It verifies the Quick-Saved base and its replay proof, preregisters a fixed
batch of eight validation episodes from the already validated soma-only BAP
recipe, assigns new Random123 seeds and new conditioned snapshots, and retains
all eight outcomes.  Only the 640 new transitions are replayed.  Acceptance
requires at least one new positive BAP episode and the minimum support contract
to pass on the logical union.  Failed positive-intent episodes are not relabeled
as hard negatives, event thresholds are unchanged, and no post-hoc seed
selection is permitted.  The composite manifest binds both physical shards by
SHA-256 while keeping the 6 GiB base in Kaggle input storage.  When KaggleHub
packages the multi-file base as `archive.zip`, notebook `03b` extracts it under
`/kaggle/temp` with progress and ETA; the temporary copy is never included in
the notebook output archive.
Top-up seed allocation is itself bound into the preregistered plan: a fixed
gap of 10,000 is added to the maximum seed present in the base episode table,
then one contiguous block is reserved.  This makes disjointness deterministic
and outcome-blind instead of relying on an assumed unused numeric range.
The top-up session also refuses to instantiate the teacher when any NEURON
sections already exist.  A fresh Python process is mandatory because loading a
second 196-section Hay cell would leave 392 sections and make the imported
SaveState structurally incompatible; variable persistence must therefore be
disabled for this notebook.

## Release identifiability and flow-map v1.1

`notebooks/04_release_identifiability_and_flowmap_v1_1.ipynb` consumes the
validated base plus BAP top-up only through `composite_dataset_manifest.json`.
Its shard-aware reader exposes one logical index while retaining the physical
shard and local row, keeps the top-up in validation, and reads full states
lazily without merging the HDF5 stores.  The four preregistered SHA-256 values,
369 episodes, 29,880 transitions, split isolation, cross-shard seed/snapshot
disjointness, and event counts are hard preflight gates.

The experiment compares scheduled, causal-Random123, and realized synaptic
inputs.  Because the stored realized view omits failed events, the reader joins
every scheduled event to the exact pre-membrane causal release record and thus
retains both successes and failures.  Random123 stream, episode, trajectory,
and snapshot identifiers are excluded from model features.  No `S_(t+1)` value
is used to construct any input view.

B0, deterministic dual ridge, and the unchanged reconditioned B3 backbone are
evaluated with episode-aware stratification.  B3 uses the hurdle representation,
gate-logit transform, event-aware U2 encoding, three common seeds, P0 as the
primary objective, and one small selective privileged ablation.  Outputs cover
pooled event counts and PR-AUC, one-step and 2/4/8/16/32 ms rollout voltage
fidelity, regional drift, peak attenuation, near/far/release branching,
recovery, state sufficiency, and episode bootstrap intervals.  The report can
recommend the first HayFlow-Hines prototype but does not implement Hines,
latents, morphology reduction, S4, Mamba, CUDA kernels, or mixed precision.

## HayFlow-Hines recurrent prototype

`notebooks/05_hayflow_hines_prototype.ipynb` is the first implementation of a
HayFlow core rather than another B3 regression ablation.  It consumes the same
immutable logical composite through `composite_dataset_manifest.json`, uses
only causal `U_realized` inputs, and requires the notebook-04 report as a
read-only numerical reference.  No teacher simulation or dataset generation
occurs in notebook 05.

The forward path encodes the complete teacher state only at initialization.
It then maintains explicit voltage, compact per-segment calcium and authentic
synapse-state views, a persistent local latent for every one of the 642
segments, and a persistent global latent.  The synaptic front-end retains
ordered successes and failures and keeps AMPA, NMDA, GABAA, and GABAB
statistics separate.  Existing double-exponential A/B state is propagated
across the macro-step and combined with the new exact causal increments; NMDA
magnesium block is evaluated causally from the boundary voltage.

The local recurrent cell predicts a non-negative effective conductance and a
source current.  Authentic capacitance, passive leak, axial coupling, reversal
potential, and parent/child topology construct one tree system per macro-step.
`DifferentiableHinesSolve` performs functional leaf-to-root elimination and
root-to-leaf substitution.  Independent tests compare it with a dense solve,
run PyTorch gradcheck, enforce positive pivots, and exercise the full canonical
morphology.  H0 uses only this solve, H1 adds a bounded continuous closure, and
H2 also adds a morphology-masked event-conditioned jump.

Six independent event heads use dedicated anatomical masks and predict
presence, timing, region, segment, and amplitude.  Local and global GRU-like
commits receive predicted voltage and event information, so teacher state is
not re-encoded during rollout.  Selective biologically motivated state/current
decoders and five-anchor microtrace decoders are training-only.  A fixed-order
ConvGRU with a similar compact recurrent budget is included as a deliberately
non-morphological control.

Before the full curriculum, both H2 and ConvGRU must attempt a balanced overfit
canary containing somatic, BAP, calcium, NMDA, plateau, hard-negative, and
counterfactual branching support.  Full training is refused unless H2 reaches
the preregistered voltage, event, peak, and branching thresholds.  If allowed,
training progresses from teacher-forced one-step to recurrent 2/4/8 ms and
then 16/32 ms windows, using episode-and-regime-stratified sampling and a
relative branching loss.  One-step, event, and rollout checkpoints are stored
separately and fingerprinted against data, normalization, code, seed, and
configuration.  Outputs under `artifacts/hayflow_hines_prototype/` include the
canary verdict, Hines tests, H0/H1/H2 and ConvGRU metrics, B3 comparison,
regional drift, peak attenuation, branching, recovery, out-of-domain rates,
checkpoint registry, and final A/B/C/D scenario classification.

### Notebook 05b: corrected architecture canary

`notebooks/05b_hayflow_hines_canary_revision.ipynb` is the controlled follow-up
to the first 05 canary. The original run showed that H2 outperformed the
fixed-order ConvGRU but that neither model could satisfy the absolute overfit
contract. It also exposed three confounds: 2,048 auxiliary absolute-state
targets dominated the H2 loss, the event jump was diluted by a probability
distribution over 642 segments, and checkpoints were selected by aggregate
training loss instead of the four acceptance metrics.

05b keeps the scientific thresholds unchanged and corrects the diagnostic:

- the canary is staged into voltage/peak, events/branching, and joint phases;
- biological auxiliary decoders predict normalized 1 ms deltas, with sparse
  regression evaluated only on active variables;
- event localisation uses sharpened attention with unit peak, while a separate
  signed boundary-voltage decoder is attenuated by predicted event timing;
- the ConvGRU control can express the full configured 120 mV macro-step change;
- each epoch records unweighted loss components and the pre-clipping gradient
  norm;
- the retained checkpoint minimises a score derived from voltage RMSE, boundary
  peak error, minimum event F1, and branching retention;
- event support uses two independent training episodes per class;
- the notebook ends after the canary and packages its model checkpoints.

Only the targeted v1.1 base dataset and BAP validation top-up v3 are required.
The B3 result is not an input because 05b performs neither final comparison nor
full training.

The completed 05b run is registered in
`experiments/hayflow/05b_hayflow_hines_canary_v2/result.json`. Generated ZIP and
checkpoint files remain external, while the record commits their SHA-256
identities, dataset fingerprint, exact acceptance metrics, `NO-GO` decision,
and the authorized 05c causal-isolation follow-up.

### Notebook 05c: causal isolation of the boundary error

`notebooks/05c_hayflow_hines_causal_isolation.ipynb` consumes the unchanged
targeted composite and the cryptographically bound 05b artifact. The latter
may be supplied as the original ZIP or as Kaggle's extracted directory: the
ZIP hash is checked when available and all required member hashes are checked
in both forms. It cannot run
the full curriculum. First, it reloads the H2 checkpoint and identifies the
exact transition and segment responsible for the maximum boundary-peak error.
For that transition it exports the teacher boundary, Hines `V_star`, continuous
closure, event jump, final prediction, event localisation, raw event-boundary
delta, and timing-attenuated delta.

The notebook then trains fresh H2 instances on nested 1/8/32/76 transition
sets. It compares the 05b timed/morphology-masked event path against a direct
per-segment boundary residual. Auxiliary biological targets are excluded, so
the experiment answers only whether the event bottleneck or the shared
encoder/optimizer prevents exact voltage memorisation. A final two-transition
experiment isolates the authentic training counterfactual pair with explicit
branching supervision.

05c writes per-transition and per-segment Parquet diagnostics, progressive and
branch histories, micro-checkpoints, a final non-authorizing diagnosis, and an
SHA-256 artifact index. Its result chooses the scope of 05d; it never converts
diagnostic success into permission for full training.

The completed 05c run is registered in
`experiments/hayflow/05c_hayflow_hines_causal_isolation/result.json`. It
classified the failure as `ENCODER_OR_OPTIMIZATION_BOTTLENECK` and retained the
full-training prohibition. The worst transition had no teacher event, and
removing the entire event jump left the 55.35 mV maximum peak error unchanged;
the timed event bottleneck is therefore not the primary cause. Neither path
overfit one transition. The direct residual control was itself numerically
confounded by very large clipped gradients and its bounded random
initialization, so 05c does not yet separate representation from optimization.
The next authorized diagnostic is a zero-initialized residual-conditioning
test, beginning with a free 642-value oracle and a frozen base path.

### Notebook 05d: residual conditioning ladder

`notebooks/05d_hayflow_hines_residual_conditioning.ipynb` is the controlled
follow-up to the registered 05c diagnosis. It consumes the same logical
composite, the exact 05b checkpoint, and the exact 05c artifact. Both external
artifacts are accepted as original ZIPs or Kaggle-extracted directories and
are bound by their preregistered member hashes.

The first gate fits a free boundary residual with one independent parameter
per segment. It uses the quadratic `0.5 * sum(error^2)` objective and unit-step
SGD, whose exact solution is reached in one update, so target magnitude cannot
create a false optimizer failure. It is a target, loss, update, and metric
plumbing check rather than a neural-model result. If it fails, no neural conditioning run occurs and
the diagnostic artifact remains downloadable. If it passes, nine
zero-initialized shared decoders compare linear, scaled-linear, and tanh
parameterizations at three learning rates while the complete 05b base and its
features remain frozen.

The best frozen parameterization then enters an independent unfreezing ladder
on the worst transition and the authentic counterfactual pair. Every stage
restarts from the same 05b checkpoint and zero decoder: head only, local feature
path, then base dynamics. There is no gradient clipping; non-finite loss or
gradient terminates a run, while per-run histories retain gradient norms and
tanh saturation fractions. Passing requires both absolute voltage accuracy
and 0.9--1.1 counterfactual retention. The notebook cannot invoke full
training and its final report only selects the scope of 05e.

The completed 05d run is registered in
`experiments/hayflow/05d_hayflow_hines_residual_conditioning/result.json`.
The free 642-value residual reached numerical precision on both the worst
transition and the authentic branch pair, validating the target, loss, update,
and metric path. All nine frozen shared decoders failed the absolute gates, and
progressively unfreezing local features and base dynamics improved but did not
resolve either memorisation or counterfactual retention. The resulting
`SHARED_REPRESENTATION_BOTTLENECK` diagnosis is deliberately limited to the
current 05b features, shared 97-parameter decoder, and registered optimization
budget; it does not reject the HayFlow architecture family. Full training
remains prohibited. The authorized 05e diagnostic is a closed-form and
segment-conditioned capacity probe.
