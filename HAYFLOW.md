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

### Notebook 05e: closed-form segment-capacity probe

`notebooks/05e_hayflow_hines_segment_capacity_probe.ipynb` consumes the same
logical composite and the exact 05b, 05c, and 05d artifacts. Every upstream
binary is accepted as an original ZIP or Kaggle-extracted directory and is
verified against its registered SHA-256 contract before any probe runs.

05e removes iterative optimization from the experiment. In float64 it solves a
shared linear decoder, a segment-bias-only control, and a shared decoder with
explicit segment biases. It records the design rank, complete singular
spectrum, nonzero condition number, and irreducible least-squares residual.
For the authentic counterfactual pair it then separates the mean residual of
each segment from the branch-dependent residual. Per-segment dynamic
coefficients are solved in closed form and truncated by SVD at preregistered
ranks from 1 through 96. This produces a deployable low-rank factorization of
segment identity by frozen H2 features and reveals the smallest rank that can
meet the unchanged 05d voltage and branching gates.

The segment-bias control may memorize one transition by construction; this is
reported explicitly as static memorization and is never interpreted as causal
branch discrimination. Only the centered counterfactual component measures
whether the frozen features distinguish the two future inputs. 05e cannot run
or authorize full training. Even a passing closed-form probe only authorizes a
fresh zero-initialized segment-conditioned neural micro-canary.

The completed 05e run is registered in
`experiments/hayflow/05e_hayflow_hines_segment_capacity/result.json`. The
shared linear, segment-bias-only, and segment-bias-plus-shared probes all
failed the authentic branch pair. The segment-conditioned path improved
monotonically; rank 64 nearly passed but exceeded the 5 mV maximum-error gate,
while rank 96 reproduced the pair at numerical precision with branching
retention 1.0. This is a positive in-sample capacity result, not a
generalization result: rank 96 is the maximum tested rank, uses 71,490
parameters, and was fitted on two transitions. It establishes that the frozen
features retain pair-discriminating information when coupled to explicit
segment-specific coefficients, but it does not establish a compact surrogate.
Full training remains prohibited. The only authorized follow-up is the 05f
zero-initialized segment-conditioned neural micro-canary with held-out
counterfactual evaluation.

### Notebook 05f: segment-conditioned neural micro-canary

`notebooks/05f_hayflow_hines_segment_micro_canary.ipynb` is the generalization
test authorized by 05e. It consumes the immutable composite, exact 05b H2
checkpoint, and the registered 05c--05e artifacts. The 05e development pair is
excluded from optimization. Multiple counterfactual pairs are selected from
`train`, while held-out pairs come only from the dedicated branching and
release-identifiability test splits. Every pair must have matching complete
boundary state, different causal `U_realized`, distinct episodes, and a
nontrivial teacher separation. Episode overlap across train and held-out roles
is forbidden and the complete pair plan is hashed.

The H2 base and feature extractor remain frozen. Train-only data determine the
feature normalization and the spectral feature bases. Rank-64 and rank-96
segment-conditioned heads then restart independently with zero segment factors
and zero segment biases, so their initial contribution is exactly 0 mV. Only
those factors and biases are optimized. The held-out targets do not influence
normalization, spectral initialization, optimization, checkpoint selection, or
early stopping; they are read only for the final fixed-checkpoint evaluation.

Passing requires every held-out pair to meet the unchanged absolute voltage
and branching gates. Even success authorizes only a separate multistep
micro-rollout experiment. 05f contains no rollout and no full-training path;
failure distinguishes train overfit from an optimization failure and keeps
full training prohibited.

The completed 05f run is registered in
`experiments/hayflow/05f_hayflow_hines_segment_micro_canary/result.json`. Its
pair plan was leakage-free and contract-valid, but both rank-64 and rank-96
heads failed to fit even the eight training pairs: RMSE remained about
15.79 mV and branching retention about 0.236 after 1,200 epochs. Held-out
absolute predictions then diverged into millions of millivolts. The eight
training pairs were episode-independent but all came from the same targeted-BAP
protocol. More importantly, each 96-feature local design had rank only 13--15;
the unregularized per-segment pseudoinverses produced a coefficient spectrum of
order 1e8--1e9. The resulting train-only spectral basis was therefore severely
underdetermined and unsafe for extrapolation.

The registered diagnosis is
`SEGMENT_CONDITIONED_MICRO_CANARY_OPTIMIZATION_FAILURE`, not a rejection of the
segment-conditioned architecture family. Full training remains prohibited.
The authorized 05g diagnostic must audit projected-feature scales, replace the
unregularized spectral basis with a regularized bounded construction, add
multi-pair oracle controls, and determine whether protocol-diverse training
pairs are available before any further held-out claim.

### Notebook 05g: regularized optimization audit

`notebooks/05g_hayflow_hines_optimization_audit.ipynb` is the controlled
follow-up to the 05f optimization failure. It consumes the immutable composite,
the exact 05b--05e upstream artifacts, and the member-hashed 05f diagnostic.
It searches farther through `train` for independent counterfactual support,
excludes the registered development pair, and reports explicitly whether more
than one protocol family is actually available instead of assuming diversity.

The H2 base and boundary-feature extractor remain frozen. Feature location and
scale are fitted only on train inputs, standardized values are clipped to a
declared range, and raw plus standardized norms are audited for train,
development, and held-out inputs. Boundary-voltage targets for held-out pairs
remain sealed during this stage. A direct per-transition residual oracle first
checks multi-pair target and metric plumbing; a segment-bias control separately
measures how much can be explained by static memorization.

The actual audit uses float64 dual ridge regression, which is appropriate for
the small number of examples relative to 96 local features, across a fixed
regularization path. Every coefficient field is independently truncated to
rank 64 and rank 96, with the centering correction folded into its segment
intercept. Predictions are bounded to +/-120 mV and candidates are rejected for
non-finite scales, excessive coefficient norm, any boundary clipping, or
failure of the unchanged absolute voltage and branching gates on train and the
pre-registered development pair.

Only a candidate that passes all of those gates may cause held-out boundary
voltages to be loaded. The held-out set is then evaluated once; it is never used
for normalization, regularization selection, coefficient fitting, or safety
thresholds. 05g contains neither rollout nor a full-training path. Its output
is therefore a diagnostic decision about optimization, representation, and
generalization, and full HayFlow training remains prohibited regardless of the
outcome.

The completed 05g run is registered in
`experiments/hayflow/05g_hayflow_hines_optimization_audit/result.json`. It
found 776 valid train counterfactual candidates and selected 12 independent
pairs spanning all six available protocol-family combinations, so the 05f
protocol-homogeneity concern was resolved. The direct residual oracle reproduced
all train targets exactly, whereas a segment-bias-only control failed with
11.83 mV RMSE and branching retention 0.254. Target and metric plumbing are
therefore working and the multi-pair task is not reducible to static segment
memorization.

None of the 16 float64 ridge candidates passed the train gates. The best train
fit used rank 96 and ridge lambda 1e-8, reaching 9.49 mV RMSE, 72.81 mV maximum
segment error, and branching retention 0.689; its coefficient Frobenius norm
was also unsafe at about 785,024. Numerically safer regularization reduced the
coefficient norm but made the already insufficient fit worse. Ranks 64 and 96
were nearly indistinguishable on the safe part of the path, while local design
ranks remained only 13--23. Thus the 05f failure is not attributable to Adam
alone: a bounded linear segment-conditioned residual cannot fit the diverse
train support using the exact frozen H2 features. This conclusion is scoped to
that frozen linear representation and does not reject nonlinear heads,
controlled feature adaptation, causal feature augmentation, or HayFlow as a
whole.

No candidate reached the held-out reveal gate, so held-out boundary-voltage
targets remained sealed. A separate raw-scale caveat was discovered: although
the implemented post-clipping gate reported `scale_safe=true`, the maximum raw
held-out feature was about 32.2 million versus 1,215 on train, and the maximum
raw segment-feature norm was about 29,172 times the train maximum. Clipping to
+/-8 concealed this excursion. It does not change the train-fit diagnosis, but
it is an explicit OOD blocker for the next experiment. The authorized 05h must
perform pre-clipping scale and projection-residual forensics, then compare
bounded nonlinear or tightly adapted causal representations on train and
development only. Rollout and full training remain prohibited.

### Notebook 05h: representation and raw-scale forensics

`notebooks/05h_hayflow_hines_representation_forensics.ipynb` implements the
forensic follow-up authorized by 05g. It consumes and member-verifies the exact
05g artifact, then reuses its hashed 12-pair, six-family train support without
performing a new selection. The registered development pair remains separate.
The frozen H2 checkpoint is evaluated on held-out inputs only to extract the
feature surfaces required by the OOD audit. Held-out boundary-voltage targets
and held-out event labels are never materialized, and no newly fitted candidate
head is run on held-out inputs for predictive evaluation. The shared batch
loader exposes an explicit input-only mode for this purpose.

The first stage measures three surfaces before clipping: frozen H2 boundary
features, the same H2 path with all causal synaptic/current inputs set to zero,
and the direct causal input tensor. Train-only centers and scales are used to
report unbounded standardized excursions, clipping fractions, raw norm ratios,
and the exact logical index, segment, and feature of the largest outliers. The
direct causal tensor includes both authentic synaptic state already present at
S_t and realized events in the next millisecond. Normalized teacher-state,
initial voltage, H2 boundary voltage, and zero-causal boundary voltage are
reported separately. The zero-causal counterfactual therefore distinguishes
normalizer/state OOD, frozen state-path amplification, causal-front-end OOD,
and causal-drive amplification inside H2.

The second stage computes an unrestricted, segment-local linear projection
oracle on train only. Per-segment design rank, condition number, irreducible
projection error, coefficient norm, region, and morphology location are saved.
This determines whether the 05g ridge failure is merely a safety/regularization
tradeoff or whether the frozen feature surface lacks a linear direction needed
by the teacher residual.

Finally, three compact zero-output nonlinear controls compare bounded H2,
bounded direct causal features, and their concatenation. Each uses a shared
two-layer local head plus a small segment embedding and a +/-120 mV output
bound. Three fixed seeds are trained on the exact train support; checkpoint
selection and early stopping use only the registered development pair. H2 is
never updated. Regardless of the result, 05h cannot reveal held-out targets,
perform rollout, or authorize full training. A raw held-out OOD finding takes
precedence over apparent train/development success and requires a separate
scale-repair experiment.

The completed 05h run is registered in
`experiments/hayflow/05h_hayflow_hines_representation_forensics/result.json`.
Its artifact and all 36 indexed members passed hash and size verification. The
input-only held-out contract also held: future boundary voltages and event
labels were never materialized, and the fitted candidate heads were never run
on held-out examples.

The dominant OOD source was localized to the normalized full teacher state,
not to the causal synaptic frontend. The maximum normalized teacher-state value
was 8.13 on train, 4.89 on development, and about 6.97e8 on held-out input
states. The causal-input maximum norm was actually smaller on held-out than on
train (ratio 0.684), whereas H2 features still diverged when all causal inputs
were zeroed (maximum-norm ratio about 3.01e6). A post-hoc audit of the saved
normalizer found that 11,888 of 17,220 coordinates had been assigned the
minimum scale 1e-8 because they were constant in the train normalization
sample. Counterfactual held-out states activate some of those coordinates,
creating the order-1e8 transformed values before H2. The physical boundary
voltage remained bounded, but the internal H2 surface did not.

The train-only unrestricted projection oracle also refines the 05g diagnosis.
It interpolated all 12 pairs at 0.00257 mV aggregate RMSE with maximum error
0.220 mV and branching retention approximately 1.0. Thus a linear direction is
present on the 24 train transitions. The fit is not usable evidence of
generalization: local ranks are only 20--24 for 96 features, median condition
number is about 9.44e7, the maximum is 4.49e9, and coefficient norms reach
8.31e8. The 05g result is therefore a regularization-versus-ill-conditioning
tradeoff, rather than proof that frozen H2 contains no train-discriminating
direction.

None of the nine bounded nonlinear controls passed train or development. The
median train/development RMSE values were respectively 10.85/16.04 mV for H2,
11.29/16.82 mV for causal-only, and 10.82/15.58 mV for H2 plus causal inputs.
The small improvement of the combined input remains far outside the absolute
gates and does not authorize a larger training run. The next experiment is
therefore 05i state-normalization repair: it must identify the exact offending
schema coordinates and introduce semantic, transform-aware scale floors before
any candidate head is retrained. Held-out future targets, rollout, and full
training remain prohibited.

### Notebook 05i: teacher-state normalization repair

`notebooks/05i_teacher_state_normalization_repair.ipynb` implements the scoped
repair required by 05h. It preserves the teacher state, state centers, semantic
transforms, delta normalization, dataset, support plan, and frozen H2 weights.
Only the scale used to encode the current teacher state is replaced. The fit is
strictly train-only: scales with usable train variation are pooled
hierarchically by exact `(category, mechanism, variable, transform)`, then by
mechanism, category, and transform family. Each pool contributes a
pre-registered fraction of its lower-quartile scale, with a final absolute
floor determined by the semantic transform. Development and input-only
held-out states cannot influence any fitted quantity.

The notebook emits a row for every one of the 17,220 state coordinates. Each
row records category, scope, owner/segment, mechanism, variable, transform,
original and repaired scales, floor source and support count, plus raw,
transformed, and pre-clipping standardized support for fit-train, audit-train,
development, and held-out inputs. Group and top-outlier Parquet tables make the
repair auditable rather than collapsing it into a single maximum. No clipping
is applied to obtain the reported support metrics.

After the coordinate contract is measured, the registered H2 checkpoint is
run once with repaired state inputs and once with all causal inputs zeroed.
H2 remains frozen. This checks both direct hidden-feature support and whether a
state-path excursion remains without the synaptic frontend. Held-out future
voltages and event labels are never loaded, and candidate heads are never
trained or evaluated. Passing 05i can authorize only a separate 05j
train/development representation recheck; it cannot authorize rollout or full
training. A failed input contract instead requires a new, explicitly registered
semantic scale policy and must not be patched post hoc in the same run.

The completed 05i run is registered in
`experiments/hayflow/05i_teacher_state_normalization_repair/result.json`. Its
archive and all 17 indexed members passed hash and size verification. The
train-only repair lifted 12,507 coordinates and reduced the held-out maximum
standardized teacher-state value from about `6.97e8` to `113.73`, a factor of
about `6.13e6`. No raw, transformed, or standardized value was nonfinite. The
global held-out fraction above `|z|=8` fell to `0.0813%`, below the registered
`1%` gate, but the absolute `|z| <= 100` gate still failed.

The remaining failure is narrow and semantically localized. Exactly two
coordinates exceed 100: `NetCon.weight[4]` for inhibitory `ProbUDFsyn2`
synapses 767 and 883, both carrying the absolute last-event timestamp `tsyn`.
The raw label `weight[4]` is ambiguous: in `ProbAMPANMDA2` it denotes the
bounded release probability `Pr`, while in `ProbUDFsyn2` it denotes `tsyn` in
milliseconds. The first repair pooled these physically different variables by
raw slot name. Raising the pooling multiplier or relaxing the threshold would
hide this schema defect rather than repair it.

Importantly, the repaired frozen-H2 contract already passes. The held-out/train
maximum-norm ratio is 0.331 with authentic causal input and 0.698 with causal
input zeroed; maximum train-standardized held-out features are 18.86 and 14.09
respectively, with no nonfinite values and bounded physical voltages. The
catastrophic H2 OOD excursion identified in 05h is therefore removed. The next
authorized activity is a separate 05i-b semantic NetCon repair: decode slots by
point-process class and represent `tsyn` causally relative to the current
boundary time (or through registered recovery variables). The raw teacher
snapshot must remain unchanged, thresholds must remain fixed, and no head
training or rollout is authorized yet.

### Notebook 05i-b: class-aware NetCon semantic state repair

`notebooks/05i_b_netcon_semantic_state_repair.ipynb` implements the narrow
semantic correction localized by 05i. It resolves every raw NetCon
`weight[index]` through `KNOWN_NET_RECEIVE_STATE_LAYOUT` and the owning
synapse's point-process class. For `ProbAMPANMDA2`, slots 1--6 become
`weight_AMPA`, `weight_NMDA`, `Pv`, `Pr`, `u`, and `tsyn`; for
`ProbUDFsyn2`, slots 1--4 become `Pv`, `Pr`, `u`, and `tsyn`. Thus identical
raw indices can no longer pool physically different quantities.

Probability slots use the registered bounded logit transform. AMPA/NMDA
amplitude weights remain nonnegative. The absolute `tsyn` timestamp is replaced
only in the model-facing state view by causal last-event age,
`age_ms = boundary_time_ms - tsyn_ms`, where `boundary_time_ms` comes from the
transition's stored `start_time_ms` metadata. The inverse is explicit:
`tsyn_ms = boundary_time_ms - age_ms`. The notebook checks this round trip on
fit-train, audit-train, development, and input-only held-out states at an
absolute tolerance of `1e-9`; the raw teacher snapshot itself is never edited.

The original 05i gates remain unchanged. After the semantic round-trip audit,
05i-b repeats the full 17,220-coordinate pre-clipping support audit and the
frozen-H2 authentic/zero-causal checks. It verifies the exact 05i artifact and
all prior provenance. Held-out future voltages and event labels remain sealed;
there is no candidate-head training, rollout, threshold relaxation, or global
scale-multiplier shortcut. A complete pass can authorize only the separate 05j
train/development representation recheck, never full training directly.

The completed 05i-b run is registered in
`experiments/hayflow/05i_b_netcon_semantic_state_repair/result.json`. The
downloaded archive and all 19 indexed members passed independent hash and size
verification. The semantic mapping itself passed completely: all 6,390 NetCon
coordinates were decoded by point-process class, no slots were unmapped, and
the boundary-relative `tsyn` transform reconstructed the raw timestamps with
maximum absolute error `0.0`. The frozen-H2 authentic and zero-causal audits
also passed with finite features and bounded voltages.

The overall input contract did not pass. The input-only held-out teacher state
reached `|z| = 223.622915` against the unchanged limit of 100, although only
`0.0813%` of values exceeded the clipping diagnostic threshold of 8. Exactly
seven coordinates exceeded 100. Four are correctly decoded `tsyn` ages: their
fit-train support is dominated by old ages around 1061--1140 ms, whereas the
held-out inputs contain recent events at ages 2.35--3.75 ms. The other three
are authentic `ProbAMPANMDA2` AMPA/NMDA trace states that are exactly zero in
the selected fit-train support but active in held-out inputs. Consequently,
05i-b resolves the raw-slot collision but demonstrates a remaining
inactive/active and recency representation problem.

The next authorized activity is a separate 05i-c input-representation
revision. It must preregister a bounded, causal recency/recovery coordinate for
`tsyn` and domain-calibrated scaling for nonnegative synaptic trace states.
The teacher snapshot, fixed thresholds, train-only policy and sealed held-out
future targets must remain unchanged. Relaxing the gate, training a candidate
head, or starting rollout is not authorized by this result.

### Notebook 05i-c: bounded recency and synaptic-domain repair

`notebooks/05i_c_synaptic_domain_repair.ipynb` implements the separate
representation revision authorized by the failed 05i-b input gate. It does
not reinterpret the raw NetCon layout again. Instead it replaces model-facing
last-event age by the bounded causal coordinate
`recency = tau / (tau + age)`. For each synapse, `tau` is the largest of the
registered 1 ms minimum and its positive authentic `Dep` and `Fac`
parameters. The inverse reconstructs age and `tsyn` at the current boundary;
the raw teacher snapshot remains unchanged.

Because recency lies in `(0, 1]`, its scale floor is preregistered as `1/50`.
The largest possible full-domain displacement is therefore 50 standardized
units, leaving margin below the unchanged limit of 100 independently of the
held-out examples. Dynamic double-exponential point-process states
`A_AMPA`, `B_AMPA`, `A_NMDA`, `B_NMDA`, `A`, and `B` retain `log1p`; their
scale floor is `log1p(1)/35`, derived from a unit release increment. These
domain floors supplement rather than replace the 05i train-only hierarchical
repair and are fixed before the held-out input audit.

05i-c verifies the exact 05i-b artifact, repeats the reversible recency audit,
the complete 17,220-coordinate pre-clipping support audit, an explicit domain
floor audit and the frozen-H2 authentic/zero-causal audit. Thresholds are
compared byte-for-value with 05i-b and cannot be relaxed. Held-out future
voltages and event labels remain sealed; no candidate head, rollout or full
training path exists. A complete pass can authorize only a separate 05j
train/development representation recheck.

The completed 05i-c run is registered in
`experiments/hayflow/05i_c_synaptic_domain_repair/result.json`. The downloaded
archive and all 20 indexed members passed independent SHA-256 and size
verification. The bounded-recency transform reconstructed the raw timestamps
with maximum absolute error `0.0`; all 1,278 recency coordinates and all 3,834
dynamic synaptic traces satisfied their preregistered domain-floor contracts.

The complete input contract passed without changing any threshold. The
input-only held-out maximum standardized teacher-state value fell from
`223.622915` in 05i-b to `39.813518`, below the fixed limit of 100. Bounded
recency reached at most `14.879954` and dynamic traces at most `39.813518`.
The global held-out fraction beyond the diagnostic `|z|=8` threshold remained
`0.0813%`, below the fixed 1% gate, and no coordinate remained above 100.

The frozen-H2 contract passed with authentic and zeroed causal inputs. Their
held-out/train maximum-norm ratios were `0.331327` and `0.698655`; maximum
standardized H2 values were `18.821954` and `14.344117`. Physical voltage was
bounded at `83.379829 mV` absolute maximum and no nonfinite value was observed.
Held-out future targets, candidate heads and rollout remained sealed.

05i-c therefore authorizes the separate
`05j_repaired_representation_train_development_recheck`. It does not authorize
full training directly: 05j must recheck the repaired representation using
train/development only before any candidate-head path is reopened.

### Notebook 05j: repaired representation train/development recheck

`notebooks/05j_repaired_representation_train_development_recheck.ipynb`
reopens only the compact diagnostic head path after the complete 05i-c input
contract pass. The exact 12 counterfactual train pairs and the separate single
development pair registered in 05g are reused, with episode-disjointness
checked again. The repaired 05i-c normalizer is reconstructed solely from its
train-only rules and must reproduce the fingerprint contained in the exact
verified 05i-c artifact.

05j deliberately does not extract held-out inputs at all. It materializes
voltage targets only for train and development and does not request event
targets. Frozen-H2, causal-only and combined H2-plus-causal bounded heads use
the same architecture, seeds and pair gates as the original 05h controls.
Heads train on train only; development selects the checkpoint. A feature
family is considered robust only if at least two of the three registered seeds
pass both roles. The unrestricted local projection remains a train-only
diagnostic rather than a deployable candidate.

Passing 05j can authorize only a separate 05k repaired-representation
micro-rollout. It cannot authorize full training, test evaluation or held-out
reveal. Failure is interpreted as a scoped compact-head/representation result;
it does not revoke the successful 05i-c numerical input contract.

The completed 05j run is registered in
`experiments/hayflow/05j_repaired_representation_recheck/result.json`. The
archive and all 38 indexed artifacts passed independent integrity checks. The
05i-c normalizer fingerprint was reproduced exactly, train and development
episodes were disjoint, normalization used train only, and no held-out input
or target was accessed.

The representation recheck failed. Across H2, causal-only and combined
H2-plus-causal families, zero of three seeds passed jointly on train and
development; the robust gate required at least two. Median development RMSE
remained `15.6233--16.8328 mV`, development branching retention only
`0.2960--0.4049`, and development maximum segment errors
`69.20--76.89 mV`. All nine runs also failed the train pair gates.

An unrestricted train-only linear projection interpolated the tiny support
with maximum per-segment RMSE `0.0636 mV`, but its design condition number
reached `4.60e9` and coefficient norms were correspondingly extreme. This is
evidence of algebraic information on the sampled surface, not a stable
learnable or deployable mapping, and cannot supersede the failed compact-head
gate. The valid 05i-c input contract remains intact, while 05k rollout and full
training remain prohibited. The next authorized experiment is the separate
`05j_b_repaired_representation_revision`.

### Notebook 05j-b: repaired representation revision

`notebooks/05j_b_repaired_representation_revision.ipynb` investigates the
specific discrepancy left by 05j: an unrestricted local projection can
interpolate the tiny train support, but all compact shared heads fail even on
train. It verifies the exact 05j artifact and reuses the exact 12 train pairs
and one episode-disjoint development pair. The repaired 05i-c normalizer is
reconstructed again from train-only quantities. No held-out input or target is
read, and neither rollout nor full training is present.

05j-b separates feature geometry from decoder sharing. The unchanged
`tanh(z/4)` representation is compared with the monotone tail-preserving map
`asinh(z)/asinh(8)`. For each transform it fits segment-specific affine ridge
decoders for H2, causal-only and H2-plus-causal inputs. This removes the shared
MLP and low-dimensional segment embedding as confounders without pretending
that 642 independent decoders are the final HayFlow architecture.

The residual target uses an invertible bounded coordinate with the unchanged
120 mV physical limit. Ridge values are selected using leave-one-pair-out on
train only; the development pair is excluded from hyperparameter selection and
evaluated once after fitting the selected candidate on all train pairs. Each
ridge fit uses both pointwise observations and paired-future difference rows
with the same unit branch weighting used in 05j. The
original RMSE, maximum-error and branching-retention gates remain unchanged,
and candidates must also satisfy explicit condition-number and coefficient-norm
limits.

If only `asinh` candidates pass, the saturating feature map is localized as the
primary blocker. If both transform families pass, decoder sharing/capacity is
localized instead. If train fits but development fails, the evidence points to
support/generalization instability. A complete pass authorizes only the
separate 05k micro-rollout, never full training; failure routes to 05j-c.

The completed 05j-b run is registered in
`experiments/hayflow/05j_b_repaired_representation_revision/result.json`. The
archive and all 21 indexed members passed integrity verification. The exact
05i-c normalizer fingerprint, train-only fitting, episode-disjoint development
pair and sealed held-out contract were all preserved.

All six candidates failed the unchanged gates, including on train. The best
candidate was segment-local H2 with `asinh`: train RMSE `9.4567 mV`,
development RMSE `12.9842 mV`, maximum development segment error `52.6778 mV`
and branching retention `0.4971`. Its `tanh` control was nearly identical at
`9.4938/13.0637 mV` and retention `0.4954`; therefore tail saturation is not
the principal explanation. Causal-only and combined surfaces remained worse.

Every selected ridge solution was numerically inside the preregistered
condition-number and coefficient-norm limits. The geometry audit instead found
median paired-future local H2 distances of only `2.17e-7` (`tanh`) and
`2.80e-7` (`asinh`), while causal-local features reached `7.63e-8` and median
rank three. This points to a missing non-local/spatial information path for a
segment-local decoder, not simple optimizer instability. The scoped next step
is `05j_c_support_and_decoder_revision`; 05k rollout and full training remain
prohibited.

### Notebook 05j-c: spatial context and support revision

`notebooks/05j_c_spatial_support_revision.ipynb` tests the two scoped factors
left by 05j-b: non-local morphology context and train-support size. It verifies
the exact 05j-b artifact, preserves the original 12 train pairs and selects 36
additional episode-disjoint train pairs round-robin across the six available
protocol families. The resulting 48-pair support remains train-only and excludes
the development episode; held-out inputs and targets remain sealed.

Frozen H2 and causal channels are reduced to deterministic train-only PCA
sketches. Local features are compared against fixed symmetric axial-neighbour
diffusion at 0, 1, 2, 4, 8, 16 and 32 tree steps, with a third context adding
causal region summaries broadcast across the morphology. Crossing these three
contexts with original and expanded support produces six controlled candidates.

Every candidate retains the 120 mV bounded target, unit branching weight,
segment-specific ridge decoder and numerical-stability limits of 05j-b. Ridge
selection uses six-fold grouped pair cross-validation on train only; development
is evaluated after selection and cannot affect the chosen hyperparameter. The
original RMSE, maximum-error and branching-retention gates are unchanged.

A preregistered 20% improvement threshold separates topology and support
effects, and must be met on both train cross-validation and development; a
gain on only one role is reported as inconclusive. A complete pass can authorize
only 05k micro-rollout. A material but
incomplete topology improvement can authorize only a separate trainable
topology-decoder micro-canary. Neither outcome directly authorizes full
training.

The completed 05j-c run is registered in
`experiments/hayflow/05j_c_spatial_support_revision/result.json`. The archive
and all 22 indexed members passed integrity verification. The original 12
pairs were preserved and expanded to 48 episode-disjoint train pairs spanning
all six available protocol families, with no development overlap and no
held-out access.

Both preregistered factors produced material, role-consistent gains. Expanded
support improved RMSE by `58.87%` in grouped train cross-validation and
`64.46%` on development. Non-local morphology context improved the same roles
by `58.57%` and `84.89%`; all values exceed the fixed 20% threshold.

No candidate passed every original pair gate. The expanded multiscale-tree
candidate nevertheless reduced cross-validation/train/development RMSE to
`2.3932/1.4017/2.6936 mV` and recovered development branching retention to
`0.9739`. Its remaining blockers were voltage accuracy and worst-segment error:
development maximum error was `16.9684 mV`, and only 14/48 cross-validation
pairs passed jointly. The tree-plus-global variant reached `2.6397 mV` and
`14.3959 mV` on development but generalized less robustly across train folds.

05j-c therefore identifies topology and support as necessary rather than
sufficient. It authorizes only
`05j_d_trainable_topology_decoder_micro_canary`, using the expanded support and
fixed tree context. 05k rollout and full training remain prohibited.

### Notebook 05j-d: trainable topology decoder micro-canary

`notebooks/05j_d_trainable_topology_decoder_micro_canary.ipynb` implements the
narrow follow-up authorized by 05j-c. It verifies the exact registered 05j-c
artifact, reconstructs the same 48 train-only pairs and makes a deterministic,
family-stratified, episode-disjoint split into 36 fit pairs and 12 internal
calibration pairs. The verified 05i-c synaptic normalizer remains frozen, while
every newly introduced topology-design normalizer and PCA sketch is fit on the
36 fit pairs only.

The notebook freezes the multiscale axial-tree representation and first fits a
grouped-CV ridge reference without using calibration or development for ridge
selection. It then compares two small nonlinear shared decoders over three
seeds: a direct bounded topology head and a bounded correction initialized
exactly at the ridge prediction. Their loss combines pointwise voltage,
paired-future differences and a worst-segment tail term.

Only the internal-calibration subset selects an epoch. Development inference
occurs after each checkpoint is frozen and cannot affect model selection. A
family must pass the unchanged pairwise RMSE, maximum-error and branching gates
on fit, calibration and development for at least two of three seeds. Even a
robust pass authorizes only a separate 05k micro-rollout. Held-out inputs,
rollout and full training remain sealed.

The completed 05j-d run is registered in
`experiments/hayflow/05j_d_trainable_topology_decoder_micro_canary/result.json`.
The archive and all 35 indexed members passed integrity verification, and the
fit/calibration/development separation remained valid. No model family passed
the absolute gate, so neither 05k nor full training is authorized.

The ridge-corrected decoder did not improve its same-split ridge reference.
The direct decoder did show a reproducible secondary signal: its median
calibration and development RMSE improved by `21.84%` and `28.74%` relative to
that reference, reaching `1.5814 mV` and `2.9638 mV`, with development
branching retention near one. However, worst-segment development errors stayed
between `18.29` and `19.56 mV`; all three seeds failed the original `1 mV`
RMSE and `5 mV` maximum-error limits. This is not a gate pass and does not
outperform the stronger all-48-pair fixed-tree diagnostic from 05j-c.

The registered route is therefore `05j_e_architecture_reassessment`. That
reassessment must explain both facts: nonlinear topology decoding contains
real signal, while the present learning target/support contract remains far
from the required segment-level accuracy.

### Notebook 05j-e: architecture reassessment

`notebooks/05j_e_architecture_reassessment.ipynb` is a frozen-checkpoint
forensic experiment. It verifies the exact 05j-d artifact, reconstructs the
registered 36/12/development design and reproduces all six saved model metrics
within a fixed numerical tolerance. It does not train or select a new model.

The direct-tree seed ensemble is analyzed by segment, region and morphology
frequency. Seed disagreement distinguishes optimization variance from a shared
systematic error; top-segment energy identifies spatial concentration. A
fit-only per-segment affine oracle tests whether static offsets or gains could
repair the result, but is explicitly prohibited from authorizing a candidate.
A paired-future audit then compares multiscale feature distance with teacher
branch separation to detect remaining representational collisions.

The resulting causal diagnosis selects the narrow 05j-f branch. Development
is diagnostic only and fits no transform or checkpoint. Held-out data, rollout,
05k and full training remain sealed.

The completed 05j-e run is registered in
`experiments/hayflow/05j_e_architecture_reassessment/result.json`. All six
frozen checkpoints reproduced their 05j-d metrics within `1e-4`, and all 31
indexed artifact members passed integrity verification.

The dominant failure is spatially localized and tied to a regenerative mixed
BAP/calcium regime. The top 10% of segments account for `73.40%` of development
error energy, led by segments 274, 273, 272, 305, 560 and nearby segments.
`other` and `apical_trunk` contribute `65.49%` and `18.56%` of that energy,
whereas tuft RMSE is only `0.5728 mV`. Branching retention remains `0.9943`, so
the model distinguishes the paired futures while missing their absolute local
voltage in the affected morphology.

Simple alternatives were ruled out. A fit-only per-segment affine oracle
improves diagnostic RMSE by only about 1% and worsens maximum errors. No large
teacher branch separation appears among nearly identical feature pairs, and
predicted-versus-teacher branch amplitudes correlate above `0.984`. The error
is also spatially smooth rather than a high-frequency morphology artifact.

The registered next step is therefore
`05j_f_region_mechanism_expert_revision`: a scoped region/mechanism expert
revision aimed at regenerative trunk and adjacent compartments. This does not
authorize 05k rollout or full training.

### Notebook 05j-f: region/mechanism expert revision

`notebooks/05j_f_region_mechanism_expert_revision.ipynb` tests whether the
localized 05j-e failure requires biological specialization rather than merely
more decoder parameters. For each seed it freezes the registered direct-tree
prediction and trains a zero-initialized bounded correction.

The factorial control holds capacity constant. Both families contain eight
identical expert MLPs. The uniform control averages all experts at all segments;
the structured candidate gates them using only canonical region and mechanism
metadata: general, apical trunk, basal, tuft, soma/axon, calcium regenerative,
sodium regenerative and repolarization/Ih. No target or development error map
is used to construct these masks.

The 36/12/development split, voltage/branch/tail loss and three seeds remain
unchanged. Calibration alone chooses checkpoints. A robust absolute pass can
authorize only 05k micro-rollout. A weaker signal must improve each seed over
its own frozen baseline and outperform the parameter-matched uniform control
on both calibration and development. Held-out data, rollout and full training
remain sealed.

## Phase 06: explicit-state architecture redesign

The verified 05t consolidated result closes the fixed-boundary-state GraphGRU
branch. Semantically aligned state improved the 8 ms rollout, but both frozen
candidates failed the preregistered 16--32 ms stability gate. The result is not
evidence that teacher state is useless: 05q--05s established that its identity
is predictive, localized most of the useful signal to mechanism STATE
variables and measured a robust benefit from semantic alignment. The failure
is that this information is injected only at the first boundary rather than
updated as part of the autoregressive state.

Phase 06 introduces the `HayFlow-ESI` explicit-state integrator. Its canonical
state is `(voltage, mechanism STATE, ions, synaptic state)` at every 1 ms
boundary. The intended full model separates causal synaptic kinetics, shared
local mechanism-state evolution, privileged membrane-current supervision and
an authentic morphology-aware implicit/Hines voltage solve. No global latent,
morphology reduction or aggressive state compression is introduced before
the explicit-state baseline is learnable.

`notebooks/06a_atomic_state_dynamics_playground.ipynb` implements the first
bounded question. It uses only original train episodes and creates
seed/snapshot-disjoint fit, calibration and development roles. A
zero-initialized semantic residual updater predicts each transformed
mechanism-state delta from its current value, causal local ion concentrations,
realized input and voltage. Two parameter-identical arms compare causal start
voltage with a diagnostic-only teacher interval-voltage oracle. Recursive
mechanism-state rollouts are measured at 1, 2, 4 and 8 ms while membrane
voltage remains teacher-forced.

06a is a single-seed technical pilot and performs no candidate selection. A
failure even with teacher interval voltage directs work back to the state
contract, normalization or missing local context. Success only in that arm
directs 06b toward explicit voltage/state coupling. Success in the causal arm
authorizes a multi-seed explicit-state updater canary. Validation, tests,
fresh-test generation, full training and mass data remain prohibited.

The completed 06a run is registered in
`experiments/hayflow/06a_atomic_state_dynamics_playground/result.json`. All ten
indexed members passed integrity verification, and only train-derived state
and outcomes were read. The causal and teacher interval-voltage arms improved
one-step normalized-delta RMSE over persistence by `0.89%` and `1.41%`, below
the preregistered `2%` technical gate. The registered diagnosis is therefore
`ATOMIC_STATE_UPDATE_NOT_YET_LEARNABLE`; it is not a pass and cannot authorize
the multi-seed 06b canary.

The result narrows the next question. Supplying the teacher endpoint voltage
adds only about `0.52` percentage points, while neither arm observes the
intra-ms voltage path that determines voltage-gated kinetics. Calibration was
still improving at step 300. Recursive state gains were positive at several
horizons, but each horizon used independently selected windows and therefore
does not form a comparable scaling curve. One bounded train-only
`06a_b_atomic_voltage_path_identifiability` forensic is authorized to separate
optimization budget from fixed voltage-path information on nested windows.
Model-capacity growth, validation access and full-neuron composition remain
prohibited.

### 06a-b atomic voltage-path identifiability

`notebooks/06a_b_atomic_voltage_path_identifiability.ipynb` implements the
bounded forensic authorized by the registered 06a failure. It retains the
exact train-only role partition, paired seed and optimizer, and crosses a
300/1200-step checkpoint budget with two equally wide privileged voltage
representations: linear interpolation between teacher boundary voltages and
the authentic voltage microtrace sampled every 0.125 ms. The hidden width is
automatically reduced as necessary so both arms share one parameter count no
larger than the 7,238-parameter 06a updater.

The one-step primary metric remains improvement over persistence. A 2% absolute
gain and a preregistered one-percentage-point paired factor effect are required
to identify optimization or voltage-path information. Semantic-macro and
active-coordinate scores are secondary. Recursive 1/2/4/8 ms measurements now
use prefixes of the same deterministically selected 8 ms windows. All voltage
microtraces are diagnostic teacher information; no deployment claim is made.
Validation/test access, capacity sweeps, full-neuron training, fresh tests and
mass data remain prohibited.

The completed 06a-b artifact is registered in
`experiments/hayflow/06a_b_atomic_voltage_path_identifiability/result.json`.
All 13 indexed members passed integrity verification. Raising the optimizer
budget from 300 to 1200 steps increased one-step gain over persistence from
about `2.8%` to `16.1%` in both equally sized arms. The paired budget effects
were `13.31` and `13.08` percentage points, whereas the authentic teacher path
was `0.16` points worse than linear endpoint interpolation. The preregistered
diagnosis is therefore `ATOMIC_STATE_WAS_OPTIMIZATION_LIMITED`, not a voltage
microtrajectory information deficit.

The long-budget result remains positive under semantic-macro (`7.44%`) and
active-coordinate (`16.60%`) summaries. Common-window recursive gains reached
`14.1/16.9/17.4/21.1%` at 1/2/4/8 ms for the linear endpoint arm, without
non-finite or state-domain violations. Neither calibration curve had plateaued
at 1200 steps.

Both 06a-b voltage arms are privileged because they know the teacher endpoint.
Consequently this result does not yet establish a deployment-compatible causal
state updater. The authorized 06b canary must restore `causal_start_voltage` as
its primary arm, retain the endpoint representation only as a paired reference,
use multiple optimization seeds, preserve common nested windows and report
semantic-group robustness. Full-neuron training and held-out access remain
unauthorized.

### 06b optimized explicit-state updater canary

`notebooks/06b_optimized_explicit_state_updater_canary.ipynb` tests the causal
gap left by 06a-b. Three paired optimization seeds train a deployment-compatible
`causal_start_voltage` updater and an equally sized privileged endpoint
reference for 1200 steps. The primary arm reads no teacher endpoint or voltage
microtrace. All state/outcome roles remain train-derived and disjoint.

The preregistered component gate requires every causal seed to beat persistence
by at least 2%, a 10% median gain, 3% semantic-macro gain, 10% active-coordinate
gain, positive improvement in at least 70% of semantic groups, 70% retention
relative to the endpoint reference, positive rollout gain for every seed and
horizon, and at least 10% median gain at 8 ms. All runs use the same nested
development windows. A pass authorizes only a coupled voltage/state
micro-canary; full-neuron training, held-out access and mass data remain
prohibited.

The completed 06b artifact is registered in
`experiments/hayflow/06b_optimized_explicit_state_updater_canary/result.json`.
All 19 indexed members passed integrity verification. The causal updater
learned consistently across all three seeds (`7.64--7.93%` one-step gain),
improved `94.4%` of semantic groups and reached `13.10--15.66%` gain at 8 ms
without numerical or domain violations. This is robust evidence that causal
boundary information contains useful mechanism-state dynamics.

The preregistered component gate nevertheless failed: median one-step gain was
`7.85%` versus `10%`, active-coordinate gain was `8.20%` versus `10%`, and
retention relative to the robust endpoint reference was `44.3%` versus `70%`.
The decision-grade component diagnosis is
`ATOMIC_STATE_REQUIRES_EXPLICIT_VOLTAGE_COUPLING`. The missing endpoint is a
material causal variable rather than seed noise or an aggregate-only effect.

The authorized next experiment is one bounded train-only
`06b_b_causal_voltage_state_coupling_forensic`. It must reuse frozen 06b
checkpoints as controls, separate coupling information from additional capacity
or optimization, and preserve paired seeds and common nested rollout windows.
It cannot yet train the full neuron or access held-out state.
