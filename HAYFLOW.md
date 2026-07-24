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
  confirmed paired hot-zone calcium spike, near-threshold controls, and five
  futures from a common branching state;
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
- produces stimulus-relative figures with separate axes for voltage, calcium,
  calcium current, NMDA conductance, and NMDA current.

The `01b` ZIP or its extracted directory is a required input. This keeps the
provenance check real: `01c` does not silently trust copied protocol names when
the reference traces and their hashes are unavailable.
