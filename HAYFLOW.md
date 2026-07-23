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
