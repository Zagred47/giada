# GIADA paper-scale track on RunPod

This directory is an operationally isolated track for scaling the validated
one-step soma comparison. It does **not** replace or modify the Kaggle
architecture-research notebooks. Historical Python modules may still contain
the technical prefix `hayflow_*`; the project and every new artifact here are
named **GIADA**.

## Scientific scope

The RunPod track answers one narrow paper question: does the previously
observed GIADA advantage over the 8,002-trainable-parameter Branch-ELM survive
as the amount of authentic teacher time increases?

- Both models consume the same numeric tensor.
- Both predict the raw authentic NEURON soma transition
  `V(t+1 ms) - V(t)`.
- Both use the same samples, order, loss, optimizer family, and three seeds.
- This comparison is one-step only; it does not make an autoregressive claim.
- S1/S1b reproduce the published NeuronIO NMDA sampler as a stochastic control.
  They are not the primary GIADA data methodology.
- The primary scaling candidate is a hybrid: stochastic background plus
  canonical-weight targeted events, hard negatives, and counterfactual arms
  restored from the same teacher state and Random123 seed.
- Probabilistic release remains the authentic teacher mechanism. `U_realized`
  is obtained causally before membrane integration and verified at the next
  boundary by the already validated instrumentation.

The large corpus stores only the local raw boundary fields required by the
registered soma comparison plus a sparse release table. It deliberately omits
41-sample microtraces and per-millisecond native snapshots. Those remain in
the small diagnostic corpus. Replicating the rich 6 GiB/29,240-transition
format at 230.4 million transitions would be methodologically unnecessary and
operationally prohibitive.

## Registered stages

| Stage | Teacher time | 1 ms transitions | Decision |
|---|---:|---:|---|
| S0 | historical diagnostic budget | 29,880 | systems smoke test |
| S1 | 10 minutes | 600,000 | first scaling check |
| S2 | 1 hour | 3,600,000 | stability across more trajectories |
| S3 | 8 hours | 28,800,000 | medium-scale confirmation |
| S4 | 64 hours | 230,400,000 | NeuronIO train+validation time parity |

Advancement is sequential: benchmark first, then S1; proceed to S2/S3/S4 only
if the paired advantage and data integrity survive the previous stage. This is
a scaling law, not a single all-or-nothing 64-hour generation.

### S1 distribution audit and corrective pilot

S1 passed byte-level and schema validation, but the mandatory post-generation
distribution audit found that its three validation trajectories were entirely
subthreshold: zero transitions had `|delta V| >= 1 mV` and there were zero
somatic `-55 mV` upcrossings.  The paired S1 result therefore remains an
authentic subthreshold one-step comparison, but it cannot support an active or
spiking claim and does not authorize S2.

S1 is frozen rather than repaired post hoc.  Before any further scaling,
`s1b_event_support_pilot.yml` runs a development-only 2x2x2 factorial pilot
over excitation range, inhibitory balance and temporal bandwidth.  Every arm
is a conditional slice inside the published NeuronIO ranges; morphology,
weights, mechanisms and authentic probabilistic release are unchanged.  Each
cell has one discovery and one independent confirmation trajectory.  The
pilot identifies a support-generating protocol; it is not a final paper test.

The first operational attempt (`s1b-event-support-pilot-v1`) was aborted after
an early hard failure exposed an invalid lower-inhibition interval when the
full-excitation arm approached zero.  No partial v1 shard is selection
eligible.  Protocol revision v2 uses the canonical-support slice
`[-600, 0]` for the inhibitory-rate difference, validates interval feasibility
before sampling, and must run under a fresh output root.

The measured S1 execution configuration is eight independent CPU workers. S1
uses one 6,000-ms trajectory per shard (100 shards total), so modulo assignment
gives each worker 12 or 13 trajectories. This replaces the earlier four-
trajectory shard layout, which would have assigned four shards to one worker
and three to the others and introduced a 33% static load imbalance. The change
affects only scheduling and restart granularity; seeds, trajectories, splits,
teacher dynamics, stored fields, and the scientific comparison are unchanged.

### S1c: prospective GIADA hybrid pilot

S1b established that random high-excitation/low-inhibition streams can recover
somatic activity, but it does not reproduce the evolved GIADA sampling
methodology. It is frozen as a control. `s1c_giada_hybrid_pilot.yml` is the
prospective correction and must run under a new output root.

The 15,360-transition pilot contains eight train and eight validation
replicates for each of 12 arms. Targeted arms within a causal family share the
exact equilibrium snapshot, Random123 seed, duration, and pre-intervention
history. The stochastic backgrounds are separate controls rather than a
same-history counterfactual pair. Registered contrasts are canonical versus
active stochastic background; 8 versus 12 local tuft synapses; unpaired,
paired-n8 and paired-n12 hot-zone stimulation; weak versus calibrated somatic
current; and soma-only, assist-only and combined bAP candidates. Synaptic
weights remain canonical. Seven anatomical probes are sampled at 0.025 ms.

This is still a development pilot: its outcomes decide the composition of a
larger hybrid corpus, not a paper test. It does not reuse the old sealed fresh
test outcomes, and it does not alter any Kaggle architecture experiment.

S1c completed all 192 shards and passed its technical audit. The NMDA and
calcium families produced the intended cross-split contrasts. The audit also
exposed two protocol-transcription errors: the single-pulse somatic positive
was written as 3 nA although the immutable calibration selected 6 nA, and the
BAP candidate did not use the selected n12/b3/w400 assist or the registered
pulse/current bracket. S1c is frozen; these errors do not invalidate its NMDA
or calcium observations, but they block large-scale generation.

### S1d: prospective somatic/BAP protocol repair

`s1d_protocol_repair_pilot.yml` reruns only the missing causal axes. It copies
the parameters from the v1.1.2 teacher artifacts: a 3-versus-6 nA one-pulse
somatic boundary, the n12/b3/w400 canonical-weight subthreshold assist,
`p2-factor3` versus `p3-factor3` soma-only BAP controls, and the historical
`p3-factor2` assisted candidate. A compact 3-by-2 BAP matrix measures pulse
count/current strength and assist presence in one run. The exploratory
combined arms cannot override failure of the preregistered controls.

S1d contains 144 independent 80-ms trajectories (11,520 transitions), split
equally between train and validation with eight replicates per arm per split.
It is a development top-up, not a paper test and not a replacement for S1c.

## Storage and interruption contract

Each CPU process owns an independent NEURON interpreter. Threads must not
share HOC state. A shard is first written as `*.h5.partial`; the final HDF5 and
its `*.done.json` marker appear only after row-count validation and SHA-256.
Restarting the same plan skips matching completed shards. A mismatching
completed shard is a hard error, never silently overwritten.

Generated data, logs, model checkpoints, and credentials are excluded from
Git. Only source code, immutable configurations, schemas, and documentation
belong on this branch.

## First CPU Pod: exact sequence

Use a **Secure Cloud CPU Pod** and attach a **network volume at creation**.
RunPod documents that a network volume lives independently from a Pod and is
mounted at `/workspace`; it can later be attached to the GPU Pod. Network
volumes cannot be attached or detached after Pod creation. Start with a
compute-oriented CPU configuration, but do not rent a fleet before measuring
one real 6-second trajectory.

Official references:

- <https://docs.runpod.io/pods/storage/types>
- <https://docs.runpod.io/storage/network-volumes>
- <https://docs.runpod.io/pods/manage-pods>
- <https://docs.runpod.io/pods/configuration/use-ssh>

Recommended first settings:

- Pod type: CPU, Secure Cloud.
- CPU: compute-optimized, initially 8 vCPU if available.
- RAM: at least 16 GiB; 32 GiB is safer for the first parallel test.
- Container disk: 30 GiB.
- Network volume: 100 GiB for S1, in a region where a later GPU is available.
- Expose TCP 22 and connect by SSH for long-running commands.

After connecting:

```bash
export GIADA_ROOT=/workspace/giada
export GIADA_TEACHER_ROOT=/workspace/neuron_as_deep_net
export GIADA_REF=runpod/paper-scale-data
export GIADA_PYTHON=/workspace/.giada-venv/bin/python

git clone --branch runpod/paper-scale-data https://github.com/Zagred47/giada.git "$GIADA_ROOT"
bash "$GIADA_ROOT/runpod_scale/scripts/bootstrap_cpu.sh"
```

For browser-console resilience, keep a named `tmux` session for interactive
work while running scientific jobs with `nohup` and persistent log files:

```bash
tmux new -s giada
# Detach without stopping the session: Ctrl-b, then d
tmux attach -t giada
```

Use `tail -F <log>` inside `tmux` to resume live log streaming after a browser
disconnect. `tmux` preserves the terminal view; `nohup` is the independent
process-lifetime guarantee. Both end if the Pod itself is stopped, so restart
contracts still depend on completed shard markers stored under `/workspace`.

Create S1's immutable plan:

```bash
export GIADA_OUTPUT_ROOT=/workspace/giada-data/s1-v2
mkdir -p "$GIADA_OUTPUT_ROOT"
cd "$GIADA_ROOT"
"$GIADA_PYTHON" -m src.giada_runpod.cli plan \
  --config runpod_scale/configs/s1_soma.yml \
  --output "$GIADA_OUTPUT_ROOT"
```

Before S1, run one full canonical trajectory benchmark:

```bash
"$GIADA_PYTHON" -m src.giada_runpod.cli benchmark \
  --config runpod_scale/configs/s1_soma.yml \
  --output /workspace/giada-data/benchmark-s1 \
  --elm-repo "$GIADA_ROOT" \
  --teacher-repo "$GIADA_TEACHER_ROOT" \
  --duration-ms 6000
```

Read `benchmark_report.json`. It extrapolates wall time and storage from the
actual machine. Worker count is chosen only after also watching peak RAM and
CPU utilization (`htop`). Start with one worker, then test two, four, and at
most the available physical vCPUs. Keep the highest count whose throughput is
near-linear and whose memory has a safety margin.

After the single-worker benchmark, use the registered concurrency harness
instead of manually launching overlapping probes. For example, four workers
with 3,000 transitions each:

```bash
GIADA_BENCHMARK_WORKERS=4 \
GIADA_BENCHMARK_DURATION_MS=3000 \
GIADA_BENCHMARK_OUTPUT=/workspace/giada-data/cpu-concurrency-4 \
bash "$GIADA_ROOT/runpod_scale/scripts/benchmark_cpu_concurrency.sh"
```

The harness writes independent logs and reports cold-start throughput
separately from the sum of the workers' steady-state generation rates. Its
parallel efficiency and projected S1 wall time use the latter, so repeated
teacher construction and burn-in do not bias the long-lived-worker estimate.
It also reports per-process peak RSS.

Generation writes one compact progress line about every 30 seconds with the
completed transition count, percentage, throughput, and ETA. Follow a
background log with `tail -F`; after reconnecting, `tmux attach -t giada`
restores the live view, while `tail -n 50 <log>` resynchronizes recent history.

Launch the selected workers in a disconnect-safe shell:

```bash
export GIADA_WORKER_COUNT=8  # selected by the recorded 1/4/8-worker benchmark
mkdir -p "$GIADA_OUTPUT_ROOT/logs"
nohup bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" \
  >"$GIADA_OUTPUT_ROOT/logs/supervisor.log" 2>&1 &
```

Compact status (no notebook-output flooding):

```bash
bash "$GIADA_ROOT/runpod_scale/scripts/status.sh"
tail -n 30 "$GIADA_OUTPUT_ROOT/logs/worker-0.log"
```

Validate all shards before stopping CPU compute:

```bash
cd "$GIADA_ROOT"
"$GIADA_PYTHON" -m src.giada_runpod.cli validate \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT"
```

Do not delete the network volume. RunPod warns that data outside `/workspace`
is lost on restart and that shared-volume concurrent writes require explicit
coordination. GIADA workers only write disjoint shard and status filenames;
the GPU phase starts after CPU writers have stopped. Critical results should
also be copied to external object storage because RunPod does not position Pod
storage as long-term archival storage.

Integrity validation must now always be followed by a target-distribution
audit.  The audit prints bounded progress and records split- and protocol-level
support:

```bash
"$GIADA_PYTHON" -m src.giada_runpod.cli audit-corpus \
  --corpus "$GIADA_OUTPUT_ROOT" \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT/distribution_audit.json"
```

Do not start GPU training unless the registered active-transition and somatic
upcrossing minima pass.  Empty strata are reported as unsupported (`null`),
never as a zero-error metric.

The corrective pilot uses a fresh output root and fresh seeds:

```bash
export GIADA_OUTPUT_ROOT=/workspace/giada-data/s1b-event-support-pilot-v2
"$GIADA_PYTHON" -m src.giada_runpod.cli plan \
  --config runpod_scale/configs/s1b_event_support_pilot.yml \
  --output "$GIADA_OUTPUT_ROOT"
```

The primary GIADA hybrid pilot is launched separately:

```bash
export GIADA_OUTPUT_ROOT=/workspace/giada-data/s1c-hybrid-pilot-v1
"$GIADA_PYTHON" -m src.giada_runpod.cli plan \
  --config runpod_scale/configs/s1c_giada_hybrid_pilot.yml \
  --output "$GIADA_OUTPUT_ROOT"

nohup bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" \
  >"$GIADA_OUTPUT_ROOT/logs/supervisor.log" 2>&1 &

"$GIADA_PYTHON" -m src.giada_runpod.cli validate \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT"

"$GIADA_PYTHON" -m src.giada_runpod.cli audit-hybrid \
  --corpus "$GIADA_OUTPUT_ROOT" \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT/hybrid_audit.json"
```

After preserving the S1c audit, launch the prospective repair under a new
root. Do not reuse a partially generated directory:

```bash
cd "$GIADA_ROOT"
git fetch origin runpod/paper-scale-data
git checkout --detach origin/runpod/paper-scale-data
"$GIADA_PYTHON" -m pytest tests/test_giada_runpod_scale.py -q

export GIADA_OUTPUT_ROOT=/workspace/giada-data/s1d-protocol-repair-pilot-v1
"$GIADA_PYTHON" -m src.giada_runpod.cli plan \
  --config runpod_scale/configs/s1d_protocol_repair_pilot.yml \
  --output "$GIADA_OUTPUT_ROOT"

mkdir -p "$GIADA_OUTPUT_ROOT/logs"
nohup bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" \
  >"$GIADA_OUTPUT_ROOT/logs/supervisor.log" 2>&1 &

"$GIADA_PYTHON" -m src.giada_runpod.cli validate \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT"

"$GIADA_PYTHON" -m src.giada_runpod.cli audit-hybrid \
  --corpus "$GIADA_OUTPUT_ROOT" \
  --plan "$GIADA_OUTPUT_ROOT/plan.json" \
  --output "$GIADA_OUTPUT_ROOT/hybrid_audit.json"
```

The final audit prints a bounded summary plus
`scientific_outcome_assessment`. Large-scale hybrid generation remains blocked
unless both the technical audit and every preregistered S1d control pass.

S1d passed every registered control. The next prospective stage is S1e, a
600,000-transition hybrid corpus with two physical components: long
stochastic background trajectories and short confirmed targeted episodes.
They are generated and validated by one resumable supervisor and then sealed
as a logical composite. Start it in `tmux` or with `nohup`:

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach origin/runpod/paper-scale-data
"$GIADA_PYTHON" -m pytest tests/test_giada_runpod_scale.py -q

export GIADA_ROOT=/workspace/giada
export GIADA_TEACHER_ROOT=/workspace/neuron_as_deep_net
export GIADA_PYTHON=/workspace/.giada-venv/bin/python
export GIADA_WORKER_COUNT=8
export GIADA_S1E_ROOT=/workspace/giada-data/s1e-hybrid-production-v1

mkdir -p "$GIADA_S1E_ROOT/logs"
nohup env PYTHONUNBUFFERED=1 \
  bash "$GIADA_ROOT/runpod_scale/scripts/launch_s1e_hybrid_corpus.sh" \
  >"$GIADA_S1E_ROOT/logs/pipeline.log" 2>&1 &
echo $! | tee "$GIADA_S1E_ROOT/logs/pipeline.pid"
tail -F "$GIADA_S1E_ROOT/logs/pipeline.log"
```

The supervisor is connection-independent. A rerun with the same root resumes
completed shards after checking their plan hash. Completion requires
`composite/composite_manifest.json` and `composite/production_audit.json` with
`valid: true`. Do not point GPU training at either component alone; its corpus
argument is the verified composite directory.

## GPU phase

Stop the CPU Pod, create a GPU Pod in the network-volume region, and attach the
same volume during deployment. Use an official current PyTorch template with
SSH. Then:

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach origin/runpod/paper-scale-data
python -m pip install -r runpod_scale/requirements-gpu.txt

# Confirm that the template's CUDA build was preserved and NumPy is ABI-safe.
python -c "import numpy, torch; assert numpy.__version__.split('.')[0] == '1'; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'numpy': numpy.__version__, 'gpu': torch.cuda.get_device_name(0)})"

python -m src.giada_runpod.cli train \
  --config runpod_scale/configs/matched_training.yml \
  --corpus /workspace/giada-data/s1-v2 \
  --output /workspace/giada-results/s1-matched \
  --elm-repo /workspace/giada
```

The original command above documents the frozen S1 run. S1e has now passed
all generation and support gates. On a GPU pod attached to the same network
volume, launch its preregistered information-matched comparison as one
connection-independent job:

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach FETCH_HEAD
python -m pip install -r runpod_scale/requirements-gpu.txt

export GIADA_ROOT=/workspace/giada
export GIADA_GPU_PYTHON=python
export GIADA_S1E_CORPUS=/workspace/giada-data/s1e-hybrid-production-v1/composite
export GIADA_S1E_RESULTS=/workspace/giada-results/s1e-matched-v1

mkdir -p /workspace/giada-results
nohup env PYTHONUNBUFFERED=1 \
  bash "$GIADA_ROOT/runpod_scale/scripts/launch_s1e_matched_training.sh" \
  >/workspace/giada-results/s1e-matched-v1.log 2>&1 &
echo $! | tee /workspace/giada-results/s1e-matched-v1.pid
tail -F /workspace/giada-results/s1e-matched-v1.log
```

The run saves paired checkpoints at 100, 300, 1,000 and 3,000 steps for each
seed. Completed seeds are restartable; a partially interrupted seed is
retrained from its frozen initialization. The final 3,000-step checkpoint is
preregistered and validation is never used for checkpoint selection. In the
same evaluation pass the code reports global, active, spike-transition,
component, family and protocol metrics. These strata are diagnostic outputs
only and do not change training. The held-out event-rich and OOD corpora remain
separate final tests.

## S2: frozen sixfold hybrid scale-up

S2 must use `s2_hybrid_background.yml` and `s2_hybrid_targeted.yml`; the old
single-protocol `s2_soma.yml` is historical scaffolding and is not eligible
for the paper comparison. The CPU pipeline creates 2.16 million long-
background and 1.44 million targeted transitions, validates both components,
then enforces exact protocol counts and preregistered distribution-fidelity
bands before sealing the logical composite.

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach FETCH_HEAD
/workspace/.giada-venv/bin/python -m pytest tests/test_giada_runpod_scale.py -q

export GIADA_ROOT=/workspace/giada
export GIADA_TEACHER_ROOT=/workspace/neuron_as_deep_net
export GIADA_PYTHON=/workspace/.giada-venv/bin/python
export GIADA_WORKER_COUNT=8
export GIADA_S2_ROOT=/workspace/giada-data/s2-hybrid-production-v1

mkdir -p "$GIADA_S2_ROOT/logs"
nohup env PYTHONUNBUFFERED=1 \
  bash "$GIADA_ROOT/runpod_scale/scripts/launch_s2_hybrid_corpus.sh" \
  >"$GIADA_S2_ROOT/logs/pipeline.log" 2>&1 &
echo $! | tee "$GIADA_S2_ROOT/logs/pipeline.pid"
tail -F "$GIADA_S2_ROOT/logs/pipeline.log"
```

After `composite/production_audit.json` reports `valid: true`, stop the CPU
Pod and attach the same network volume to a GPU Pod. S2 uses 18,000 steps,
which preserves S1e's effective sample exposure, and five paired seeds. Every
checkpoint is diagnostic; only the final preregistered checkpoint determines
the decision.

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach FETCH_HEAD
python -m pip install -r runpod_scale/requirements-gpu.txt

export GIADA_ROOT=/workspace/giada
export GIADA_GPU_PYTHON=python
export GIADA_S2_CORPUS=/workspace/giada-data/s2-hybrid-production-v1/composite
export GIADA_S2_RESULTS=/workspace/giada-results/s2-matched-v1

mkdir -p /workspace/giada-results
nohup env PYTHONUNBUFFERED=1 \
  bash "$GIADA_ROOT/runpod_scale/scripts/launch_s2_matched_training.sh" \
  >/workspace/giada-results/s2-matched-v1.log 2>&1 &
echo $! | tee /workspace/giada-results/s2-matched-v1.pid
tail -F /workspace/giada-results/s2-matched-v1.log
```

The S2 report includes medians, means and sample standard deviations across
seeds, a paired one-sided test, a deterministic seed-bootstrap interval, and
component/family/protocol breadth. These uncertainty summaries do not turn
development validation into a fresh test and do not authorize architecture
selection.

## What remains in the Kaggle track

Architecture exploration, recursive-state repair, event-specific ablations,
and Allen-Zhu-style atomic playgrounds remain on `main`/Kaggle. This RunPod
branch may consume a frozen candidate selected there, but it must not invent a
new architecture based on S1--S4 validation outcomes. That separation prevents
paper-scale confirmation data from becoming an architecture-development set.
