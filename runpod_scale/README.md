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
- The input generator reproduces the published NeuronIO NMDA rate ranges,
  piecewise temporal rates, Gaussian smoothing, length weighting, spatial
  randomization, and exponential-then-Bernoulli sampling.
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

The measured S1 execution configuration is eight independent CPU workers. S1
uses one 6,000-ms trajectory per shard (100 shards total), so modulo assignment
gives each worker 12 or 13 trajectories. This replaces the earlier four-
trajectory shard layout, which would have assigned four shards to one worker
and three to the others and introduced a 33% static load imbalance. The change
affects only scheduling and restart granularity; seeds, trajectories, splits,
teacher dynamics, stored fields, and the scientific comparison are unchanged.

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

## GPU phase

Stop the CPU Pod, create a GPU Pod in the network-volume region, and attach the
same volume during deployment. Use an official current PyTorch template with
SSH. Then:

```bash
cd /workspace/giada
git fetch origin runpod/paper-scale-data
git checkout --detach origin/runpod/paper-scale-data
python -m pip install -r runpod_scale/requirements-gpu.txt

python -m src.giada_runpod.cli train \
  --config runpod_scale/configs/matched_training.yml \
  --corpus /workspace/giada-data/s1 \
  --output /workspace/giada-results/s1-matched \
  --elm-repo /workspace/giada
```

The run saves paired checkpoints at 100, 300, 1,000 and 3,000 steps for each
seed. These checkpoints are the mini scaling law. The registered primary
output is the median raw soma RMSE and GIADA's relative RMSE reduction versus
Branch-ELM. The held-out event-rich and OOD corpora remain separate final
tests; they are not used for normalization, tuning, or checkpoint selection.

## What remains in the Kaggle track

Architecture exploration, recursive-state repair, event-specific ablations,
and Allen-Zhu-style atomic playgrounds remain on `main`/Kaggle. This RunPod
branch may consume a frozen candidate selected there, but it must not invent a
new architecture based on S1--S4 validation outcomes. That separation prevents
paper-scale confirmation data from becoming an architecture-development set.
