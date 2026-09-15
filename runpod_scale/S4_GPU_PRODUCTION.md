# S4 GPU: one GPU per paired seed

Approved workflow: five independent jobs for seeds **61017, 61029, 61043,
61071, 61103**. Each job trains both unchanged models on the same minibatches.
No distributed gradients, no architecture change, no new teacher generation.
The contract is in
`../experiments/giada_runpod_paper_scale/s4_matched_training_preregistration.json`.

Per seed: **1,152,000 paired updates**, batch4096, nominal exposure25.6.
576k updates at1e-3, then fresh AdamW and the registered new random stream for
576k at3e-4. This is the agreed exposure-scaled S3 schedule, not an already
measured S4 performance result. Intermediate checkpoints are diagnostic only;
the final endpoint evaluates all46.08M development-validation transitions.
No endpoint shortening or best-checkpoint selection based on observed scores.

## First pod only

Initially deploy only one GPU, with the existing network volume mounted at
`/workspace`. Use the qualified PyTorch2.4/CUDA12.4 environment. All five jobs
must use the same exact torch/numpy/Python versions recorded during preparation.
Do not change shared source or the teacher venv while any job is running.

Use one clean published revision for code and instructions. Before other jobs
start, fetch `runpod/paper-scale-data` and check out the agreed commit. Do not
execute another Git checkout midway through a shared-volume run.

On each pod, configure Git trust for this known repository:

```bash
git config --global --add safe.directory /workspace/giada
cd /workspace/giada
```

Install into the GPU image's `python`, **not** `/workspace/.giada-venv`:

```bash
python -m pip install -r runpod_scale/requirements-gpu.txt
python -m pytest tests/test_giada_s4_training.py -q
```

The tests include CPU checkpoint/restart checks using installed PyTorch. The
launcher also performs a tiny real-CUDA checkpoint check before paid production
training. This is an operational check, not hyperparameter search.

Only once, on the first GPU pod:

```bash
bash /workspace/giada/runpod_scale/scripts/s4_gpu.sh start-prepare
```

This returns a PID and log path under `/workspace/giada-results/setup` and
continues after browser disconnection. Follow the exact returned log with
`tail -F LOG_PATH` (replace LOG_PATH). Ctrl-C exits the tail, not the detached
process. Preparation authenticates the CPU seal, indexes the corpus, fits the
unchanged train-only normalization, and freezes a global diagnostic sample.
It finishes with `[S4 GPU prepare] ready`; **no seed starts automatically**.
`run.json` is published only after preparation artifacts are complete. If
preparation fails, inspect its log; never delete or overwrite an output blindly.

Then start the first paired seed:

```bash
bash /workspace/giada/runpod_scale/scripts/s4_gpu.sh start --seed 61017
```

The launcher checks both real models and checkpoint restart on this GPU,
then returns a detached PID/log. Physical shard hashes are verified before
training. Indexing and hash verification have progress logs. Do not launch
another seed while the same GPU is claimed.

Observe the first production progress lines and RAM/GPU usage before renting
the remaining pods. The prefix remains part of this run, not a disposable
training experiment. ETA is measured paired updates/s, and explicitly excludes
future validation. All checkpoints, sampling and method settings remain frozen.

## Remaining four pods

Use the same mounted volume and revision; install GPU dependencies per pod.
**Do not rerun prepare**. Assign each seed to one distinct pod:

| GPU pod | Seed argument |
|---|---:|
| first | 61017 |
| second | 61029 |
| third | 61043 |
| fourth | 61071 |
| fifth | 61103 |

For example on the second pod only:

```bash
bash /workspace/giada/runpod_scale/scripts/s4_gpu.sh start --seed 61029
```

The same command with the appropriate seed is used on each other pod. Seed
claims block duplicate ownership across pods; device claims block two seeds
on one pod. Reconnecting requires no exported variables and no relaunch:

```bash
bash /workspace/giada/runpod_scale/scripts/s4_gpu.sh status
```

Start one finalizer on a pod that will stay on until aggregation:

```bash
bash /workspace/giada/runpod_scale/scripts/s4_gpu.sh start-finalizer
```

It waits for exactly five complete seed reports, validates identities,
checkpoint hashes and evaluation coverage, and writes
`/workspace/giada-results/s4-matched-v1/final_report.json`.
It reports failed/stopped jobs rather than silently accepting partial results.
It does not stop/bill-control pods automatically. After completion, use `sync`
and stop idle GPU pods manually; retain the volume and finalizer host until
the final report exists. If a seed required recovery, restart a finalizer that
exited with an attention-needed message.

## Interruptions and limits

- A browser disconnect does not stop detached preparation/training/finalizer.
- SIGTERM/SIGINT requests a checkpoint after the current update; wait for the
  stopped message before shutting down. Never send signals to unverified PIDs.
- Unexpected process/pod death leaves claims for manual inspection. There is
  deliberately no automatic cross-host claim stealing. Confirm the old owner
  is dead before controlled recovery; do not remove a live claim.
- A resumed seed restores both models, both optimizers, phase and all RNG
  states. A corrupt/mismatched checkpoint fails closed, not from step0.
- Intermediate checkpoints use a fixed262,144-row sample; final metrics use
  the full split. Do not interpret them as a same-sample learning curve.
- Bounded HDF5 handles prevent accumulating23,040 open files. The corpus is
  still read from the shared volume; this version does **not** introduce a
  local cache. CPU RAM, network I/O and final evaluation remain measurable costs.
- Identical software and checkpoints preserve restart state; bitwise identity
  across different GPU architectures/drivers is not promised.
- At the user's quoted$0.74/GPU-hour, five active GPUs cost$3.70/hour plus
  storage. Compute-only hours/seed=1,152,000/(measured updates/s)/3600.
  Earlier4–8h estimates are provisional, not a measured S4 guarantee.

The S4 seal and its hashes are read on the mounted RunPod volume, not inferred
from console text. Local tests use fixtures and real CPU PyTorch models; the
real CUDA/corpus end-to-end check happens on the first pod before fleet expansion.
