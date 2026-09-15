# S4 GPU preparation — proposal, not a launch authorization

**Historical proposal, superseded on 2026-09-15:** the user approved the
1,152,000-update paired-seed budget and publication. Use
[S4_GPU_PRODUCTION.md](S4_GPU_PRODUCTION.md) and the checked-in S4 GPU
preregistration for the implemented contract. The local-cache idea below is
deferred, not an implemented feature. This draft is retained as design history.

Prepared 2026-09-15. No GPU has been provisioned and no training configuration
has been frozen by this document. CPU completion is supported by the user's
terminal output (`sealed: true`, 23,040 shards, no failure markers). The actual
S4 seal, audit and fingerprint files have not yet been imported locally.

## Scientific contract and budget decision

Keep the 60/40 hybrid corpus, whole-trajectory splits, 76 causal inputs,
authentic one-step soma target, raw readout, both existing architectures,
paired minibatches, AdamW family, loss, batch size 4096 and five seeds
61017, 61029, 61043, 61071, 61103. This remains voltage-bridge comparison,
not full-state prediction or rollout validation.

S3's independently evaluated candidate was NOT trained from scratch with
constant learning rate 0.0003. Its path was 72,000 initial steps at 0.001,
followed by 72,000 continuation steps at 0.0003, with fresh AdamW and a new
deterministic sampling stream at the boundary. The forensic extension then
continued the optimizer and RNG states exactly. Sources:

- `configs/s3_matched_training.yml`
- `configs/s3_late_optimization_forensic.yml`
- `configs/s3_matched_exposure_extension.yml`
- `../experiments/giada_runpod_paper_scale/s3_matched_exposure_extension_preregistration.json`
- `../src/giada_runpod/optimization_forensic.py`, `_run_arm`

S4 contains 184,320,000 train and 46,080,000 development-validation rows.
Nominal exposure means sampled examples / train rows; replacement sampling
does not guarantee that every row was visited.

| Budget per paired seed | Sampled examples per model | Nominal exposure |
|---|---:|---:|
| 144,000 updates (same compute as S3) | 589,824,000 | 3.2 |
| 1,152,000 updates (same exposure as S3) | 4,718,592,000 | 25.6 |

Proposed primary endpoint: 1,152,000 updates per paired seed, with 144,000
retained as a diagnostic checkpoint on the same run. Proposal for the adapted
schedule: 576,000 steps at 0.001, then 576,000 at 0.0003 with explicitly
specified fresh AdamW and deterministic continuation stream. This is an
eightfold exposure-scaled transfer, NOT an already tested S4 schedule. Obtain
user agreement before implementing/finalizing this choice. A 144k checkpoint
on this schedule is not equivalent to a separately optimized 144k run.
Do not choose a winning checkpoint using S4 validation or silently truncate
the endpoint because runtime is inconvenient.

Use a frozen global representative development sample for intermediate
diagnostics, with sampling manifest and support counts; full 46.08M validation
at the final endpoint. Do not use the first N physical rows: background-first
file order would bias a truncated evaluation. Freeze exact checkpoint list,
sample seed and sample size before launch. Preserve all five seed reports,
including losses. Final acceptance gates must also be preregistered.

## Parallelism and data path

Prefer independent paired-seed jobs over distributed gradients: one process
owns both models for one seed, separate output directory and exclusive claim.
Five GPUs can run the five seeds concurrently; fewer GPUs run successive
waves. This does not change batch size or optimizer semantics. A single GPU
may ultimately handle concurrent seeds efficiently, but this requires a
measurement; do not assume five processes yield fivefold throughput.

DDP synchronizes gradients between replicas. For these roughly 8k/9k-parameter
models, seed parallelism is the first engineering candidate, not a measured
claim that DDP must be slower:
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html

Prefer a verified local read-only shard cache on each training host to repeated
network HDF5 random reads, subject to sufficient disk space. Hash-check copied
content against the sealed manifest; never modify or replace the original
corpus. A cache is an acceleration artifact, not a new dataset.

Do not materialize the entire transformed corpus in RAM/GPU: 230.4M x 76 x
float32 alone is 70,041,600,000 bytes (~65.23 GiB), before labels and copies.
The current reader already streams evaluation, but retains all visited HDF5
handles and stores row indices and Python trajectory labels. Bound handles
with an LRU and measure host memory. Preserve exact sampling order and values.

## Required implementation and verification before paid production

1. Import and validate `SEALED.json`, `s4_run.json`, both distributed manifests
   and validation reports, composite manifest, production audit and fingerprint.
   Freeze actual hashes; no placeholder hashes in a runnable config.
2. Require the complete hash contract for S4 in `MatchedTrainingConfig.validate`.
   Currently the explicit mandatory-contract stage list contains S2 and S3 only.
3. Add exact resumable, atomic checkpoints containing BOTH models, optimizers,
   phase/step, normalization, all relevant RNG states and frozen identities.
   Current base trainer skips completed seeds but cannot exactly resume a
   partially trained seed from its model-only checkpoint.
4. Implement disjoint seed launch claims, attempt-specific logs, final aggregation
   requiring exactly the five expected seeds and identical contracts. A config
   with one seed must not weaken the five-seed decision or masquerade as the
   aggregate result.
5. Bound HDF5 handles, avoid duplicate expensive scans, provide progress during
   normalization, verification, cache preparation and final evaluation. Never
   replace physical verification with trusting file existence.
6. Test deterministic batches before/after reader changes, phase-boundary and
   interrupted-run state equivalence, duplicate launches, corrupt cache and
   contract mismatches, and aggregate rejection of incomplete/mixed runs.
7. Measure a short operational prefix on ONE GPU after configuration is frozen.
   If exact checkpoint continuation is verified, retain that prefix as part of
   production; no hyperparameter search and no disposable teacher generation.
   Expand to other GPUs only after throughput, memory and cost are acceptable.

Do not enable mixed precision, alter batch size, replace sampling, or adopt
compile/fused optimizer variants merely to meet a time estimate without
separate equivalence checks and explicit method decisions.

## Runtime and cost: sensitivity, not a measured ETA

Let r be measured paired training updates/second (one update trains BOTH models).
For the proposed full budget:

- compute hours per seed = 1,152,000 / r / 3600;
- total compute GPU-hours = 5 times that value;
- with G identical GPUs and one seed at a time per GPU, compute makespan is
  ceil(5/G) times the per-seed time;
- add measured verification/cache startup, checkpoint/evaluation and final audit
  time. Network contention and unequal devices can prevent ideal scaling.

| Hypothetical paired updates/s | Compute hours/seed | Total compute GPU-hours |
|---|---:|---:|
| 50 | 6.4 | 32 |
| 100 | 3.2 | 16 |
| 200 | 1.6 | 8 |

These rates are scenarios, not measurements of S4. Historical S3 ETA messages
and planned time budgets do not establish S4's runtime. Full evaluation is also
eight times larger. No promise of a sub-day single-GPU run is justified yet.

Cost = sum(actual hourly pod rate x active billed hours) + storage/cache charges.
Use the current deployment quote, not an old screenshot; RunPod directs users
to the console for current prices: https://docs.runpod.io/pods/pricing
Parallelism reduces ideal wall time, not total required GPU-hours. Do not rent
the full fleet before the software, identity contract and spending limit are
ready. The previous 24-hour deadline explicitly covered CPU generation; ask
before treating it as a new GPU budget or spending authorization.

## Status

This is a reviewed-code proposal only. No GPU launch script/config or new
scientific preregistration is claimed ready. No source/model/data changes,
Git commit or remote publication are included in this preparation document.
