# S4 production CPU runbook

Use this runbook only after the qualified canary. Do not repeat the canary or
use its data for training. Results and evidence limits are recorded in
`experiments/giada_runpod_paper_scale/s4_cpu_distributed_canary_result.json`.
The tested recovery killed a worker, not the entire pod; no claim about
physical-server independence is inferred from different pod hostnames.

## Frozen data contract

The existing `s4_hybrid_background_candidate.yml` and
`s4_hybrid_targeted_candidate.yml` retain their filenames for traceability;
their packing is now selected by the recorded canary result. No numerical
config, protocol, teacher, seed or stored-field setting changes in this update.
The coordinator freezes their actual canonical plan identities and the clean
Git revision in `s4_run.json` before any production writer can start.

- 138,240,000 background + 92,160,000 targeted transitions.
- 60/40 composition; 80/20 split; the same 14 protocols as S3.
- 11,520 shards per component; 90 per global worker per component.
- Exactly 128 immutable global worker indices; each seed is 7000001 + index.
- Support gates are prospectively eight times the preregistered S3 bounds,
  preserving that allowed density envelope (not tuned to any S4 output):

| Split / metric | Minimum | Maximum |
|---|---:|---:|
| Train absolute delta >= 5 mV | 3,173,040 | 5,288,400 |
| Validation absolute delta >= 5 mV | 795,264 | 1,325,440 |
| Train -55 mV upcrossings | 1,164,912 | 1,941,520 |
| Validation -55 mV upcrossings | 292,512 | 487,520 |

The streaming audit checks exact planned trajectory order, steps, seeds,
splits, protocol labels, physical SHA-256, completion metadata, finite fields,
exact component/protocol/split counts, activity support, extra files and stale
claims. Quantiles are **not** calculated by this large-scale audit; no
quantile was an acceptance gate. It retains no full-corpus voltage arrays.
Training configuration and authorization are a later, separate operation.

## Before starting paid generation

Consult `S4_CPU_24H_PLAN.md`. The 24-hour CPU target is conditional, not a
guarantee or an unlimited-spending authorization. The reference fleet is four
32-CPU pods at 30 workers each plus one 8-CPU pod, but actual sizes, hostnames,
rates and prices must be recorded when available. All need the existing shared
network volume. Canary packing suggests roughly 44 GB of S4 HDF5 alone; check
actual network-volume quota/headroom for logs, plans and existing data as well.
`df` on this network filesystem may describe the storage pool, not your quota.

Deploy the SAME clean published commit on all pods BEFORE generation starts.
Do not fetch/checkout or install into the shared Python environment while
workers are active. The new commands require only CPU requirements, not torch.
On each pod configure Git safe-directory exceptions for the two known repos
if needed. CPU affinity and common cgroup v1/v2 CPU/memory limits are checked
before launch; missing limits remain unknown, not proof of unlimited memory.

## 1. Coordinator: prepare once

From `/workspace/giada`, using the published revision:

```bash
bash runpod_scale/scripts/s4_cpu.sh prepare
```

This creates `/workspace/giada-data/s4-hybrid-production-v1`. Existing roots
are rejected, not overwritten. An interrupted preparation must be inspected;
do not delete it blindly or generate a replacement plan while workers run.
The command builds plans only, not NEURON data. It prints progress per component.

## 2. Register actual pods on the coordinator

Read `hostname` and `nproc` on each pod, then register its unique range. The
following is a TEMPLATE; replace the example hostname with the actual value:

```bash
bash runpod_scale/scripts/s4_cpu.sh register --hostname ACTUAL_HOSTNAME --start 0 --count 30 --cpus 32
```

Example full layout: 0--29, 30--59, 60--89, 90--119, 120--127. These are not
five launches on one pod. Overlapping indices, a duplicate hostname, invalid
counts and worker counts above declared CPU capacity are rejected. A 16-CPU
alternative can own a smaller disjoint interval. `status` lists unassigned
indices; a fleet with unassigned indices cannot complete production.

There is deliberately no automatic reassignment or stale-claim recovery in
this production wrapper. After an interruption, inspect exact original owner
processes on all implicated pods before a supervised recovery/handoff. The
qualified low-level CLI supports explicit recovery, but it is not safe to use
that flag against live owners. Do not edit `fleet.json` while any pod is active.

## 3. Start: the same short command on EVERY registered pod

```bash
bash /workspace/giada/runpod_scale/scripts/s4_cpu.sh start
```

No copied worker offsets or exported environment blocks are needed. The
hostname chooses the assignment. Unknown hosts, a changed revision, a sealed
corpus or duplicate pod claims are rejected. Every attempt gets a separate log
and the command prints its PID and absolute log path. It starts a detached
session with stdin closed; browser disconnection does not terminate it.
Each slot runs its 90 background shards then its 90 targeted shards; there is
no whole-fleet background barrier and no duplicate teacher threads.

The supervisor prints a heartbeat every 30 seconds. Detailed per-shard progress
is in `logs/HOST/ATTEMPT/background-worker-NNNNN.log` and the corresponding
targeted logs. A child failure stops the remaining local pipelines and leaves
claims for diagnosis. SIGTERM also requests child shutdown; SIGKILL cannot run
cleanup and always requires checking whether child processes survived.

## 4. Coordinator: start final verification in the background

```bash
bash /workspace/giada/runpod_scale/scripts/s4_cpu.sh start-finalizer
```

The finalizer survives console disconnection, refuses another finalizer, waits
for both components, and refuses to seal while pod claims remain. It stops for
reported failures; it never repairs or steals claims. Complete generation
triggers the exact streaming audit and sealing automatically. Its printed log
path can be followed with `tail -F`. Above 24 hours it reports the missed target
but does not destroy data or silently kill paid compute.

## 5. Status: identical on any pod, including after reconnect

```bash
bash /workspace/giada/runpod_scale/scripts/s4_cpu.sh status
```

Optional display refresh:

```bash
watch -n 30 bash /workspace/giada/runpod_scale/scripts/s4_cpu.sh status
```

Missing preparation is an error, not a misleading 0/N counter. Status reports
unassigned indices, component counts, per-pod completion, claims and elapsed
generation time. Claims alone do not prove owner liveness. This counter is not
the audit and does not certify integrity. A partially provisioned fleet's ETA
must include unstarted ranges; no automatic 128-worker throughput is assumed.

## Completion and shutdown

Final completion is `SEALED.json`, not just 23,040 completion markers. It locks
the hashes of both distributed manifests, both component validations,
the logical composite, support audit and physical fingerprint. Legacy
`compose-s3` and monolithic `plan.json` commands are not appropriate for S4.
The shard reader supports the distributed plan directory in composite entries;
the future GPU config must lock the resulting manifest/audit/fingerprint hashes.

If the detached finalizer was not started, run `s4_cpu.sh seal` only after all
writers have finished. An existing seal is never overwritten. Preserve the
entire output, including distributed plans, claims/history, logs and markers.
Before stopping pods confirm no workers/finalizer remain and run `sync`.
No command in this runbook rents or terminates a pod or changes the data recipe.
