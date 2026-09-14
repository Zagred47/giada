# S4 CPU capacity plan: 24-hour completion target

Recorded 2026-09-14. Planning only: no fleet has been rented, no production
plan has been sealed by this document, and no production process was launched.
The user specified a time limit, not an unlimited financial budget.

## Scope and invariants

Interpret the requested 24 hours as the CPU phase, including startup,
generation, validation and corpus sealing; GPU training is separate. Count
startup and provisioning delays against the available time, not as free time.

Preserve the existing S4 hybrid candidate configs: 138,240,000 background
transitions and 92,160,000 targeted transitions; 60/40 composition; 80/20 split;
the same two background and twelve targeted protocols. Do not use the legacy
`s4_soma_parity.yml`. Do not shorten trajectories, remove difficult cases,
change seeds, or relax integrity/support gates to meet the deadline.

With the qualified packing, each component has 11,520 shards. The immutable
128-worker namespace therefore assigns exactly 90 background and 90 targeted
shards to each global worker. It is a logical namespace, not evidence that
128 workers are running. Unassigned indices leave the corpus incomplete.

## Observations supplied through the RunPod console

These are user-supplied execution observations, not newly downloaded or
independently rehashed artifacts. Different pod hostnames do not prove that
their underlying physical servers are different.

| Pod | Local concurrency | Component | Completed transitions | Wall seconds | Aggregate transitions/s |
|---|---:|---|---:|---:|---:|
| e374159d3660 (32 visible CPUs) | 30 | background | 360,000 | 342 | 1052.632 |
| e374159d3660 | 30 | targeted c30-v2 | 240,000 | 261.918 | 916.317 |
| 5b20edd19fd3 (8 visible CPUs) | 8 | background crosspod | 96,000 | 333.020 | 288.271 |
| 5b20edd19fd3 | 8 | targeted crosspod | 64,000 | 249.612 | 256.398 |

Do not use the earlier targeted c30-v1 manual 355-second observation as the
precise benchmark: its end timestamp included an unknown reporting delay.
Eight or sixteen processes on the 32-CPU pod do not measure the performance
of an independently provisioned 8- or 16-CPU pod.

Reported integrity outcomes already obtained (do not repeat these canaries):

- Full background shared-v2 and targeted shared-v1 outputs passed their
  distributed validations (30 shards each, no blockers) and their HDF5 hash
  lists matched the single-pod references.
- A duplicate worker on the second pod failed with exit 1 and an exclusive
  worker-claim error. The original paused HDF5 partial, worker claim, and shard
  claim all retained their SHA-256 values.
- Worker 8856 was killed on e374159d3660 and its absence was checked before
  explicit recovery on 5b20edd19fd3. Recovered background shard-00000 matched
  the reference byte for byte; no partial files or claims remained.
- The recovery was after confirmed process death, not a whole-pod shutdown.
  The preregistration's wording about stopping the old pod is therefore
  narrower than the actual test; do not claim a whole-pod-loss experiment.

These observations establish tested operational behaviors, not a throughput
guarantee at full fleet scale or a scientific audit of the ungenerated S4 data.

## Conditional reference fleet

This is an example assignment, not a purchase instruction or a claim of
current availability. Actual pod hostnames and costs must be recorded before
launch. Never reuse old hostname assumptions after replacing a pod.

| Slot | Example pod size | Active workers | Global indices |
|---|---:|---:|---|
| A | 32 CPUs | 30 | 0--29 |
| B | 32 CPUs | 30 | 30--59 |
| C | 32 CPUs | 30 | 60--89 |
| D | 32 CPUs | 30 | 90--119 |
| E | 8 CPUs | 8 | 120--127 |

This example rents 136 nominal CPUs for 128 active single-threaded workers.
It uses tested local concurrency levels rather than assuming that 32 workers
on a 32-CPU pod have been benchmarked. Heterogeneous 16- or 8-CPU alternatives
may fill the same disjoint global ranges; their actual progress must determine
the revised ETA. Do not change the global count or worker seed on migration.

## Reproducible timing calculation

Use whole-batch wall throughput, including startup, rather than summing
individual worker rates. Assume each pod runs its assigned background and
targeted components sequentially and all pods operate in parallel.

For each 30-worker pod:

    background = 138240000 * 30/128 / (360000/342)
    targeted   =  92160000 * 30/128 / (240000/261.918)
    total      = (background + targeted) / 3600 = 15.09795 hours

For the 8-worker pod:

    background = 138240000 * 8/128 / (96000/333.020)
    targeted   =  92160000 * 8/128 / (64000/249.612)
    total      = (background + targeted) / 3600 = 14.5658 hours

The forecast is the slowest assigned pod (15.09795 hours), not total work
divided by a naive sum of rates. A 25% increase in elapsed generation time
gives 18.8724375 hours, leaving 5.1275625 hours out of 24 for provisioning,
coordination, audits and recovery. This is an engineering allowance, NOT a
confidence interval or a measured upper bound. It may be insufficient.

Important uncertainty: short cold-start batches, variation in worker/protocol
mix, unequal host capacity, shared-volume contention at 128 writers, validation
cost at 23,040 shards, availability delays, and failed/restarted workers.

## Deployment and deadline checks

1. Complete the production plan, distribution-audit/sealing path and
   hostname-checked launch instructions before renting idle capacity.
2. Record the actual hostname, CPU allocation, assigned global range, pinned
   revision and displayed hourly price for each pod. Time urgency does not
   authorize unknown or unlimited spending.
3. Keep recovery disabled for ordinary launches. Reassign a claimed range only
   after confirming its old processes cannot write; never steal live claims.
4. Evaluate the ETA from completed production shards per assigned range and
   component. Account for every unstarted range and time already spent.
   Do not rerun the already passed canary suite as a routine pretext to scale.
5. If the forecast no longer fits 24 hours, report this immediately. A move to
   faster capacity requires safe range handoff; launching duplicate workers
   cannot accelerate an occupied range. Additional rented capacity is not
   useful unless it can own an idle range or safely replace a slower owner.
6. Completion still requires exact coverage, physical hashes, protocol/split
   and activity-support gates, composite sealing, and no live writers. Never
   mark the corpus ready merely because a time limit or done count was reached.
