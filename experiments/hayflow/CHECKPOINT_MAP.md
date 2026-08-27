# HayFlow checkpoint and experiment map

Status date: 2026-08-27. The machine-readable source of this map is
`checkpoint_registry.json`.

## Executive decision

The strongest defensible checkpoint family for the first report is **05j-n,
evaluated by the preregistered 05j-o fresh test**. It is not the newest
experiment. It is the only learned candidate that combines frozen checkpoint
selection, a disjoint fresh test, three passing seeds and an explicit model
authorization. Its authorization is strictly **one-step only**.

The canonical reported unit is the three-seed family and its ensemble mean,
not a seed selected after reading fresh-test outcomes:

| Model unit | Fresh-test RMSE | Status |
| --- | ---: | --- |
| Three-seed ensemble | **0.3308 mV** | Canonical one-step report model |
| Seed 17 | 0.4037 mV | Passed |
| Seed 29 | 0.3774 mV | Passed; best internal calibration before fresh test |
| Seed 43 | 0.3775 mV | Passed |
| Persistence | 2.5763 mV | Baseline |
| Frozen H2 | 2.7445 mV | Baseline |

If a single checkpoint file is operationally required, seed 29 is the cleanest
representative because it already had the lowest internal-calibration RMSE
(0.8125 mV) before the fresh test was opened. The report must not say that the
fresh test selected seed 29. The registered candidate remains the complete
three-seed family.

## Canonical one-step architecture

The system is a frozen HayFlow-Hines H2 core followed by the registered
direct-tree residual decoder:

- canonical 642-segment Hay morphology;
- causal `U_realized` input through the frozen H2/tree feature path;
- 226 fixed multiscale tree features per segment;
- learned 16-dimensional segment embedding;
- shared `242 -> 96 -> 96 -> 1` SiLU MLP;
- output represented as a 120 mV tanh-bounded residual;
- 43,009 trainable decoder parameters, derived from the registered dimensions;
- frozen H2 parameters are additional and are not counted by that decoder-only
  number.

The exact reconstruction chain is:

1. `hayflow_hines_canary_v2.zip`, containing the frozen H2 checkpoint
   `checkpoints/canary_models.pt`;
2. `hayflow_hines_regenerative_decoder_refit.zip`, containing
   `direct_tree_refit_seed17.pt`, `seed29.pt` and `seed43.pt`;
3. `hayflow_hines_regenerative_fresh_test.zip`, containing the immutable 05j-o
   evaluation evidence.

The archive names and SHA-256 identities are recorded in
`checkpoint_registry.json`.

## Why this is not the best free-running model

The same frozen 05j-n family failed 05k autoregressive rollout. At 8 ms the
three checkpoints reached 76.70, 95.47 and 62.72 mV RMSE. The 0.3308 mV result
therefore establishes an excellent one-step flow-map decoder, not a stable
autonomous simulator.

At present there is **no promoted autonomous free-running checkpoint**.

## Recursive frontier

Different experiments answer different questions and must not be collapsed
into one leaderboard:

| Category | Experiment / arm | 8 ms RMSE | Interpretation |
| --- | --- | ---: | --- |
| Best numeric nonlearned baseline | 06b-o passive Hines checkpoint zero | **7.427 mV** | Physical prior; neural source was not learned |
| Best frozen scalar train-only audit | 06b-h, source scalar `alpha=0.75` | 11.537 mV | Sealed train component; severe quiet/moderate and soma weaknesses |
| Best analytic noncheckpoint | 06b-j, `region_raw_voltage` lookup | 13.538 mV | Causal analytic lookup, not a trainable checkpoint |
| Best learned recursive diagnostic | 06b-r, auxiliary off + pushforward 4 ms + stability off | **14.720 mV** | Zero physical violations, but not promoted |
| Support-matched passive reference for 06b-r | 06b-r | 18.162 mV | Baseline |
| Previous learned 06b-q recursive arm | 06b-q | 46.938 mV | Recursively unstable |

The 06b-r arm is the current learned rollout frontier: it improves over its
support-matched passive baseline by 18.95% and reduces the previous 06b-q
error by 68.64%, with zero physical-voltage violations. It nevertheless fails
the registered no-seed-regression calibration gate because seed 61043 regresses
by 10.08%. It also receives teacher mechanism state during rollout, which is
not yet deployment-ready. It is therefore a valuable architecture result, not
the canonical checkpoint for the report.

## Minimal lineage for the report

1. **05b:** the original HayFlow-Hines H2 canary failed its full gate, but
   outperformed ConvGRU on voltage and branching.
2. **05j-n / 05j-o:** expanded regenerative support plus a refit direct-tree
   decoder produced the validated one-step champion (0.3308 mV ensemble).
3. **05k / 05k-b:** frozen rollout exposed closed-loop state-distribution shift
   and invalidated the one-step family as an autonomous simulator.
4. **05l--05t:** recurrent and representation canaries found real signals but
   no robust deployable topology/state contract.
5. **06a--06b-h:** atomic experiments isolated learnable state and voltage
   directions, optimization scaling and a useful fixed amplitude correction.
6. **06b-o--06b-q:** the passive Hines prior proved stabilizing; effective-source
   and ordered-event learners remained unstable or causally submaterial.
7. **06b-r:** pushforward boundary exposure causally repaired much of the
   recursive instability, but cross-seed calibration robustness remains open.

## Claims that are safe now

- HayFlow has a decision-grade, fresh-tested **one-step** checkpoint family.
- Its ensemble one-step RMSE is 0.3308 mV on the preregistered 05j-o fresh test.
- This one-step success did not compose autoregressively.
- Pushforward exposure is the strongest causally supported recursive training
  intervention found so far.
- No autonomous free-running HayFlow checkpoint is currently validated.

These qualifiers should remain attached to every table, abstract sentence and
figure caption in the first report.
