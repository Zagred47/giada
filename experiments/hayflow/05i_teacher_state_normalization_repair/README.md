# HayFlow 05i - teacher-state normalization repair

Status: completed on Kaggle; the registered input contract did not pass.

This experiment is an input-contract repair, not a model-training experiment.
It consumes the immutable targeted composite and the exact 05b--05h artifacts.
The 05h archive is accepted only after its archive hash, artifact-index hash,
final-report hash, and every indexed member are verified.

The new state scale is fitted on train states only. It pools non-degenerate
train scales by exact variable, mechanism, category, and transform family and
uses pre-registered semantic absolute floors as the last fallback. Teacher
values and semantic transforms are not changed. Development and held-out
inputs are audit-only and never contribute to the fit.

The result is decision-grade only for the numerical input contract. H2 is
frozen and is evaluated with authentic and zeroed causal inputs. Held-out
future voltages and event labels remain sealed. There is no candidate-head
training, rollout, or path to full training in this notebook.

Expected artifact: `hayflow_hines_state_normalization_repair.zip`.

## Registered result

The archive and all 17 indexed members passed size and SHA-256 verification.
The repair lifted 12,507 coordinates and reduced the held-out maximum
standardized teacher state from approximately `6.97e8` to `113.73`, an
improvement factor of approximately `6.13e6`. There were no nonfinite values,
and the held-out global fraction above `|z|=8` was only `0.0813%`, below the
registered `1%` limit. Nevertheless, the absolute `|z| <= 100` gate failed.

Only two coordinates remained above the absolute limit. Both are
`NetCon.weight[4]` belonging to inhibitory `ProbUDFsyn2` synapses, for which
that slot is the absolute `tsyn` timestamp. The same raw slot name denotes
`Pr` for `ProbAMPANMDA2`, so pooling by `weight[4]` mixed two different
physical variables. This is a semantic state-encoding defect. The threshold
must not be relaxed and a global multiplier-only patch is not recommended.

The frozen-H2 audit itself passed after repair. Held-out/train maximum-norm
ratios were `0.331` with authentic causal inputs and `0.698` with causal inputs
zeroed; the corresponding maximum train-standardized features were `18.86`
and `14.09`, with no nonfinite values. Thus the catastrophic H2 excursion from
05h has been removed, but candidate-head training remains blocked until the
two timestamp coordinates are represented causally and semantically.
