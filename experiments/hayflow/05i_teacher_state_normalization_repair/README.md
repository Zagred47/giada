# HayFlow 05i - teacher-state normalization repair

Status: implementation ready; execution pending on Kaggle.

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
