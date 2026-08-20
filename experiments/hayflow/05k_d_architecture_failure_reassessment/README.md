# HayFlow 05k-d - architecture failure reassessment

Status: completed and independently verified.

This small decision notebook consumes only the exact 05k-c development
artifact. It performs no neural inference or training and never loads 05j-o.
It checks whether H2 free recurrence is much worse than persistence, whether
state-consistent recommit is immaterial, and whether teacher boundary resets
recover a useful one-step map.

If all conditions hold, the free-running H2 latent recurrence is retired as
the autoregressive candidate. The only authorized next experiment is a fixed
rollout-aware canary comparing a morphology-aware GraphGRU with an ordered
ConvGRU control. Both must use the authentic causal synaptic front-end and no
future teacher state. This decision does not authorize full training or a new
fresh test.

Expected artifact: `hayflow_hines_architecture_failure_reassessment.zip`.

## Registered result

The downloaded archive SHA-256 is
`f6d036156f9fbec1344388759632f060dc7d6126a552274f6c97d290f4de8685`.
All three members declared by its artifact index passed independent size and
SHA-256 verification.  The run used code revision
`cb7c6a68cbbb157923323d28fdefe3a2bc695caa` and verified all 41 indexed
members of the exact 05k-c prerequisite.

Every preregistered condition passed.  At 8 ms the free-running H2 recurrence
had `70.0338 mV` RMSE versus `7.17037 mV` for persistence, a ratio of
`9.76711`.  The state-consistent recommit improved the three seeds by only
`3.64%`, `3.05%`, and `0.53%`, all below the fixed 10% materiality threshold.
Conversely, resetting the boundary to the teacher recovered a useful local
map, with oracle-to-persistence ratios between `0.3953` and `0.5412`.

The registered diagnosis is therefore
`RETIRE_FREE_RUNNING_H2_LATENT_RECURRENCE`.  H2 is retired as an
autoregressive candidate.  No fresh test, full training, or mass generation is
authorized.  The only authorized next experiment is
`05l_rollout_aware_graphgru_vs_convgru_canary`.
