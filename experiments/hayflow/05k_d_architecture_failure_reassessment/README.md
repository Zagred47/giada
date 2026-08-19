# HayFlow 05k-d - architecture failure reassessment

Status: implemented and locally verified; execution pending.

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
