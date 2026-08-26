# 06b-m — continuous mixture-state architecture playground

This is the first architecture revision authorized by the terminal 06b-l
diagnostic.  Persistence and the frozen dynamic voltage update remain fixed;
only a small causal controller is trained to emit a per-segment convex mixing
coefficient.

The synchronized matrix isolates four factors: physiological observability,
temporal recurrence, authentic morphology and oracle-target distillation.  A
relabelled-tree control has exactly the same parameters, initialization and
minibatches as the authentic-tree arm.  Width and checkpoint sweeps form a
small scaling law without duplicated training trajectories.

All roles are historically reused train-only playground roles.  No result is
an independent confirmation and no outcome directly opens validation, test,
full training, mass-data generation or the coupled 06c canary.

The full width-16 matrix contains six arms. Only the local, authentic-tree
and relabelled-tree arms are repeated at widths 8 and 32, for 12 trajectories
per seed in total. Checkpoints at steps 0, 100, 200 and 400 expose both
capacity and optimization trends from the same trajectories.

The corresponding Kaggle entry point is
`notebooks/06b_m_continuous_mixture_state_playground.ipynb`. Its final cell
uses the project's browser-Blob ZIP download path and keeps notebook output
compact.

## Registered result

The verified run completed the full matrix but selected no candidate. Rich
physiological input and width both carry measurable signal; recurrence and
authentic tree messages do not outperform their aligned controls. The
optimal-blend auxiliary target makes alpha heterogeneous and partly
predictable, but conflicts with the recursive rollout objective. The formal
diagnosis is `MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS` and
the authorized continuation is a bounded mixture objective/coupling revision.
