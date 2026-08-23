# HayFlow 06b - optimized explicit-state updater canary

Status: implemented and preregistered; awaiting Kaggle execution.

06a-b established a large optimization-budget effect when the teacher endpoint
voltage was available, but it did not establish a causal updater. This canary
restores `causal_start_voltage` as the primary arm and retains the linear
teacher-endpoint representation only as a privileged paired reference.

Both arms use the same 7,238-parameter ceiling, 1200-step budget, train-derived
fit/calibration/development roles and three paired optimization seeds. No
voltage microtrace is read. Each causal run must clear the 2% floor; the median
must reach 10%, retain at least 70% of the endpoint gain and remain positive in
at least 70% of the 18 semantic mechanism groups. Semantic-macro, active-state
and common-window recursive rollout gates prevent an aggregate-only success.

Passing 06b does not authorize the full neuron. It authorizes only a small
coupled voltage/state canary in which the explicit state updater and membrane
solver are composed and tested for autoregressive consistency. Validation,
tests, fresh data, capacity sweeps and mass generation remain sealed.
