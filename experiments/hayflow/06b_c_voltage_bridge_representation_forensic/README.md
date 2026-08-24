# HayFlow 06b-c - voltage-bridge representation forensic

This is the primary experiment after the registered 06b-b result.  It keeps
the six mechanism-STATE updaters and all three 06b-b voltage bridges frozen.
It compares, on train-derived disjoint roles and three paired seeds:

- the frozen local bridge;
- additional optimization of the same local bridge;
- a zero-initialized residual using the authentic NEURON segment tree;
- the same residual with a fixed relabelled tree while segment features remain
  unmoved.

The authentic and relabelled residuals have the same parameters,
initialization, minibatch stream, and optimizer budget.  Therefore their
difference tests whether the optimizer exploits authentic morphology rather
than merely benefiting from added capacity.  This component experiment does
not authorize a full neuron, a fresh test, or mass data generation.

Notebook: `notebooks/06b_c_voltage_bridge_representation_forensic.ipynb`.
