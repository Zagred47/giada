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

## Registered result

The archive produced at code revision `3cbd913` passed independent ZIP, index,
member-size, member-SHA-256, and CRC verification.  The preregistered median
optimizer-budget gate passed: continuing the same local bridge improved
voltage gain over the frozen bridge by 2.69 percentage points at the median.

This signal is not uniform: seeds 61017 and 61029 improved by 2.69 and 5.87
points, while seed 61043 worsened by 3.29 points.  The correct conclusion is
therefore a conditional optimization signal, not an all-seed robust repair.

The downstream STATE result is more consistent than the global voltage RMSE:
continued local optimization improves STATE gain in all three seeds by 0.68,
0.89, and 1.51 percentage points.  Thus the bridge continuation contains a
robust causally useful signal even though global delta-V RMSE is not monotonic.
This exposes a likely objective-alignment issue that the next optimization
forensic must isolate.

Authentic and relabelled topology residuals were practically identical.  The
median authentic advantage was only 0.004 percentage points for voltage and
slightly negative for downstream STATE.  This rejects the tested
1,409-parameter topology-residual representation; it does not establish that
morphology is irrelevant to the neuron or to every possible architecture.

No full training, fresh-test generation, or mass dataset generation is
authorized. The logically independent professor-requested Branch-ELM sidecar
has now been closed by the corrective information-matched comparison. The
next primary activity is the nested coupling-aware local bridge optimization
scaling forensic.
