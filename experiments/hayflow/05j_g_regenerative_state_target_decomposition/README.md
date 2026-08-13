# HayFlow 05j-g - regenerative state-target decomposition

Status: implemented; Kaggle execution pending.

05j-f established that adding region/mechanism expert capacity does not repair
the localized regenerative voltage error. This notebook asks a narrower causal
question before another neural architecture is trained: is the missing voltage
correction already identifiable from the boundary state at `t`, or is it tied
to the internal state transition that unfolds during the next millisecond?

The frozen three-seed direct-tree ensemble is retained. Non-voltage teacher
state coordinates are partitioned a priori into calcium-channel, calcium
homeostasis, sodium, potassium, Ih, NMDA-synaptic and other-synaptic groups.
For each segment and group, signed mean, RMS and maximum
absolute standardized values are computed for:

1. the causal boundary state at `t`;
2. the teacher state delta from `t` to `t+1 ms`, used only as a diagnostic
   oracle;
3. a capacity-matched spatially shifted version of that oracle.

The future voltage coordinate is excluded exactly once for every one of the
642 segments. Fixed per-segment ridge probes are selected only by grouped-pair
cross-validation inside the fit support. Development is evaluated afterward;
held-out inputs and rollout remain sealed. Individual state-group oracle probes
localize any signal. A per-segment intercept-only probe removes apparent gains
from static offsets, while the aligned-versus-shifted comparison tests whether
the effect depends on the correct spatial correspondence rather than merely on
extra feature width.

The oracle can diagnose a missing transition closure but can never authorize
itself as a deployable input. The output routes the next experiment toward an
explicit causal state input, a joint regenerative state-transition head,
support expansion, or a regime-conditioned voltage objective. No candidate
training, micro-rollout, or full training is authorized by 05j-g.

Expected artifact: `hayflow_hines_regenerative_state_decomposition.zip`.
