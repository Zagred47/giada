# HayFlow-ESI architecture specification (phase 06)

## Registered architectural decision

The canonical autoregressive state is no longer a hidden GRU initialized once
from a compressed teacher sketch.  It is the explicit structured state

```text
S_t = (V_t, X_mechanism_t, X_ion_t, X_synapse_t)
```

and every dynamic component must be updated and supervised at each 1 ms
boundary.  Semantic identities are shared across segments; spatial ownership
and the authentic morphology remain explicit.

The intended full architecture is operator-split:

1. causal synaptic front-end driven by `U_realized`;
2. shared local mechanism-state updater;
3. local membrane-current/source decoder with privileged current supervision;
4. fixed morphology-aware implicit/Hines voltage coupling;
5. explicit propagation of the predicted state to the next macro-step.

Synaptic kinetics that admit an exact cheap update should remain analytic.
The first implementation keeps the full morphology and does not introduce a
global latent, morphology reduction, Mamba/S4 block or aggressive state
compression.

## Allen-Zhu-style experimental decomposition

Architecture feasibility is decomposed into atomic capabilities:

- state relaxation and equilibrium, including causal local ion context;
- voltage-conditioned mechanism kinetics;
- causal synaptic release and decay;
- passive axial propagation;
- regenerative threshold transitions;
- closed-loop composition.

Difficulty is measured over horizon (1--32 ms), spatial extent (coordinate,
segment, branch, full tree) and biological regime (rest, subthreshold, spike,
regenerative).  Comparisons must align data, shuffling, initialization,
capacity and optimization, and must report curves rather than one endpoint.

The planned gates are:

- `06a`: train-only atomic state playground and technical learnability pilot;
- `06b`: multi-seed state-updater canary and mini scaling law;
- `06c`: state/current/Hines operator-split composition canary;
- `06d`: frozen full-tree validation-only autoregressive go/no-go;
- sealed evaluation only after a genuine 06d GO.

Each failure maps to a different cause.  Failure with teacher interval voltage
implicates the state contract or local updater.  Success there but failure
with causal boundary voltage implicates state/voltage coupling.  Local success
followed by branch failure implicates the spatial operator.  Full-tree
short-horizon success followed by long-horizon drift implicates numerical or
training stability rather than state observability.
