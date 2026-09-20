# GIADA atomic data contract

- Teacher commit: `074c4666300a8ad246601dab179a97a6942f0f29`
- Canonical macro-step: **1.0 ms**
- Families: **7**
- Mechanism bindings: **20**

| Family | Causal state | Inputs | Solver | Sufficiency boundary |
|---|---|---|---|---|
| `voltage_dependent_gate` | gate occupancy x_t for every declared STATE gate | membrane-voltage control path V(tau), tau in [t,t+dt] | exponential/Rush-Larsen update when voltage is held; path-aware or jointly coupled update otherwise | (x_t,V_t) is sufficient only for the voltage-clamp constant-V playground; coupled use requires V(tau) or a joint voltage-state solver. |
| `calcium_dependent_gate` | calcium-dependent gate occupancy x_t | intracellular-calcium control path cai(tau); membrane voltage for analytic current readout | exponential gate update under held calcium; path-aware or jointly coupled update otherwise | Endpoint cai alone is not assumed sufficient when calcium changes materially inside the step. |
| `ion_concentration_dynamics` | intracellular ion concentration at t | causal ionic-current path over the step | exponential decay with current forcing; integrate forcing on the native or audited micro-grid | Initial concentration plus the within-step current forcing and fixed parameters is the atomic Markov contract. |
| `event_driven_synapse` | all synaptic decay/rise STATE; short-term-plasticity NET_RECEIVE weights/state; synapse RNG state | ordered presynaptic events with intra-step timestamps and weights; postsynaptic voltage path for voltage-dependent current | exact ordered event jumps plus analytic/exponential inter-event decay | Event order, exact timestamps, complete synaptic/STP state and RNG state are required for replay. |
| `prescribed_time_dependent_point_process` | phase/time relative to onset when the source remains active; otherwise empty | absolute or relative time over the step; onset/delay control | analytic waveform evaluation | The contract is Markov only if phase/time-since-onset is represented explicitly or derivable from the supplied clock and onset. |
| `passive_membrane` | segment voltage V_t | total causal transmembrane-current path; axial-current contribution or joint axial solve; externally applied current path | implicit or structure-preserving membrane update; coupled with axial tree solve in deployment | V_t is sufficient only together with all causal current forcing and the axial boundary/coupling contract. |
| `axial_coupling` | voltage vector on the complete instantiated segment tree | local membrane source for every segment over the step | Hines/tree-structured implicit solve or numerically equivalent differentiable solver | The complete voltage vector, full tree coefficients and causal local membrane sources are required; isolated segment transitions are insufficient. |

## Global anti-leakage rule

Teacher endpoint values and privileged observables are labels or diagnostics. They are not deployment inputs unless an upstream causal component has generated them before the downstream update.

## Voltage-path distinction

A constant-voltage atomic playground may use the exponential gate solution from `(x_t, V_t)`. The coupled neuron may not silently reuse that assumption: it must receive an intra-step control path or solve voltage and internal state jointly.
