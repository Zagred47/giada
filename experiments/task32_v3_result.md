# GIADA Task 32 v3 — dynamic calcium feedback

The preregistered phase-aligned v3 completed on Kaggle at revision `05bdabb` after a technical process-isolation repair. The archive and exact report are in `experiments/results/task32_v3_kaggle_05bdabb/`. The ZIP report equals the separate JSON byte-for-byte after parsing; `process_status.returncode=0`; source checkout was clean. No retraining or fresh-data selection occurred.

**Decision: pass within the registered one-compartment fixed-E_Ca domain.** Native E_Ca deviation was 0 mV. Formula-vs-native confirmation floor at 40 ms: pooled voltage RMSE 0.000962566 mV, worst episode 0.002056732 mV, worst calcium RMSE 4.043604e-8 mM, no occupancy or physical-voltage violations. All six frozen family×seed candidates passed the unchanged 40 ms gates:

| Frozen family | Seeds | Pooled V RMSE range (mV) | Worst episode V RMSE max (mV) | Worst calcium RMSE max (mM) | Worst SK-gate RMSE max |
| --- | --- | ---: | ---: | ---: | ---: |
| Independent | 17, 29, 43 | 0.013437–0.023818 | 0.062181 | 5.0203e-6 | 0.0004723 |
| Shared heads | 17, 29, 43 | 0.025345–0.044835 | 0.122577 | 7.6746e-6 | 0.0015905 |

The frozen-calcium negative control showed maximum differences of 0.012896 mM calcium, 0.99909 SK gate, and 0.36613 mV voltage RMSE, confirming an active coupled feedback effect. The 80 ms results remain diagnostic, not the registered primary gate.

This confirms a narrow autonomous voltage–calcium–SK integration with frozen learned rates and analytic currents. It does **not** validate synapses, axial coupling, morphology, stochasticity, long autoregressive stability, a full multicompartimental neuron, or hardware speedup. The first `d1ed300` attempt and instrumented `1aa0583` attempt were technical interruptions, not model failures; both partial packages remain archived. The latter already established the same passing native floor, but only `05bdabb` produced model metrics.
