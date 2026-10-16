# 🔬 IE-P6 — audited result

Kaggle v1 COMPLETE; scientific revision `c5b3819eeaa6b8e4522129ee309e468cb4fa2d8a`
verified in worker revision file, final report and setup log. About9.5minutes
notebook runtime,30ktraining steps.210timing cells and210numeric preflights,
zero graph failures. Read-only local audit replayed all35maps on CPU with maximum
path-metric discrepancy1.17e-6. Regenerated path float64 discrepancy2.84e-14mV
(cross-platform elementary math); saved paths used for CPU replay.

## ✅ Main result: learned local maps

All six learned piecewise-linear coefficient operators pass ALL registered
quality gates and the batch642 CUDA Graph resident-cost criterion (median
saved fraction>=10%, paired interval lower bound>0). No automatic promotion.

| Map | Fresh gate RMSE | Path gate RMSE | Historical V RMSE mV | Graph saving vs formula |
| --- | --- | --- | --- | --- |
| local81 seed17 | 4.02e-5 | 8.67e-5 | 0.000713 | 10.12% |
| local81 seed29 | 3.97e-5 | 9.22e-5 | 0.001115 | 10.05% |
| local81 seed43 | 3.98e-5 | 8.69e-5 | 0.001010 | 10.63% |
| local161 seed17 | 1.92e-5 | 5.12e-5 | 0.001007 | 10.59% |
| local161 seed29 | 2.16e-5 | 4.26e-5 | 0.001312 | 10.49% |
| local161 seed43 | 1.95e-5 | 6.83e-5 | 0.000979 | 10.70% |

Local81 occupies11,084bytes,local16121,964bytes including scale, versus9,420bytes
for MLP64 and87,176bytes for fine table deployment. This is NOT a reduction in
parameters versus MLP64. It is a local learned numerical coefficient table,
initialized from neural predictions and optimized against analytic teacher
targets, not a dense neural-network speedup. Initialization projection error
is separately retained. First selected local161 checkpoints are1k/10k/1k;
local813k/30k/30k. More updates are not monotonically better: dev scores can
deteriorate, especially local161. Freeze before fresh prevented fresh selection.

The analytic .25mV LUT still passes quality and saves25.32% in the SAME graph
benchmark. Thus learned local maps do not dominate the best numerical baseline;
they provide a smaller-memory cost/quality tradeoff. Ordinary compiled local
costs are near parity with formula, with intervals spanning parity, not a
confirmed speed advantage. The two local resolutions have similar costs.
Historical error below formula in one case is compatible with error cancellation,
not proof of greater physical accuracy than the analytic equations.

## 🧪 Factorial matrix

Standard continuation A reduces joint fresh gate/path score versus its P5parent
30.2%,34.7%,12.1% for17/29/43. A passes all quality gates2/3; seed17 still fails
historical V maximum (0.1111mV versus0.1). Continuation is useful but does not
isolate additional updates from new training data/optimizer reset/LR change.

Targeted B improves the registered worst-case joint score versus A by
21.9%,25.3%,24.2% (median24.2%,consistent3/3). However, fresh gate RMSE and path
RMSE worsen slightly3/3, historical V RMSE improves only1/3, and B passes joint
quality1/3. This supports targeted worst-case coverage, NOT universal accuracy.

Temporal C worsens the joint score versus A3/3; D worsens versus B3/3. Factorial
interactions are positive0.0823/0.1080/0.0505 (lower score is better), contrary
to the proposed complementarity. C passes quality1/3; D0/3. This rejects the
tested8step normalized temporal objective/weight, not all recurrent supervision.
All12dense members remain slower than formula in graph mode. Fine tabulation
passes conversion12/12 and retains the respective quality classifications;
that is deployment of each learned function, not repair of its bias.

## 🔍 Attribution and objective probes

On the NEW P5standard parents, exact z swaps reduce historical V RMSE to
0.002013/0.001297/0.001243mV and Ca maxima to1.45e-7/2.75e-7/1.36e-7mM.
Exact calcium-group coefficients also strongly reduce Ca error. Exact u alone
does not reliably repair the integrated error. These are conditional oracles,
not deployable candidates or a unique mechanism proof.

The old balanced logit term has gradient norm about84x/219x/169x the base
objective at these probes; cosine is near zero (-.0375,.0108,.0466).
Constant400 gradients are positively aligned (.335–.451). This is stronger
evidence for scale imbalance than for a universal conflicting-gradient account.
It is not a causal ablation of the P5training trajectory.

## 🔒 Scope

Same17V-dependent gates and explicit18state; SK,calcium,readout,synapses,solver
and dt0.1ms unchanged. New fresh gate/path tests only after freeze; historical
Task35 already opened and diagnostic. One T4/run, no fresh native confirmation,
full-neuron proof, end-to-end speed claim or operational promotion. Archive
includes all curves,35map outputs,210timing cells,18oracle interventions
and checkpoints, including unsuccessful branches.
