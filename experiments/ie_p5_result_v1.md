# IE-P5 — Accuracy repair and inference deployment

Kaggle v1 completed successfully at scientific revision
`5825ea21be144342e16102f42b84bc79f2c789ba`. All 138 timing preflights passed,
with zero CUDA Graph failures. Local audit independently replayed all 23 maps;
maximum CPU/GPU forced-path metric discrepancy was 5.96e-7.

## Accuracy

The standard refit reduced historical closed-loop voltage RMSE by
4.08x / 4.84x / 3.48x for seeds 17/29/43, to 0.00958 / 0.00826 / 0.00912 mV.
Fresh gate RMSE was 0.000628 / 0.000749 / 0.000581. All standard refits passed
imposed-path gate criteria. However, fresh maximum gate errors remained
0.01168 / 0.01364 / 0.01041 (limit 0.01); calcium maxima remained
5.047e-6 / 5.182e-6 / 6.107e-6 mM (limit 5e-6). Seed43 also failed historical
voltage maximum (0.11471 mV versus 0.1). No learned candidate passed joint quality.

The balanced-loss package was worse than standard in all three seeds on fresh
gate RMSE, imposed-path RMSE and historical voltage RMSE. This contradicts the
registered package, not each extra term separately. Development selected 30k
steps for standard and 10k for balanced. Gains against old frozen checkpoints
conflate additional budget, new data and identically reset Adam moments;
they do not establish that budget alone caused the improvement.

## Execution

All six 0.25mV neural tables passed sampled conversion tolerance (maximum map
delta 2.16e-5–2.32e-5, limit 5e-5); all 1mV tables failed it
(3.45e-4–3.71e-4). Fine tables largely preserve neural bias, not repair it.
At batch642, CUDA Graph fine deployment saves about 16.5–17.3% versus formula
and about 37% versus its MLP. Ordinary compiled deployment remains slower than
formula. Fine tables occupy 87,176 bytes versus 9,420 bytes of network weights
and scales: about 9.25x memory. This is tabulation of learned predictions,
not a new neural architecture, parameter compression or direct MLP speedup.
The analytic LUT passes quality and saves about 17.0% in the same graph backend.
These are resident-core measurements, not end-to-end simulation speedups.

## Attribution and limits

Oracle coefficient interventions apply ONLY to the old frozen IE-P4 models.
Replacing z with exact coefficients reduces their historical voltage RMSE
4.00–7.37x. Replacing the calcium coefficient group strongly reduces calcium
error. Replacing u alone does not repair it and worsens calcium maxima in all
three seeds; compensation and nonlinear interaction remain possible.
These interventions neither identify a unique cause nor attribute errors in
the new refits, which were not subjected to these swaps. Oracles are not
selection/cost eligible. Opened Task35 traces are diagnostic, not a fresh native
confirmation. No operational candidate is promoted and no new run is launched.

Raw results, checkpoints, per-family probes, timing blocks, selection freeze
and independent audit are retained in `results/ie_p5_kaggle_v1/`.
