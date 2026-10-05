# GIADA Task 29 — external-clamp ionic/passive diagnostic

Kaggle run `giada-task29-external-clamp-1245b70` at immutable commit
`1245b70c021e7c67270144c11f26addba0b1c4e4` completed. The archived ZIP,
source hashes, contract, row support and pass logic were independently checked
by `scripts/audit_task29_result.py`.

The native NEURON passive-current audit matched the analytic passive current
at all six preregistered voltages (maximum absolute error 0 mA/cm²). All 240
frozen hybrid comparisons passed: four paths × five conductance panels × four
family/arm combinations × three seeds. Both-arm primary: 120/120 passed, with
zero gate-occupancy violations.

| Both-arm family | Worst gate RMSE | Worst individual normalized current RMSE | Worst normalized clamp-demand RMSE |
| --- | ---: | ---: | ---: |
| Independent | 0.000255 | 0.001452 | 0.000373 |
| Shared heads | 0.000370 | 0.000852 | 0.000179 |

The preregistered limits were 0.005, 0.01 and 0.01, respectively. The
none-arm ablations also passed, but were not model selectors or alternatives
for rescuing a failed both-arm result.

This establishes a narrow scientific result: under externally imposed
voltage/calcium, the frozen ionic block remains accurate when analytic
passive and capacitive terms are included in the requested external current
budget. The balance equation itself is an algebraic identity; its residual
does not validate membrane dynamics. The informative measurement is the
hybrid-versus-formula clamp-demand difference. No authentic full NEURON
voltage-clamp trajectory, autonomous voltage, calcium feedback, axial
coupling, synapses, or speedup was tested. Gate D performance remains NO-GO;
Task30 is not automatically authorized by this report.
