# GIADA Task 0.5 — atomic domain splits

No point-wise random split is used. Complete voltage windows and complete state/`dt` values are held out.

| Stratum | Role | Isolated axis | Cases |
|---|---|---|---:|
| `train` | `fit` | `support` | 8,096 |
| `interpolation_voltage_development` | `development` | `voltage` | 1,232 |
| `interpolation_voltage_test` | `sealed_test` | `voltage` | 1,232 |
| `interpolation_state_development` | `development` | `state` | 80 |
| `interpolation_state_test` | `sealed_test` | `state` | 80 |
| `interpolation_dt_development` | `development` | `dt` | 220 |
| `interpolation_dt_test` | `sealed_test` | `dt` | 110 |
| `boundary_singularity_test` | `sealed_test` | `voltage_boundary` | 264 |
| `boundary_state_test` | `sealed_test` | `state_boundary` | 160 |
| `boundary_support_edge_test` | `sealed_test` | `support_edge` | 176 |
| `ood_voltage_test` | `sealed_test` | `voltage_ood` | 704 |
| `ood_dt_test` | `sealed_test` | `dt_ood` | 220 |
| `factorial_stress_test` | `sealed_test` | `multi_axis_ood` | 8 |

## Leakage audit

- Total cases: **12,582**
- Cross-stratum duplicate cases: **0**
- Fit/evaluation overlap: **0**
- Guard bands adjacent to held-out voltage windows are excluded from both fitting and scoring.
- Single-axis OOD tests keep the other axes in support; the multi-axis stress test is reported separately.

## Embedded confirmation

No atomic case is relabelled as embedded evidence. Coupled-teacher confirmation remains a future, independently generated and sealed protocol.
