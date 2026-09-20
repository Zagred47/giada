# GIADA teacher mechanism inventory

- Teacher commit: `074c4666300a8ad246601dab179a97a6942f0f29`
- Combined source SHA-256: `6d74150e95f3606fe01aeb70310fe5365f9897cd594f97b75efc24839b5dbba1`
- NMODL files: **18**

| Mechanism | Kind | STATE | Ions read/write | Currents | Solver | NET_RECEIVE |
|---|---|---|---|---|---|---|
| `Ca_HVA` | SUFFIX | m, h | ca: R=eca W=ica | ica | states:cnexp | no |
| `Ca_LVAst` | SUFFIX | m, h | ca: R=eca W=ica | ica | states:cnexp | no |
| `CaDynamics_E2` | SUFFIX | cai | ca: R=ica W=cai | — | states:cnexp | no |
| `epsp` | POINT_PROCESS | — | — | i | — | no |
| `Ih` | SUFFIX | m | — | ihcn | states:cnexp | no |
| `Im` | SUFFIX | m | k: R=ek W=ik | ik | states:cnexp | no |
| `K_Pst` | SUFFIX | m, h | k: R=ek W=ik | ik | states:cnexp | no |
| `K_Tst` | SUFFIX | m, h | k: R=ek W=ik | ik | states:cnexp | no |
| `Nap_Et2` | SUFFIX | m, h | na: R=ena W=ina | ina | states:cnexp | no |
| `NaTa_t` | SUFFIX | m, h | na: R=ena W=ina | ina | states:cnexp | no |
| `NaTs2_t` | SUFFIX | m, h | na: R=ena W=ina | ina | states:cnexp | no |
| `ProbAMPANMDA2` | POINT_PROCESS | A_AMPA, B_AMPA, A_NMDA, B_NMDA | — | i_AMPA, i_NMDA | state:cnexp | yes |
| `ProbAMPANMDA3` | POINT_PROCESS | A_AMPA, B_AMPA, A_NMDA, B_NMDA | — | i_AMPA, i_NMDA | state:cnexp | yes |
| `ProbAMPANMDA_EMS` | POINT_PROCESS | A_AMPA, B_AMPA, A_NMDA, B_NMDA | — | i | state:unspecified | yes |
| `ProbGABAAB_EMS` | POINT_PROCESS | A_GABAA, B_GABAA, A_GABAB, B_GABAB | — | i | state:unspecified | yes |
| `ProbUDFsyn2` | POINT_PROCESS | A, B | — | i | state:cnexp | yes |
| `SK_E2` | SUFFIX | z | k: R=ek W=ik; ca: R=cai W=- | ik | states:cnexp | no |
| `SKv3_1` | SUFFIX | m | k: R=ek W=ik | ik | states:cnexp | no |

> This inventory is descriptive. Causal classification is intentionally deferred to Task 0.2.
