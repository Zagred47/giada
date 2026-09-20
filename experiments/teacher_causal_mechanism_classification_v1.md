# GIADA teacher causal mechanism classification

- Teacher commit: `074c4666300a8ad246601dab179a97a6942f0f29`
- NMODL mechanisms: **18**
- Runtime mechanisms made explicit: **2**

| Mechanism | Primary causal family | Secondary behavior | Evidence |
|---|---|---|---|
| `Ca_HVA` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=eca; WRITE=ica |
| `Ca_LVAst` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=eca; WRITE=ica |
| `CaDynamics_E2` | `ion_concentration_dynamics` | — | STATE=cai; READ=ica; WRITE=cai |
| `epsp` | `prescribed_time_dependent_point_process` | — | — |
| `Ih` | `voltage_dependent_gate` | — | STATE=m; v |
| `Im` | `voltage_dependent_gate` | — | STATE=m; v; READ=ek; WRITE=ik |
| `K_Pst` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=ek; WRITE=ik |
| `K_Tst` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=ek; WRITE=ik |
| `Nap_Et2` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=ena; WRITE=ina |
| `NaTa_t` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=ena; WRITE=ina |
| `NaTs2_t` | `voltage_dependent_gate` | — | STATE=h,m; v; READ=ena; WRITE=ina |
| `ProbAMPANMDA2` | `event_driven_synapse` | stochastic_release, synaptic_conductance_dynamics, voltage_dependent_nmda_current | STATE=A_AMPA,A_NMDA,B_AMPA,B_NMDA; v; NET_RECEIVE; RNG=erand,setRNG |
| `ProbAMPANMDA3` | `event_driven_synapse` | stochastic_release, synaptic_conductance_dynamics, voltage_dependent_nmda_current | STATE=A_AMPA,A_NMDA,B_AMPA,B_NMDA; v; NET_RECEIVE; RNG=erand,setRNG |
| `ProbAMPANMDA_EMS` | `event_driven_synapse` | stochastic_release, synaptic_conductance_dynamics, voltage_dependent_nmda_current | STATE=A_AMPA,A_NMDA,B_AMPA,B_NMDA; v; NET_RECEIVE; RNG=setRNG,urand |
| `ProbGABAAB_EMS` | `event_driven_synapse` | stochastic_release, synaptic_conductance_dynamics | STATE=A_GABAA,A_GABAB,B_GABAA,B_GABAB; v; NET_RECEIVE; RNG=setRNG,urand |
| `ProbUDFsyn2` | `event_driven_synapse` | stochastic_release, synaptic_conductance_dynamics | STATE=A,B; v; NET_RECEIVE; RNG=erand,setRNG |
| `SK_E2` | `calcium_dependent_gate` | — | STATE=z; READ=cai,ek; WRITE=ik |
| `SKv3_1` | `voltage_dependent_gate` | — | STATE=m; v; READ=ek; WRITE=ik |
| `Passive membrane` | `passive_membrane` | — | C_m dV/dt plus passive membrane current |
| `Axial coupling` | `axial_coupling` | morphology_dependent_cable_dynamics | tree topology, segment geometry and axial resistance |

## Methodological boundary

This taxonomy identifies causal subsystem families. It does not yet define the minimal atomic data contract or select a neural architecture; those decisions belong to Task 0.3 and later tasks.
