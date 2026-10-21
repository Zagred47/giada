# GIADA Task40 — audit dei 22 fallimenti

Audit post hoc; soglie e checkpoint invariati. Nessun nuovo training.

## Risultati

- **outcome**: Task40 technical execution valid; J1 264/264 imposed and autonomous trajectory checks pass, but 22/168 autonomous intervention case-seed checks fail. No GO to Task41.
- **localization**: All 22 failures concern calcium channels: 18 Ca_LVAst, 4 Ca_HVA. I3 and J1 fail identical cases. All are early_high protocols: 10 distinct intervention/source cases, not 22 independent protocols.
- **numerical_floor**: All 672 saved intervention verdicts reproduced. Recalculation maximum numeric discrepancy: 5.59e-10. Casting before subtraction changes no verdict. Formula FP32 versus FP64 passes 56/56 differential controls; candidates still fail 22/168 against FP64. Numerical roundoff alone is not supported.
- **causal_channel_control**: Frozen CPU replay matches 168/168 GPU differential flags. Learned calcium-only has 22 failures; learned all-except-calcium has 0. Restoring only Ca_LVAst leaves 4, only Ca_HVA leaves 18. Canonical substitutions are diagnostic oracles, not eligible candidates.
- **kinetic_control**: At fixed reference V/Ca and common initial state, learned Ca coefficients fail 22; oracle u still 22; oracle z gives 0; oracle m-z alone leaves 3. In this diagnostic, correcting relaxation coefficients suffices to remove failures; correcting equilibrium alone does not. This is a sufficiency result, not proof of a unique cause or learnability.
- **readout_control**: Reference V in readout leaves 22; reference m leaves 2; reference h leaves 17; reference m+h leaves 0. Dominant activation kinetics with a secondary inactivation contribution, not evidence that h can be ignored.
- **why_J1_did_not_fix**: Task39j learned corrections target only gates 8,9,12,13 (Nap_Et2,K_Pst). Calcium gates 0..3 remain I3. Their residual failure is not a regression introduced by 39j. Its atomic frozen-function method is a promising hypothesis to transfer, not proof of Ca repair.
- **scope**: The 32 historical native paths and 56 formula interventions are opened diagnostics, not fresh NEURON confirmation. Imposed mode mixes native base V/Ca with formula intervention V/Ca; consistent formula-path coefficient controls also reproduce 22 failures. Original NO-GO preserved.
- **stress**: Both known stress failures remain: Ca_LVAst offset (seed17, same_v_history case1) and Ca_HVA NRMSE (seed29, activation_boundary case0). Shared channels motivate testing a common repair but do not prove an identical cause.
- **decision**: Propose Task40b calcium kinetic precision diagnosis, not Task41. No training launched and no current threshold relaxed.

## Tutti i 22 confronti falliti

| Caso | Seed modello | Stimolo | Intervento | Canale | NRMSE | NMAX | Metriche fallite |
|---|---|---|---|---|---|---|---|
| 33 | 17 | source 1 | block_Ca_HVA | Ca_LVAst | 0.020315 | 0.056055 | nrmse, nmax |
| 35 | 17 | source 3 | block_Ca_HVA | Ca_LVAst | 0.011536 | 0.023744 | nrmse |
| 41 | 17 | source 1 | block_Nap_Et2 | Ca_LVAst | 0.028936 | 0.069605 | nrmse, nmax, peak_error_ms |
| 43 | 17 | source 3 | block_Nap_Et2 | Ca_LVAst | 0.013552 | 0.026878 | nrmse |
| 47 | 17 | source 7 | block_Nap_Et2 | Ca_LVAst | 0.012817 | 0.039603 | nrmse |
| 49 | 17 | source 1 | block_K_Pst | Ca_LVAst | 0.011976 | 0.035470 | nrmse |
| 57 | 17 | source 1 | block_SK_E2 | Ca_HVA | 0.013480 | 0.054042 | nrmse, nmax |
| 59 | 17 | source 3 | block_SK_E2 | Ca_HVA | 0.011438 | 0.045839 | nrmse |
| 73 | 17 | source 1 | injection_0.99 | Ca_LVAst | 0.011305 | 0.031700 | nrmse, offset_error_ms |
| 81 | 17 | source 1 | injection_1.01 | Ca_LVAst | 0.010698 | 0.029251 | nrmse |
| 33 | 29 | source 1 | block_Ca_HVA | Ca_LVAst | 0.016446 | 0.045651 | nrmse |
| 41 | 29 | source 1 | block_Nap_Et2 | Ca_LVAst | 0.025787 | 0.068732 | nrmse, nmax |
| 47 | 29 | source 7 | block_Nap_Et2 | Ca_LVAst | 0.011296 | 0.039262 | nrmse |
| 57 | 29 | source 1 | block_SK_E2 | Ca_HVA | 0.010238 | 0.042237 | nrmse |
| 73 | 29 | source 1 | injection_0.99 | Ca_LVAst | 0.011276 | 0.032818 | nrmse, offset_error_ms |
| 81 | 29 | source 1 | injection_1.01 | Ca_LVAst | 0.010718 | 0.030290 | nrmse |
| 33 | 43 | source 1 | block_Ca_HVA | Ca_LVAst | 0.016276 | 0.045526 | nrmse |
| 41 | 43 | source 1 | block_Nap_Et2 | Ca_LVAst | 0.025204 | 0.066851 | nrmse, nmax |
| 47 | 43 | source 7 | block_Nap_Et2 | Ca_LVAst | 0.013877 | 0.044087 | nrmse |
| 57 | 43 | source 1 | block_SK_E2 | Ca_HVA | 0.011397 | 0.046322 | nrmse |
| 73 | 43 | source 1 | injection_0.99 | Ca_LVAst | 0.010842 | 0.039212 | nrmse |
| 81 | 43 | source 1 | injection_1.01 | Ca_LVAst | 0.010061 | 0.039183 | nrmse |

## Proposta Task40b (da preregistrare)

{
  "status": "proposed_not_preregistered_or_launched",
  "task": "40b",
  "thresholds_changed": false,
  "baseline": "FrozenJ1",
  "candidate_scope": "LearnedCa_HVA/Ca_LVAst coefficient corrections only;noanalyticalgatefallback inpromotablecandidate.",
  "arms": [
    {
      "id": "B0",
      "variant": "FrozenJ1 no-training control"
    },
    {
      "id": "B1",
      "variant": "z-only m+h;32basis;1024independentuniformFITanchors;coefficientloss"
    },
    {
      "id": "B2",
      "variant": "B1 + independentpairedresponse/sensitivityloss"
    },
    {
      "id": "B3",
      "variant": "B1 with4096FITanchors"
    },
    {
      "id": "B4",
      "variant": "B2 with4096FITanchors"
    },
    {
      "id": "B5",
      "variant": "B1 with64basis"
    },
    {
      "id": "B6",
      "variant": "B1 butu+z;capacitymatchiffeasibleotherwiseexplicitparameterconfound"
    },
    {
      "id": "B7",
      "variant": "B1 butm-onlyz negative/restrictedcontrol"
    }
  ],
  "primary_contrasts": [
    "B1/B2andB3/B4:objective",
    "B1/B3andB2/B4:coverage",
    "B1/B5:capacity",
    "B1/B6:jointcorrection",
    "B1/B7:m-onlyrestriction"
  ],
  "checkpoints": [
    0,
    1000,
    3000,
    10000,
    30000
  ],
  "seeds": [
    17,
    29,
    43
  ],
  "safeguards": [
    "SharedFITstreams;independentholdout;noexactfailedpathtraining",
    "Samefrozenparent;nochangeabstractionlevel",
    "Coefficient,heldrelaxation,pairedresponseandautonomousmetrics",
    "Original40criteriaand72developmentchecksunchanged",
    "Canonicaloraclesdiagnosticonly",
    "Bothstressfamiliesafterfreeze;nonewauthorizationfromaverages",
    "Task41independentconfirmationonlyaftercausalqualification",
    "GPUvectorizedarmsifparitypreflightpasses;costandexposurereported"
  ]
}

I file JSON contengono tutte le metriche, incluse quelle che passano, i controlli e i limiti.
