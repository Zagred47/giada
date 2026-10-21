# GIADA Task40b — audit e scelta per Task41

Audit post hoc; nessun training, nessuna modifica delle soglie.

| Variante | Development | Stress | Risposte autonome | Risposte imposte |
|---|---:|---:|---:|---:|
| B0 | 72/72 | 70/72 | 146/168 | 148/168 |
| B1 | 72/72 | 71/72 | 168/168 | 168/168 |
| B2 | 72/72 | 71/72 | 168/168 | 168/168 |
| B3 | 72/72 | 71/72 | 168/168 | 168/168 |
| B4 | 72/72 | 71/72 | 168/168 | 168/168 |
| B5 | 72/72 | 71/72 | 168/168 | 168/168 |
| B6 | 72/72 | 72/72 | 168/168 | 168/168 |
| B7 | 72/72 | 70/72 | 165/168 | 165/168 |

Tutte le varianti:264/264 traiettorie in ciascuna modalità. Sono casi×seed, non altrettanti protocolli indipendenti.

## outcome

B1-B6 repair all22Task40 autonomous intervention failures:146to168/168;imposed148to168/168. All8arms pass264/264 trajectories inbothmodes and72/72development. B7 m-only retains3Ca_LVAst differential failures inbothmodes.

## stress

B0/J1 reproduces70/72stress;B1-B5 reach71/72;B6alone72/72;B7 remains70/72. Allmodelseeds preserved;24uniqueprotocols perrole,not72independentprotocols.

## timing_repair

same_v_history case1 seed17 Ca_LVAst offset0.5to0.1ms forB1andB6,unchangedlimit0.3ms. Relaxationcorrection resolves this timingresidual.

## equilibrium_repair

activation_boundary case0 seed29 Ca_HVA NRMSE0.01178899to0.01205136forB1(stillfails)and0.00145030forB6(limit0.01). Jointu+z repairs the activation-boundary residual.

## ablation

FrozenB6 z-only passes11/12focusedstress,losingCa_HVA boundary;u-only passes11/12,losingCa_LVAst offset. Bothcomponents required for this frozenpackage ontheseopenedcases,notproofthatallpossiblemodelsneedbothcorrections.

## contrast

No registered incremental contrast meets the20percent medianseverityrule inbothdomains. Globaldevelopmentseverity canbe dominatedbyunchangedchannelsandexcludes timing. This doesnotnegatethecase-specificB6repair. Extra pairedloss,4096anchors,64basisdo not establish addedregisteredbenefit here;notgeneralproofthatdataorcapacityneverhelp.

## nomination

NominateB6 withall3frozenmodelseeds17/29/43,notonebestseed. 32basis,1024FITanchors,coefficientloss,u+zCa-onlycorrection;256activeweights/seedversusB1 128. Parentbitwiseunchanged,18FP32states,sameabstractionlevel,noanalyticalgatefallback.

## audit

Allsavedcheckpoints/Adamandautonomoustracesfinite;freezehash,checkpointSHA,per-memberargminandchosenweightsverified;independentatomicrecipeandselectionlossrecomputed. 2112autonomoustrajectoriesand1344interventionmetricsrecomputedfromsavedtraces;CPU144recurrenceverdictsmatchB6GPU. Imposedtracesnotsaved,soimposedauditchecksallstorednumericcomponents,notfreshfulltrace replay.

## membrane

B6autonomousworstVrmse0.0128588mVandCa3.22957e-7mM;0spike-countmismatches,0spike-timingerror,0domainexits. AbsoluteVqualitydoesnotreplaceionicdifferentialgates.

## provenance_erratum

ExecutedpreregistrationlimitscopiedaTask40lineassertingnotraining/noselection. Explicit40btraining/budget/selectionfieldsandcheckpointsshowactualexecution. Separateerratumpreserved;DBeffectiveprocedureannotatedwithoutchangingexecutedpreregbytes,hash,thresholdsorrun.

## decision

GO to prepareTask41independentlevel1confirmationofB6,notGOtodeploymentormulticompartment. Newprotocols,seeds,parametricpanelsandmatchedcostuncertaintyarerequired;freezeB6andthresholdsbeforefreshaccess. NoB6speedupclaim;Task41mayconcludeaccuratebutslow.

## Due stress originali

| Variante | Ca_LVAst offset (ms; limite0.3) | Ca_HVA NRMSE (limite0.01) |
|---|---:|---:|
| B0 | 0.5 | 0.01178899 |
| B1 | 0.1 | 0.01205136 |
| B6 | 0.1 | 0.00145030 |

GO alla preparazione della conferma indipendente Task41 con B6. Nessuna conferma indipendente già acquisita; nessuno speedup B6 dimostrato.
