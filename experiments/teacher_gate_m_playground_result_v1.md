# GIADA Task 1 — risultato del gate `m`

## Decisione preregistrata

**NO-GO per i candidati correnti al budget registrato.** L'artefatto è valido, il test sigillato è stato aperto una sola volta e gli hash di checkpoint e rapporto finale coincidono. Il NO-GO non equivale a rigettare l'ipotesi physical-τ.

| Braccio | RMSE sealed macro |
|---|---:|
| formula oracle | 0 |
| physical-τ | 0,002110 |
| direct-z | 0,002624 |
| direct MLP | 0,004336 |
| LUT lineare delle rate | 0,008644 |
| persistenza | 0,362057 |

La soglia preregistrata era `1e-3`. Physical-τ è il miglior modello appreso ma manca la soglia di circa 2,1 volte.

## Informazione causale acquisita

- La struttura physical-τ aiuta realmente: supera MLP diretto e direct-z, pur usando soltanto 338 parametri.
- Non emerge instabilità ricorsiva: il suo RMSE passa da `0,001701` a 10 passi a `0,001018` a 1.000 passi, superando il gate rollout `5e-3`.
- Tutti i bracci strutturati rispettano `[0,1]` senza violazioni.
- L'errore physical-τ sugli strati non-OOD è `0,001314`; lo strato OOD-voltage sale a `0,009724` e domina il macro aggregato.
- Le tre repliche physical-τ hanno RMSE macro `0,002577`, `0,001779` e `0,001975`: esiste dispersione tra seed, ma nessun seed viene escluso o selezionato.
- La curva development continua a migliorare fino all'ultimo checkpoint: `0,1944 → 0,0509 → 0,0235 → 0,0090 → 0,00467 → 0,00134`. Non è stata osservata saturazione a 10.000 step.

## Interpretazione

Il fallimento registrato è principalmente un problema di accuratezza one-step, extrapolazione OOD e budget/ottimizzazione non saturato; non è evidenza di una dinamica ricorsiva instabile. Prima di passare al gate `h`, è giustificato un esperimento mirato che separi capacità, budget, parametrizzazione delle rate e copertura OOD, mantenendo sigillato ogni nuovo test indipendente.

Il risultato vale soltanto per `Ca_HVA.m` con voltaggio mantenuto costante nel passo. Non autorizza conclusioni sul percorso di voltaggio variabile, sul neurone accoppiato o sulla sostituzione embedded del teacher.
