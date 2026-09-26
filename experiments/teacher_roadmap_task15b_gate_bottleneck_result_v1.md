# GIADA supplemento Task 15b — risultato sul collo di bottiglia dei gate

Artefatto utente: `giada_roadmap_task15b_gate_bottleneck_confirmation.zip`, SHA-256 `dd1fdb1e6d33eab179baa6c3711b76dd55d11279e93cec85bd55c3b3cc35152d`; report interno SHA-256 `a20c6536a5f852fe9ba3ae323d00b717d1d4c371e5433519567aa36f364dba97`. Codice `4262849a7fd7c321734eacef61f8d03b89817f5d`, coerente con la preregistrazione. `valid=true`; SHA del freeze e dei 24 checkpoint verificati, prerequisito Task15 esatto, nessuna selezione su sealed/controfattuali. Tutti i quattro bracci hanno scelto seed17 mediante RMSE m development a step800. Sono stati usati seed di dati nuovi, dunque le misure sealed qui **non** sono una replica appaiata numericamente della Task15 originale.

| Braccio | m RMSE sealed | h RMSE | I RMSE (mA/cm²) | V RMSE one-step (mV) | V RMSE rollout16ms (mV) | Mediana RMSE delta-I (mA/cm²) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline @400 | 0,02708 | 0,01476 | 2,50e-5 | 0,204 | 0,761 | — |
| Baseline @800 | 0,01988 | 0,01187 | 2,17e-5 | 0,156 | 0,560 | 1,35e-5 |
| Gate ×16 @800 | 0,01877 | 0,01222 | 2,08e-5 | 0,252 | 0,794 | 1,18e-5 |
| Corrente supervisionata @800 | 0,03211 | 0,07009 | 1,67e-5 | 0,474 | 2,238 | 6,15e-6 |
| Gate ×16 + corrente @800 | 0,02424 | 0,01586 | 1,48e-5 | 0,463 | 2,126 | 5,64e-6 |

**H1, budget:** nella *stessa traiettoria* baseline, 400→800 step riduce m RMSE sealed del 26,6%, migliora I RMSE del 13,3% e V rollout di circa 26,4%. Il gate preregistrato per il budget passa, ma con margine piccolo sulla soglia del 25%; due checkpoint non dimostrano che lo scaling possa eliminare l'intero gap né che il training abbia convergito. Sul development m migliora a 800 in tutti e tre i seed baseline (17/29/43).

**H2, peso gate:** ×16 rispetto a ×4 porta m solo 5,6% più in basso a step800, mentre V rollout peggiora del 41,7%; il gate composito fallisce. **H3, loss corrente:** migliora I one-step del 23,1% e dimezza circa la mediana dell'errore delta-I, ma m peggiora del 61,5%, h di quasi 6× e V rollout di circa 4×. **Interazione H2×H3:** la combinazione ottiene il miglior I one-step (−31,7%) e la migliore mediana delta-I (−58,1%), ma m peggiora del 21,9% e V rollout di circa 3,8×. Nessuna delle tre modifiche di loss soddisfa i criteri preregistrati di riparazione senza sacrifici importanti.

Il probe analitico lungo il V della baseline @800 dà m RMSE 0,000581 contro 0,01988 dei gate **appresi**; resta quindi un grande gap diagnostico. Il probe usa le equazioni del teacher e non è un candidato appreso selezionato. Una possibile interpretazione del trade-off è che la corrente scalare non identifica da sola m, h e V separatamente: più combinazioni producono I simile. Questa è un'**ipotesi causale plausibile**, non una dimostrazione isolata da questo run; anche scala della loss, supporto dei dati e ottimizzazione possono contribuire. Il peso ×16 testato non esclude pesi intermedi o schedule diversi.

La validità tecnica della 15b e il suo miglioramento col budget **non** autorizzano Task16 o Gate C come promozione. La Task16 rimane il prossimo test *embedded congelato* della roadmap qualora si scelga consapevolmente un candidato da diagnosticare nel contesto reale; non deve essere descritta come prova che il collo di bottiglia dei gate sia risolto. Un'eventuale nuova ottimizzazione dei gate richiede protocollo separato, con confronto a budget e input appaiati, anziché ritoccare retroattivamente questa 15b.
