# GIADA Task 17e — conferma indipendente della sostituzione Ca_HVA

Stato: preregistrato prima di aprire qualsiasi risultato delle condizioni qui elencate. Questa è la verifica conclusiva proposta per la Task 17 della roadmap, con ambito limitato al `Ca_HVA` del teacher canonico a 642 segmenti. Non è una prova su altri neuroni o un benchmark GPU. Il candidato è la LUT float64 a 2049 nodi dei soli rate `m`, costruita come nella Task 17c e osservata nella 17d-b. La tabella è congelata tramite SHA-256 `e15f712faccc68dafbc7d861a14e23262388b9715ae5b564c51c06f261251765`; se l'impronta cambia, stop prima dei trial. Nessuna scelta di iperparametri dopo i risultati.

## Casi e bracci fissati

Seed nuovi `171001`, `171019`, `171043` (assenti negli esperimenti versionati prima di questo protocollo); cinque programmi: quiete, NMDA tuft originale n10/b2/finestra 0,4 ms, calcio hot-zone originale n12/b3/finestra 0,8 ms, NMDA variante n8/b2/finestra 0,6 ms, calcio variante n10/b3/finestra 0,6 ms. Per ognuno, `gCa_HVAbar` ×0,5/1/1,5. Ogni episodio dura 60 ms, campionato ogni 0,025 ms. Totale: 45 condizioni × nativo, clone formula e LUT2049 = 135 episodi. Il protocollo può riusare famiglie di stimoli note; l'indipendenza qui è nei nuovi seed e nelle combinazioni di perturbazione congelate, non in una nuova morfologia.

Tutti i bracci usano CVode `atol=1e-7`, `rtol=1e-8`, senza scale selettive. Stessi SaveState iniziali, schedule, Random123 e gbar per contrasto; nuova istanza di snapshot a ogni cambio di struttura NMODL. La LUT sostituisce realmente `Ca_HVA` in ogni sezione che lo esprime e scrive la corrente ionica nel teacher accoppiato. Non riceve V finale o stati futuri. Il confronto resta raw, senza allineare a posteriori i picchi.

## Controlli e soglie di Gate C

Prima della matrice: compilazione e probe delle tabelle; ripetizione nativa con massimo errore V ≤1e-5 mV e decisioni di rilascio identiche. Ogni clone formula deve avere RMSE V ≤0,05 mV, massimo errore m/h ≤0,002 e RMSE `ica_hva` ≤1e-4 mA/cm² in ogni episodio/sito, oltre a eventi e rilasci coincidenti. Fallimento del clone interrompe l'attribuzione alla LUT.

Per la LUT2049, in **ogni** episodio e in ciascuno dei siti 0/387/460/469: RMSE V ≤2 mV, massimo errore m/h ≤0,01, RMSE della corrente analitica `ica_hva` ≤1e-4 mA/cm². Il controllo di corrente è un'aggiunta preregistrata al microcanary originale. Stato finito, occupanza dei gate in [0,1], pesi NetCon e morfologia invariati sono precondizioni tecniche.

Eventi estratti con il detector diagnostico versionato sui rappresentanti soma, AIS, tronco, hot zone, nexus e tuft. Per ogni classe presente, il numero di eventi e lo stato di censura devono coincidere. Errore massimo di onset ≤0,2 ms per spike soma/AIS e ≤0,5 ms per eventi dendritici. Il supporto nativo deve includere almeno un somatic spike, calcium spike e NMDA spike nell'intera matrice; se manca, Gate C resta non informativo. Rilasci: stessa identità degli eventi, draw Random123, successo e quantità realizzata entro 1e-10.

Sensibilità a gbar: per ogni protocollo/seed/sito con effetto nativo identificabile (RMSE del contrasto ×1,5−×0,5 ≥0,05 mV), errore relativo del contrasto candidato ≤20%. Deve esserci almeno un contrasto identificabile. Si riportano tempo totale dei bracci e costo di compilazione separatamente; il tempo su CPU Kaggle non è usato come prova di accelerazione futura.

`GATE_C_PASS_BOUNDED_CAHVA` richiede **tutti** i controlli. Solo allora la Task 17 è completata per questo meccanismo nel teacher canonico e la Task 18 della roadmap può cominciare. Ogni fallimento conserva la matrice e indica quale criterio non è passato; non si cambiano seed, candidate, soglie o eventi dopo averla vista. Anche un passaggio non dimostra generalizzazione a diversa morfologia, ad altri meccanismi o speedup hardware.
