# ⚡ IE-P4 — costo, accuratezza e supervisione interna

Kaggle v1, T4/FP32, codice `e73d454a9acf0c243e8ed43412682a62c5db657e`.
Run tecnicamente valido:84celle di timing,84preflight,zero errori graph.
Audit locale: freeze/hash/selezione verificati,24path rigenerati e14bracci
riprodotti su CPU senza retraining. Massimo scarto nelle RMSE dei path
CPU/GPU8,382e-9; nei massimi8,941e-8, entro tolleranza2e-5.

## 🎯 Qualità: progresso reale, ma nessuna rete promossa

| Famiglia | RMSEgate fresh | RMSEgate path40ms | RMSEV storico mV |
|---|---:|---:|---:|
| Formula vettorizzata |0|0|0,0009993|
| LUT0,25mV |1,235e-6|2,697e-6|0,0010539|
| Transition32/64,6seed |0,001589–0,002156|0,008748–0,012624|0,4239–0,5576|
| Internal32/64,6seed |0,001528–0,002313|0,001629–0,003793|0,03171–0,04510|

Formula e LUT superano tutti i gate di qualità; nessuna delle12reti li supera.
I sei bracci con supervisione interna passano la sola RMSEV≤0,05mV ma non
i massimiV≤0,1mV né Ca max≤5e-6mM. Vmax0,257–0,477mV e Camax
1,212e-5–2,113e-5mM. Tutte le reti falliscono anche il fresh atomico completo
(RMSEgate≤0,001 e max≤0,01). Non cambiamo soglie dopo il test.

La supervisione interna riduce l'errore dei path in **6/6** contrasti appaiati,
e la RMSEV storica di un fattore **10,29–16,10**. Migliora invece la RMSE
fresh a un passo solo in3/6: l'ipotesi congiunta fresh+path ha esito misto.
Questo mostra perché una sola metrica a un passo è insufficiente per scegliere
un modello dinamico. Non prova da sola quale singolo coefficiente sia causale.

Raddoppiare la larghezza non dà un vincitore robusto né supera i gate.
Tutti selezionano10000update, con curve ancora migliorabili: convergenza
incompleta resta plausibile, ma "basta altro budget" non è dimostrato.

## ⏱️ Costo del core residente al batch642

| Famiglia | Compiled ordinario ms | CUDA Graph ms |
|---|---:|---:|
| Formula vettorizzata |0,289652|0,050840|
| LUT0,25mV |0,314346|0,041403|
| Reti32,6seed |0,477395–0,484975|0,061379–0,062335|
| Reti64,6seed |0,478589–0,485771|0,065098–0,065726|

LUT: **18,57%** di risparmio appaiato in graph, CIinterno[18,45%;18,73%].
Con chiamate ordinarie è invece8,60% più lenta: vantaggio condizionato al
backend. A batch1 risparmia59,21%graph/39,72%ordinario; a4096
4,47%graph ma perde9,43%ordinario. Non è uno speedup universale.

Le reti perdono20,8–29,4% contro le formule già ottimizzate in graph e
circa66–69% nelle chiamate ordinarie. Più training può cambiare la qualità,
non rimuove da solo questo costo del medesimo grafo di inferenza.

I12membri sono stati addestrati insieme: circa31secondi dal checkpoint0
al10000; notebook completo circa2minuti48secondi. Nessun timing in contesa
con il training, nessun padding nel costo di inferenza.

## 🧭 Decisione e informazione guadagnata

- H_direct: contraddetta per le famiglie/budget provati: nessuna rete soddisfa
  insieme qualità e risparmio. Non è un teorema contro operatori appresi.
- H_internal: mista; beneficio ricorsivo coerente, non sul fresh a un passo.
- H_capacity: nessun guadagno sufficiente a giustificare il costo aggiuntivo
  secondo i criteri registrati, non impossibilità dello scaling in generale.
- H_lut: supportata nel dominio contro le reti testate; qualità e costo
  migliori. Vantaggio sulle formule dipendente da batch/backend.

Conservare formula vettorizzata e LUT come riferimenti competitivi. Prima
di un nuovo run va affrontato separatamente il problema dei peggiori errori
e il costo della rappresentazione, non aumentata alla cieca la stessa MLP.
Nessuna nuova esecuzione o promozione è autorizzata dal solo risultato.

Dominio:17gateV-dipendenti, SK canonicoCa→SK, stato esplicito, dt0,1ms.
Fresh sintetico a target formula e24path imposti; Task35 è già aperto.
Non è una nuova conferma NEURON, non copre una morfologia completa, non
misura eventi/RNG/trasferimenti né runtime online. Nessun candidato promosso.
