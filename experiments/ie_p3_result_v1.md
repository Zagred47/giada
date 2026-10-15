# ⚡ IE-P3 — risultato verificato

Kaggle v1, codice `f6db29bfae7e3e5d2481b77092a6732e95c5043f`, TeslaT4.
Completato:48celle di timing,51preflight superati,zero errori CUDA Graph.
Freeze/checkpoint SHA-256 e preregistrazione verificati; nessuna promozione.

## ⏱️ Costo residente: batch642, singolo modello

| Core con readout | Compiled ordinario ms | CUDA Graph ms |
|---|---:|---:|
| Algebra storica |0.412387|0.102743|
| Algebra vettorizzata esatta |0.285076|0.053139|
| MLP16, tre seed |0.523558–0.536182|0.099340–0.100284|
| MLP32, tre seed |0.520368–0.525247|0.110013–0.110477|

L'algebra vettorizzata risparmia **30,92%** ordinario e **48,26%** graph
rispetto alla storica. Bootstrap appaiato graph[48,21%;48,33%], solo
incertezza interna al run. Riduzione graph anche a batch1:28,71%,4096:28,45%.
Nessuna somma con lo speedup IE-P2: sono confronti diversi sullo stesso core.
Le reti sono87,6–108,0% più lente del vettorizzato in graph al batch642.

## 🎯 Qualità

Tutte6reti falliscono i criteri preregistrati. Fresh atomico RMSEapertura
0,02092–0,03143 contro limite0,005; max0,1945–0,2338 contro0,03.
Nelle traiettorie storiche40ms: V RMSE0,544–0,915mV contro limite0,05;
massimo2,30–5,39mV. Anche il calcio supera il proprio limite.

Le due algebre hanno identiche metriche diagnostiche storiche:
V RMSE0,0009993mV/max0,005253mV; Ca max1,189e-7mM.
Questo è l'errore rispetto al teacher storico, **non** l'errore introdotto
dalla vettorizzazione. I controlli di compilazione restano entro tolleranze;
massimo scartoV eager/compiled/graph1,526e-5mV.

## 🧭 Informazione guadagnata e limiti

Il costo del readout non richiedeva necessariamente apprendimento: una
riorganizzazione esatta recupera qui quasi metà del tempo con graph.
Il semplice MLP condiviso sui descrittori dei canali non è competitivo
in questo budget, né in qualità né in costo. Non è una confutazione del
deep learning per un operatore ionico più ampio: qui si apprende soltanto
la mappa gate→apertura, mentre dinamiche dei gate e solver restano canonici.

Tutti i seed selezionano3000update; la curva ancora in discesa suggerisce
ottimizzazione incompleta, ma non prova che lo scaling risolva tutto.
Il training vettorizzato0→3000 termina in circa6secondi dopo il primo log;
il notebook completo richiede circa3minuti. Nessuna promessa sul runtime
online con eventi/RNG/trasferimenti o su un'intera morfologia.

Gate uniformi indipendenti sono un dominio atomico, non occupazione
fisiologica. Task35 è già aperto: diagnostica, non nuova conferma sigillata.
La baseline esatta è una candidata ottimizzazione locale da conservare;
integrazione operativa e conferma indipendente non sono state eseguite qui.
