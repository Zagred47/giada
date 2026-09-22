# ✅ GIADA Task 3c v2 — conferma degli esatti checkpoint Task 3b

La revisione v1 ha dimostrato che il training multi-seed fuso è numericamente
equivalente al primo update, ma non conserva necessariamente la stessa
traiettoria dopo 50.000 step. Il seed 29 ha quindi fallito il gate development;
il fresh non è stato aperto.

La v2 elimina completamente il riaddestramento. La Task 3b possiede già i tre
checkpoint `symmetric_rates` a 50k, scelti tramite soli dati development e
salvati prima di qualsiasi conferma fresh. Sono quindi i candidati corretti da
confermare.

Il notebook:

1. verifica gli SHA-256 dell’archivio, del report e dei checkpoint Task 3b;
2. estrae esclusivamente i pesi `symmetric_rates` a 50k per i seed 17, 29 e 43;
3. riproduce gli score development con tolleranza `1e-10`;
4. congela gli stessi pesi senza alcun training;
5. apre una sola volta il fresh Task 3c, rimasto finora sigillato.

Questa soluzione è più veloce di qualsiasi parallelizzazione e più forte dal
punto di vista confermativo: evita che una replica numericamente diversa venga
scambiata per il modello selezionato nella Task 3b.
