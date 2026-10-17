# 🧪 Task38 — Risultato e limiti

Versione Kaggle2, codice9fe269fcfda669d825ce68ef61852ce8dc4c6a74.
Training30k update completato in370s; ricorrenze complete e report in circa500s.
Freeze e checkpoint verificati,432 righe univoche di valutazione.

| Modello | Gate completo development, casi×seed | Peggior RMSE gate |
|---|---:|---:|
| joint width32 |0/72|0,00958324|
| joint width64 |1/72|0,0164121|
| frozen local161 + SK canonico |53/72|0,000190388|

I72 non sono esperimenti indipendenti:24 traiettorie ×3 seed. La baseline
è una sostituzione parziale, non un modello interamente appreso. Nessun
non-finito e nessuna violazione delle occupazioni in tutte432 valutazioni.
Questi invarianti non garantiscono la fedeltà delle correnti ricorrenti.

La loss comune one-step continua a scendere fino30k per tutti i membri;
non sappiamo se altro budget basta a risolvere gli errori lunghi. Le code
a V uguale conservano un contrasto di memoria ma con errore. Molti worst
NRMSE riguardano correnti quasi nulle: riportare sempre anche errore assoluto
e scala, senza cancellarli dal gate. Non sono errori in mV.

Nota esecutiva: width32 usa compile verificato; width64 ha usato fallback
eager dopo una differenza della loss oltre la tolleranza numerica preregistrata.
Il risultato non dimostra che la maggiore capacità sia inutile in generale.
Task39 usa uno stesso backend per l'intera matrice e confronti interni appaiati.

**Decisione:** NO-GO per la sostituzione/Task40; GO alla diagnosi Task39 già
prevista. Separare copertura, budget, esposizione ricorrente, loss e dipendenze
di stato. Nessun test indipendente aperto, nessuno speedup dimostrato.
