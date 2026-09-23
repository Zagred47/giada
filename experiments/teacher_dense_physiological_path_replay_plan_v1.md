# 🔬 GIADA Task 9c — verifica autentica del campionamento

Riproduciamo **12 transizioni train** già selezionate nella Task 9: per ciascuno
dei quattro segmenti, due spike e un quiet, scegliendo gli ID più piccoli prima
di leggere gli esiti della nuova acquisizione. Ripristiniamo i loro SaveState e
gli RNG del dataset v1.1, mantenendo input e meccanismi originali.

Per ogni transizione eseguiamo tre replay indipendenti con osservazioni ogni
`0,025`, `0,005` e `0,001 ms`. Il primo deve riprodurre i 41 punti salvati.
**Tutte** le categorie dello stato finale e le sequenze RNG devono coincidere
entro `1e-5`; in caso contrario il notebook si ferma. Il confronto è tra
formula Ca_HVA su voltaggio autentico più fitto, formula sui 40 campioni e
interpolazione lineare della Task 9b.

La formula densa supera il gate diagnostico solo se la differenza fra le
risoluzioni `0,005` e `0,001 ms` è ≤`0,001` per ogni gate e l'RMSE di `m,h`
verso il teacher finale è ≤`0,005` in ciascun gruppo sito/regime. Un risultato
positivo vale **solo per questi percorsi train**, non è una nuova validazione
indipendente né un risultato di rollout autonomo. Nessuna rete viene allenata.
