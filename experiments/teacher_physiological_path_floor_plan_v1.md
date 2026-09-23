# 🔍 GIADA Task 9b — origine del *reference floor* negli spike

La Task 9 ha mostrato che la formula esatta applicata ai 40 campioni salvati
non coincide abbastanza con il gate finale autentico nei gruppi spike. La
Task 9b riusa **gli stessi 350 percorsi già aperti**, senza nuova simulazione
NEURON, nuovi dati test, scelta di modello o riaddestramento.

Confrontiamo sei integrazioni della stessa formula Ca_HVA: voltaggio destro
(controllo che riproduce Task 9), destro con cinque sottopassi costanti
(controllo negativo: deve dare lo stesso risultato), sinistro, punto medio,
interpolazione lineare con 5 e con 20 sottopassi per intervallo registrato.
Per ogni sito/regime riportiamo RMSE e bias di `m,h,m²h` verso il gate finale
autentico e la frazione di percorsi migliorata.

Prima di interpretare, la modalità destra deve riprodurre la Task 9 entro
`1e-10`, e il raffinamento a voltaggio costante deve essere invariato entro
`1e-10`. Il criterio diagnostico preregistrato richiede, in **tutti** i gruppi
spike, almeno il 50% di riduzione dell'RMSE di `m`, RMSE massimi di `m,h`
non superiori a `0,005` e convergenza 5-vs-20 entro `0,001`.

⚠️ Un miglioramento sarebbe coerente con perdita d'informazione tra campioni,
ma non proverebbe quale fosse il vero voltaggio continuo. Se il criterio
fallisce, servirà una piccola cattura autentica più fitta del teacher.
