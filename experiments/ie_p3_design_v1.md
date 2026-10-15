# ⚡ IE-P3 — Algebra vettorizzata o readout appreso?

## 🎯 Domanda atomica

Il margine del readout rilevato da IE-P2 viene prima confrontato con una
riorganizzazione **esatta**: gather dei gate, potenze intere mediante prodotti,
riduzione vettorizzata. In parallelo, due MLP di larghezza16/32 apprendono
l'apertura di ciascun canale dagli stessi gate e descrittori fisici. Le
conduttanze massime rimangono moltiplicatori analitici: azzerarle azzera il
canale, e la loro linearità è preservata. Non si apprende il solver.

Questa è una prova sul **readout**, non ancora un nuovo operatore di evoluzione
ionica completo. I18gate rimangono espliciti; non si rivendica compressione
dello stato. Il segnale Ca→SK mantiene l'ordine verificato in IE-P2.

## 🧪 Disegno comune

Quattro famiglie: algebra storica, algebra vettorizzata, MLP16, MLP32.
Tre seed per famiglia appresa; sei reti trainate simultaneamente con bmm,
minibatch identici, momenti Adam separati e clipping per membro. Il test CPU
confronta predizioni e singolo aggiornamento contro esecuzioni separate.
Il padding del training viene eliminato nell'inferenza misurata.

Dominio atomico: gate indipendenti uniformi[0,1]. Tre insiemi indipendenti
fit/development/fresh, senza selezione sul fresh. Il dominio è intenzionalmente
controllato e non rappresenta la distribuzione di occupazione biologica.
Checkpoint0/100/500/1500/3000 lungo la medesima traiettoria, selezione per seed
su development. Si salvano anche errori massimi, non solo medie.

I rollout chiusi40ms sugli stati storiciTask35 sono diagnostici già aperti:
non sono un nuovo test sigillato e non autorizzano reinserimento. Tutte le
leggi, sinapsi e correnti di ingresso sono identiche; si cambia solo readout.
L'errore del riferimento formula viene riportato insieme a quello appreso.

## ⏱️ Esecuzione e decisione

Stesso core a cinque output, FP32/TF32off, dt0,1ms, Inductor fullgraph e
CUDA Graph, batch1/642/4096. Compilazione/capture escluse;30blocchi da20call,
ordine intercalato. Ogni rete viene misurata singolarmente: il confronto è
per un modello contro una formula, non ensemble contro singolo modello.

Prima si verificano eager/compiled/graph. Soglie prospettiche nel JSON:
RMSEapertura≤0,005/max≤0,03 e diagnosticaV RMSE≤0,05mV/max≤0,1mV,
Ca max≤5e-6mM. Il margine di costo interessante è ≥10% rispetto all'algebra
vettorizzata, con intervallo appaiato positivo, al batch642. Tutti i seed
vengono riportati, anche se non qualificati; non si seleziona sul runtime.

Un fallimento numerico della rete è un risultato scientifico; un crash di
compilazione è tecnico. Graph mancante rende inconclusiva quella modalità.
Non si promette accelerazione online: questi timing usano ingressi residenti
fissi, senza trasferimenti o RNG host. Nessun kernel custom/OpenEvolve.
