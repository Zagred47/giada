# 🧩 Task22 originale — condivisione controllata

Ih, Im e Nap_Et2 **h isolato** condividono un contratto V→inf,tau con solver
esponenziale analitico. SK_E2 non entra: il calcio cambierebbe il contratto
informativo e confonderebbe l'effetto della condivisione.

## 🔬 Matrice

Tre famiglie, due width, tre seed = **18 sistemi a tre gate**, non18 singoli
MLP. Ogni seed è un bundle indipendente; le tre curve restano sempre valutate
separatamente. Si confrontano:

- tre MLP indipendenti;
- un trunk condiviso V→features e tre head inf/tau;
- un MLP condizionato da V e identità onehot del gate.

L'identità è nota in tutti i casi: routing nei primi due, onehot nel terzo.
Non stiamo dando informazioni privilegiate al modello condizionato. Stato e
dt sono comuni e usati dal solver, non aggiunti solo a una delle reti.

## ⚖️ Equità e parametri

Stessi tuple numeriche, minibatch, seed, durate multiscala, loss media a peso
uguale per gate e clipping norm1 perbundle; Adam indipendente perseed.
Trunk/head inizializzati allo stesso modo dove le forme lo consentono;
l'inizializzazione non può essere identica per matrici di forma differente.

| Width | Separati (bundle) | Trunk+head | Condizionato |
|---|---:|---:|---:|
|16|1014|406|386|
|32|3558|1318|1282|

Width allineate, **non stesso numero di parametri**. Il confronto misura il
tradeoff esplicito: passare le stesse soglie con meno parametri. Non autorizza
una dichiarazione di superiorità a budget parametrico esattamente uguale.

## 📏 Decisione preregistrata

Checkpoint0/1k/5k/15k/30k, selezione worst-seed/worst-gate/worst-domain sul
solo development. Freeze prima del fresh. Rollout1ms fino10000ms, congiuntivo
per gate, seed e orizzonte. OOD separato. Identità/head permutate costituiscono
un controllo negativo; non sono usate per scegliere il modello.

La condivisione è confermata soltanto se il controllo indipendente passa3/3
e una famiglia condivisa passa3/3 con meno parametri selezionati. Se passa
solo il controllo, i gate restano apprendibili ma lo sharing non è dimostrato.
Non si avvia automaticamente la Task23.

## ✅ Preflight

Test conteggi, forme, boundedness, identità scambiata, Adam vettorizzato vs
indipendente e target canonici. Smoke completo CPU attraverso training,
freeze, fresh, rollout e report; audit NEURON Ih/Im e Nap_h in processi distinti.
Lo smoke verifica eseguibilità, non qualità scientifica. Nessun claim corrente
completa, voltaggio accoppiato o accelerazione hardware.
