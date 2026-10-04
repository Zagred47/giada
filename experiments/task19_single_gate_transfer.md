# 🧬 GIADA — Task 19 originale: gate singolo

## 🎯 Domanda

La cella fisicamente vincolata già studiata per Ca/Na/K è apprendibile anche
per i singoli gate canonici di **Ih** e **Im**? La Task18d autorizza questa
fase. Il successo resta riferito al voltaggio costante e al dominio dichiarato.

## ⚗️ Matrice nello stesso run

Due meccanismi × due obiettivi × due capacità × tre seed = **24 modelli**.
I dodici modelli di ciascuna capacità sono vettorizzati sulla GPU; i pesi,
i momenti Adam e il clipping restano indipendenti. Width16 e width32 avanzano
con gli stessi minibatch; checkpoint0/1k/5k/15k/30k lungo le medesime traiettorie.

Il controllo `transition_only` riceve gli stessi input numerici e parte dalla
stessa funzione del braccio `rate_supervised`; quest'ultimo aggiunge le etichette
di equilibrio e costante temporale alla loss. Le due capacità hanno338 e1186
parametri, rispettivamente304 e1120 MAC densi, oltre alle operazioni elementari.
La condivisione di pesi tra meccanismi resta nella Task22 della roadmap.

## 🧱 Stato e equazioni

Ingresso `(V,x,dt)`; rete `1→w→w→2`, SiLU, output `inf=sigmoid`,
`tau=exp(logtau)` positivo. Solver: `z=-expm1(-dt/tau)` e
`x_next=(1-z)*x+z*inf`. Ogni gate rimane in[0,1] per costruzione.
Conduttanza `gbar*x` e corrente `gbar*x*(V-E)` sono analitiche.
Per questi due canali la frazione aperta coincide con il gate: la metrica
open-RMSE non costituisce una misura indipendente dal gate-RMSE.

Teacher canonico074c466: Ih shift0 ed equilibrio corrente−45mV;
Im qt fisso2.3^1.3 ed ek−85mV per la corrente diagnostica. Non si varia la
temperatura. In Ih il ramo esatto `v==vhalf` corregge il valore locale usato
nei rate di+0.0001mV, mentre la tensione osservata della sezione rimane al
valore richiesto; questa semantica è verificata e registrata dal doppio oracle.
Nessuna modifica alle equazioni originali oltre all'esposizione RANGE dei rate.

## 🔍 Contrasti e probe

- Supervisione dei rate contro transizioni sole, a capacità/budget/seed uguali.
- Width32 contro16 e checkpoint successivi: capacità e ottimizzazione.
- Grid degli stati iniziali0/1: bound del gate per tensione e dt fissati.
- Rollout ricorsivo1/10/100/1000 passi da1ms a V costante, dopo freeze.
- LUT513 e2049 f64 sulle identiche tuplefresh interne.
- Latenze eager CUDA di MLP/formulaTorch/LUT2049, batch4096, insieme all'errore.

Fit24k:50% uniforme[-135,75]mV,25% regioneIh[-120,-95],25% regioneIm[-45,-25].
Development e fresh usano stream e griglie di tensione disgiunti; gli stati
iniziali comprendono condizioni lontane dall'equilibrio. Il test fresco viene
materializzato soltanto dopo freeze di selezioni e hash di tutti i checkpoint.
La selezione è comune ai tre seed per ciascun meccanismo/obiettivo e usa solo
il development, con penalità anche per il massimo errore.

## ✅ Decisione preregistrata

Per autorizzareTask20 entrambi i canali rate-supervised devono passare in
tutti3seed e nei dominiuniforme/coda/griglia: gateRMSE≤.001, massimo≤.01,
infRMSE≤.002, logtauRMSE≤.02, openRMSE≤.002, zero violazioni e valori finiti.
Ogni orizzonte del rollout deve avere gateRMSE≤.002 e massimo≤.01.
OOD negativo[-155,-135] e positivo[75,95] restano diagnostici separati.

Il confronto fattoriale development non viene descritto come confermafresh.
Le latenze misurano primitive isolate su CUDA: nessun confronto wall-clock con
NEURON, né vantaggio del neurone completo, viene inferito automaticamente.
Teacher nativo e training GPU usano processi separati; errori esecutivi producono
failure_report, distinti da un eventuale NO-GO scientifico con run valido.
