# ✅ GIADA Task 3c — conferma della cella congiunta riparata

La Task 3b ha identificato la supervisione simmetrica dei rate come riparazione
più efficiente: raggiunge a 50k lo stesso regime della baseline a 100k, con
minore dispersione fra seed. La Task 3c congela questa scelta senza ulteriori
screening.

## 🚀 Esecuzione GPU

I tre seed non vengono più addestrati in sequenza. Parametri, input e gradienti
hanno un asse iniziale `seed`, così le operazioni dei tre modelli vengono fuse
in matrici GPU più grandi. Adam mantiene momenti elemento per elemento e il
clipping viene calcolato separatamente per ogni seed: non esiste comunicazione
statistica fra le repliche.

Prima del training viene eseguito un confronto sequenziale-vettoriale su forward
e primo update. La tolleranza massima è `1e-5`; un fallimento arresta il run.
`torch.compile` viene usato soltanto se supera lo stesso controllo numerico,
altrimenti il notebook ripiega automaticamente sul modello vettorializzato eager.

## 🔒 Freeze e conferma

- architettura width-23 physical-τ;
- tre seed `17, 29, 43`;
- checkpoint fisso a 50k, senza selezione;
- development medio `≤0,15` e peggiore seed `≤0,20`;
- fresh completamente nuovo e disgiunto, aperto una sola volta dopo il freeze;
- stessi gate biologici e di rollout della Task 3.

Il superamento di tutti i gate autorizza la Task 4. Il risultato resta limitato
al voltage clamp costante e alla probabilità di apertura `m²h`.
