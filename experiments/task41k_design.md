# GIADA — Task 41k: riparazione e compressione separate

## Evidenza e domanda

La41j non ha promosso alcun candidato. H e catena riottimizzata congelata hanno
gli stessi6 fallimenti sul pannello appaiato. Per tutti i seed della catena e
dei compatti, la selezione conserva step0: per i compatti questo segue4096
aggiornamenti di distillazione. Non significa assenza di training.
La loss dei coefficienti migliora mentre la dinamica development peggiora.
Ipotesi da distinguere, non causa unica già dimostrata: obiettivi in conflitto,
perdita di comportamenti utili, copertura insufficiente, distillazione inadatta.

## Matrice preregistrata

| Bracci | Obiettivo | Contrasto |
|---|---|---|
| K_point / K_oldmix | Coefficienti / precedente mix sparso | Controlli di ottimizzazione |
| K_dyn_uniform / K_dyn_uniform_retain | Dinamica canonica / + replay H | Protezione dei comportamenti |
| K_dyn_target / K_dyn_target_retain | Come sopra, campionamento FIT mirato | Matrice2×2 copertura×replay |
| K_copy_coeff / K_copy_dynamic | Copiare H: coefficienti / dinamica | Compressione senza riparazione |

Riparazione: stessa catena H, tutti i parametri addestrabili, inizializzazione
identica. Compressione: stesso compatto64, inizializzazione identica dal tronco
base di H; nessun target canonico nel gradiente. H resta controllo congelato.
Tre seed vettorizzati per braccio; bracci eseguiti nello stesso worker con stream
appaiati. Non dichiariamo esecuzione simultanea degli otto grafi GPU.

Stessi18stati, dipendenze V/Ca, ricorrenza convessa e passo0,1ms. Nessun fallback
cinetico. Correnti, calcio, solver, sinapsi e soglie scientifiche invariati.

## Probe e budget

Gradienti coefficienti/dinamica: coseno e perturbazioni normalizzate con clone
scartato. Checkpoint0/16/64/256/1024; nuovo Adam, lr1e-5 con decadimento10×.
Budget diagnostico, non prova di saturazione: stessi update non sono stessiFLOPs.
I bracci dinamici ottimizzano ogni update, contro una loss ausiliaria ogni16
del vecchio mix. Correnti su prefisso completo e coda1200ms; risposte accoppiate
di40passi. Targeting50%uniforme+50%quartile FIT peggiore, mai dai test aperti.
Replay H sulla metà inferiore degli errori FIT per seed: non equivale a una
certificazione di traiettorie completamente corrette. Peso replay unitario.

## Selezione e risultati

La riparazione mantiene il criterio lessicografico originale development.
La compressione seleziona fedeltà dinamica al parent, poi coefficienti al parent.
Non può nominarsi come riparazione. Tutti gli otto bracci vengono valutati dopo
freeze con l'intera suite originale: stress, ricorrenza, interventi, regressioni,
timing, correnti e costo appaiato dell'operatore/intero sistema. Nessuna soglia
rilassata e nessuna riselezione dai test aperti. La42 non è autorizzata da questo
esperimento: un eventuale candidato richiede conferma indipendente.

## Durabilità

Un notebook versionato, fasi64/256/1024 e valutazioni congelate. Prima della
fase successiva: ZIP persistito Kaggle, download locale fuori runtime, SHA256,
CRC e controllo dei tensori, poi dataset privato di ripresa verificato in lettura.
Il primo backup scientifico si produce al checkpoint64, non prima. Errori tecnici
arrestano la pipeline; DB e backup del DB esclusivamente locali. Preflight CPU
e reale GPU obbligatori prima del run scientifico.
