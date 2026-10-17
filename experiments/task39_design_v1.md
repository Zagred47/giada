# 🧪 Task39 — Diagnosi fattoriale della learnability

## 🎯 Domanda e confini

La Task38 ha completato training e valutazione ma non ha passato il gate
ricorrente. Non promuoviamo il candidato alla Task40: separiamo le spiegazioni.
Il livello resta18 gate fisici, dt0,1ms; V e Ca imposti, readout invariato.
Non è una dimostrazione di speedup o di comportamento del compartimento.

## 🔬 Matrice nello stesso run

Otto bracci: copertura fit/fit+fit_extra × teacher forcing/ricorrenza libera
× contesto di stato congiunto/fattorizzato. Tre seed appaiati:24 membri.
Due ancore aggiungono gate-only contro gate+correnti e una loss relativa
ausiliaria per correnti piccole: altri6 membri. Ogni contrasto cambia un solo
fattore rispetto al suo controllo; niente screening architetturale generico.

Tutti supervisionano gli stessi8 target per update; il controllo teacher-forced
vede lo stato vero a ogni tick, quello ricorrente solo all'inizio. Il numero
di target non viene moltiplicato soltanto per favorire la loss ricorrente.
Width32 fissata, stessa inizializzazione per seed e momenti Adam indipendenti.
Il mascheramento del contesto è un bias di dipendenza, non un vantaggio di costo.

## 📈 Budget, dati e prove

Checkpoint0/3k/10k/30k/60k sulla stessa traiettoria. Campioni bilanciati per
famiglia, metà uniformi e metà arricchiti presso cambi degli ingressi, con
log delle esposizioni e delle finestre realmente distinte. Duplicare tick
non equivale ad aggiungere storie indipendenti. Il confronto di copertura
aggiunge4 storie fit_extra per famiglia, a update e target invariati.

Selezione identica per tutti: massimo tra loss comune TF e FR su finestre
development. I rollout completi fino a10s e lo stress arrivano dopo il freeze.
Non si riseleziona il modello per nascondere un successivo fallimento.
Le soglie della Task37 non cambiano, incluse correnti individuali, timing,
carica, occupazioni, massimo errore e recuperi. Le correnti quasi nulle sono
stratificate, mai rimosse dalla congiunzione del gate.

Si mantengono la baseline frozen local161/SK canonico e i due candidati della38.
Quest'ultima è un confronto storico, non un controllo appaiato: campionamento
e budget per update differiscono. Le affermazioni causali usano i contrasti39.

## 🧠 Probe e decisione

Stati casuali fuori dalla manifold delle storie fit; sensibilità incrociata
tra gate a V/Ca fissi; contrasto di memoria nelle code a V uguale; errori
piccoli/attivi. Sono diagnostiche, non test fresh. Un miglioramento medio
non supera un fallimento raro: ogni contrasto conserva anche il caso peggiore.

Effetto diagnostico preregistrato: riduzione mediana≥20%, segno favorevole
in tutti3 seed e peggioramento del worst≤5%. Non è significatività statistica
di popolazione. Il trend30k→60k riguarda la loss breve, non prova che più
compute risolverà la dinamica lunga. Task40 richiede qualità completa e audit.
