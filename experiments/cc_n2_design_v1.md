# 🧠 GIADA — N2 · CC-05–CC-08

## 🎯 Domanda

Una dinamica unificata del compartimento può imparare a conservare i futuri
nativi con una memoria più piccola dei 18 gate? Confrontiamo nove bracci × tre
seed nello stesso run; nessuna task originale viene rinumerata.

## 🧪 Matrice preregistrata

| Task | Contrasto | Che cosa può dirci |
|---|---|---|
| CC-05 | Rete condivisa per tutte le 18 cinetiche, gate espliciti; formule e due ibridi congelati | Condivisione/ottimizzazione prima della perdita di stato |
| CC-06 | Corrente aggregata con z=0/4/8/16 | Primo profilo memoria–fedeltà nel budget provato |
| CC-07 | z=8: corrente e flusso Ca / quattro conduttanze / transizione V-Ca | Utilità del bilancio di membrana e di una struttura fisica più forte |
| CC-08 | z=8: encoder fisico iniziale / zero / warmup osservabile | Dipendenza dall'informazione iniziale; reset e permutazione diagnosticano l'uso di z |

La cella latente usa una ricorrenza convessa
`z' = z + (sigmoid(a)-z)*(1-exp(-dt/tau))`, con tau positiva.
La rete condivisa riceve V, log(Ca), corrente causale, parametri e z. Il
controllo esplicito limita invece le cinetiche a V/Ca e conserva la costruzione
delle 11 correnti e il bilancio di membrana. Non conserva sei canali analitici:
tutte le 18 cinetiche sono prodotte dalla stessa rete.

La testa a corrente produce separatamente corrente ionica netta e flusso Ca;
la testa a conduttanze distingue Ca/Na/Ih/K con segni e reversal fissati;
la testa diretta apprende gli incrementi V e log(Ca). Non è assunto che una
testa vinca. Le formule economiche sono bias candidati, non un dogma.

## 📦 Supporto e informazione

112 episodi NEURON di 40 ms, dt=0,1 ms: fit 64, development 24, diagnostica 24.
Quattro pannelli, correnti a blocchi e preparazioni ioniche continue indipendenti
per ruolo. La diagnostica è aperta soltanto dopo freeze; rimane un playground,
non una conferma sealed. N1 e Task35 sono supporti storici già aperti.

Questa prima matrice isola **il core ionico senza sinapsi**. La competenza con
sinapsi resta da ricomporre: non viene dichiarata conservata automaticamente.
Usiamo futuri NEURON reali anche dove il floor analitico N1 era insufficiente;
formule e ibridi sono misurati come riferimenti, non sostituiscono la verità.

Il modello con encoder riceve lo stato fisico **una sola volta all'inizio del
tratto valutato**, dopo 10 ms. Il warmup riceve soltanto V/Ca/corrente/parametri
nei 10 ms precedenti, senza gate. Successivamente entrambi sono autonomi per
8 e 30 ms, senza reset teacher. L'encoder è quindi un'interfaccia privilegiata
diversa dal warmup: il loro confronto non è a informazione identica.

## ⚖️ Controlli e limiti

Seed, episodi, minibatch e budget appaiati. Checkpoint 0/150/500/1500 lungo la
stessa traiettoria; scelta su sviluppo a 8 ms. Il controllo esplicito ha anche
una loss sui gate: è un controllo diagnostico privilegiato, non un confronto
loss-matched con il latente. La memoria del latente non viene supervisionata
come gate; una ridge probe fit-only controlla la decodificabilità su sviluppo.

Tutti i membri sono vettorizzati con pesi, momenti Adam e clipping indipendenti.
La larghezza è fissa, gli stati sono mascherati in un grafo padded comune:
parametri allocati e stato logico sono riportati distintamente. Questo run
non dimostra un vantaggio di costo della cella non-padded. Compilazione
facoltativa con equivalenza numerica; fallback documentato.

Le soglie di fedeltà CC-01 restano 0,05 mV pooled, 0,1 mV peggior episodio,
5e-6 mM Ca. Un fallimento scientifico produce comunque report completi;
non autorizza conclusioni universali sulla memoria minima o sullo scaling.
Il dominio è deliberatamente piccolo: assenza di spike sufficienti non
certifica proprietà di spike, adattamento o refrattarietà.

## ➡️ Dopo N2

Leggere tutte le curve e i contrasti, distinguere gate espliciti, memoria,
inizializzazione e bias. N3 affronta dati/budget/capacità, distillazione,
esposizione ricorsiva e supporto. Nessuna promozione automatica, nessuna Task36.
