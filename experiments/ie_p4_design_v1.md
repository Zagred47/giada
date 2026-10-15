# ⚡ IE-P4 — Operatore discreto congiunto, stato esplicito

## 🎯 Bersaglio e lezioni precedenti

IE-P3 conserva l'algebra vettorizzata: apprendere semplici prodotti non ha
dato vantaggi. Qui si sostituisce invece la valutazione dei tassi di10canali
e la conversione delle loro costanti temporali in coefficienti discreti.
È il confronto IE-06/07/08, non una task originale rinumerata e non il
ritorno al modello monolitico CC. Nessuna memoria viene compressa.

I17gate dipendenti daV mantengono il proprio stato fisico:

`x_next = (1-z_theta(V))*x + z_theta(V)*u_theta(V)`.

Due sigmoid garantiscono u,z in[0,1]; z è scalato con massimi per-gate da
una griglia del dominio di training. È un prior dichiarato, comune ai bracci,
che evita di trattare allo stesso modo variazioni rapide e molto lente.
Le sigmoid possono contenere esponenziali: non diciamo "inferenza senza exp".

SK_E2 resta canonico: gateCa aggiornati→correnteCa→nuovoCa→nuovoSK.
LeggeCa, readout vettorizzato, conduttanze massime, sinapsi e solver identici.
Il dominio resta localeTask35, gbar dichiarati, dt0,1ms. Non è ancora
la sostituzione completa degli11canali né una verifica multicompartmentale.

## 🧪 Matrice nello stesso notebook

| Braccio | Calcolo dei coefficienti |
|---|---|
| Formula vettorizzata | Formule canoniche verificate |
| LUT0,25mV | Interpolazione lineare di u,z |
| Rete32/64, transition | Sole transizioni, bilanciate per scala del gate |
| Rete32/64, internal | Stesse transizioni più u e logz supervisionati |

Quattro famiglie apprese×3seed=12membri contemporanei su GPU, minibatch
appaiati, momentiAdam e clipping indipendenti. Il preflight verifica
predizioni e un aggiornamento contro esecuzioni isolate. Padding rimosso
nel timing. Due larghezze, due supervisioni e checkpoint crescenti separano
capacità, informazione interna e convergenza, senza cambiare il budget dopo
aver visto il fresh. Il successo dello scaling non viene assunto.

Fit/development atomici indipendenti inV[-100,60]mV e gate[0,1]. I target
sono formula, cross-check NumPy64/GPU: nessuna nuova conferma NEURON nativa.
La selezione per ogni seed usa solo lo stesso errore normalizzato di sviluppo.
Checkpoint e freezeSHA prima dei risultati fresh.

## 🔬 Prove dopo freeze

Fresh atomico indipendente: errore dei gate, probeu/logz. Ventiquattro nuovi
pathV imposti di40ms: costanti, gradini, sinusoidi, chirp, impulso/recupero,
randomwalk. Split per path/famiglia, quattro esempi per famiglia; nessun punto
del fresh viene utilizzato nel training. Misure per famiglia, gate, correnti
dei10canali e carica; SK escluso da questo test imposto e conservato invece
nella diagnostica integrata. Vincoli di occupazione e uscite dal dominio
sono riportati esplicitamente. La LUT mantiene i valori di bordo fuori dominio,
non è una garanzia di estrapolazione.

Task35 resta una diagnostica aperta40ms con solver/sinapsi invariati, non un
nuovo test sigillato. Il floor formula viene riportato insieme ai candidati.
I limiti registrati nel JSON devono valere per tutti i seed e famiglie prima
di considerare interessante il costo. Nessuna promozione operativa automatica.

## ⏱️ Costo e decisione

Stesso core osservabile a5output, Inductor ordinario/CUDA Graph, FP32/TF32off,
batch1/642/4096,30blocchi×20call intercalati. Preflight eager/compiled/graph;
compilazione e capture escluse. La rete deve risparmiare almeno10% contro
le **formule vettorizzate**, con intervallo appaiato positivo al batch642,
oltre a soddisfare la qualità. Il risultato LUT viene riportato come
controllo competitivo, non ignorato se superiore alla rete.

Si tratta di ingressi fissi residenti, non di speedup online con eventi/RNG,
trasferimenti e morfologia completa. Fallimenti tecnici e NO-GO scientifici
restano distinti. OpenEvolve e kernel custom non entrano in questo studio.
