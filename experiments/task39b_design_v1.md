# 🧪 Task 39b — diagnosi del training e del bias architetturale

## 🎯 Domanda

Il NO-GO Task39 dipende da orizzonte ricorrente, identificabilita' del campo,
loss delle correnti o parametrizzazione/interferenza tra gate?
Non abbassiamo l'astrazione: stessi18 gate espliciti, V/Ca imposti, dt0.1ms,
readout economico invariato, nessuna cinetica canonica nel forward appreso.
La Task40 resta bloccata: questa e' una sottotask39, non una nuova fase.

## 🔀 Matrice preregistrata

| Bracci | Fattori | Membri |
|---|---|---:|
| 8 base | orizzonte8/64 tick x collocazione off/on x loss relativa off/on | 24 |
| 2 grouped | ancore8/off/off e64/on/on | 6 |
| 2 multiscale | stesse due ancore | 6 |

Tre seed17/29/43, inizializzazioni appaiate, finestre e stream condivisi.
Le interazioni architettura-training sono osservabili sulle due ancore,
non su tutte le combinazioni dei tre fattori.

## 🧠 Varianti senza formule canoniche in inferenza

**Grouped:** hidden32 suddiviso in16 unita' condivise,8 rapide e8 lente;
connettivita' delle teste mascherata. Gli8 output piu' rapidi sono assegnati
dal mediano delle frazioni teacher su32 finestre per ogni famiglia fit.
Non e' una classificazione biologica universale e non usa development.

**Multiscale:** solo la frazione d'aggiornamento cambia da sigmoid(raw) a
media di4 sigmoid(raw+offset), offset0,-3,-6,-9 con traslazione comune
che uguaglia il valore iniziale del controllo a raw=-3. Nessun parametro
appreso aggiuntivo, nessun tau canonico, nessuna esponenziale del teacher.
E' un'ipotesi di condizionamento, non un aumento garantito di espressivita'.

Nominali1772 parametri per membro; attivi strutturalmente1220 base/multiscale,
948 grouped, per via di input azzerati e maschere. Confronto non equiparabile
a un aumento di capacita'; costi e connettivita' effettiva dichiarati.
Aggiornamenti convessi e dipendenza corretta da V/Ca preservati.

## 📦 Dati e supervisione

Ogni update usa16 finestre contigue da64 tick =1024 target per membro.
I bracci corti ripartono dallo stato fornito ogni8 tick, quelli lunghi soltanto
al tick0. Nessun reset nelle valutazioni finali autoregressive.
Fit e fit_extra mescolati50/50;512 finestre/famiglia/ruolo, meta' campionate
uniformemente e meta' arricchite vicino ai cambi degli input.
Il controllo e' comparabile internamente, non una replica esatta del sampler
Task39 a2048 finestre di8 tick. Il vecchio candidato e' un riferimento congelato.

Collocazione: sostituzione del25% delle finestre con stesse sequenze V/Ca e
stati iniziali fisici indipendenti, target generati offline dal teacher.
Stati e pre-stati restano coerenti anche ai reset dei bracci corti.
Non si sommano label extra ai bracci trattati; esposizione registrata.
Teacher offline ammesso per supervisione, mai nel candidato in inferenza.

## ⚖️ Selezione, metriche e controlli

Checkpoint0/3k/10k/30k/60k sulla stessa traiettoria. Selezione per membro
con minimo del massimo della loss comune fra famiglie e orizzonti8/512 tick,
quattro finestre development per famiglia, comuni a tutti i bracci.
Nessuna selezione con l'auxiliary loss relativa, stress o test indipendente.
Soglie Task37 immutate. Valutazione finale su12 famiglie, gate/correnti,
carica, timing, occupazione, blocking, memoria e recuperi fino a10s.

12 contrasti fattoriali +4 contrasti architetturali,3 rapporti seed appaiati.
Supporto: mediana rapporto<=0.8, tutti<1, peggior caso non oltre1.05.
Interazioni descrittive, nessuna significativita' di popolazione dichiarata.
Errori per gate, gradienti delle teste, frazioni, equilibri, pendenze del
Jacobiano diagonale e stati fisici fuori traiettoria come probe interni.
I dettagli temporali sono diagnostici; eventi piatti/multipli non autorizzano
a cambiare il criterio originale. Nessuna certificazione di biforcazioni
o spike emergenti: V/Ca sono ancora imposti.

## ⚡ Esecuzione e arresti

36 membri vettorizzati, Adam e clipping indipendenti. CUDA graph nativo
comprensivo di forward/backward/optimizer riduce overhead del loop Python.
Preflight obbligatori prima del budget: equivalenza36 ottimizzatori isolati,
teacher/collocazione/fasi, ricorrenza eager/graph, due passi training
captured/eager, benchmark tecnico20 update con reset completo pesi/momenti.
Fallimenti tecnici producono failure_report, non NO-GO scientifico.
Output compatto ogni1000 update. Nessuno speedup d'inferenza rivendicato.

Budget60k fisso; tempi, target e memoria riportati separatamente.
Un candidato che passa development richiede audit prima della Task40;
stress gia' aperto non diventa improvvisamente una conferma indipendente.
