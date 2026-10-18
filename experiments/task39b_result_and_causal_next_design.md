# 🔬 Task 39b — risultato, memoria interrogata e proposta 39c

Stato: risultato 39b acquisito e verificato; **39c proposta, non eseguita né preregistrata**.
Task40 resta bloccata. Il candidato storico39 resta conservato.

## ✅ Esito verificato

Kaggle versione2, revisione scientifica04d6ed5f9c32073ad8d828656d7c2f1823c1ec7d.
36 membri,12bracci×3seed,60k update,48.5min training e56.7min inclusa valutazione.
2016 casi univoci e22176 misure per canale, tutti finiti, zero violazioni di
occupazione. SHA del freeze e del checkpoint verificati; conteggi ricalcolati.
Non è una ricostruzione indipendente dell'intera valutazione CUDA.

Nessun braccio supera la congiunzione. Migliore nuovo per numero di casi
development: base_h8_c1_r0,16/72; storico39:24/72. Il confronto storico non
isola una singola causa: sampler, finestre e selezione sono cambiati.
I72 casi sono24traiettorie×3seed, non72 repliche indipendenti.

Nel nuovo braccio16/72:45casi falliscono RMSE di corrente,37 massimo,
26carica,34timing,18gate;10falliscono soltanto timing. Le categorie si sovrappongono.
Canali più spesso fuori criterio:Ca_LVAst35/72,Ca_HVA27/72,Im26/72,K_Pst21/72.
Quindi il problema non è spiegato soltanto da picchi temporalmente ambigui.

Tutti36 membri scelgono60k e migliorano lo score development rispetto30k:
riduzione mediana71.98%. Questo score copre al massimo51.2ms; non dimostra
che il solo budget ripari le traiettorie fino10s.
Nessuno dei16contrasti soddisfa tutti i requisiti preregistrati insieme.
Il termine relativo riduce tutti12worst severity ratios dei quattro confronti,
circa61–93%, ma non migliora coerentemente il caso tipico. Severity non include
timing; i criteri di promozione invece lo includono.

## 🧠 Query esplorative effettivamente eseguite

Script riproducibile:`scripts/explore_task39b_memory.py`.
SQL, parametri,15precedenti completi e23catene ipotesi→evidenza→risultato in
`results/task39b_kaggle_v2/memory_exploration.json`. Ricerca limitata a GIADA,
includendo codici legacy che non contengono la parola GIADA. Nessuna lettura Airtable.
Query JSON1 dei nuovi casi/canali verificano che le metriche siano interrogabili
nel DB, non soltanto reperibili tramite il percorso di un file.

| Precedente | Risultato registrato | Implicazione per39b |
|---|---|---|
| Task2b | Rate supervision e multi-orizzonte riparano separatamente h physical-tau; rollout1000 circa.0026/.0025 | Identificazione dinamica e accuratezza endpoint non coincidono |
| Task3b | Supervisione simmetrica migliora61.5% a50k; baseline migliora63.1% raddoppiando budget; PCGrad non ripara | Informazione nella loss e budget possono produrre miglioramenti simili; non assumere conflitto gradienti |
| Task3d→3e | Supporto esteso e prior condizionale riparano shared; width maggiore non era necessario; conferma3/3 | Cercare supporto mancante prima di dichiarare impossibile la condivisione |
| Task21 | short/transition0/3, durate multiscala/transition3/3 | La scala temporale dei target conta; non basta chiamare lungo un rollout |
| Task24b | Sostituire tau aiuta soprattutto1s, inf soprattutto10s; budget/capacità aiutano; FP64 e clipping non spiegano il vecchio errore | Separare velocità di rilassamento, equilibrio, ottimizzazione e precisione |
| Task26 |94/300coseni gradienti negativi, senza attribuzione causale | Un coseno negativo da solo non dimostra che separare le reti risolva |
| IE-P5/P6 | Loss aggiuntive talvolta peggiorano; gradiente del termine logit84–219×base; mappe locali accurate | Controllare la scala dei gradienti; località/capacità può contare, ma tabulazione non è soluzione automatica |

Questi sono precedenti con domini e contratti diversi, non repliche della39b.
La Task21 riguardava h isolato e V costante; i vecchi risultati non autorizzano
a sostituire le soglie attuali né a riaprire test come se fossero indipendenti.

## 🔎 Perché gli interventi possono essere rimasti insufficienti

Il modello corrente implementa x_next=x+z(V,Ca)*(u(V,Ca)-x), con u,z appresi.
La sensibilità locale a u è z. Per z piccolo, una transizione può essere
accurata pur lasciando errori importanti su equilibrio o velocità. Sotto
input costante, dopo n passi la sensibilità a u è1-(1-z)^n. Gli stati iniziali
diversificati migliorano l'informazione su z ma non eliminano automaticamente
il piccolo fattore z della supervisione di u a un passo.

Probe CPUfloat64 sui pesi congelati, V=-55mV (coda storica già diagnostica):
Nap_Et2.h ha tau teacher1917.755ms;6.4ms di training sono appena0.00334tau.
La selezione51.2ms osserva circa0.0267tau. Per base_h8_c1_r0, Im.m ha equilibrio
teacher.017986 e predetto.010227/.011588/.015035 nei tre seed; Ca_LVAst.m ha
teacher.075858 e predetto.069314/.066661/.063130. Sono errori reali della funzione
appresa a questo V; non attribuiscono da soli l'intero errore di traiettoria.
Il probe estrae direttamente le teste, evitando la cancellazione di1-slope.

Le varianti architetturali39b sono test limitati:

- grouped rimuove connessioni:948parametri attivi contro1220base, pur con1772
  allocati per entrambi. Non è un test puro di interferenza a capacità identica.
- multiscale applica una miscela fissa di sigmoid a un unico scalare per gate.
  Conserva una sola frazione effettiva e lo stesso stato; non crea una banca di
  memorie. Il NO-GO non confuta ogni architettura con tempi diversi.
- gradient cosine attuale è un probe locale post-freeze. Non registra frequenze
  di clipping né l'equilibrio dei gradienti durante il training: restano ignoti.

## 🧪 Proposta 39c: un notebook diagnostico, tre livelli di confronto

Obiettivo: distinguere informazione insufficiente, copertura, budget e capacità
senza ridurre il livello di astrazione. Stessi18stati, dt0.1ms, readout e input
causali; formule teacher solo per etichette e controlli diagnostici.
La supervisione interna in training non inserisce formule nell'inferenza.

### A. Attribuzione senza training, sugli stessi checkpoint

Sostituire separatamente u, z ed entrambi con valori teacher; aggiungere swap
per gruppo di canali, misurando errore lungo, carica e timing. Oracoli sempre
non selezionabili e fuori benchmark costo. Confrontare lo stesso modello in
FP32/FP64 e teacher nelle stesse precisioni. Registrare interazioni: la somma
degli effetti di due swap non è garantita. Se entrambi gli swap non raggiungono
il floor previsto, fermare il training e diagnosticare interfaccia/valutatore.
Controllare timing sui casi timing-only con tracce sovrapposte e annotazioni
di eventi multipli/piatti, senza rimuoverli né cambiare i criteri attuali.

### B. Matrice principale2×2×2,8bracci×3seed

1. Informazione: loss corrente/stato oppure la stessa + supervisione simmetrica
   di u e z con scale congelate su fit. Testa se rendere osservabili i coefficienti
   rimuove la debole sensibilità del target breve. Gradienti dei termini misurati.
2. Supporto: pool esistente oppure pool addizionale stratificato V/Ca e stati
   estremi, generato esclusivamente da fit/teacher. Stesso numero di target;
   differenza in diversità/supporto, non nel numero di update. Nessun riuso stress.
3. Durata: ancore held brevi oppure durate log-spaziate0.1,1,10,100,1000,10000ms
   allo stesso V/Ca. Entrambi i livelli mantengono le stesse sequenze V variabili
   e lo stesso numero di ancore; cambia soltanto la durata delle ancore.

Per le ancore costanti si può comporre esattamente la mappa APPRESA:
x_n=u+(x_0-u)*(1-z)^n, usando calcolo numericamente stabile. Questo consente
target lunghi senza backpropagare100000passi. Verificare equivalenza su passi
brevi e lunghi, con floor di precisione. Non usarlo su V variabile né per
affermare che l'inferenza abbia un nuovo passo temporale.

Tutti i bracci usano seed/minibatch appaiati, normalizzazioni fit e la stessa
loss comune di selezione. Contare distintamente target, nodi unici, durate,
update e FLOP/tempo: stesso numero di target non significa stesso costo.
Una ladder30k/60k/120k sulla stessa traiettoria distingue miglioramento residuo
da plateau, senza reset dell'ottimizzatore. Il120k è un budget proposto da
congelare dopo benchmark tecnico, prima di osservare qualità; non una promessa.

### C. Quattro bracci di capacità/condivisione agli stessi due estremi

Alle condizioni minima e completa della matrice, aggiungere:

- trunk condiviso più largo (capacità aumentata, costo esplicito);
- teste/reti per gate realmente separate e vettorizzate con budget di parametri
  attivi approssimativamente appaiato, dichiarando differenze di capacità locale.

Totale12bracci×3seed=36modelli. Sono confronti condizionali agli estremi,
non un fattoriale completo delle architetture. Non riciclare la maschera grouped
come se fosse un controllo indipendente. Misurare gradienti per gate, rapporto
tra termini, clipping e curve per regime. Tutte le reti restano allo stesso
livello di astrazione; separare i pesi non ripristina cinetiche analitiche.

### Predizioni che distinguono le ipotesi

- Se u/z supervisionati riparano dove più budget endpoint non ripara, l'informazione
  dell'obiettivo è il candidato principale; se held-multiscala ottiene lo stesso
  beneficio, abbiamo due interventi convergenti come2b/21.
- Se solo supporto addizionale ripara regimi specifici a budget uguale, priorità
  ai dati. Un beneficio soltanto combinato è un'interazione, non tre cause provate.
- Se tutte le curve migliorano ancora e120k chiude il divario, il budget era
  insufficiente nel dominio misurato. Nessuna conclusione di scaling illimitato.
- Se largo aiuta e separato appaiato no, capacità/risoluzione più plausibile;
  se separato aiuta dove largo no, interferenza/specializzazione più plausibile.
  Sono indizi da interpretare con parametrizzazione e costo, non teoremi.
- Se FP64 elimina il residuo a pesi identici, componente numerica; se resta e
  gli swap lo eliminano, errore della funzione appresa. Il floor va rimisurato.
- Se il solo criterio di selezione sceglierebbe checkpoint diversi, riportare
  confronto diagnostico prospettico; non cambiare retroattivamente il freeze39b.

La nuova selezione deve includere sia durata breve sia held-lungo e un sottoinsieme
V-variabile lungo, su development; il test indipendente viene aperto solo dopo
freeze. Comparazioni39/39b rimangono storiche. Soglie37, unità, eventi, worst-case,
stati e correnti individuali conservati. Nessuno speedup inferito dal throughput
del training; timing core/operativo della rete congelata nella stessa esecuzione
con workload/backend appaiati. Task40 richiede qualità e audit, non il solo RMSE.
