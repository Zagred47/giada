# Task 41c — audit delle risposte autonome e degli stress

Esito: **NO-GO scientifico valido**, soglie invariate. Task 42 non autorizzata.
P110 resta congelata. Nessun training o selezione eseguito durante questo audit.

## Che cosa è confermato

P110 supera 528/528 controlli di traiettoria sia a ingressi imposti sia in
autonomia, 336/336 risposte imposte, 334/336 risposte autonome e 31/36 stress.
I 528 controlli sono 176 casi fisici per tre seed di modello; non sono 528
protocolli indipendenti. I cinque stress falliti appartengono a due storie
iniziali/pannelli dello stesso input seed, valutate nei tre seed di modello.

SHA-256 e CRC dell'archivio esterno, checkpoint, controlli nativi, fase delle
correnti e floor FP32/FP64 sono verificati. Tutte le tracce salvate sono finite.
Il replay CPU riproduce i verdetti per canale e il timing di tutti i 36 stress
P110. Sulle risposte, le metriche sono ricalcolate direttamente dalle tracce
CUDA originali. I successivi controfattuali CPU sono diagnostici separati.

## Le due risposte autonome

| Caso | Perturbazione e protocollo | Seed modello | Nap NRMSE | Limite |
|---|---|---:|---:|---:|
| 122 | blocco SK_E2, reversed_staircase | 17 | 0.0113569 | 0.01 |
| 108 | blocco K_Pst, irregular_train | 43 | 0.0100642 | 0.01 |

Entrambi partono da −79 mV nel pannello canonico. I seed degli input sono
rispettivamente 241129 e 241117. Fallisce solo il criterio NRMSE della risposta
di Nap_Et2: massimo, carica e altri canali passano. Il timing di queste piccole
risposte è esplicitamente non valutabile secondo il contratto esistente.
La normalizzazione usa il floor registrato di 1e−7 mA/cm²; l'errore assoluto
RMS è circa 1.14e−9 e 1.01e−9 mA/cm². Il floor non viene modificato.

La traiettoria di voltaggio non diverge: gli errori RMS dei casi base e
perturbati sono circa 0.000037–0.000132 mV. Tuttavia, una differenza fra due
traiettorie vicine può essere inaccurata anche quando entrambe passano i
criteri assoluti. Le differenze di voltaggio indotte arrivano a circa
0.000252 mV: la precisione locale e l'aritmetica meritano controlli dedicati.

Sul medesimo dataset, B6 fallisce due controlli diversi: caso 122, seed 29/43.
P110 li ripara e ne introduce due nuovi. Il conteggio aggregato 334/336
nasconde quindi **due miglioramenti e due regressioni**.

Interventi sul modello congelato:

- Sostituire solo l'equilibrio di Nap.m con quello canonico porta le due
  risposte a NRMSE 0.00815 e 0.00799 nel replay CPU.
- Rimuovere soltanto la correzione P110 su Nap.m, conservando le correzioni
  NaTs.h e Im, fa passare i due protocolli in tutti e tre i seed nel confronto
  CPU appaiato. Non abbiamo qualificato questa variante sugli altri casi.
- Rimuovere le altre correzioni produce effetti misti: esistono interazioni
  attraverso il voltaggio. Non è giustificata una causa unica per tutto il
  sistema, né il ritorno automatico a B6.
- Sui voltaggi visitati, P110 peggiora rispetto a B6 sia l'errore di equilibrio
  di Nap.m sia quello delle differenze locali fra ingressi vicini. Questo
  motiva supervisione appaiata e regolarità della funzione; non dimostra da
  solo che i nodi della base lineare siano la causa esclusiva.

I controfattuali CPU conservano i fallimenti originali nelle baseline, ma
non sono bitwise CUDA: massimo scarto 3.05e−5 mV, 1.46e−6 nei gate e
1.15e−9 mA/cm² nelle correnti assolute. I valori ufficiali restano quelli
dell'archivio CUDA. Il secondo probe include anche riferimento formula CPU
appaiato, con gli stessi esiti qualitativi delle reversioni.

## I cinque stress

Tutti riguardano **Ca_HVA**, ritorno lento al voltaggio −37.441528 mV,
input seed 241173. L'offset deve restare entro 0.3 ms.

| Storia | Seed modello | Offset originale | Con solo equilibrio Ca_HVA.m canonico |
|---|---:|---:|---:|
| 0, mixed_diagonal | 17 | 0.5 ms | 0.0 ms |
| 0, mixed_diagonal | 29 | 0.5 ms | 0.0 ms |
| 1, low_mixed | 17 | 3.8 ms | 0.0 ms |
| 1, low_mixed | 29 | 3.7 ms | 0.1 ms |
| 1, low_mixed | 43 | 0.8 ms | 0.0 ms |

Gli errori NRMSE di corrente sono appena 6.37e−5–1.11e−4. Passano anche
occupazione e accuratezza dei gate. Il problema è la precisione del crossing
di una coda molto piatta. Non c'è un secondo evento spurio lontano: le tracce
hanno un singolo intervallo attivo continuo.

Un bias nell'equilibrio del gate di attivazione m di circa 1.06e−4 nei seed
17/29 e 2.02e−5 nel seed 43 riscalarebbe la corrente m²h. Il rapporto fra
residuo di corrente e pendenza della coda predice ritardi di circa
0.484, 0.471, 3.800, 3.718 e 0.775 ms, coerenti con quelli osservati.
La sostituzione del solo equilibrio m ripara tutti i cinque casi; sostituire
il rilassamento m, l'equilibrio h o il rilassamento h non basta. Accumulare
lo stato in FP64 lascia tutti i cinque fallimenti.

Questi cinque errori di Ca_HVA sono identici in B6 e P110. La correzione 41b
non modificava Ca_HVA. P110 invece corregge l'ulteriore stress Im di B6,
portando il conteggio appaiato da 30/36 a 31/36.

Le due storie differiscono sia per stato iniziale sia per pannello di
conduttanze: la loro differenza non isola causalmente uno di questi fattori.

## Confronto con 41b e memoria storica

41b aveva riparato tutti gli stress già aperti (36/36), oltre alle risposte
e alle regressioni storiche. 41c usa protocolli, seed, stati iniziali,
pannelli e forme di stress nuovi. Il passaggio 36→31 è una lacuna di
generalizzazione a pesi immutati, non un peggioramento causato da nuovo
training. I tre bersagli precedenti Nap.m, NaTs.h e Im passano tutti i nuovi
stress imposti; il problema di Nap ricompare invece nella risposta autonoma.

Le query SQLite, conservate con i risultati in `memory_patterns.json`,
richiamano tre precedenti utili: il gate veloce Nap.m spostava già una coda
lenta nella Task 41; le ablazioni 40b mostravano effetti diversi di equilibrio
e rilassamento; 39j mostrava che una loss sui coefficienti più bassa non
garantisce la dinamica. Questi precedenti motivano i probe, ma l'attribuzione
attuale deriva dagli interventi sulla P110 effettiva.

Il costo resta un problema distinto: rispetto a B6, P110 impiega circa
19–60% in più sul blocco ionico e 7–41% in più sul compartimento integrato,
nei batch misurati. Nessuno speedup qualificato dalla qualità viene dichiarato.

## Proposta per 41d — da preregistrare

Un notebook, due pannelli mirati sullo stesso livello di astrazione:

1. **Ca_HVA.m:** 2×2 fra supporto uniforme/sensibile e risoluzione locale
   32/64. Stesso numero di dati e aggiornamenti, RNG separati per evitare che
   cambiare il campionamento cambi anche gli altri fattori.
2. **Nap.m:** 2×2×2 fra correzione lineare/liscia, loss puntuale/con differenze
   appaiate, training atomico/con risposte brevi in circuito autonomo.
   Stessa informazione, budget parametrico comparabile, controllo delle
   norme dei gradienti durante tutto il training.

I 12 bracci apprendibili possono condividere una sola esecuzione vettorizzata
per tre seed, se il preflight di equivalenza passa. Checkpoint lungo la
medesima traiettoria distinguono budget insufficiente da plateau. Le parti
non interessate rimangono congelate. Le formule canoniche negli oracle
servono alla diagnosi e ai target di training, non diventano l'inferenza
dei candidati.

Selezione su nuovi FIT/development, con tutte le soglie attuali e vincoli di
regressione. Dopo il freeze dei due moduli, nello stesso notebook si valuta
la composizione con quattro controlli: parent, solo Ca, solo Nap, entrambi.
41/41b/41c sono ormai pannelli aperti di regressione, non nuova conferma.
Qualsiasi candidato riparato richiederà poi una conferma indipendente nuova.

L'audit non autorizza automaticamente questo training. Registra una proposta
concreta e falsificabile; non garantisce che il prossimo esperimento passerà.
