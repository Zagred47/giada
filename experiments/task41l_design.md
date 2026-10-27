# Task 41l — Da dove arrivano le regressioni?

## Stato e perimetro

**Protocollo scientifico proposto, non ancora eseguibile o lanciato.** Sottotask
della Task 41 originale; non autorizza Task 42. Prima del lancio servono worker,
notebook unico, preregistrazione macchina con hash, test CPU/GPU e stima di costo.
Nessun risultato della 41l è già disponibile. Nessun cambiamento alle soglie,
al passo 0.1 ms, alle 18 occupazioni, alle dipendenze causali V/Ca, alla ricorrenza
affine limitata o al readout, alle sinapsi e al solver. Nessun fallback cinetico
analitico in inferenza. La compressione NON è parte della riparazione.

## Evidenza che motiva il disegno

Archivio 41k finale v5 SHA-256:
`ee629c24da75676e735670b4a0c5d22b223c932f9d394a4cd9a4ca738257e0c4`.
Audit: `results/task41k_audited_v1/comparison_with_J.json`.

- H: 6 valutazioni fallite; K_dyn_target: 8. Non sono casi biologici indipendenti.
- Target risolve due stress Ca_HVA slow_boundary_return (seed 17/29), ma introduce
  tre fallimenti Nap_Et2 e uno K_Tst. La correzione locale non è un GO globale.
- Le traiettorie aggregate e i test storici dei bracci repair passano; ciò non
  annulla gli errori di risposta, timing o ricorrenza.
- La retention 41k copiava H sui punti FIT nella metà inferiore dell'errore: non
  certificava una traiettoria interamente corretta e non proteggeva il teacher.
- Il probe iniziale usa un piccolo minibatch: coseni coefficiente/dinamica
  circa 0.0323, 0.0246, 0.00289, non negativi. Non prova un conflitto globale.
  Il passo dinamico normalizzato 1e-4 migliora la loss sui tre seed, 1e-3 la
  peggiora sui tre seed. Indizio di curvatura/ampiezza, non diagnosi di Adam.
- I compatti restano veloci ma non qualificati; nessuno speedup utilizzabile
  dimostrato. Il budget 1024 non dimostra impossibilità rappresentazionale.
- Il sidecar numerico già salvato nella41k mostra un'interazione importante:
  sul pannello41c le risposte fallite H/K_target sono1/2 nella politica primaria,
  2/3 con il solo solve64,1/2 con il solo voltaggio64 e0/0 con entrambi64.
  Sul41 K_target passa da1 a0 con solve64. Sono SOLO risposte autonome di questi
  pannelli: non prova che stress, ricorrenza o tutta la suite passino. Neural,
  calcio e gate non diventano tutti FP64. Il riferimento numerico va inoltre
  controllato in modo appaiato: nessun uso di questi numeri come deroga al FP32.

## A. Attribuzione causale congelata, senza training

Confrontare H e K_dyn_target già congelati. Non selezionare un nuovo candidato
su questi pannelli già aperti. Eseguire un fattoriale 2×2×2: coefficienti appresi
di Ca_HVA, Nap_Et2 e K_Tst presi da H oppure K, tutti gli altri da H. Aggiungere
K completo come controllo: la differenza rispetto al trapianto dei tre canali
misura il contributo congiunto dei canali restanti, non lo identifica singolarmente.
Scambiare INSIEME u/z di tutti i gate di ciascun canale; non correnti o stati a
posteriori. Testare identità all-H e all-K completo, occupazioni e precisione.

Per ogni combinazione usare gli stessi ingressi, stato iniziale, eventi e RNG:

1. V e Ca imposti dalla traccia di riferimento: isola errore locale dell'operatore.
2. V/Ca autonomi con lo stesso solver: espone amplificazione e interazioni.
3. Probe V libero/Ca imposto e V imposto/Ca libero, SOLO diagnostici: identificano
   quale feedback trasferisce l'errore. Non sono architetture candidabili.

Riportare correnti con segno, gate, Ca, V, carica, timing, primo istante di
divergenza e margini rispetto a OGNI soglia originale. Effetti principali e
interazioni sono calcolati su errori continui appaiati, oltre ai pass/fail.
Un Nap fallito in autonomo ma sano sotto ingressi imposti NON prova un difetto
intrinseco di Nap: il trapianto distingue la modifica di Nap dalla nuova traiettoria
causata dal calcio. Nessuna conclusione da soli conteggi aggregati.

### Controllo numerico incrociato obbligatorio

Riutilizzare il sidecar41k senza rifarlo inutilmente. Per H/K e i trapianti,
sui casi diagnostici interessati registrare la politica primaria e quella con
voltaggio+solve64; per H/K conservare anche le due combinazioni intermedie.
Controllare i riferimenti canonici con la MEDESIMA politica e contro il floor
nativo prima di attribuire le differenze alla rete. Misurare margini continui,
primo istante di divergenza, errori locali e risposta al feedback.
Non nominare la politica che passa più casi; il candidato ufficiale rimane
FP32 e deve superare tutti i predicati originali. Il sidecar distingue fragilità
numerica, errore appreso e loro interazione; non è una riparazione autorizzata
del solver. Confrontare anche il carrier FP64 usato dal training con quello FP32
di inferenza per quantificare il disallineamento, senza modificarli tacitamente.

## B. Matrice di riparazione: 8 bracci, 3 seed, un notebook

Tutti partono dallo stesso H, stessa architettura completa 4004 parametri,
stessi seed 17/29/43, stesso ottimizzatore/clipping e loss dinamica base della
41k target. Nessuna compressione, nessuna nuova testa o bypass. Tre fattori:

| Fattore | Livello 0 | Livello 1 | Domanda |
|---|---|---|---|
| A: ampiezza | LR iniziale 1e-5 | LR iniziale 1e-6 | Stessa direzione utile, passo troppo grande? |
| D: copertura | supporto FIT 1× | supporto FIT 4× annidato | Mancano esempi, a parità di aggiornamenti? |
| P: protezione | loss dinamica base | base + peggior gruppo canonico | La media sacrifica canali/regimi rari? |

Il supporto 1× ha le stesse cardinalità di quello della 41k ma nuovi seed. Il 4×
contiene esattamente l'1× più tre campioni indipendenti delle stesse famiglie.
Le probabilità delle famiglie restano uguali. La soglia di errore di H per il
targeting è stimata una volta sul FIT 1× e riusata nel 4×, senza ridefinire lì
il quartile; mantenere il mix 50% uniforme/50% sopra soglia della 41k. Registrare
eventuali gruppi privi di supporto e arrestare il preflight, non colmarli dai test.
Niente copie degli episodi test falliti nel FIT.
Usare numeri casuali appaiati per campionamento, non fingere identiche righe
tra dataset di dimensioni diverse. P0/P1 e A0/A1 condividono le righe esatte
all'interno di ogni livello D. Se il 4× migliora, distinguere maggior diversità
da maggior numero di visite ai regimi tramite contatori espliciti.

Protezione P1: aggiungere con peso 1 il massimo delle loss dinamiche canoniche
normalizzate per gruppo (canale × famiglia), su un minibatch FIT di protezione
stratificato, fisso nella composizione e rinnovato nei campioni. Normalizzazioni
uguali alla base, nessuna riscalatura appresa sui test. Non inseguire H e non
mascherare canali perché già passano. Calcolare anche P0 sugli stessi campioni
per diagnostica senza inserirli nel suo gradiente. Definire per codice gruppi
vuoti/ties/riduzioni prima del lancio. P1 cambia obiettivo e ponderazione dei
gruppi: un miglioramento non prova da solo interferenza nella rappresentazione.

Checkpoint comuni: 0,16,64,256,1024,4096 lungo la stessa traiettoria;
`lr(step)=lr_iniziale * 0.1**(min(step,1024)/1024)`: prima parte come41k,
poi mantenimento del minimo, senza adattamenti ai test. Fissare i seed dei
generatori nella preregistrazione eseguibile prima del training. A 1024 confronto
col budget41k (non appaiato sui nuovi FIT), a4096 mini scaling del budget.
Uguale numero di aggiornamenti non significa uguali FLOP o tempo: misurare
tempo, esposizioni ai dati e costo della protezione per braccio.

## C. Probe che distinguono le ipotesi

- Ampiezza: misurare update Adam effettivo, norme per blocco e loss per gruppo
  prima/dopo; replay diagnostico dello stesso update a frazioni 0,1/4,1/2,1 senza
  conservarlo. Non modificare i momenti del candidato attraverso il probe.
- Interferenza: gradienti per Ca_HVA/Nap_Et2/K_Tst/altri, coseni per blocco e
  per regime ai checkpoint. Un coseno negativo da solo non è una diagnosi:
  abbinarlo all'effetto finito sulle loss e alle regressioni osservate.
- Copertura: errori e conteggi per famiglia/regime; confronto D a parità di
  aggiornamenti e curve di budget. 4× insufficiente non esclude dati migliori.
- Protezione: verificare se P riduce nuove regressioni mantenendo la correzione
  Ca_HVA, oppure congela anche la riparazione. Questo distingue conservazione
  utile da semplice rinuncia a imparare.
- Feedback: confronto A congelato identifica se i difetti compaiono già a
  ingresso imposto o solo dopo ricomposizione autonoma.
- Precisione: controlli numerici appaiati distinguono errore locale della rete
  da sensibilità dell'accumulo di V/solve e interazione tra i due. Risolvere
  un'identità/floor invalido è precondizione tecnica; superare solo un sidecar
  numerico non autorizza a cambiare la politica primaria o la selezione.

Tutti gli effetti sono appaiati per seed/caso con ampiezza, segno e dispersione;
tre seed non sono molte repliche indipendenti. Nessuna nuova soglia scientifica
dedotta a posteriori, nessuna promessa che il prossimo run risolva tutto.

## D. Selezione, valutazione, compressione e velocità

FIT e development di selezione nuovi, disgiunti e fissati nella preregistrazione.
Selezione tramite criterio lessicografico canonico già usato nella 41k, senza
sostituirlo con la nuova loss P. Congelare checkpoint/hash prima delle suite
aperte 41/41c, stress, storiche e ricorrenza. Stessi predicati, soglie, supporto
eventi e orizzonti della 41k, copiati e verificati per uguaglianza strutturale.
Confrontare sempre H nello stesso run; development di run diversi non appaiato.

La matrice congelata A è diagnostica e non compete nella selezione B. Nessuna
conferma indipendente rivendicata con i pannelli già aperti. Se un candidato
supera tutto, servirà conferma nuova prima del passaggio autorizzato dalla roadmap.
Nessun braccio scelto a posteriori in base ai test aperti.

**Compressione parcheggiata**: K_copy_coeff/K_copy_dynamic restano evidenza
separata della 41k; non nuovi bracci addestrati qui. La loro velocità non compensa
il fallimento scientifico. Compressione futura solo da padre congelato esplicito,
con fedeltà al padre e qualità canonica misurate separatamente.

Misurare comunque i candidati repair con lo stesso benchmark ionic/integrated,
batch 1/642/4096 per seed, 30 ripetizioni appaiate, controlli formula/LUT/H e
parità compilata. Non presentare il costo dei trapianti diagnostici come costo
di una rete finale. Soglia di risparmio minimo 10% invariata. Una riparazione
scientifica senza velocità è un risultato parziale, non un GO completo.

## E. Condizioni tecniche prima del lancio

Worker e notebook unici, vettorizzazione dei seed con Adam/clipping indipendenti;
parallelizzare bracci solo dopo benchmark di memoria/tempo. GPU preflight delle
otto configurazioni, equivalenza sequenziale/vettorizzata, gradienti e optimizer
finiti, isolamento dei probe, scanner senza cicli, resume identico, export report
completo, hash/CRC backup fuori runtime. Registrare prereg e hash nel SQLite
prima di qualsiasi training; checkpoint esterno per fase prima di proseguire.
Il backup completo DB resta locale. Non modificare i sorgenti hash-pinned 41k.
