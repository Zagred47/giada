# 🧭 GIADA — Dropshot per la compressione appresa del compartimento

> **Stato:** direzione approvata; N1 completato. Vedi
> [risultati CC-01…CC-04](cc_n1_result_v1.md) e [registro delle verifiche](cc_n1_progress_v1.md).
>
> **Piattaforma corrente:** GPU. Kernel custom e Tenstorrent sono differiti.
>
> **Punto di partenza:** compartimento ibrido locale validato nella Task35.
>
> **Perimetro:** un solo compartimento attivo. Task36 rinviata, non invalidata.
>
> **Numerazione:** CC-01…CC-20; nessuna rinumerazione delle task originali.
>
> **Primo notebook preparato:** [CC-N1 — CC-01…CC-04](../notebooks/cc_n1_contract_memory.ipynb),
> con [protocollo numerico](cc_n1_contract_memory_preregistration_v1.json).
>
> **Secondo notebook preparato:** [CC-N2 — CC-05…CC-08](../notebooks/cc_n2_unified_cell.ipynb),
> con [matrice e limiti](cc_n2_design_v1.md) e
> [preregistrazione](cc_n2_unified_cell_preregistration_v1.json). Nessun risultato N2 ancora acquisito.
>
> **Principio guida:** comprimere stato e calcoli, preservare le proprietà
> computazionali, recuperare con l'apprendimento ciò che è recuperabile.

---

## 🎯 Obiettivo

Costruire una **piccola dinamica appresa del compartimento**, senza dover
aggiornare separatamente ogni meccanismo del teacher durante l'inferenza.

La rete deve apprendere gli effetti collettivi dei meccanismi, mantenendo
soltanto lo stato e i vincoli fisici necessari. Il costo può essere trasferito
al training, ma il vantaggio finale deve essere misurato a parità di fedeltà.

```text
Compartimento ibrido validato — riferimento congelato
                         ↓
Informazione necessaria e memoria del compartimento
                         ↓
Cella unificata → stato compatto → dinamica appresa
                         ↓
Dati, distillazione e training ricorsivo
                         ↓
Proprietà dinamiche e generalizzazione causale
                         ↓
Conferma indipendente + costo reale
                         ↓
Decisione di rientro nella roadmap originale
```

Le tre quantità da distinguere sono **parametri appresi**, **stato dinamico** e
**costo per passo**. Ridurne una non garantisce di ridurre le altre.

Il passo iniziale resta **0,1 ms**: compressione dello stato e aumento del passo
temporale vengono studiati separatamente.

---

# 🧱 I. Contratto, costo e sufficienza dell'informazione

> **Notebook previsto N1 — CC-01…CC-04.** Prima di scegliere la rete,
> stabilire che cosa deve conservare e che cosa possiamo eliminare.

- **Task CC-01 — Contratto di equivalenza computazionale** → Definire dominio,
  ingressi causali, inizializzazione, orizzonti, proprietà da preservare e
  margini di non inferiorità. Separare accuratezza, stabilità e prestazioni.

- **Task CC-02 — Baseline e mappa dei costi** → Confrontare teacher nativo,
  formule GPU e ibridi congelati Task35. Misurare floor numerico, latenza,
  memoria, operazioni e costo end-to-end a batch rappresentativi.

- **Task CC-03 — Collisioni dello stato ridotto** → Costruire situazioni con
  stessi V/Ca e ingressi futuri, ma storie e gate differenti. Verificare quali
  rappresentazioni rendono indistinguibili stati con futuri diversi.

- **Task CC-04 — Quale memoria serve** → Confrontare vista istantanea,
  finestre causali e statistiche lente. Distinguere informazione mancante
  da difficoltà del decoder e definire i requisiti dell'inizializzazione.

### ✅ Gate CC-A — Contratto della compressione

Si procede alla cella soltanto con:

- baseline e floor numerico verificati;
- unità, segni e ingressi disponibili online dichiarati;
- split per episodi e domini, senza leakage tra finestre adiacenti;
- una mappa delle insufficienze delle viste ridotte;
- requisiti espliciti di memoria, inizializzazione e metriche;
- soglie, seed e budget del notebook successivo preregistrati.

Una vista insufficiente viene scartata o arricchita: non si attribuisce il suo
fallimento alla sola capacità della rete.

---

# 🧠 II. Cella unificata e stato compatto

> **Notebook previsto N2 — CC-05…CC-08.** Separare l'apprendimento condiviso
> dalla perdita di informazione introdotta comprimendo lo stato.

Un possibile stato operativo è:

\[
s_t=(V_t,Ca_t,z_t),
\]

dove \(z_t\) è una piccola memoria appresa, non necessariamente una copia
dei gate del teacher.

- **Task CC-05 — Controllo unificato a stato esplicito** → Apprendere tutte
  le cinetiche con una rete condivisa, mantenendo i gate espliciti e lo stesso
  passo. Confrontare con l'ibrido congelato per isolare il costo della condivisione.

- **Task CC-06 — Scala della memoria latente** → Sostituire lo stato ionico
  esplicito con V/Ca/z. Confrontare poche dimensioni latenti preregistrate,
  controllando capacità, parametri e stabilità del rollout.

- **Task CC-07 — Quale bias fisico conservare** → Confrontare correnti
  aggregate, conduttanze raggruppate e transizione diretta. Stessa informazione
  in ingresso; ablazioni mirate dei vincoli; flusso Ca distinto dalla corrente netta.

- **Task CC-08 — Inizializzazione senza teacher online** → Confrontare encoder
  fisico soltanto iniziale, warmup causale e stato standard. Usare reset e
  permutazione di z per verificare che la rete sfrutti davvero la memoria.

### 🧪 Bracci fondamentali

| Braccio | Stato | Parte appresa | Ruolo |
|---|---|---|---|
| Formule canoniche | fisico esplicito | nessuna | Riferimento numerico e di costo |
| Ibrido Task35 congelato | fisico esplicito | tassi di cinque canali | Baseline già validata |
| Cella unificata esplicita | gate e variabili fisiche | tutte le cinetiche previste dal contratto | Controllo della condivisione |
| Cella a correnti aggregate | V, Ca, z | memoria, corrente ionica e flusso Ca | Compressione con bilancio fisico |
| Cella a conduttanze raggruppate | V, Ca, z | memoria e conduttanze effettive | Bias sui potenziali di inversione |
| Cella a transizione diretta | V, Ca, z | avanzamento dello stato | Controllo meno strutturato |

Il controllo a gate espliciti è un ponte diagnostico, **non la destinazione
finale obbligatoria**. Inizialmente le sinapsi restano esplicite per non cambiare
contemporaneamente dinamica ionica e informazione sinaptica.

### ✅ Gate CC-B — Learnability della cella

La fase deve stabilire:

- quale memoria compatta è utilizzabile nel dominio;
- quali vincoli aiutano davvero rispetto ai controlli;
- se l'inizializzazione è disponibile senza reset teacher durante il rollout;
- se i segnali necessari al calcio sono conservati;
- se gli errori sono attribuibili a condivisione, compressione o apprendimento.

Non è sufficiente ricostruire bene lo stato del teacher con un autoencoder:
la rappresentazione deve sostenere anche la dinamica futura.

---

# 📈 III. Learnability, distillazione e mini scaling laws

> **Notebook previsto N3 — CC-09…CC-12.** Verificare quanto possiamo recuperare
> aumentando il lavoro di training senza aumentare quello di inferenza.

- **Task CC-09 — Mini scaling dati × budget** → Usare sottoinsiemi annidati
  di dati e checkpoint crescenti lungo le stesse traiettorie. Separare una
  piccola scala di capacità per distinguere dati, ottimizzazione e rappresentazione.

- **Task CC-10 — Distillazione delle variabili interne** → Confrontare loss
  sulle sole osservabili con target ausiliari di correnti, gate e flusso Ca.
  Le informazioni privilegiate sono target di training, non nuovi input online.

- **Task CC-11 — Apprendimento del rollout** → Confrontare training one-step
  con orizzonti ricorsivi crescenti. Contabilizzare transizioni e costo effettivo
  del training, non soltanto il numero di step dell'ottimizzatore.

- **Task CC-12 — Supporto dei regimi rari** → Confrontare stream uniforme e
  bilanciato per regimi/perturbazioni, a pari numero di campioni. Testare
  l'interazione preregistrata con l'esposizione ricorsiva.

### ✅ Gate CC-C — Diagnosi del limite di apprendimento

Distinguere sperimentalmente:

- miglioramento con più dati;
- miglioramento con più ottimizzazione;
- necessità di maggiore capacità o memoria;
- beneficio della supervisione interna;
- beneficio dell'esposizione ricorsiva e del supporto raro;
- eventuale saturazione nel budget testato.

Nessuna conclusione «basta fare scaling» senza curve appaiate che la sostengano.
Una saturazione locale non dimostra impossibilità universale.

---

# 🔬 IV. Generalizzazione e proprietà computazionali

> **Notebook previsto N4 — CC-13…CC-16.** Sfide ortogonali sul candidato
> sviluppato; conservare decisioni separate per ciascuna proprietà.

- **Task CC-13 — Generalizzazione causale dei parametri** → Variare parametri
  visibili, blocchi dei canali e combinazioni non viste. Misurare gli effetti
  delta degli interventi oltre all'errore sulle traiettorie assolute.

- **Task CC-14 — Compressione delle sinapsi** → Confrontare stati sinaptici
  espliciti e memoria aggregata. Prima eventi e draw casuali appaiati, poi RNG
  indipendenti per distribuzioni, correlazioni e risposta ai burst.

- **Task CC-15 — Passo temporale e ingressi intra-passo** → Confrontare 0,1 ms
  con pochi passi maggiori. Dichiarare la disponibilità causale degli eventi;
  nessun voltaggio o rilascio futuro teacher come ingresso.

- **Task CC-16 — Dinamiche lunghe e perturbazioni** → Verificare rollout lunghi,
  recupero, adattamento, refrattarietà e risposte frequenziali dove presenti
  nel teacher. Valutare timing e conteggio spike soltanto con supporto adeguato.

### ✅ Gate CC-D — Fedeltà dinamica nel dominio

Il candidato deve rispettare i requisiti dichiarati per:

- V, Ca e correnti/carica pertinenti;
- regimi attivi e silenti, senza nascondere errori nelle medie;
- interventi sui parametri e condizioni iniziali;
- memoria lenta e recupero;
- stochasticità nel contratto pathwise o distribuzionale pertinente;
- passo temporale e interfaccia effettivamente disponibili.

Il successo di due modifiche separate non ne certifica la combinazione:
la variante integrata scelta deve essere verificata su development prima del freeze.

---

# ⚙️ V. Interfaccia componibile e utilità operativa

> **Notebook previsto N5 — CC-17…CC-18.** Riutilizzare i checkpoint per
> verificare interfaccia e costo; non avviare ancora il multicompartmentale.

- **Task CC-17 — Porta per la futura composizione** → Verificare risposta
  locale a corrente esterna e trial-V, sensibilità/conduttanza e bilancio.
  Definire l'interfaccia per la futura Task36 senza simulare due compartimenti attivi.

- **Task CC-18 — Pareto di inferenza misurato** → Confrontare candidati,
  ibrido e formule con ottimizzazione simmetrica. Misurare latenza per ms
  biologico, memoria e throughput a batch rappresentativi, includendo eventi e RNG.

### ✅ Gate CC-E — Utilità del modello compresso

Registrare separatamente:

- correttezza dell'interfaccia;
- fedeltà scientifica;
- costo reale a pari lavoro e accuratezza;
- vantaggio materiale e sua incertezza nel workload preregistrato.

Un candidato accurato ma più lento può essere informativo scientificamente:
non è però una soluzione all'obiettivo prestazionale. Non rivendicare speedup
dal solo numero di parametri, dalla memoria GPU occupata o da baseline non ottimizzate.

---

# 🔒 VI. Conferma indipendente e rientro nella roadmap

> **Notebook previsto N6 — CC-19…CC-20.** Nessun aggiustamento sui dati
> di conferma dopo il congelamento del candidato.

- **Task CC-19 — Conferma indipendente congelata** → Congelare modello,
  interfaccia, seed e soglie prima di nuovi test. Verificare risultati per
  regime e caso peggiore, separando validità tecnica, scientifica e computazionale.

- **Task CC-20 — Decisione e rientro nella roadmap** → Registrare il modello
  e il dominio promossi, oppure il limite localizzato. Aggiornare DB e artefatti;
  proporre esplicitamente la ripresa della Task36, senza lancio automatico.

### ✅ Gate CC-F — Promozione finale

Promuovere soltanto le affermazioni sostenute dalla conferma:

- non inferiorità sulle proprietà richieste;
- causalità e inizializzazione deployabile;
- robustezza nei domini e orizzonti dichiarati;
- eventuale vantaggio operativo effettivamente misurato;
- tracciabilità di modello, dati, revisione e decisioni.

Sono esiti ammissibili: promozione, utilità scientifica senza vantaggio operativo,
oppure NO-GO circoscritto. Il piano non obbliga a far passare una particolare rete.

---

# 🔬 VII. Protocollo trasversale di ogni task

## 📋 Prima del run

Ogni notebook deve preregistrare:

- ipotesi, contrasti e fattori isolati;
- stato, ingressi causali e inizializzazione;
- baseline e floor numerico;
- split per episodi, domini, famiglie di stimolo e seed;
- capacità, budget e checkpoint della mini scaling law;
- metriche, soglie e margini di non inferiorità;
- controlli negativi e probe interni;
- criteri di arresto e massimo numero di bracci;
- artefatti da conservare e dipendenze tra le task.

## 📊 Output obbligatori

- errore one-step e rollout per orizzonte;
- errori V/Ca e correnti/carica pertinenti;
- comportamento dinamico e supporto effettivo degli eventi;
- effetti degli interventi e uso della memoria;
- mappe di errore per regime, dominio e seed;
- parametri, stato persistente, memoria temporanea e operazioni;
- costo GPU reale quando previsto dal contratto;
- esito tecnico distinto dalla decisione scientifica/prestazionale;
- decisione GO, CONDITIONAL GO o NO-GO, con limiti espliciti.

## 🚨 Regole di interpretazione

- Più dati non recuperano automaticamente un'informazione assente dagli input.
- Compressione dello stato e aumento del passo temporale non sono la stessa cosa.
- Una bassa loss di ricostruzione non certifica una buona dinamica latente.
- Corrente totale corretta non implica flusso di calcio corretto.
- Target teacher nel training non autorizzano input teacher durante l'inferenza.
- Una media buona non compensa un regime essenziale fallito.
- I vecchi sealed, ormai conosciuti, sono benchmark storici, non nuovi test indipendenti.
- Nessuna soglia viene allentata retroattivamente per far passare un candidato.

## ⚡ Parallelismo informativo

- Riunire nello stesso notebook i contrasti indipendenti che condividono dati e metriche.
- Appaiare seed, inizializzazione e minibatch dove il confronto lo consente.
- Salvare checkpoint crescenti senza riaddestramenti ridondanti.
- Non eseguire il prodotto cartesiano indiscriminato di tutte le opzioni.
- Conservare stop/go espliciti anche tra celle dello stesso notebook.
- Verificare equivalenza di predizioni, gradienti, Adam e clipping prima di
  vettorizzare più modelli sulla GPU.

Una sola tornata di riparazione diagnostica per blocco nel piano corrente;
oltre il budget preregistrato si rivede esplicitamente il piano. I guasti tecnici
restano distinti dai fallimenti scientifici.

---

# 🚀 Sequenza principale

I numeri romani sono sezioni; gli identificativi CC sono le venti task.
Le dipendenze puntuali restano quelle del piano strutturato, non una catena
obbligatoriamente seriale tra tutte le task.

```text
Task35 validata — baseline da conservare
↓
N1 · CC-01…04 — Contratto, costo, informazione e memoria
↓ Gate CC-A
N2 · CC-05…08 — Cella unificata, compressione e bias
↓ Gate CC-B
N3 · CC-09…12 — Scaling, distillazione ed esposizione
↓ Gate CC-C
N4 · CC-13…16 — Generalizzazione, sinapsi, tempo e proprietà
↓ Gate CC-D
N5 · CC-17…18 — Interfaccia e costo reale
↓ Gate CC-E
N6 · CC-19…20 — Conferma congelata e decisione
↓ Gate CC-F, secondo il contratto
Proposta di ripresa Task36 — due compartimenti attivi
```

---

# 🥇 Prima milestone concreta

> Preparare N1: definire le proprietà del compartimento da preservare,
> profilare le baseline e costruire controfattuali appaiati che rivelino quale
> memoria manca quando osserviamo soltanto V e Ca.

Questa milestone deve chiarire:

- che cosa significa «stesso comportamento computazionale» nel nostro dominio;
- dove il compartimento attuale spende tempo e memoria;
- quali variabili possiamo aggregare senza ambiguità eccessiva;
- quale inizializzazione è necessaria;
- quali ipotesi architetturali portare alla prima cella compatta.

Non richiede ancora di scegliere una rete vincente o di cancellare le formule
dal modello. Serve a rendere informativo il primo confronto di compressione.

---

## 📎 Documenti collegati

- [Piano strutturato: 20 task, dipendenze e 8 ipotesi](giada_compartment_compression_plan_v1.json).
- [Approfondimento scientifico, equazioni e motivazioni](giada_compartment_compression_roadmap_v1.md).
- [Roadmap originale: numerazione invariata](giada_modular_surrogate_roadmap.md).

Questo dropshot è la presentazione operativa della stessa proposta CC v1,
non una nuova campagna o una modifica delle task registrate nel mirror SQLite.
I gate CC sono riepiloghi dei criteri di avanzamento: non sono task aggiuntive
né criteri numerici già preregistrati. Airtable resta congelato.
