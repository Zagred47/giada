# 🧭 GIADA — Roadmap modulare per surrogate biofisici

> **Direzione corrente — 17 ottobre 2026:** aumentare gradualmente
> l'astrazione del **solo blocco ionico**, dai gate alle correnti, cercando
> fedeltà causale e accelerazione misurata. Sei nuovi blocchi, **Task 36–71**,
> alternano singolo compartimento e più compartimenti a tre livelli di
> astrazione. Sono nella **sezione VI-ter** di questo stesso dropshot.
> La prima attività è la **Task 36 — Contratto e scala di astrazione ionica**.
> Le task sono decisioni scientifiche, non trentasei run obbligatori.
>
> **Continuità:** Task0…35 e i prerequisiti conservano identità ed evidenze.
> Il [percorso CC](giada_compartment_compression_dropshot_v1.md) resta storico:
> N1 è concluso; N2 è terminato tecnicamente senza candidati promossi nelle
> soglie CC. [Audit IE-01 e risultati N2](cc_n2_result_v1.md): 108 valutazioni
> ricalcolate da array salvati e selezione verificata, nessun nuovo training.
> N3…N6 sono sospesi come sequenza operativa, non dichiarati eseguiti o falliti.
> Le vecchie Task36–49 diventano **Task72–85**, con corrispondenza esplicita
> sotto. Gli identificativi nei vecchi artefatti e nel DB restano storici;
> non si rietichettano risultati o protocolli già registrati.
> IE e CC conservano le evidenze ma non sono più la sequenza operativa corrente.

> **Stato:** documento di direzione sperimentale, da usare per pianificare i run.
>
> **Piattaforma corrente:** GPU. Tenstorrent è deliberatamente rinviato.
>
> **Teacher canonico:** modello Hay/L5PC NEURON, con **642 segmenti effettivi** nel manifest verificato.
>
> **Principio guida:** isolare, verificare, comporre; non aumentare la complessità prima di avere localizzato il failure mode.

---

## 🎯 Obiettivo

Costruire surrogate neurali capaci di sostituire progressivamente le componenti
del neurone multicompartmentale, preservandone stato, causalità, eventi e
stabilità di rollout.

Il progetto non parte più da un modello monolitico del neurone completo. Il
teacher viene decomposto in sottosistemi con interfacce esplicite:

```text
Spettri atomici indipendenti
├── cinetiche dei gate
├── meccanismi ionici
├── calcio e dinamiche lente
├── passivo e capacitivo
├── sinapsi deterministiche e stocastiche
├── coupling assiale
└── morfologia

                ↓ composizione controllata

meccanismi → compartimento → ramo → subtree → neurone
```

Lo scopo non è soltanto trovare un modello che funzioni, ma costruire una mappa
causale di **dove**, **quando** e **perché** ogni rappresentazione fallisce.

---

# 🧱 0. Fondazioni e contratto sperimentale

> Questa fase precede qualunque training. Il contratto dati deve essere
> sufficiente per il task prima che venga scelta l'architettura.

- **Task 0.1 — Inventario meccanicistico del teacher** → Estrarre automaticamente
  dai file `.mod` variabili `STATE`, `PARAMETER`, `ASSIGNED`, ioni letti e
  scritti, correnti, dipendenze e metodo di integrazione.

- **Task 0.2 — Classificazione causale dei meccanismi** → Distinguere gate
  voltage-dependent, gate calcium-dependent, dinamiche delle concentrazioni,
  passivo, sinapsi event-driven, rilascio stocastico e coupling assiale.

- **Task 0.3 — Contratto atomico dei dati** → Dichiarare, per ogni sottosistema,
  stato sufficiente, ingressi, parametri, target, passo temporale, solver,
  precisione e variabili privilegiate.

- **Task 0.4 — Doppio oracle indipendente** → Confrontare formula estratta dal
  `.mod` ed esecuzione NEURON isolata. Nessun dataset viene accettato finché i
  due percorsi non coincidono entro tolleranza preregistrata.

  **Completata (2026-09-20):** 7.360/7.360 confronti superati con zero
  fallimenti e massimo errore assoluto `2.22e-16`, contro soglia
  `atol=rtol=1e-10`. Il risultato certifica il target atomico di `Ca_HVA` a
  voltaggio controllato; non certifica ancora il trasferimento accoppiato.

- **Task 0.5 — Split per dominio** → Separare interpolation, boundary, OOD e
  conferma embedded. Non dividere casualmente punti adiacenti della stessa
  curva tra train e test.

  **Completata (2026-09-20):** contratto deterministico di 12.582 casi in 13
  strati, con 8.096 casi di fit, finestre contigue di sviluppo/test, boundary
  fisici e numerici, OOD a singolo asse e stress multifattoriale separato.
  Overlap tra strati e tra fit/evaluation: zero. La conferma embedded resta
  intenzionalmente vuota e riservata a protocolli teacher-coupled futuri.

- **Task 0.6 — Baseline GPU comune** → Stessi batch, seed, precisione, budget,
  misure temporali e stream di dati per tutti i bracci.

  **Completata (2026-09-20):** contratto GPU comune con tre seed appaiati,
  stream minibatch persistibili, float32 primario senza AMP, griglia LR e
  checkpoint uguali, firewall development/test e benchmarking CUDA distinto
  tra eager e compiled. Helper runtime deterministici inclusi.

### ✅ Gate 0 — Integrità del contratto

Si procede soltanto se:

- ogni target è determinato dagli input dichiarati;
- unità e convenzioni di segno sono esplicite;
- formule e NEURON concordano;
- split e domini sono privi di sovrapposizioni;
- il dataset registra quanto serve alle metriche previste;
- tutti gli artefatti sono identificati da hash e revisione del teacher.

---

# ⚛️ I. Spettro dei gate ionici

## 🔹 A. Gate sotto voltage clamp costante — Priorità massima

Con voltaggio costante durante l'intervallo:

\[
\dot x=\frac{x_\infty(V)-x}{\tau_x(V)}
\]

ha update esponenziale:

\[
x_{t+1}=x_\infty(V_t)+[x_t-x_\infty(V_t)]e^{-\Delta t/\tau_x(V_t)}.
\]

- **Task 1 — Gate `m` di Ca-HVA** → Imparare
  \((V_t,m_t,\Delta t)\mapsto m_{t+1}\).

- **Task 2 — Gate `h` di Ca-HVA** → Ripetere l'esperimento per
  \((V_t,h_t,\Delta t)\mapsto h_{t+1}\).

- **Task 3 — Cella congiunta `m+h`** → Condividere la rappresentazione di
  \(V\), mantenendo stati e output distinti e interpretabili.

- **Task 3b — Diagnosi dell’ottimizzazione condivisa** → Dopo il NO-GO
  circoscritto della Task 3, distinguere con una matrice appaiata supervisione
  asimmetrica dei rate, conflitto dei gradienti, warm-start e budget. La fase è
  development-only e non autorizza da sola la Task 4.

- **Task 3c — Conferma della riparazione simmetrica** → Congelare il trunk
  condiviso physical-τ con supervisione simmetrica dei quattro rate, addestrare
  i seed come ensemble GPU vettorializzato e aprire una nuova conferma fresh
  disgiunta soltanto dopo il superamento del gate development.

- **Task 4 — Matrice di primitive appaiate** → Confrontare nello stesso run:
  formula originale, lookup table, interpolazione, Chebyshev/polinomio, MLP
  diretto, GRU, cella physical-\(\tau\) e cella direct-\(z\).

- **Task 5 — Mini scaling laws** → Salvare checkpoint a capacità e budget
  crescenti lungo le stesse traiettorie di training.

- **Task 6 — Stress test del dominio** → Coprire tutto l'intervallo di
  voltaggio, punti numericamente delicati, stati iniziali indipendenti
  dall'equilibrio, diversi \(\Delta t\) e rollout molto lunghi.

### 🧪 Bracci fondamentali

| Braccio | Output appreso | Struttura imposta | Proprietà principale |
|---|---|---|---|
| Formula `.mod` | nessuno | completa | oracle e costo reale |
| LUT/spline/polinomio | curve cinetiche | parziale | forte baseline numerica |
| MLP diretto | \(x_{t+1}\) | nessuna | controllo di capacità |
| GRU | stato successivo | ricorrenza generica | controllo espressivo |
| Physical-\(\tau\) | \(x_\infty,\tau\) | solver esponenziale | interpretabilità e \(\Delta t\) variabile |
| Direct-\(z\) | \(x_\infty,z\) | combinazione convessa | efficienza a \(\Delta t=1\) ms |

### ✅ Gate A — Learnability atomica

La fase deve stabilire:

- architettura minima stabile;
- errore dei gate e della corrente;
- comportamento OOD;
- vantaggio o svantaggio rispetto a LUT e polinomi;
- compromesso errore–MAC–latenza GPU;
- generalizzazione della variante physical-\(\tau\) a passi diversi;
- eventuale vantaggio del direct-\(z\) a 1 ms fisso.

---

## 🔹 B. Voltage path esogeno variabile — Priorità massima

L'update precedente è esatto soltanto quando \(V\) è costante. Nel neurone
accoppiato il gate dipende dall'intero percorso \(V(s)\) nell'intervallo.

- **Task 7 — Sequenze di voltage step** → Percorsi piecewise-constant noti.
  **Conclusa nel dominio esogeno** con `07r_roadmap_voltage_step_sequences.ipynb`:
  contrasti temporali appaiati e gate superati. I notebook storici `07`, `07b`,
  `07c` restano studi differenti su Ca-HVA closed-loop e semantica dei confini.

- **Task 8 — Rampe e chirp** → Percorsi continui con velocità differenti.

- **Task 9 — Percorsi fisiologici registrati** → Usare voltage path provenienti
  dal teacher, mantenendoli teacher-forced.

- **Task 10 — Test di sufficienza dell'ingresso** → Confrontare solo \(V_t\),
  \((V_t,V_{t+1})\), statistiche intra-ms, pochi campioni del path e substep.
  **Conclusa nel perimetro diagnostico**: 350 path già aperti, matrice 8×5;
  prima vista campionata testata al gate: 21 campioni oracle a 40 passi.
  Non è un limite universale né un contratto causale online validato.
  Lo studio supplementare `TG-01` resta distinto da questa Task 10.

- **Task 11 — Operatore path-aware** → Introdurlo soltanto se \(V_t\) risulta
  causalmente insufficiente.
  **Eseguita nel sistema minimo Ca-HVA+pas** con
  `11_roadmap_causal_operator.ipynb`: contrasto sugli ingressi pianificati,
  integratori accoppiati, path compatto, coefficienti integrati e sforzo
  adattivo. Esito scopiato in
  `teacher_roadmap_task11_causal_operator_result_v1.md`; nessuna prova sul
  teacher multicompartimentale o sugli spike.

### ✅ Gate B — Contratto temporale minimo

Determinare l'informazione minima sul percorso di voltaggio necessaria per
aggiornare correttamente i gate a 1 ms. Un fallimento con il solo \(V_t\) non
viene attribuito alla capacità della rete senza questo controllo.

---

# 🧲 II. Meccanismo Ca-HVA completo

Il teacher implementa:

\[
I_{\mathrm{CaHVA}}=\bar g_{\mathrm{CaHVA}}m^2h(V-E_{\mathrm{Ca}}).
\]

- **Task 12 — Corrente analitica** → Calcolare la corrente dai gate predetti;
  non farla apprendere senza una precisa ablation.

- **Task 13 — Allineamento temporale** → Stabilire se corrente e conduttanza
  usano stato pre-update, post-update o la semantica effettiva di
  `BREAKPOINT/SOLVE`.

- **Task 14 — Variazione dei parametri** → Variare \(\bar g\),
  \(E_{\mathrm{Ca}}\), stato iniziale, presenza e maschera del meccanismo.

- **Task 15 — Decomposizione contro end-to-end** → Confrontare gate neurali più
  corrente analitica, rete diretta per la corrente e rete con target ausiliari.

- **Task 16 — Conferma embedded congelata** → Inserire il candidato nel teacher
  mantenendo ancora il voltaggio teacher-forced.

- **Task 17 — Sostituzione causale** → Sostituire davvero `Ca_HVA` e misurare
  l'effetto sul compartimento senza riaddestrare il modello.

### ✅ Gate C — Sostituzione del meccanismo

Il meccanismo è promosso soltanto se mantiene gate, corrente, vincoli e rollout
nel playground e nella conferma embedded.

---

# 🧬 III. Generalizzazione tra famiglie di meccanismi

Non si assume che una singola cella sia universale.

- **Task 18 — Famiglia HH standard** → Ca-HVA, NaTa e un meccanismo del
  potassio con gate voltage-dependent.
  **Conclusa nel contratto isolato**: Ca/Na superati nella18; K_Pst confermato
  nella18d (feature cuspide,3/3seed, tutti domini richiesti). OOD e sostituzione
  embedded di Na/K restano fuori da questa promozione.

- **Task 19 — Gate singolo** → `Ih`, `Im` o meccanismo equivalente.
  **Conclusa nel contratto isolato**: Ih e Im rate-supervised superati3/3seed,
  stati estremi e rollout1000 a V costante. MLP eager più lento di formule/LUT;
  nessuna accelerazione o sostituzione embedded implicita.

- **Task 20 — Dipendenza dal calcio** → `SK_E2`, separato dai gate puramente
  voltage-dependent.
  **Conclusa nel contratto isolato**: calcio log/lineare superati3/3seed;
  controllo soloV fallito. Gate e percorsi di calcio imposto confermati;
  tau canonica1ms analitica. Non CaDynamics né speedup.

- **Task 21 — Dinamica lenta** → Meccanismo con costante temporale molto più
  lunga.
  **Conclusa nel contratto isolato**: h di Nap_Et2, tau circa350–2194ms;
  multiscala con/senza rate3/3seed, breve/transition0/3, breve/rate2/3.
  Primario confermato con rollout1ms fino10s; floorfloat32 non spiega il
  residuo learned. Non intero canale Nap.
  `task21_slow_gate_transfer.json`, `21_roadmap_slow_gate_transfer.ipynb`.

- **Task 22 — Condivisione controllata** → Confrontare modelli distinti, trunk
  condiviso con head separate e modello condizionato dall'identità del
  meccanismo.
  **Conclusa**: confronto Ih/Im/Nap_h a V imposto; tre famiglie, width16/32,
  tre seed, stessi dati e loss perbundle. Parametri non uguali, tradeoff
  esplicito; controllo identità/head scambiate. SK_E2 escluso per non cambiare
  contemporaneamente il tipo di ingresso. `22_roadmap_controlled_gate_sharing.ipynb`.
  Riferimento indipendente confermato3/3, condivisione non confermata nel budget
  registrato. Risultati verificati: `experiments/results/task22_kaggle_eb494b8/`.

La domanda centrale è:

\[
\text{quanto compute può essere condiviso senza perdere stabilità?}
\]

---

# ⚡ IV. Spettri indipendenti eseguibili in parallelo

> Questi filoni possono procedere nello stesso capitolo sperimentale quando
> condividono infrastruttura, ma mantengono contratti e decisioni separati.

### 🗂️ Registro operativo e dipendenze (riconciliazione 2026-10-04)

Le Task1–35 conservano numero e significato; dal 17 ottobre le precedenti
Task36–49 sono rinviate e rinumerate Task72–85 (tabella in VI-ter).
Questa sezione usa
identificativi `IV-A1` … `IV-D3`, registrati come **azioni proposte**, non come
esperimenti già preregistrati o conclusi. Il piano strutturato e i criteri di
completamento sono in [giada_section_iv_plan.json](giada_section_iv_plan.json).
Prima di ogni run si congelano domini, soglie numeriche, seed, budget e controlli.
Il riuso di risultati precedenti richiede una verifica esplicita del contratto:
per esempio, la Task20 su SK con calcio imposto non valida CaDynamics.

| ID stabile | Esperimento da preparare | Prerequisito IV | Criterio di completamento |
|---|---|---|---|
| IV-A1 | RC, leak, capacità e convenzioni | — | Doppio oracle, unità, equilibrio e costante temporale |
| IV-A2 | Correnti variabili e voltage update passivo | IV-A1 | Bilancio capacitivo, rollout, convergenza e interfaccia correnti |
| IV-B1 | CaDynamics_E2 con corrente imposta | — | Concentrazione, decadimento, unità e risposta agli impulsi |
| IV-B2 | CaDynamics → SK a V imposto | IV-B1 + SK validato | Attribuzione dell'errore e timing; feedback elettrico nella Task32 |
| IV-C1 | AMPA/NMDA/GABA deterministiche | — | Stati, correnti, carica e timing eventi intra-ms |
| IV-C2 | Plasticità breve e rilascio stocastico | IV-C1 | Stato/RNG, replay e distribuzioni di rilascio |
| IV-C3 | Interfaccia sinaptica integrata | IV-C1, IV-C2 | Burst misti, rollout, ripristino e stato sufficiente |
| IV-D1 | Due compartimenti passivi | IV-A2 | Bilancio assiale, transiente e confronto solver |
| IV-D2 | Catene passive 4–8–16 | IV-D1 | Scaling dell'errore e costo per taglia |
| IV-D3 | Biforcazione e albero passivo | IV-D2 | Bilancio ai nodi, rinumerazione e rollout |

**Punti di ingresso obbligatori nelle sezioni successive:**

- Task23–27: moduli ionici isolati validati e voltaggio imposto. La Task23 deve
  includere un preflight isolato Ca_LVA: il flag della Task22 permette la
  preparazione, non certifica Ca_LVA o il completamento della sezione IV.
- Task28: dichiarare le concentrazioni come input se imposte; se sono generate
  dal modello o il feedback calcio è attivo, richiedere IV-B1/B2.
- Task29–30: IV-A2 prima della composizione ionico-passiva.
- Task32: IV-B2 prima del feedback con dinamiche lente.
- Task33: IV-C3 prima dell'integrazione sinaptica.
- Task35: IV-A2, IV-B2 e IV-C3 per il compartimento locale completo dichiarato.
- Task72 (ex36): IV-D1; Task73–74 (ex37–38): IV-D2;
  Task75–80 (ex39–44): IV-D3. Gli stessi prerequisiti passivi si applicano
  alle geometrie dei nuovi blocchi multicompartmentali 2, 4 e 6 in VI-ter.

Questi sono prerequisiti aggiuntivi: restano necessari i gate ionici e le
validazioni del sistema ricevente. IV-A1, IV-B1 e IV-C1 possono condividere un
notebook; i passi successivi partono quando passa il proprio prerequisito.
La Task23 può procedere in parallelo a questi filoni sotto V imposto.
Ogni famiglia conserva il proprio esito; un successo non chiude le altre.
Formule economiche e solver classici validati possono restare componenti finali;
una rete è giustificata dal confronto accuratezza/costo.

## 🔋 A. Passivo e capacitivo

- RC puro;
- leak più capacità;
- corrente esterna;
- variazione di area, \(C_m\), leak ed equilibrio;
- rollout verso l'equilibrio corretto.

Le soluzioni analitiche sono baseline obbligatorie.

## 🧪 B. Calcio e dinamiche lente

- `CaDynamics_E2` isolato;
- input di corrente di calcio controllato;
- feedback `CaDynamics + SK`;
- adaptation e rollout lungo;
- conservazione dei tempi di decadimento.

## ⚡ C. Sinapsi

- AMPA deterministica;
- NMDA con blocco voltage-dependent;
- GABA;
- plasticità a breve termine;
- rilascio probabilistico con RNG esplicito;
- eventi multipli intra-ms.

La dinamica sinaptica event-driven non viene forzata nella cella dei gate HH.

## 🌿 D. Coupling assiale

- due compartimenti passivi;
- catena 4–8–16;
- biforcazione;
- albero passivo;
- confronto con Hines/tree solver.

Il solver classico è una baseline, non soltanto un teacher.

---

# 🧩 V. Composizione ionica

La prima composizione viene valutata sotto voltaggio teacher-forced.

- **Task 23 — Ca-HVA + Ca-LVA** → Prima coppia correlata.
  Protocollo originale: `experiments/task23_calcium_pair_composition.json`;
  notebook `notebooks/23_roadmap_calcium_pair_composition.ipynb`.
  Audit isolato Ca-LVA obbligatorio, riferimento indipendente e due forme di
  condivisione; gate e correnti individuali/sommate sotto V imposto. Nessuna
  autorizzazione implicita a CaDynamics o voltaggio autonomo.
  **Conclusa (0f79081)**: indipendenti e shared-heads passano3/3; conditioned
  fallisce la congiunzione. Shared-heads usa440 vs744parametri (-40.86%), senza
  claim di speedup misurato. 10094 osservazioni scalari registrate; artefatti
  verificati in `experiments/results/task23_kaggle_0f79081/`.

- **Task 24 — Famiglia del sodio** → Ricerca di compute condiviso.
  **Conclusa (4fad2c6), NO-GO del controllo indipendente**:
  `experiments/task24_sodium_family_composition.json` e
  `notebooks/24_roadmap_sodium_family_composition.ipynb`. NaTa_t/NaTs2_t/Nap_Et2,
  tutti6gate, controllo indipendente, shared-heads e conditioned; due width,
  tre seed e ladder appaiata. Audit nativo preliminare di tutti3canali, metriche
  per canale e correnti, percorsi25/2500ms e held-V10s. Latenza frozen misurata
  come diagnostica, separata dalla selezione. Nessun V autonomo.
  Shared-heads e conditioned passano3/3 (1516 e420parametri); indipendenti0/3.
  Tutti gli sforamenti sono nel gate h di Nap_Et2 (one-step o held1000/10000ms).
  L'esecuzione è valida; Task25 resta bloccata. Evidenza verificata in
  `experiments/results/task24_kaggle_4fad2c6/`.
  **Task24b preregistrata**: matrice2x2 schedule LR x clipping percanale/bundle,
  width16/32 e budget fino60k, stessi dati/inizializzazione. Selezione development
  con probe held composto (non rollout iterativo), nuovo fresh dopo freeze;
  probe oracle inf/tau eFP32/FP64 non selezionabili. Braccio primario
  `decay_channel` fissato prima del test, soglie invariate; modelli condivisi
  precedenti congelati. `notebooks/24b_sodium_control_diagnosis.ipynb`.
  **Task24b conclusa (bde23cc), GO del riferimento riparato**: sei bracci
  passano3/3; decay_channel width16/60k usa1116parametri. Il conditioned
  congelato420 passa e comprime; shared_heads1516 non riduce parametri.
  LR annealing/budget/capacità migliorano i contrasti tardivi; clipping mai
  attivo. L'errore del vecchio Nap_h è soprattutto nei rate, non risolto dal
  solo solverFP64; held-aware e one-step selezionano gli stessi checkpoint.
  28713 osservazioni collegate; `experiments/results/task24b_kaggle_bde23cc/`.

- **Task 25 — Meccanismi eterogenei** → Combinare cinetiche differenti.
  **Conclusa (bdb75e6), GO primario**: indipendenti e calcio compresso passano
  3/3 seed; sodio compresso ed entrambi compressi2/3. Nel seed43, sul supporto
  activation_boundary, NaTa_t/NaTs2_t m_RMSE superano0.001; correnti, held e path
  passano. Nuovo dominio: non modifica retroattivamente l'esitoTask24b.
  22456 osservazioni e6conclusioni collegate nel mirrorSQLite. Nessun training,
  V imposto; `experiments/results/task25_kaggle_bdb75e6/`.

- **Task 26 — Joint vs independent** → Confronto appaiato tra modelli separati
  e trunk condiviso.
  **Conclusa (2a6bd21), GO**: entrambe le famiglie superano 3/3 seed e tutte
  le soglie fresh/held/path. Selezionati independent width32, 6260 parametri,
  e shared_heads width16, 644 parametri (-89.7%). Il beneficio verificato è
  la compressione del numero di parametri nel dominio V imposto, non uno
  speedup del neurone completo. Nessun target corrente nella loss.
  Risultati in `experiments/results/task26_kaggle_2a6bd21/`.

- **Task 27 — Correnti individuali e totale** → La corrente totale può essere
  un target, ma le componenti individuali restano target ausiliari e probe.
  **Conclusa (c072e69), GO circoscritto**: due architetture x quattro loss
  (none, individual, total, both), tre seed appaiati, budget identico. Tutti
  gli otto bracci superano 3/3 seed e le soglie fresh/held/path. La loss
  congiunta individual+total riduce la mediana del peggior errore di corrente
  del 37,45% nell'architettura indipendente e del 10,88% in quella condivisa
  rispetto a none: entrambe superano la soglia preregistrata del 10%, ma il
  margine condiviso è piccolo. Nel modello condiviso la loss solo individuale
  o solo totale non domina stabilmente la congiunta; mantenere dunque il
  confronto fattoriale, senza dedurre una legge universale. La corrente resta
  formula analitica, i width sono fissati dalla Task26, i checkpoint sono
  congelati prima del fresh e l'audit indipendente verifica 40 hash. Task28 è
  preparabile; Gate D, voltaggio autonomo e closed loop non sono dimostrati.
  Archivio completo: `experiments/results/task27_kaggle_c072e69/artifact_bundle.zip`;
  audit: `experiments/results/task27_kaggle_c072e69/result_audit.json`.

- **Task 28 — Blocco ionico completo teacher-forced** → Tutti i gate e le
  correnti locali con voltaggio ancora fornito dal teacher.
  **Eseguita, GO nello scope teacher-forced**: `experiments/task28_ionic_block_teacher_forced.json`,
  notebook `notebooks/28_roadmap_ionic_block_teacher_forced.ipynb`. Include
  esplicitamente gli 11 meccanismi ionici con gate dell'inventario: cinque
  moduli neurali congelati dalla Task27 e sei moduli a formula canonica,
  verificati separatamente. Gli ingressi (V), (cai), stato iniziale e passo
  sono imposti; nessun CaDynamics, training o generazione autonoma. Quattro
  bracci frozen, tre seed fresh indipendenti, pannelli di corrente firmata e
  percorsi imposti. Il GO di composizione e Gate D restano decisioni diverse:
  senza benchmark di compute appaiato Gate D non è dichiarato completo.
  Esecuzione Kaggle `giada-task28-ionic-block-4c3cba7`, versione 2,
  revisione `4c3cba7`: audit indipendente valido, 144 righe fresh e 48
  percorsi imposti, tutti i quattro bracci frozen passati. Peggior RMSE
  fresh dei gate 0,000659; peggior RMSE normalizzato di corrente individuale
  0,000740; di corrente totale 0,000257. Le sei formule canoniche superano
  il confronto nativo NEURON. Il blocco è ibrido, non un sostituto neurale
  completo: V e cai imposti, nessun CaDynamics, passivo, solve assiale o
  sinapsi. Archivio verificato in
  `experiments/results/task28_kaggle_4c3cba7/artifact_bundle.zip`, audit in
  `experiments/results/task28_kaggle_4c3cba7/result_audit.json`. Nessun
  benchmark appaiato di compute: Gate D resta aperto e Task29 non autorizzata.

- **Gate D compute audit post-Task28 — preregistrato**:
  `experiments/task28b_gate_d_compute.json`, notebook
  `notebooks/28b_gate_d_compute.ipynb`. Confronto formule canoniche complete
  contro blocco ibrido congelato sullo stesso CUDA, medesimi input residenti,
  dtype e output di 18 gate + 11 correnti. Cronometro con eventi CUDA
  alternati e controllo numerico preliminare. Primario: batch 642, riduzione
  mediana appaiata >=10% in **entrambe** le famiglie; batch 4096/16384 solo
  diagnostici. Il benchmark non autorizza inferenze sul throughput end-to-end
  del neurone o su voltage/Ca closed-loop.
  L'esecuzione eager su Tesla T4, commit `59c91e7`, supera la soglia: riduzione
  mediana appaiata 42,57% (independent) e 43,02% (shared_heads) al batch 642,
  con accuratezza intatta. Il risultato è archiviato in
  `experiments/results/gate_d_compute_kaggle_59c91e7/`. Prima della promozione
  definitiva si verifica una baseline analitica compilata simmetricamente:
  `experiments/task28c_gate_d_compiled_confirmation.json`. Questa conferma è
  distinta e motivata dal risultato eager, non una modifica retroattiva alla
  preregistrazione. **Conferma compilata: NO-GO per Gate D** su Tesla T4,
  batch 642, Inductor simmetrico. Le formule esatte impiegano 0,397 ms contro
  0,689 ms dell'ibrido independent; 0,375 ms contro 0,786 ms per shared_heads.
  Il confronto compilato è numericamente equivalente all'eager entro le
  tolleranze registrate. Le flag di accuratezza compilata sono comunque false;
  i RMSE restano bassi e la causa più probabile è il test di occupazione sui
  confini [0,1], ma il report non conserva i conteggi e non permette una
  quantificazione retrospettiva. Il fallimento del compute è già decisivo:
  **Gate D resta incompleto e Task29 non autorizzata dal Gate D originale**. Archivio:
  `experiments/results/gate_d_compiled_kaggle_e66dfea/`.

  **Emendamento prospettico di scope scientifico (Task29)**:
  `experiments/task29_scientific_track_amendment.json` separa la progressione
  scientifica dall'eventuale promozione prestazionale. Il NO-GO compilato del
  Gate D e la soglia originale del 10% restano invariati. Si autorizza soltanto
  una Task29 *diagnostica* con voltaggio/calcio imposti, controllo canonico a
  formule e nessun claim di speedup o voltaggio autonomo. Un'eventuale Task30
  richiederà una decisione separata basata sui risultati della Task29.

### ✅ Gate D — Componibilità ionica

Non si passa al closed loop se:

- i moduli isolati non sono validi;
- il modello congiunto degrada correnti importanti;
- il risparmio di compute non è materiale;
- non è possibile attribuire l'errore ai singoli meccanismi.

---

# 🔁 VI. Compartimento attivo closed-loop

Soltanto qui il modello comincia a generare il proprio voltaggio.

- **Task 29 — Ionico + passivo con clamp esterno**. Autorizzata soltanto come
  studio diagnostico dall'emendamento scientifico; non dal Gate D prestazionale.
  **Eseguita, GO diagnostico nello scope imposto**: quattro percorsi di V/cai,
  cinque pannelli di conduttanza, quattro bracci frozen e tre seed producono
  240/240 confronti validi, di cui 120/120 primari both-arm; nessuna violazione
  di occupazione. Il controllo passivo nativo NEURON concorda esattamente con
  la formula nei sei voltaggi registrati. Peggior RMSE normalizzato di corrente
  di clamp nei both-arm: 0,000373 (independent) e 0,000179 (shared_heads), sotto
  soglia 0,01. Audit e limiti in
  `experiments/results/task29_kaggle_1245b70/result_audit.json` e
  `experiments/task29_external_clamp_result.md`. La conservazione di corrente
  è un'identità del bilancio imposto: non implica voltage rollout autonomo.
  Gate D prestazionale resta NO-GO e Task30 richiede decisione separata.
- **Task 30 — Voltage update autonomo**. Primo microcanary eseguito ma
  **non decision-grade**: la metrica primaria a 8 ms precedeva ogni stimolo
  iniettato (zero campioni attivi). Il report resta archiviato; il pass
  numerico non promuove Task31. Audit in
  `experiments/results/task30_kaggle_fb08789/result_audit.json`.
  **Task 30b — conferma prospettica con esposizione attiva**: condizioni
  disgiunte con stimoli entro 1–8 ms, stessa soglia primaria e checkpoint
  congelati. Protocollo `experiments/task30b_active_exposure_confirmation.json`.
  Conferma completata e verificata: 24/24 episodi non-rest esposti prima di
  8 ms, sei crossing attivi di riferimento, peggior RMSE per episodio nel
  both-arm 0,0097 mV (limite 2 mV). GO scientifico **solo** per la mappa
  autonoma interna a formule esatte del singolo compartimento. Il riferimento
  attivo NEURON completo non è stato validato; calcio e ingresso sono imposti.
  Task31 non è automaticamente autorizzata; Gate D speed resta NO-GO.
  Dettagli in `experiments/task30b_active_exposure_result.md`.
  **Task 30c — conferma nativa attiva preregistrata**: stesso disegno 30b,
  undici meccanismi NMODL originali e checkpoint congelati. Prima si misura
  il floor formula–NEURON; solo se passa si giudica il modello. Il calcio
  resta imposto e Gate D speed resta NO-GO. Contratto in
  `experiments/task30c_native_active_confirmation.json`.
  Conferma completata: floor formula–NEURON a 8 ms 5,41e-13 mV pooled,
  tutti i sei bracci `both` famiglia × seed sotto 1/2 mV pooled/peggior
  episodio, sei crossing nativi; audit indipendente valido. GO scientifico
  soltanto per un compartimento attivo con cai imposto; non per il neurone
  multicompartimentale, CaDynamics o la velocità. Task31 ancora da
  autorizzare separatamente. Dettagli in
  `experiments/task30c_native_active_result.md`.
- **Task 31 — Scheduled sampling diagnostico**. Preflight retrospettivo sui
  checkpoint congelati 30b/30c: il feedback sul gate è misurabile, ma l'errore
  assoluto di V resta molto basso fino a 80 ms (peggior episodio nativo
  0,0271 mV). Nessun nuovo training/test o selezione: continuation con
  scheduled sampling rimandato finché un dominio indipendente non mostri un
  collo di bottiglia di esposizione. Rapporto e limiti in
  `experiments/task31_exposure_diagnostic_result.md`. Task 32 resta una nuova
  decisione causale, non una promozione automatica; Gate D speed NO-GO.
- **Task 32 — Dinamiche lente in feedback**. L'avvio del feedback elettrico resta
  subordinato ai prerequisiti IV-B1 e IV-B2 del piano originale. Protocollo
  appaiato v1 in `experiments/iv_b1_b2_calcium_prerequisite.json`:
  CaDynamics_E2 isolato passa 48/48, ma la composizione SK non passa perché
  il riferimento leggeva cai vecchio. L'audit post-hoc delle tracce native
  identifica l'ordine causale effettivo: CaDynamics aggiorna cai, poi SK usa
  quel valore nello stesso passo. La v1 resta non promossa. Conferma v2
  preregistrata con due protocolli nuovi e soglie invariate in
  `experiments/iv_b1_b2_calcium_prerequisite_v2.json`. La conferma prospettica
  Kaggle su revisione `8b3a4e5` passa 72/72 IV-B1 e 12/12 IV-B2, inclusi
  `late_single` e `triplet`; ZIP, tracce e metriche sono stati controllati
  indipendentemente in `experiments/results/iv_b_v2_kaggle_8b3a4e5/`.
  È quindi autorizzata la preparazione del feedback elettrico della Task32,
  **non** dichiarata conclusa la Task32: Ca e SK avevano ancora ica e V imposti.
  Il protocollo Task32 v1 ora preregistrato in
  `experiments/task32_dynamic_calcium_feedback_preregistration.json` combina
  gli undici canali, CaDynamics e SK nel singolo compartimento nativo con V e
  cai autonomi. Separa calibrazione del timing della sorgente di calcio e
  conferma disgiunta; giudica i checkpoint congelati solo dopo un floor
  formula–NEURON ammissibile. È un nuovo run, non un esito già acquisito.
  Il primo run `f8c5d25` ha correttamente bloccato il giudizio del modello:
  NEURON ricalcolava automaticamente `eca` mentre il riferimento lo teneva
  fisso, producendo un floor 40 ms non ammissibile. Audit e tracce sono in
  `experiments/results/task32_kaggle_f8c5d25/`; diagnosi in
  `experiments/task32_first_run_diagnosis.md`. La v2 preregistrata in
  `experiments/task32_dynamic_calcium_feedback_v2.json` forza e misura `eca`
  fisso senza cambiare checkpoint o soglie. Nessun NO-GO del modello è stato
  dichiarato dal primo run.
  La v2 `e22b669` verifica `eca` fisso e un floor di voltaggio molto basso,
  ma non supera il gate preregistrato del calcio per una differenza di fase
  fra gli stati formula e nativi; i checkpoint non sono giudicati neppure lì.
  L'audit locale dei gate porta alla v3 prospettica in
  `experiments/task32_dynamic_calcium_feedback_v3.json`: sorgente del calcio
  dai gate aggiornati **della mappa formula**, senza usare uno stato futuro del
  teacher. Evidenza e limiti in `experiments/task32_v2_timing_diagnosis.md`.
  La conferma v3 a revisione `05bdabb` supera il floor nativo e tutti i sei
  checkpoint congelati sul gate primario a 40 ms. Il controllo con calcio
  congelato mostra che il feedback Ca–SK–V è effettivamente attivo. Due tentativi
  precedenti della v3 erano interruzioni tecniche, non NO-GO dei modelli; la
  valutazione GPU è stata isolata dal processo NEURON senza modificare soglie
  o checkpoint. Report, archivio e limiti scientifici in
  `experiments/task32_v3_result.md`. Task32 è conclusa **solo nel dominio
  monocopartimentale preregistrato**: sinapsi, accoppiamento assiale e
  morfologia restano aperti per le task successive.
- **Task 33 — Sinapsi con ingressi completamente osservabili**. Ingresso
  subordinato a IV-C3, che richiede IV-C1 e IV-C2. IV-C1 è preregistrato in
  `experiments/iv_c1_deterministic_synapse_preregistration.json`: confronto
  analitico/nativo dei kernel AMPA, NMDA, GABAA e GABAB con un evento per
  sinapsi, V imposto, calibrazione del timing separata da conferma e controlli
  negativi. La preregistrazione non equivale a completamento del prerequisito.
  Il run v1 `5920b73` ha conduttanze/correnti/cariche esatte al rumore numerico,
  ma fallisce il gate sugli stati A/B. La spiegazione iniziale di una
  convenzione pre-jump uniforme era un'ipotesi, non un fatto acquisito.
  Nessuna promozione retroattiva. La v2 preregistrata in
  `experiments/iv_c1_deterministic_synapse_v2.json` ha usato schedule nuove
  con soglie invariate, ma ha fallito ancora il gate A/B in tutte le 12 celle;
  conduttanze, correnti, cariche e controlli restano corretti. Vedi
  `experiments/iv_c1_v1_state_phase_diagnosis.md` e
  `experiments/iv_c1_v2_result.md`. L'audit per evento e per sinapsi ha
  identificato il vero discrimine: il tempo effettivo `h.t` può trovarsi appena
  prima o dopo il timestamp dell'evento, benché la griglia nominale mostri
  lo stesso tempo. La conferma v3 preregistrata su schedule disgiunte ha
  superato tutte le 12 celle, i gate A/B, conduttanza, corrente, carica e i
  controlli negativi. **IV-C1 è conclusa soltanto nel suo dominio isolato e
  deterministico**; evidenza in `experiments/iv_c1_event_clock_forensic_result.md`
  e `experiments/iv_c1_v3_result.md`. IV-C2 ha poi superato il replay
  nativo appaiato di plasticità/RNG/ripristino in 64 celle e la conferma
  prospettica delle frequenze native di recupero/rilascio in 16 celle con
  256 nuovi seed per cella. Il perimetro resta quello di sinapsi isolate a
  voltaggio imposto; dettagli e limiti in
  `experiments/iv_c2_v1_result_and_v2_rationale.md`. IV-C3 ha superato
  prospetticamente 36/36 celle miste, i tre controlli negativi, la
  ricostruzione causale online delle correnti e il restart nativo delle
  quattro sinapsi. Il risultato è documentato in
  `experiments/iv_c3_integrated_synaptic_interface_result.md`.
  IV-C3 da sola dimostrava solo l'interfaccia isolata a voltaggio imposto.
  Task33 v1 si è fermata alla calibrazione per la fase degli eventi fuori
  griglia; la v2 ha verificato le sinapsi in 16 casi ma non il floor del
  calcio, senza giudicare i modelli. La v3 prospettica (`21e3c35`) ha
  corretto la fase della conduttanza nel solver, mantenuto soglie e checkpoint
  congelati e usato seed/schedule nuovi: floor formula–NEURON, 16 casi,
  controlli e tutti e sei i checkpoint hanno superato i gate.
  **Task33 è conclusa solo nel singolo compartimento attivo con quattro
  sinapsi completamente osservate**. Vedi `experiments/task33_v3_result.md`;
  non implica ancora accoppiamento assiale, cellula completa o speedup.
- **Task 34 — Correnti e stati come probe privilegiati**. Audit v1/v2/v3
  sui 16 casi Task33 già aperti, senza retraining o selezione. La v1 ha
  rivelato un errore di fase nel braccio oracle degli stati teacher; la v2
  lo ha corretto prospetticamente. La v3 ha separato modello–formula da
  formula–stato nativo sotto lo stesso V imposto: il baseline Task33 si
  riproduce esattamente, la scomposizione delle correnti chiude e gli
  errori grezzi maggiori rispetto al nativo non sono prevalentemente
  difetti dei gate appresi. Task34 è **diagnosticamente completa** ma non è
  un test indipendente. Dettagli in `experiments/task34_v3_diagnostic_result.md`.
- **Task 35 — Full compartment surrogate (compartimento locale dichiarato)**.
  Contratto prospettico in `experiments/task35_full_local_compartment_preregistration.json`:
  32 nuovi casi appaiati nei pannelli canonico, calcio×4, sodio×4 e potassio×4.
  Riusa senza riaddestramento i sei checkpoint Task32 e l'interfaccia causale
  Task33, ma varia le conduttanze per distinguere la conferma locale dalla
  sola ripetizione del regime canonico. Gate per pannello, floor formula–native,
  shadow sinaptico e controlli negativi sono obbligatori. Non comprende
  accoppiamento assiale, morfologia, cellula intera o speedup: questi non sono
  implicati dal nome della task e restano fuori dalla rivendicazione.
  **Esito v1 confermato** su 32 casi indipendenti e tutti i sei checkpoint,
  con gate per pannello e nessun riaddestramento. Vedi
  `experiments/task35_full_local_compartment_result.md`. Il prossimo salto
  causale strutturale del piano precedente era l'accoppiamento assiale
  dell'exTask36, ora Task72. La priorità corrente è la scala di astrazione
  ionica Task36–71, che include propri test di trasferimento assiale.

### 🔬 Matrice causale obbligatoria

| Voltaggio | Stato | Significato |
|---|---|---|
| teacher | surrogate | isola l'errore dello STATE updater |
| surrogate | teacher | isola l'errore del voltage updater |
| surrogate | surrogate | misura l'interazione closed-loop |
| formula originale | voltage updater appreso | controllo meccanicistico |

---

# ⚡ VI-bis. Efficienza ionica appresa — percorso IE conservato

> **Archivio di direzione, aggiornato il 17 ottobre 2026:** il percorso IE
> documenta le ipotesi e i contrasti precedenti. La sequenza attiva è ora
> Task36–71 in VI-ter. Le attività IE non eseguite non diventano concluse:
> i loro quesiti pertinenti confluiscono nei nuovi contratti, senza rinominare
> i notebook IE-P1…P7 o riutilizzarne i test aperti come conferme fresh.

> **Obiettivo:** sostituire una parte sostanziale della computazione ionica
> con un modulo appreso più economico, a fedeltà dichiarata, e verificarne il
> beneficio nel compartimento Task35. Non massimizzare la percentuale di rete.
>
> **Numerazione storica:** IE significa *efficienza ionica*. IE-01…IE-16 sono
> gli ID del percorso precedente; non riutilizzano gli ID originali né CC. Una sottotask futura
> avrà un suffisso esplicito e una propria ipotesi, senza cambiare i parent.
>
> **Stato:** esistono run IE, descritti nel registro di avanzamento in fondo.
> Questo aggiornamento modifica la pianificazione, non esegue esperimenti
> e non promuove candidati. La chiusura documentale dei risultati più recenti
> va verificata nella Task36 prima di riusarli come baseline.

## 📚 A. Evidenze acquisite e bersaglio computazionale

- Task35 conferma il compartimento locale ibrido nei 32 casi registrati;
  peggior V pooled per modello/pannello 0,010606 mV. Non certifica speedup.
- [Task28d](results/gate_d_forensic_kaggle_b19aa65/result_audit.json): il core
  dei cinque canali appresi costa più delle cinque formule che sostituisce.
  La presenza dei sei canali analitici non spiega da sola il rallentamento.
- [CC-N1](cc_n1_result_v1.md): a batch642 per seed, tre seed, T4/FP32 e
  compilazione simmetrica, core formula 0,423835 ms contro 0,807482/0,904102 ms
  per i due ibridi. Lo stato resta V/Ca/18 gate. Il benchmark esclude la
  generazione host di eventi/plasticità/RNG: non è il tempo totale del simulatore.
- CC-N1 dimostra collisioni delle viste V/Ca nel dominio controllato. Non
  identifica la memoria latente minima e non prova impossibilità universale.
- CC-N2 non promuove neppure il controllo nuovo a 18 gate nelle soglie CC:
  il fallimento non può essere attribuito soltanto alla riduzione dello stato.
  Audit IE-01 eseguito: [esito](cc_n2_result_v1.md), con limiti del confronto.

- **Task IE-01 — Riconciliazione delle evidenze** → Acquisire report, curve,
  checkpoint e provenance N2; completare audit e registrazione nel mirror
  SQLite. Collegare Task28d, Task35 e CC-N1/N2, distinguendo risultati,
  ipotesi, incidenti e domini. Le soglie Task35 e CC sono diverse e non vanno
  scambiate. I dati già aperti restano development o benchmark storici.

- **Task IE-02 — Profilo causale del costo** → Misurare separatamente e nella
  composizione: valutazione dei tassi, update dei gate, aperture/conduttanze,
  correnti, Ca/SK, solver, sinapsi e gestione eventi. Usare sostituzioni
  diagnostiche a output precomputato solo per stimare margini, mai come
  candidati causali. Rilevare lanci, memoria, sincronizzazioni e operazioni.
  Le latenze isolate non sono additive per effetto della fusione del codice.

- **Task IE-03 — Bilancio di accelerazione e scelta del bersaglio** → Fissare
  workload principale, batch secondari, precisione e margini di qualità.
  Stimare il massimo beneficio globale ottenibile accelerando il blocco
  scelto; distinguere costo per passo e per millisecondo biologico. Se il
  margine è trascurabile, cambiare bersaglio prima di investire nel training.
  Soglia minima di guadagno e incertezza si preregistrano prima della ricerca;
  il vecchio Gate D/10% resta storico e non viene reinterpretato.

### ✅ Gate IE-A — Un bersaglio che valga il costo della ricerca

Servono baseline riproducibili, un blocco con margine utile, contratto di
qualità e un confronto contro **formule GPU competitive**, oltre all'ibrido.
Un miglioramento rispetto al solo ibrido lento non dimostra una soluzione
vantaggiosa rispetto alle formule. Nessuna promessa di accelerazione massiccia.

## 🔌 B. Contratto del blocco e playground isolato

- **Task IE-04 — Porta ionica e ordine causale** → Congelare stato, ingressi,
  parametri visibili, unità e fasi pre/post-step. Il modulo deve fornire le
  conduttanze richieste dal solver e la corrente Ca nella fase appropriata;
  una sola corrente netta non è sufficiente. Conservare il bilancio
  `I_k = g_k (V - E_k)` e l'ordine già verificato Ca → SK. Se un primo studio
  lascia SK canonico per isolare la dipendenza da Ca, dichiararlo: non è
  ancora la sostituzione completa degli undici canali. Variare E o parametri
  esclusi dal training richiede un dominio esplicito, non un claim implicito.

- **Task IE-05 — Playground ionico appaiato** → V/Ca imposti, stessi path e
  condizioni iniziali per tutti i bracci; target nativi e formula verificati
  nel dominio. Coprire transitori rapidi, memoria lenta, recupero, stati fuori
  equilibrio, correnti piccole/grandi e blocchi di canale. Split per path e
  famiglia, non per punti vicini. Verificare supporto di ciascuna metrica.

Il primo passo resta **0,1 ms**, coerente con Task35. Le vecchie conclusioni
sull'insufficienza del solo V a passi maggiori non vengono cancellate.
Solver, sinapsi e legge Ca rimangono congelati; l'uscita ionica alimenta Ca,
quindi l'effetto sul suo feedback sarà comunque verificato in composizione.

## 🧮 C. Ridurre il calcolo prima di ridurre la memoria

- **Task IE-06 — Riferimenti e condivisione a stato esplicito** → Confrontare
  formule, ibrido congelato, approssimatori numerici pertinenti (LUT/spline o
  polinomi) e rete congiunta dei tassi. Riutilizzare il lavoro già svolto sui
  gate; evitare di ripetere task atomiche concluse senza un nuovo contrasto.
  Tutti conservano lo stesso stato e la stessa interfaccia.

- **Task IE-07 — Operatore discreto congiunto economico** → Confrontare il
  riferimento a tassi con coefficienti di transizione appresi direttamente:
  `x_next = r_theta * x + (1-r_theta) * u_theta`, con coefficienti in [0,1].
  È un'estensione controllata del precedente direct-z, non un'idea mai testata.
  Stimare anche il costo delle attivazioni e delle letture/scritture; non
  assumere che rimuovere exp(-dt/tau) renda automaticamente veloce la rete.
  Il vincolo convesso preserva l'intervallo dei gate, non garantisce da solo
  la stabilità dell'intero compartimento.

- **Task IE-08 — Learnability e frontiera costo–errore** → Piccola matrice
  preregistrata di capacità/condivisione e supervisione (transizioni contro
  transizioni più target interni), con mini scaling dati/budget e checkpoint
  crescenti. Inizializzazioni, seed e stream appaiati ove compatibili.
  Misurare gate, conduttanze, correnti, carica e rollout forzati per regime.
  Mantenere separati pesi, stato persistente, temporanei, operazioni e latenza.
  Individuare quali termini producono l'errore, senza compensazioni nascoste
  fra correnti entranti e uscenti.

### ✅ Gate IE-B — Primo risultato locale concreto

Promozione prestazionale solo con fedeltà e risparmio misurato sul blocco,
replicati nei seed/workload preregistrati. Un braccio accurato ma lento può
restare un controllo scientifico; non raggiunge questa milestone.
Se i bracci espliciti falliscono, diagnosticare dati/ottimizzazione/capacità
prima di attribuire il limite allo stato. Un ramo latente può essere studiato
come alternativa motivata se il costo dello stato esplicito non offre margine;
non richiede che l'esplicito diventi vincente, ma richiede un controllo valido.

## 🧠 D. Compressione della memoria ionica, quando giustificata

- **Task IE-09 — Memoria sufficiente e inizializzazione** → Riutilizzare le
  collisioni CC-N1 e i probe N2. Separare encoder di stato usato una volta,
  warmup causale e stato standard; non dichiarare equivalenti informazioni
  diverse. Precisare l'inizializzazione disponibile nel simulatore reale.

- **Task IE-10 — Operatore ionico ricorrente compatto** → Poche dimensioni di
  z e un readout delle conduttanze per canale. Confrontare con il controllo
  esplicito allo stesso passo; regolare capacità separatamente da dimensione
  dello stato. Solo dopo testare gruppi di canali, mantenendo la possibilità
  di applicare i parametri e gli interventi previsti. I gruppi non possono
  nascondere la perdita di identificabilità dei singoli canali.

- **Task IE-11 — Distillazione, uso della memoria e recupero del limite** →
  Supervisione interna solo nel training; probe dei gate, reset/permutazione
  dello stato e orizzonti ricorsivi appaiati. Mini scaling per distinguere
  informazione mancante, rappresentazione e training insufficiente.
  La perdita di informazione causale non è recuperabile dal solo scaling.

Queste task si attivano se la compressione offre una specifica opportunità
di risparmio o risponde a un limite osservato. Non è obbligatorio ridurre z
per promuovere un aggiornamento esplicito già accurato e conveniente.

## 🔁 E. Reinserimento nel compartimento congelato

- **Task IE-12 — Sostituzione causale e attribuzione** → Inserire il candidato
  nel compartimento Task35 con solver/sinapsi/Ca invariati. Confrontare V/Ca
  imposti e autonomi, gruppi ionici sostituiti separatamente e insieme, con
  controlli di fase. Misurare dove nasce un'eventuale amplificazione degli
  errori. Un candidato locale fallito qui non è promosso come compartimento.

- **Task IE-13 — Fedeltà dinamica e interventi** → Nuovi stimoli di sviluppo,
  conduttanze/blocchi, recupero e orizzonti lunghi; timing/conteggio spike,
  adattamento e refrattarietà soltanto dove il teacher e il supporto dati
  li rendono valutabili. Distinguere errore numerico, qualità delle risposte
  e sensibilità agli interventi. Conservare le correnti Ca e il feedback SK.

- **Task IE-14 — Utilità integrata e freeze** → Misurare costo del blocco e
  dell'intero compartimento, includendo eventi, trasferimenti, stato e RNG
  nel perimetro operativo dichiarato. Confrontare formule e ibridi con pari
  ottimizzazione; latenza/throughput a batch rappresentativi, riscaldamento,
  sincronizzazione e ripetizioni. Nessun timing in contesa sulla stessa GPU.
  Congelare candidato, dominio, inizializzazione, codice e criteri di conferma.

- **Task IE-15 — Conferma indipendente congiunta** → Nuovi seed/path/domìni
  riservati dopo freeze; verificare qualità e vantaggio computazionale con
  intervalli d'incertezza e casi peggiori. Nessun aggiustamento sui risultati
  del test; una riparazione richiede nuova conferma. Gli stress estremi fuori
  contratto sono riportati separatamente, non esclusi dopo aver visto l'esito.

- **Task IE-16 — Decisione e rientro** → Registrare evidenze e limiti nel
  mirror SQLite. Nel piano IE, qualità e utilità avrebbero consentito di
  proporre il rientro all'exTask36 (ora Task72)
  con il modulo e la porta assiale verificati. Se emerge un altro collo di
  bottiglia dominante, motivare un nuovo blocco di lavoro prima di ampliarlo.
  Un NO-GO circoscritto conserva il riferimento Task35; nessuna promozione
  per obbligo di calendario e nessun ritorno automatico alla ricerca monolitica.

### ✅ Gate IE-C — Modulo utile nel sistema

Occorrono insieme fedeltà del compartimento, robustezza nel dominio e un
vantaggio materiale nel workload prefissato. Le soglie originali restano
immutate per i vecchi risultati; i nuovi margini di non inferiorità e il
target di speedup saranno fissati prima dei run, senza scegliere a posteriori
fra i limiti più permissivi di Task35 e quelli CC.

## 🧭 F. Estensioni condizionali, non nuovi obblighi seriali

- **Sinapsi:** aprire una compressione dedicata solo se costo/memoria lo
  giustificano; preservare eventi, plasticità, stato RNG e contratti pathwise
  o distribuzionali. Prima isolare, poi reinserire e confermare.
- **Solver:** partire da profilo e controllo numerico. Sostituirlo solo per
  un risparmio credibile, con bilancio, stabilità e floor indipendente.
  Una formula economica può rimanere nella soluzione finale.
- **Tempo:** passi maggiori richiedono un proprio esperimento di sufficienza
  degli ingressi e convergenza; non sono impliciti nella compressione ionica.
- **Coarse-graining del compartimento:** riaprire CC o un nuovo emendamento
  identificato soltanto con un limite preciso e componenti già comprese.
  Evitare sia di abbandonarlo per principio sia di riprenderlo automaticamente.
- **Hardware:** GPU corrente e compilazione simmetrica già nei benchmark;
  kernel custom, precisione ridotta e acceleratori specifici dopo un profilo
  che motivi l'intervento, con validazione numerica separata. Tenstorrent differito.

## 🧬 G. OpenEvolve — ricerca assistita futura

L'utente riferisce un'installazione di OpenEvolve sul proprio droplet
DigitalOcean. **Disponibilità dichiarata, non verificata in questo aggiornamento.**
Possibile collegamento futuro a RunPod tramite strumenti/MCP da verificare;
nessuna connessione, installazione, allocazione o spesa viene avviata qui.

L'ingresso richiede IE-02…05: bersaglio misurato, interfaccia congelata,
valutatore numerico e costo di valutazione sostenibile. Non occorre aspettare
IE-16 se questi prerequisiti sono soddisfatti. Primo pilota piccolo con budget
di tentativi, token, GPU e tempo prefissati; confronto con candidati manuali.

Evolvere solo il modulo ammesso, non teacher, dati, metriche o soglie.
Valutazione a stadi: validità → fedeltà → rollout → costo. Criteri fisici come
vincoli obbligatori; niente giudizio LLM nel punteggio scientifico. Test finale
separato dalla ricerca, esecuzione isolata, codice/checkpoint/traiettoria delle
modifiche conservati. Ogni candidato neurale richiede il suo training equo:
la ricerca evolutiva non ne annulla il costo. Una scoperta promettente richiede
ablazioni causali per capire quale modifica aiuta. Il database di programmi
OpenEvolve non sostituisce la memoria scientifica SQLite.

Riferimenti: [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve),
[valutatore](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/evaluator.py).
I risultati del progetto esterno non sono promesse di speedup per GIADA.

## ⚡ H. Organizzazione dei run e criteri di arresto

IE-01…05 condividono audit/artefatti; IE-06…08 possono essere una piccola
matrice appaiata; IE-09…11 una seconda solo se motivata. IE-12…14 possono
condividere checkpoint e dati di sviluppo, ma il freeze precede sempre IE-15.
IE-16 è una decisione. Nessuna corrispondenza obbligatoria task = notebook.

Parallelizzare contrasti indipendenti e salvare checkpoint intermedi; non
eseguire combinazioni senza un'ipotesi. Vettorizzazione verificata per Adam,
gradienti e clipping; processi NEURON/CUDA isolati come nei run precedenti.
Preregistrare budget e regola di stop per ciascun run. Una saturazione porta
a diagnosi e decisione esplicita, non a un numero indefinito di tentativi.
Fallimento tecnico, NO-GO scientifico e NO-GO prestazionale sono distinti.

---

# 🪜 VI-ter. Sei blocchi di astrazione ionica progressiva — Task 36–71

> **Stato:** Task36 conclusa come audit di contratto, con
> [esito e limiti](task36_result_v1.md). Task37 conclusa nel playground isolato
> ([risultato](task37_result_v1.md)); Task38 conclusa con NO-GO ricorrente
> ([risultato](task38_result_v1.md)). Si prepara ora Task39; Task40–71 restano
> pianificate e vincolate alla qualità. Ogni run successivo
> richiede il proprio contratto e i prerequisiti indicati. Una task può
> condividere un notebook con altre quando non dipende dal loro risultato.

## 🎯 Bersaglio e tre gradini di astrazione

Il bersaglio comprende **dinamica ionica, memoria e produzione delle correnti
individuali** degli undici canali del compartimento. L'ipotesi è che una
transizione appresa con calcolo condiviso e, gradualmente, memoria compressa
possa essere più economica della composizione meccanicistica. Il beneficio
va misurato: più astrazione, meno parametri o meno formule non garantiscono
da soli una latenza inferiore. Dati e training possono ridurre l'errore di
approssimazione; non ricostruiscono informazione causale eliminata dagli input
o da uno stato insufficiente.

| Blocchi | Task | Livello | Stato persistente del candidato | Computazione da apprendere |
|---|---|---|---|---|
| 1 → 2 | 36–41 → 42–47 | 1 — transizione congiunta esplicita | Gate fisici conservati | Aggiornamento discreto congiunto, oltre le sole curve dei rate |
| 3 → 4 | 48–53 → 54–59 | 2 — memoria per famiglie | Piccoli stati latenti per gruppi motivati | Dinamiche e aperture efficaci senza ricostruire tutti i gate in inferenza |
| 5 → 6 | 60–65 → 66–71 | 3 — memoria ionica condivisa | Un unico stato ricorrente locale, dimensione da verificare | Intero operatore ionico, senza imporre la separazione dinamica per famiglia |

In ogni coppia, il primo blocco studia **un compartimento**; il secondo
trasferisce **lo stesso livello di astrazione** a compartimenti realmente
accoppiati. I pesi possono essere condivisi, ma ogni compartimento conserva
il proprio stato e i propri parametri. Un batch di compartimenti indipendenti
non costituisce validazione multicompartmentale. Le tre verifiche spaziali
riusano geometrie e protocolli comparabili, senza aumentare contemporaneamente
astrazione, passo temporale e scala morfologica.

## 🔒 Contratto comune e criteri per salire di livello

- **Confine fisico:** solver di membrana e assiale, passivo, CaDynamics,
  sinapsi e RNG restano quelli verificati. Cambiano soltanto il modulo ionico
  e il suo adattatore esplicitamente conteggiato. Ca è un ingresso reale;
  il feedback delle correnti Ca viene testato attraverso CaDynamics invariato.
  La fase Ca → SK e la corrente Ca richiesta prima dell'update devono restare
  implementabili causalmente, anche se servono due chiamate alla rete.
- **Porta compatibile:** fornire correnti per canale e conduttanze/coefficienti
  richiesti dal solver semi-implicito, nella sua convenzione temporale.
  Predire solo I totale non basta. L'eventuale readout diretto non può
  sostituire di nascosto il solver con un integratore esplicito.
- **Tempo e informazione:** partenza a dt = 0,1 ms e precisione di riferimento
  comune. V, Ca, parametri ammessi e stato precedente sono disponibili online;
  nessun V futuro, reset periodico al teacher o stato privilegiato nel rollout.
  Encoder iniziale e warmup vanno dichiarati, valutati e inclusi nel costo.
- **Bias da conservare e mettere alla prova:** causalità, memoria multiscala,
  correnti individuali, maschere di presenza, blocco esatto con gbar = 0 e
  risposta ai potenziali di inversione. Readout primario economico:
  `g_k = gbar_k * a_k`, `I_k = g_k * (V - E_k)`, con aperture ammissibili.
  Le potenze dei gate vengono eliminate dai candidati latenti quando il
  readout impara direttamente a_k. Il readout di corrente diretto è una
  ablation controllata, non un obbligo di cancellare algebra economica.
- **Fedeltà causale:** oltre a RMSE, correnti piccole/grandi e carica integrata,
  misurare risposte a storie diverse con stesso V, recupero, inattivazione,
  adattamento, blocchi e variazioni parametriche. Valutare V/Ca, spike e
  sensibilità degli esiti agli interventi solo nei regimi dove il teacher li
  manifesta e il supporto è sufficiente. Somiglianza di forma e numero di
  spike non certificano da soli la stessa dinamica. L'equivalenza resta
  relativa al dominio e agli interventi testati, mai a tutti i comportamenti.
- **Controlli e costo:** teacher nativo verificato, formule GPU competitive,
  LUT pertinente, riferimento Task35 e miglior livello precedente congelato.
  Stessi dati/seed e budget appaiati ove compatibili; compilazione simmetrica.
  Riportare latenza ionica, core integrato e wall-clock operativo separatamente,
  memoria dello stato/pesi/temporanei e costo per ms biologico. Non attribuire
  al blocco ionico un costo host/RNG/IPC né spacciare un risparmio locale per
  speedup dell'intero neurone. Niente kernel custom necessario per dimostrare
  il primo vantaggio architetturale; le baseline restano ottimizzate equamente.
- **Decisioni:** soglie, margini di non inferiorità, target di risparmio,
  workload primario, seed, budget e regole di stop si congelano prima del run.
  Non si allentano i vecchi gate. Dopo sviluppo: freeze e nuovi test disgiunti.
  Un test già aperto può informare la fase successiva solo come evidenza
  dichiarata di sviluppo; la conferma successiva richiede un nuovo insieme.
  Separare errore tecnico, NO-GO scientifico e NO-GO di velocità. Un livello
  accurato ma lento può essere il controllo del successivo, con decisione
  motivata sul risparmio atteso; non è una promozione prestazionale. Un livello
  inaccurato richiede diagnosi prima di aggiungere un'altra astrazione.

## 🔍 Valutazione dei regimi rari e dei futuri divergenti — requisito trasversale

**Precisazione dell'utente, 17 ottobre 2026:** aumentando l'astrazione può
crescere il fabbisogno di dati e di training, soprattutto per regimi rari e
vicino alle transizioni dinamiche. Un errore medio basso, anche su un insieme
etichettato OOD, non certifica fedeltà delle risposte o dei rollout. Un
fallimento locale non dimostra da solo che l'astrazione sia sbagliata; un
successo medio non autorizza a ignorarlo. Questo requisito vale per tutti
e sei i blocchi, dalla Task36 alla Task71.

### 📚 Recuperare i valutatori storici senza trasferirli alla cieca

Nella repository esistono
[`RapidFireEventDetector` e `RapidFireSomaEvaluator`](../src/neuronio/neuronio_rapid_fire_eval_utils.py)
e [`NeuronioEvaluator`](../src/neuronio/neuronio_eval_utils.py). Il modulo
rapid-fire misura copertura degli eventi, errori sui picchi, code della
distribuzione e metriche sulle finestre rapide, con visualizzazione degli
eventi peggiori. È un precedente utile, non una validazione delle nuove correnti.
Il detector usa quantili e indici di campionamento: conversione a tempo fisico,
normalizzazioni, soglie, smoothing e definizione dell'evento vanno verificati
prima del riuso a dt = 0,1 ms. Un transiente rapido rilevato da quel codice
non equivale automaticamente a una sequenza di spike ad alta frequenza.
Servono valutatori ionici specifici e test autoregressivi dedicati.

### 🧪 Distinguere copertura, sforzo di training e stato insufficiente

- Registrare per regime il numero di traiettorie indipendenti, eventi,
  durate, combinazioni di parametri e campioni effettivamente visti durante
  il training. Ripetere un evento non crea nuova diversità; molti punti
  adiacenti non sono molte osservazioni indipendenti.
- In caso di fallimento raro, usare un contrasto mirato **copertura × budget**:
  dati correnti contro nuove traiettorie del regime, budget corrente contro
  maggiore esposizione di training. A parità di architettura e test disgiunto,
  separare l'effetto di nuova informazione da quello di più aggiornamenti.
  Riportare campioni visti e compute effettivo, non soltanto epoche.
- Distinguere inoltre sottopeso nella loss, campionamento sbilanciato,
  ottimizzazione e capacità con ablazioni mirate. Sovracampionamento e loss
  per regime vanno confrontati anche sui regimi comuni per rilevare regressioni.
  Gli interventi sono scelti sul development, mai sul test finale.
- Curve di errore per regime e checkpoint a budget crescenti guidano la
  diagnosi. Un miglioramento con dati/compute supporta quella spiegazione
  nel dominio osservato, non prova che lo scaling risolverà ogni residuo.
  Se storie fuse nello stesso stato richiedono futuri differenti, aumentare
  soltanto il budget non elimina il problema informativo.

### 📏 Metriche numeriche, comportamento e revisione qualitativa

- **Regimi separati:** dati rappresentativi della distribuzione attesa e
  suite di stress stratificata, riportati separatamente. Oltre alla media,
  errori per canale/regime, quantili di coda, casi peggiori, frequenza dei
  fallimenti e incertezza per traiettoria/seed. Soglie obbligatorie per regimi
  critici impediscono che lunghi periodi quiescenti nascondano errori rari.
  Copertura insufficiente significa «non valutabile», non «passato».
- **Correnti:** segno, ampiezza e timing dei picchi, onset/offset, carica
  integrata firmata e assoluta, inattivazione e recupero; analizzare anche
  le componenti che si cancellano nella somma. Normalizzazioni e scale
  assolute vanno dichiarate, evitando divisioni instabili vicino a corrente zero.
- **Risposte causali:** coppie con storia diversa e stesso V, impulsi
  ravvicinati e treni rapidi, recupero dopo pausa, variazioni di Ca/gbar/E
  e blocchi selettivi. Confrontare la variazione di risposta prodotta dallo
  stesso intervento, non soltanto due tracce di base somiglianti.
- **Transizioni e biforcazioni dinamiche:** scandire ampiezza/frequenza degli
  stimoli e parametri intorno ai cambi di regime osservati nel teacher;
  misurare spostamenti delle soglie, ritardi, isteresi e regime raggiunto
  quando presenti. Distinguere queste biforcazioni dalle ramificazioni
  morfologiche. Prima verificare convergenza del riferimento e sensibilità
  alle perturbazioni: la divergenza di traiettorie va interpretata rispetto
  al floor numerico e alla dinamica del teacher, non automaticamente imputata
  al surrogate né giustificata senza misura.
- **Autoregressione reale:** orizzonti crescenti, senza reset o correzioni
  teacher; errore e drift nel tempo, tempo al superamento della tolleranza,
  recupero e cambi di regime. In composizione includere spike mancanti/spuri,
  timing, intervalli inter-spike e struttura dei burst dove supportati.
- **Revisione qualitativa riproducibile:** pannelli allineati di correnti,
  V/Ca e stati/probe, diagrammi di fase e mappe stimolo–risposta; selezione
  prefissata di casi tipici, casuali, rari e peggiori, con scale comuni.
  Una checklist descrive anomalie come burst troncato, recupero errato,
  inversione spuria o deriva. L'ispezione visiva integra i gate quantitativi,
  non li sostituisce e non consente di scegliere soltanto figure favorevoli.
- **Testare i valutatori:** controlli deliberatamente difettosi, per esempio
  ritardo della corrente, picchi attenuati, memoria azzerata o drift piccolo
  ma persistente, devono essere rilevati; il riferimento corretto deve passare.
  Qualunque anomalia nuova diventa un caso documentato e un futuro test di
  regressione. Se emerge nel fresh, quel fresh resta aperto e la riparazione
  richiede una nuova conferma indipendente.

Task36 definisce questo contratto; Task37 costruisce supporto e valutatori;
Task39/51/63 diagnosticano dati e compute; le conferme di ogni blocco lo
applicano nel proprio scope. Nessuna suite garantisce di scoprire subito ogni
errore ignoto: mantenere un catalogo dei failure mode e ampliare prospetticamente
la copertura è parte del lavoro, non un motivo per reinterpretare vecchi pass.

## 🧩 Blocco 1 — Transizione ionica congiunta, singolo compartimento

**Domanda:** possiamo sostituire le cinetiche e l'integrazione dei gate con
un operatore discreto appreso, conservando inizialmente lo stato fisico?

- **Task 36 — Contratto e scala di astrazione ionica** → Riconciliare Task35,
  CC e IE, completando audit/provenance dei risultati riusati. Inventariare
  gate, correnti, Ca/SK e fasi del solver; congelare baseline, domini,
  comportamenti richiesti e profilo di costo. Separare ciò che è già noto
  dalle nuove ipotesi: non ripetere IE sotto un numero diverso.
  **Conclusa come audit locale (17 ottobre):** inventario/fasi e 22 fonti
  verificati; IE-P7 v4 riconciliata, sette ipotesi aperte e otto specifiche
  di valutazione. GO alla preparazione Task37, nessun training autorizzato
  senza i suoi valutatori, soglie e supporto. Vedi `task36_result_v1.md`.
- **Task 37 — Playground di memoria e interventi** → Generare sequenze con
  pari V ma storie diverse, step/rampe/chirp, recupero lento, Ca variabile,
  stati non all'equilibrio e blocchi/variazioni gbar. Split per traiettoria
  e protocollo, controlli nativi e test di sufficienza; riservare conferme.
  **Conclusa nello scope isolato:**168traiettorie/12famiglie,60NPZ e cinque
  ruoli; riferimento canonico e confronto numerico con il prefisso già
  validato, senza nuova simulazione NEURON. Nove test, otto controlli difettosi,
  reset reale e sette coppie storiche appaiate. Conferma non generata.
  Valutatori/soglie ioniche pronti; comportamenti emergenti e biforcazioni
  dinamiche richiedono ancora Task40–41. Vedi `task37_result_v1.md`.
- **Task 38 — Aggiornamento appreso a stato esplicito** → Confrontare update
  congiunto diretto vincolato dei 18 gate con il riferimento a coefficienti
  discreti già studiato. Includere SK con l'ingresso Ca nella fase corretta.
  Conservare aperture/readout algebraici per isolare l'astrazione dell'update;
  l'eventuale controllo SK canonico va etichettato come sostituzione parziale.
  **Protocollo prospettico pronto:** `task38_joint_transition_preregistration_v1.json`,
  notebook `giada_task38_joint_transition.ipynb`: width32/64 × tre seed,
  checkpoint lungo lo stesso training, soglie Task37 invariate. Ricorrenze
  development/stress dopo freeze. Nessuna
  conferma indipendente o accelerazione del compartimento già dimostrata.
  **Esito acquisito:** training30k completato, NO-GO ricorrente (0/72 e1/72
  per width32/64; baseline53/72). Vedi `task38_result_v1.md`.
- **Task 39 — Learnability, bias e mini scaling** → Matrice piccola di
  capacità e supervisione gate/correnti, budget crescenti sulla stessa
  traiettoria e dati a difficoltà controllata. Probe interni, rollout brevi
  e lunghi, attribuzione per canale: distinguere ottimizzazione, copertura,
  errore di fase e rappresentazione prima di aumentare lo stato nascosto.
  **Protocollo pronto:** `task39_learnability_preregistration_v1.json`;
  2×2×2 copertura/esposizione/bias + due ancore loss, tre seed e ladder fino60k,
  stesso notebook `giada_task39_learnability.ipynb`. Task40 resta bloccata
  fino a superamento della qualità e audit, senza cambiare le soglie.
- **Task 40 — Reinserimento causale e costo locale** → V/Ca imposti contro
  autonomi, sostituzioni per famiglia e congiunte nel compartimento invariato.
  Testare interventi e timing; misurare costo ionico e integrato allo stesso
  lavoro/accuratezza. Congelare candidato e contratto della conferma.
- **Task 41 — Conferma indipendente del livello 1** → Nuovi protocolli/seed
  e pannelli parametrici, qualità e costo con incertezza e casi peggiori.
  Registrare GO, controllo accurato ma lento o NO-GO motivato, per dominio.
  Autorizzare il trasferimento spaziale soltanto con controllo locale valido.

### ✅ Uscita del blocco 1

Operatore e interfaccia congelati, errore attribuibile e costo misurato.
La ricorrenza deve funzionare senza correzioni del teacher; occupazioni
limitate non sono da sole una dimostrazione di stabilità del compartimento.

## 🌿 Blocco 2 — Livello 1 in più compartimenti

**Domanda:** il guadagno locale e la fedeltà si conservano quando i voltaggi
si influenzano tramite il solver assiale classico?

- **Task 42 — Porta assiale e sistemi appaiati** → Riutilizzare IV-D1;
  verificare IV-D2/D3 prima di catene/biforcazioni. Costruire riferimenti
  attivi 2/4/8/16 compartimenti e una biforcazione piccola, con floor nativo,
  unità e bilanci verificati. Queste geometrie restano il pannello comune
  dei blocchi 4 e 6; la loro ampiezza finale si preregistra prima dei run.
- **Task 43 — Trasferimento congelato del livello 1** → Applicare i pesi
  della Task41 con stati locali distinti, parametri eterogenei e ordine dei
  nodi permutato. Prima path imposti, poi feedback assiale; nessun retraining
  nascosto nel risultato di trasferimento.
- **Task 44 — Localizzazione dell'errore di composizione** → Sostituire un
  sito, un gruppo di canali e poi tutti; confrontare accoppiamento nullo,
  debole e forte. Misurare correnti locali, V/Ca e propagazione; un batch
  disconnesso è solo il controllo negativo dell'accoppiamento.
- **Task 45 — Copertura multicompartmentale controllata** → Se necessario,
  refit con path locali prodotti da sistemi accoppiati, stessa architettura
  e stessa rappresentazione del livello 1. Confronto appaiato con frozen,
  mini scaling e morfologie riservate; nessuna informazione futura dai vicini.
- **Task 46 — Interventi spaziali e frontiera di costo** → Stimoli locali,
  blocchi selettivi e conduttanze eterogenee; ritardi/attenuazione e crossing
  dove presenti. Separare costo ionico dal solve assiale e dal traffico dati;
  freeze prima della conferma, senza modificare il livello di astrazione.
- **Task 47 — Conferma e decisione sul livello 1** → Geometrie/seed/stimoli
  disgiunti, confronto con formula e candidato locale. Registrare limiti di
  trasferimento e risparmio reale; formulare quale costo o ridondanza rende
  sensata la compressione del blocco 3, senza proclamarla già riuscita.

### ✅ Uscita del blocco 2

Validità di trasferimento nel piccolo sistema accoppiato dichiarato e
frontiera qualità–costo. Non equivale alla validazione del Hay completo.

## 🧠 Blocco 3 — Memorie ioniche per famiglie, singolo compartimento

**Domanda:** possiamo eliminare parte dei gate espliciti senza perdere la
memoria che determina correnti e risposta agli interventi?

- **Task 48 — Ridondanza e stato minimo per famiglia** → Usare i dati dei
  blocchi 1–2 per proporre raggruppamenti motivati. Cercare collisioni di
  aperture/correnti con futuro diverso, probe di recupero e tempi lenti.
  Un'apertura per canale non viene assunta sufficiente al posto dei gate.
- **Task 49 — Encoder e inizializzazione causale** → Confrontare encoding
  iniziale dello stato fisico disponibile con warmup causale e stato standard
  nel proprio dominio. Decoder dei gate solo come probe/training; verificare
  restart e storie non viste, includendo il costo di inizializzazione.
- **Task 50 — Ricorrenze compatte per famiglie** → Poche dimensioni di
  memoria per famiglia, capacità del trunk variata separatamente. Imparare
  direttamente aperture efficaci e produrre correnti individuali con readout
  economico; mantenere Ca/SK e parametri espliciti. Controllo livello 1 identico
  per dati, input e perimetro; nessuna necessità di ricostruire i gate online.
- **Task 51 — Diagnosi della compressione** → Dimensione dello stato ×
  supervisione ausiliaria in una matrice mirata, ladder dati/budget, reset e
  scambio di memoria come controlli. Misurare separatamente saturazione da
  informazione persa e training insufficiente, recupero e correnti cancellate
  nel totale. Gli oracle privilegiati non sono candidati selezionabili.
- **Task 52 — Closed loop e interventi a stato compresso** → Inserire il
  candidato nel singolo compartimento, isolare le famiglie e il feedback Ca.
  Verificare blocchi, inversioni, correnti/carica e memoria lenta; misurare
  byte di stato, latenze e costo degli adattatori. Congelare la scelta.
- **Task 53 — Conferma indipendente del livello 2** → Test fresh dopo freeze
  contro formule e livello 1, con contratti comuni. Decidere se la memoria
  ridotta conserva le risposte nel dominio e se il risparmio è materiale;
  conservare un esito accurato ma lento come tale, senza chiamarlo speedup.

### ✅ Uscita del blocco 3

Memoria per famiglie sufficiente nel dominio testato, senza accesso ai gate
teacher durante l'inferenza; qualità e riduzione di costo valutate separatamente.

## 🌳 Blocco 4 — Livello 2 in più compartimenti

**Domanda:** la memoria per famiglie rimane sufficiente con storie locali
generate dall'accoppiamento e con interventi eterogenei?

- **Task 54 — Trasferimento congelato delle memorie** → Stessi sistemi della
  Task42, checkpoint Task53, stato latente distinto per sito e famiglia.
  Verificare inizializzazione, restart e porta assiale senza cambiare solver.
- **Task 55 — Memoria sotto perturbazioni spaziali** → Stesso V locale ma
  storie di stimolazione differenti, ingresso distale/prossimale, coupling
  e blocchi selettivi. Confrontare livello 1 e 2 per corrente e risposta
  temporale, non soltanto per V medio dell'intero sistema.
- **Task 56 — Attribuzione locale e tra famiglie** → Sostituzioni miste
  formula/latente per sito e famiglia, probe della memoria, controllo con
  coupling disattivato. Distinguere errore locale accumulato, copertura e
  interazione con calcio; nessun aggiornamento appreso del solver assiale.
- **Task 57 — Riparazione del trasferimento senza nuova astrazione** →
  Eventuale refit sulla copertura accoppiata con architettura e dimensioni
  congelate; braccio frozen appaiato e budget dichiarato. Se occorre più
  memoria, tornare alla diagnosi Task51 con emendamento e nuova conferma.
- **Task 58 — Costo e robustezza spaziale del livello 2** → Misurare memoria
  per sito, throughput e latenza sui workload comuni, includendo readout e
  inizializzazione. Confrontare formule e livello 1, con tempi ionici e
  integrati distinti; freeze e dominio di validità esplicito.
- **Task 59 — Conferma e decisione sul livello 2** → Nuove geometrie/seed e
  interventi riservati; registrare quali memorie sono davvero comprimibili.
  Autorizzare lo studio della memoria condivisa sulla base di una domanda
  causale e di un controllo valido, non per automatismo di calendario.

### ✅ Uscita del blocco 4

Trasferimento causale della stessa rappresentazione compressa, con errori
e costo per sito documentati; nessuna fusione di compartimenti fisici.

## 🔗 Blocco 5 — Un operatore ionico ricorrente condiviso, singolo compartimento

**Domanda:** possiamo rimuovere la separazione dinamica per famiglie e
comprimere ulteriormente il lavoro, conservando le risposte dei singoli canali?

- **Task 60 — Contratto della memoria condivisa** → Cercare ridondanze tra
  le memorie del livello 2; fissare quali stati/interventi devono rimanere
  distinguibili. Il nuovo z è locale al compartimento e non ingloba sinapsi,
  solver o CaDynamics. Una dimensione minore è un'ipotesi da verificare.
- **Task 61 — Ricorrenza ionica unificata** → Piccolo stato multiscala
  condiviso con output per gli undici canali, condizionato su V/Ca e parametri
  ammessi. Confrontare memoria condivisa e per famiglie a budget/capacità
  controllati; mantenere la semantica delle due fasi Ca/SK nell'interfaccia.
- **Task 62 — Astrazione del readout e ablation dei vincoli** → Confrontare
  aperture apprese + algebra economica con corrente predetta direttamente e
  coefficienti compatibili con il solver. Isolare il costo/beneficio del
  vincolo di inversione, del blocco esatto e delle teste individuali; una
  sola corrente totale resta controllo diagnostico, non porta promuovibile.
- **Task 63 — Learnability e memoria emergente** → Mini scaling di stato,
  capacità, dati e budget con poche ipotesi ortogonali; distillazione dai
  gate disponibile solo in training. Probe di memoria lenta e controfattuali
  su storia, gbar e Ca per scoprire compensazioni spurie tra correnti.
- **Task 64 — Operatore completo nel compartimento** → Rollout e interventi
  con tutti gli altri moduli congelati, confronto con livelli 1/2 e formule.
  Valutare ricorrenza stabile, costo totale del modulo e beneficio integrato;
  congelare la variante selezionata prima di aprire il fresh.
- **Task 65 — Conferma indipendente del livello 3** → Nuove combinazioni di
  stimoli/parametri e storie iniziali, misure per canale e dinamiche locali.
  Decidere se la maggiore astrazione conserva la risposta causale richiesta
  e dove migliora realmente accuratezza–memoria–velocità.

### ✅ Uscita del blocco 5

Un unico modello ionico ricorrente locale candidato, con stato e contratto
verificati. Conservare algebra utile non ne riduce il livello di astrazione
della memoria; rimuoverla deve portare un beneficio misurato.

## 🌲 Blocco 6 — Livello 3 in più compartimenti e decisione complessiva

**Domanda:** lo stesso operatore condiviso conserva qualità e vantaggio
quando viene replicato in un sistema elettricamente accoppiato?

- **Task 66 — Trasferimento congelato dell'operatore unico** → Stessi
  sistemi della Task42, pesi Task65 e stato z locale per sito. Validare porte,
  permutazione dei nodi, eterogeneità e restart; nessuna memoria globale
  condivisa che cambi il livello di astrazione o anticipi i vicini.
- **Task 67 — Fedeltà causale spaziale** → Protocolli e interventi appaiati
  con i livelli 1/2: propagazione, recupero, adattamento e risposta a blocchi
  dove supportati. Analizzare correnti per sito e Ca oltre ai segnali somatici.
- **Task 68 — Diagnosi e riparazione circoscritta** → Frozen contro eventuale
  refit di copertura a struttura invariata, con sostituzioni per sito e probe.
  Distinguere generalizzazione, inizializzazione e insufficienza dello stato;
  una modifica della rappresentazione torna alla Task63 e richiede nuovo freeze.
- **Task 69 — Frontiera comune dei tre livelli** → Confrontare i candidati
  congelati su stessi workload, qualità e hardware: correnti, comportamento,
  memoria e tempi ionici/integrati/operativi. Quantificare il risparmio di
  lavoro ottenuto dall'astrazione e la quota che limita lo speedup globale.
- **Task 70 — Conferma finale indipendente** → Congelare sulla sola fase di
  sviluppo livello, candidato e soglie; nuovi sistemi/stimoli/seed per il
  giudizio finale. Selezionare poi sui risultati fresh non è consentito;
  nessun claim sul neurone completo da questi piccoli sistemi.
- **Task 71 — Sintesi nel DB e decisione di prosecuzione** → Registrare nel
  mirror SQLite successi, fallimenti, ipotesi discriminate, domini, costi e
  mapping delle task. Concludere quale livello è utile, quale va scartato e
  quale domanda resta aperta. Se nessuno offre qualità e risparmio, mantenere
  il riferimento e formulare una diagnosi: non dichiarare il programma riuscito.

### ✅ Uscita del blocco 6

Decisione basata su fedeltà causale e vantaggio misurato, non sulla quantità
di formule eliminate. La soluzione migliore può stare a uno dei tre livelli;
non è obbligatorio promuovere quello più astratto.

## ⚡ Esecuzione informativa e continuità dei numeri

Preparare insieme i contrasti indipendenti che condividono dati e controlli;
vettorizzare training/seed quando verificato, salvare budget intermedi e
misurare i timing senza contesa. Esempi di possibili notebook unici:
Task37–39, Task43–44, Task49–51, Task55–56, Task61–63, Task67–68.
La dipendenza da un risultato resta reale: dati, freeze e conferme non vengono
mescolati per ridurre il numero di run. La checklist per task conserva esito,
prove e prerequisiti anche quando il notebook ne copre più di una.
Ipotesi e protocolli entrano nel mirror SQLite prima del relativo run;
risultati, limiti e decisioni dopo ciascun audit, non soltanto alla Task71.
La Task71 sintetizza il percorso, non rimanda la registrazione delle evidenze.

| Numero precedente | Numero corrente | Contenuto preservato |
|---|---|---|
| 36 | 72 | Due compartimenti attivi |
| 37 | 73 | Quattro compartimenti attivi |
| 38 | 74 | Scaling 8 → 16 compartimenti |
| 39 | 75 | Prima biforcazione attiva |
| 40 | 76 | Branch dendritico |
| 41 | 77 | Subtree |
| 42 | 78 | Morfologia ridotta da 32 segmenti |
| 43 | 79 | Scaling 64 → 128 → 256 segmenti |
| 44 | 80 | Hay completo a 642 segmenti |
| 45 | 81 | Fusione di due segmenti |
| 46 | 82 | Piccolo blocco lineare |
| 47 | 83 | Branch operator |
| 48 | 84 | Subtree operator |
| 49 | 85 | Neurone gerarchico coarse-grained |

Gli ID già presenti nel DB, nei file e nelle conversazioni conservano il
significato della loro revisione. Le nuove registrazioni devono citare questa
revisione del dropshot e il titolo della task; un riferimento storico a
«Task36» significa «exTask36 / attuale Task72», non il nuovo contratto ionico.
I risultati multicompartmentali dei sei blocchi possono soddisfare parti
delle Task72–75 soltanto tramite audit esplicito di equivalenza del contratto;
non si duplicano run già validi e non si assegnano completamenti automatici.

---

## 📌 Promemoria — seguito del dropshot, dopo la serie ionica

**Se questa serie dimostra una strada efficace**, applicare lo stesso
percorso graduale alle altre aree del modello: membrana/solver, sinapsi e
relative memorie, dinamiche del calcio e altre componenti pertinenti.
Definire allora i nuovi blocchi sulla base delle lezioni misurate qui.
L'obiettivo di lungo periodo è arrivare eventualmente a **un unico modello
astratto che calcoli insieme le dinamiche necessarie**, con ulteriore
astrazione e composizione validate progressivamente. È una direzione di
ricerca, non una garanzia che tutta la fisica debba scomparire o che più dati
possano recuperare qualsiasi perdita informativa.

Il piano restante è conservato sotto questa linea; cambiano soltanto i
numeri delle task rinviate e i relativi rimandi. Prima di riprenderlo, Task71
decide quali prerequisiti siano già soddisfatti e quali estensioni preparare.

---

# 🌳 VII. Scaling multicompartmentale

- **Task 72 — Due compartimenti attivi (ex Task36)**. Ingresso bloccato fino alla
  verifica indipendente IV-D1, preceduta dalla riconciliazione IV-A1/A2.
  Protocollo passivo prospettico in
  `experiments/iv_ad1_passive_axial_preregistration.json`: cinque controlli
  RC/leak/correnti, tre geometrie × quattro condizioni assiali, limiti
  disconnesso/simmetrico, bilancio capacitivo e convergenza di `dt`. Questo
  run porta il proprio ID IV-D1, **non** viene rinominato Task 72.
  **Aggiornamento IV-D1:** il prerequisito passivo è passato nella v3
  (`experiments/iv_ad1_passive_axial_result.md`); il prerequisito scientifico
  è soddisfatto, ma l'esecuzione è differita dopo i sei blocchi Task36–71.
  Ripresa tramite decisione Task71; IV-D1 non è il completamento della Task72.
- **Task 73 — Quattro compartimenti attivi (ex Task37)**.
- **Task 74 — Scaling 8 → 16 compartimenti (ex Task38)**.
- **Task 75 — Prima biforcazione attiva (ex Task39)**.
- **Task 76 — Branch dendritico (ex Task40)**.
- **Task 77 — Subtree (ex Task41)**.
- **Task 78 — Morfologia ridotta da 32 segmenti (ex Task42)**.
- **Task 79 — Scaling 64 → 128 → 256 segmenti (ex Task43)**.
- **Task 80 — Hay completo a 642 segmenti effettivi (ex Task44)**.

Per ogni livello confrontare:

1. meccanismi neurali più Hines classico;
2. meccanismi neurali più solver assiale approssimato;
3. eventuale operatore coarse-grained.

Questa separazione identifica se errore e costo provengono dalle cinetiche
locali o dal coupling morfologico.

---

# 🪓 VIII. Coarse-graining controllato

Il coarse-graining inizia soltanto dopo avere una composizione esplicita valida.

- **Task 81 — Fusione di due segmenti (ex Task45)**.
- **Task 82 — Piccolo blocco lineare (ex Task46)**.
- **Task 83 — Branch operator (ex Task47)**.
- **Task 84 — Subtree operator (ex Task48)**.
- **Task 85 — Neurone gerarchico coarse-grained (ex Task49)**.

Ogni operatore aggregato viene confrontato con gli stessi moduli espliciti già
validati. In questo modo la perdita informativa è misurabile.

---

# 🖥️ IX. GPU engineering — piattaforma corrente

> La GPU è l'unica piattaforma operativa di questa fase. Il co-design futuro
> non deve interferire con la validità scientifica degli esperimenti correnti.

- **GPU 1 — PyTorch vectorizzato corretto** → Prima baseline funzionale.
- **GPU 2 — `torch.compile`** → Fusion e riduzione dell'overhead disponibile.
- **GPU 3 — Mixed precision controllata** → FP32 come riferimento, poi
  BF16/FP16 dopo la definizione delle tolleranze.
- **GPU 4 — Profiling** → Kernel launch, utilizzo, memoria e sincronizzazioni.
- **GPU 5 — Triton/CUDA** → Solo per primitive scientificamente congelate.
- **GPU 6 — Batching strutturato** → Segmenti, neuroni e protocolli.
- **GPU 7 — Confronto wall-clock con NEURON** → Stesso lavoro e stessa
  precisione scientifica.
- **GPU 8 — Scaling multi-neurone** → 1, 10, 100 e 1.000 neuroni.

### ⏸️ Tenstorrent — esplicitamente differito

Per ora si preservano soltanto proprietà portabili:

- pesi condivisi;
- stato locale compatto;
- operatori fuse-friendly;
- dipendenze causali esplicite;
- forme regolari;
- assenza di controllo dinamico superfluo.

Porting, BFP8, Tensix e kernel persistenti saranno valutati soltanto dopo la
stabilizzazione scientifica e la baseline GPU ottimizzata.

---

# 🔬 X. Protocollo trasversale di ogni task

## 📋 Prima del run

Ogni esperimento deve preregistrare:

- ipotesi causale;
- componente isolata;
- stato e ingressi causalmente sufficienti;
- baseline analitica o numerica;
- bracci e contrasti appaiati;
- seed, budget e mini scaling law;
- metriche e soglie;
- controllo negativo;
- dominio OOD;
- criteri di arresto;
- artefatti da salvare.

## 📊 Output obbligatori

- errore one-step;
- errore di rollout per orizzonte;
- errore sugli stati interni;
- errore sulle correnti;
- rispetto dei vincoli fisici;
- failure map per regime;
- parametri e MAC;
- latenza e throughput GPU reali;
- risultato per seed;
- decisione `GO`, `CONDITIONAL GO` o `NO-GO`.

## 🚨 Regole di interpretazione

- Un RMSE medio migliore non dimostra validità dinamica.
- Teacher forcing e closed loop sono risultati diversi.
- Un input insufficiente non è un fallimento dell'architettura.
- Una componente accurata non implica composizione stabile.
- Una conferma sul development set non è un fresh test.
- Un modello neurale deve essere confrontato anche con approssimatori numerici
  semplici, non soltanto con altre reti.

---

# 🚀 Sequenza principale aggiornata

La sequenza seguente descrive livelli di composizione. Il livello 6 è un
**insieme di filoni paralleli**, con ingressi obbligatori definiti nella
sezione IV, e non una barriera globale prima del livello 7. I numeri di
questo schema sono livelli descrittivi, non identificativi delle Task.

```text
0. Inventario e contratto del teacher
↓
1. Gate Ca-HVA sotto voltage clamp costante
↓
2. Gate con voltage path esogeno variabile
↓
3. Meccanismo Ca-HVA completo
↓
4. Conferma embedded e sostituzione causale
↓
5. Meccanismi rappresentativi di famiglie differenti
↓
6. Filoni IV-A/B/C/D paralleli, da chiudere ai rispettivi punti di integrazione
↓
7. Composizione ionica sotto teacher-forced voltage
↓
8. Compartimento attivo closed-loop
↓
8-bis. Evidenze IE conservate → Task36: contratto della nuova progressione
↓
8-ter. Livello 1: update congiunto esplicito — singolo (36–41), multiplo (42–47)
↓
8-quater. Livello 2: memorie per famiglie — singolo (48–53), multiplo (54–59)
↓
8-quinquies. Livello 3: memoria condivisa — singolo (60–65), multiplo (66–71)
↓ decisione Task71 e promemoria sulle altre aree del modello
9. Ripresa dello scaling multicompartmentale originale (Task72–80)
↓
10. Branch e subtree
↓
11. Morfologie ridotte
↓
12. Hay completo — 642 segmenti
↓
13. Coarse-graining controllato
↓
14. Ottimizzazione GPU
↓
15. Scaling multi-neurone
↓
16. Tenstorrent, dopo la stabilizzazione scientifica
```

---

# 🥇 Stato corrente — Task39e conclusa; proposta Task39f

> Task36 ha fissato il contratto; Task37 ha implementato il playground e
> i valutatori isolati. Task38 ha completato il tentativo congiunto senza
> superare la qualità ricorrente. Task39 è terminata con NO-GO scientifico:
> migliorano alcuni contrasti ma nessun candidato passa tutti i criteri.
> Task39b è completata con NO-GO scientifico: nessun nuovo braccio supera
> tutti i criteri; tutti36 migliorano ancora lo score tra30k e60k.
> Task39c è conclusa: migliore nuovo braccio44/72 casi development, contro16/72
> in39b; progresso descrittivo, non effetto causale di un unico intervento.
> Persistono errori di correnti e timing: NO-GO scientifico.
> Task39d è conclusa su Colab: massimo43/72, nessun nuovo vincitore; la scala
> locale riduce fortemente le code ma non soddisfa il contratto completo.
> [Risultato39d](task39d_result_and_next_hypotheses.md).
> Si prepara [Task39e](task39e_preregistration_v1.json), sempre sottotask39:
> obiettivo delle code×copertura coefficienti×teste u/z,8bracci più2sidecar
> memoria64/256tick a pari512target/update;3seed,30membri vettorizzati.
> Oracoli sui nuovi checkpoint39d prima del fit; selezione prospettica include
> timing nel pannello development, con confronto non selezionabile sulla
> vecchia regola usando gli stessi checkpoint. Soglie37 invariate.
> Esecuzione tramite CLI Colab/WSL collaudata, non automazione dell'editor browser.
> Task40 non è autorizzata dal solo miglioramento della loss media.

**Aggiornamento19 ottobre2026:** Task39e retry2 conclusa a180000 aggiornamenti,
parametri e stati Adam finali finiti, archivio esterno finale verificato.
Miglior nuovo braccio56/72 development (39d43/72),15 casi recuperati e2 persi.
Restano16 fallimenti:7 ampiezza,1 carica,10 timing, categorie sovrapposte;
9 casi falliscono soltanto sul timing. Otto scarti temporali riguardano
l'offset, uno l'onset e uno il picco. Solo1/14 contrasti preregistrati supportato.
Il confronto storico non isola un intervento: cambiano budget, selezione e
correzione della proprietà degli stati Adam nel CUDA Graph.
[Audit completo](results/task39e_colab_v1/result_audit.json) e
[ipotesi/proposta39f](task39e_residual_hypotheses.json) conservano dettagli e limiti.
Task40 resta bloccata. Task39f è proposta, non preregistrata né avviata:
audit congelato u/z e crossing, poi matrice obiettivo temporale×supporto
transizioni×capacità con soglie invariate e mini scaling laws.

Questo aggiornamento pianifica i sei blocchi; non avvia training o run Kaggle.
La registrazione dei nuovi protocolli nel mirror SQLite, con numerazione e
titoli coerenti, precede la loro esecuzione. Airtable resta congelato.

## 📚 Registro di avanzamento IE — conservato, non sequenza corrente

> Localizzare il margine di risparmio e preparare un confronto isolato del
> blocco ionico: formule competitive, baseline congelata e aggiornamento
> congiunto appreso, a stato/ingressi/passo controllati. Cercare il primo punto
> con qualità ammissibile **e** costo inferiore, poi provarlo nel compartimento.

La prima attività del precedente piano era IE-01…05, con audit N2, profilo riutilizzabile e contratto
di interfaccia. Non si lancia N3 automaticamente e non si riaddestra il
compartimento intero prima di aver circoscritto il problema. Le nuove task
sono pianificate in questo documento; la loro registrazione DB e le relative
preregistrazioni numeriche vanno completate prima dell'esecuzione.

**Avanzamento:** audit N2 concluso con ricostruzione indipendente delle
metriche salvate; [primo notebook IE-P1](../notebooks/ie_p1_ionic_cost.ipynb)
e [protocollo](ie_p1_ionic_cost_preregistration_v1.json) preparati per la prima
fase residente di IE-02…04. Il profilo globale (eventi/RNG/trasferimenti) e
il nuovo playground IE-05 restano aperti; IE-P1 non addestra candidati.

**IE-P1 concluso nella v3:** [risultato e incidenti](ie_p1_result_v1.md).
Sul core formula il buffer dei gate risparmia circa il 16% a batch642,
contro circa il 48–56% sugli ibridi. È una misura diagnostica di opportunità,
non un modello promosso o un limite universale. IE-03 deve valutare il
residuo prima di promettere un'accelerazione grande dalla sola sostituzione
dei gate; IE-02 globale e IE-04/05 restano aperte.

**IE-P2 — 16 ottobre 2026:** [disegno e contrasti](ie_p2_design_v1.md),
[protocollo](ie_p2_residual_cost_preregistration_v1.json) e
[notebook](../notebooks/ie_p2_residual_cost.ipynb) preparati per distinguere
prefisso ionico, readout, solver e overhead di invocazione, con le stesse
equazioni. Tre test locali superati, inclusa ricomposizione/cache e cattura
fullgraph; questo preflight CPU non certifica CUDA Graph o prestazioni GPU.
Nessun training e nessuna promozione prima del risultato e della decisione.

**IE-P3 preregistrato:** [confronto readout esatto/appreso](ie_p3_design_v1.md),
[notebook unico](../notebooks/ie_p3_readout.ipynb). Isola il readout dai18gate:
algebra storica, algebra vettorizzata, MLP16/32 con3seed appaiati. Non sostituisce
la dinamica dei gate e non conclude ancora IE-07 o compressione della memoria.
Esecuzione comune Inductor/CUDA Graph; qualità e costo sono criteri separati.

**IE-P3 concluso:** [risultato verificato](ie_p3_result_v1.md).48celle e51preflight
validi. Al batch642, algebra vettorizzata esatta:48,26% di tempo risparmiato
contro la storica con CUDA Graph, stessa qualità diagnostica. Tutte6reti
falliscono qualità e costo rispetto al controllo vettorizzato. Conservare
questa baseline; nessuna sostituzione appresa promossa e nessuna conclusione
di impossibilità per un operatore ionico più ampio.

**IE-P4 preregistrato:** [disegno](ie_p4_design_v1.md),
[notebook](../notebooks/ie_p4_discrete_ionic.ipynb). IE-04/06/07/08: coefficienti
discreti congiunti per17gate dipendenti daV, SK canonico nella faseCa→SK,
readout vettorizzato esatto. Formule/LUT e matrice capacità×supervisione,
3seed e budget crescenti lungo la stessa traiettoria. Nessuna riduzione di
memoria, nessuna nuova conferma nativa o promozione già dichiarata.

**IE-P4 concluso:** [risultato e audit](ie_p4_result_v1.md).84celle/preflight
validi,14bracci riprodotti suCPU. Supervisione interna: beneficio sui path6/6,
RMSEV storica10,29–16,10volte inferiore, ma nessuna rete supera tutti i gate
di qualità/costo. LUT supera qualità e risparmia18,57%graph a642; ordinario
più lento8,60%, vantaggio non universale. Nessuna promozione operativa.

**IE-P7 preregistrato:** [disegno](ie_p7_design_v1.md),
[notebook](../notebooks/ie_p7_native_operational.ipynb). Sei operatori locali
congelati;32nuovi episodi nativi e test atomici/path indipendenti. Il timing
IE-P6 includeva già il core di membrana: qui si aggiunge il perimetro operativo
dichiarato (eventi/RNG indipendenti,IPC,trasferimenti,rollout e output), contro
formule,LUT e tutti6ibridiTask35. Nessun training o promozione automatica;
copertura parziale IE-12–15, non completamento implicito IE-16 o exTask36
(ora Task72). Questa voce conserva lo stato della preregistrazione;
la riconciliazione dei successivi run è parte della nuova Task36.

**IE-P7 v4 conclusa e riconciliata in Task36:**
[risultato verificato](ie_p7_result_v1.md). 56 celle core e 56 operative,
14 classificazioni ricalcolate; otto mappe/riferimenti riprodotti sulle
32 tracce native. Sei operatori locali passano qualità e costo residente,
ma non il criterio operativo del 10%. Supporto insufficiente per un claim
generale sui comportamenti; nessuna promozione o training aggiuntivo.

**IE-P6 concluso:** [risultato e limiti](ie_p6_result_v1.md),
[disegno preregistrato](ie_p6_design_v1.md),
[notebook](../notebooks/ie_p6_residual_repair.ipynb). Matrice copertura ×
ricorrenza breve, operatori locali appresi81/161nodi, deployment tabulato,
attribuzione sui nuovi parent IE-P5. Il training comprende18modelli
(12MLP+6locali);35mappe in valutazione.
Soglie e blocchi fisici invariati. Tutti6operatori locali superano qualità
e criterio di costo graph a642 (10,0–10,7%risparmio); LUT analitica resta
più veloce (25,3%). Matrice: copertura migliora worst-case ma non ogni metrica;
obiettivo temporale testato peggiora.210preflight/celle validi,35mappe
riprodotteCPU. Nessuna nuova conferma nativa o promozione implicita.

**IE-P5 preregistrato (poi concluso, esito sotto):** [disegno](ie_p5_design_v1.md),
[notebook](../notebooks/ie_p5_accuracy_execution.ipynb). IE-08/12: refit
appaiato interno/bilanciato, attribuzione u/z e gruppi su modelli congelati,
inferenzaMLP contro tabulazione delle sue uscite. Quest'ultima è compilazione
numerica, non una nuova architettura. Nuovo fresh dopofreeze, limiti invariati,
nessuna compressione dello stato, conferma nativa o promozione già avvenuta.

**IE-P2 concluso:** [risultato e limiti](ie_p2_result_v1.md), 54 celle e
72 preflight numerici superati. A batch642, stesso full con CUDA Graph
circa 4,2× più veloce nel microbenchmark; il readout delle conduttanze
mostra un margine importante anche nel replay. Prima del training occorre
un controllo algebraico vettorizzato e una baseline di esecuzione comune.
I buffer non sono candidati causali; profilo globale e interfaccia del
playground restano aperti. La numerazione IE e originale non cambia.

## IE-P5 — esito verificato

Esperimento completato: [risultati e limiti](ie_p5_result_v1.md).
Il refit standard migliora il voltaggio diagnostico 3,48–4,84 volte; il pacchetto
balanced peggiora tutti e tre i seed. Nessun candidato appreso supera tutte
le soglie. La tabulazione fine delle predizioni migliora il costo CUDA Graph
con maggiore memoria, non dimostra accelerazione diretta della MLP.
Le attribuzioni oracle riguardano soltanto i vecchi modelli congelati.
Nessuna promozione operativa e nessuna modifica della numerazione delle task.

## 📜 Prima milestone storica — conservata come riferimento

> Costruire un playground GPU fattoriale per i gate `m` e `h` di Ca-HVA sotto
> voltage clamp, verificato sia contro le formule di `Ca_HVA.mod` sia contro
> NEURON. Confrontare nella stessa esecuzione formula, LUT, spline/polinomio,
> MLP, GRU, physical-\(\tau\) e direct-\(z\), includendo mini scaling law,
> stress test e misura GPU.

Questa milestone deve dire non soltanto quale modello ottiene l'errore più
basso, ma anche:

- se la funzione è veramente learnable con una primitive minima;
- se una rete porta vantaggi rispetto a una baseline numerica;
- quale informazione sul percorso di voltaggio sarà necessaria nel passo
  successivo;
- quali vincoli garantiscono stabilità;
- quale parte della soluzione è già pronta per una futura composizione.
