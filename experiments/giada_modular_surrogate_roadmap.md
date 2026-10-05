# 🧭 GIADA — Roadmap modulare per surrogate biofisici

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

Le task originali 1–49 conservano numero e significato. Questa sezione usa
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
- Task36: IV-D1; Task37–38: IV-D2; Task39–44: IV-D3.

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
- **Task 32 — Dinamiche lente in feedback**.
- **Task 33 — Sinapsi con ingressi completamente osservabili**.
- **Task 34 — Correnti e stati come probe privilegiati**.
- **Task 35 — Full compartment surrogate**.

### 🔬 Matrice causale obbligatoria

| Voltaggio | Stato | Significato |
|---|---|---|
| teacher | surrogate | isola l'errore dello STATE updater |
| surrogate | teacher | isola l'errore del voltage updater |
| surrogate | surrogate | misura l'interazione closed-loop |
| formula originale | voltage updater appreso | controllo meccanicistico |

---

# 🌳 VII. Scaling multicompartmentale

- **Task 36 — Due compartimenti attivi**.
- **Task 37 — Quattro compartimenti attivi**.
- **Task 38 — Scaling 8 → 16 compartimenti**.
- **Task 39 — Prima biforcazione attiva**.
- **Task 40 — Branch dendritico**.
- **Task 41 — Subtree**.
- **Task 42 — Morfologia ridotta da 32 segmenti**.
- **Task 43 — Scaling 64 → 128 → 256 segmenti**.
- **Task 44 — Hay completo a 642 segmenti effettivi**.

Per ogni livello confrontare:

1. meccanismi neurali più Hines classico;
2. meccanismi neurali più solver assiale approssimato;
3. eventuale operatore coarse-grained.

Questa separazione identifica se errore e costo provengono dalle cinetiche
locali o dal coupling morfologico.

---

# 🪓 VIII. Coarse-graining controllato

Il coarse-graining inizia soltanto dopo avere una composizione esplicita valida.

- **Task 45 — Fusione di due segmenti**.
- **Task 46 — Piccolo blocco lineare**.
- **Task 47 — Branch operator**.
- **Task 48 — Subtree operator**.
- **Task 49 — Neurone gerarchico coarse-grained**.

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
9. Piccoli sistemi multicompartmentali
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

# 🥇 Prima milestone concreta

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
