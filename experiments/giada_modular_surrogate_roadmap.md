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
  **Da verificare come task della roadmap**: i notebook storici `07`, `07b`,
  `07c` riguardano invece il microcanary Ca-HVA closed-loop e la semantica dei
  confini. Il loro numero non costituisce evidenza di completamento di questa
  Task 7.

- **Task 8 — Rampe e chirp** → Percorsi continui con velocità differenti.

- **Task 9 — Percorsi fisiologici registrati** → Usare voltage path provenienti
  dal teacher, mantenendoli teacher-forced.

- **Task 10 — Test di sufficienza dell'ingresso** → Confrontare solo \(V_t\),
  \((V_t,V_{t+1})\), statistiche intra-ms, pochi campioni del path e substep.
  **Da eseguire**: lo studio supplementare `TG-01` sulla granularità interna,
  etichettato inizialmente Task 10, non testa questa matrice di informazione.

- **Task 11 — Operatore path-aware** → Introdurlo soltanto se \(V_t\) risulta
  causalmente insufficiente.

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

- **Task 19 — Gate singolo** → `Ih`, `Im` o meccanismo equivalente.

- **Task 20 — Dipendenza dal calcio** → `SK_E2`, separato dai gate puramente
  voltage-dependent.

- **Task 21 — Dinamica lenta** → Meccanismo con costante temporale molto più
  lunga.

- **Task 22 — Condivisione controllata** → Confrontare modelli distinti, trunk
  condiviso con head separate e modello condizionato dall'identità del
  meccanismo.

La domanda centrale è:

\[
\text{quanto compute può essere condiviso senza perdere stabilità?}
\]

---

# ⚡ IV. Spettri indipendenti eseguibili in parallelo

> Questi filoni possono procedere nello stesso capitolo sperimentale quando
> condividono infrastruttura, ma mantengono contratti e decisioni separati.

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

- **Task 24 — Famiglia del sodio** → Ricerca di compute condiviso.

- **Task 25 — Meccanismi eterogenei** → Combinare cinetiche differenti.

- **Task 26 — Joint vs independent** → Confronto appaiato tra modelli separati
  e trunk condiviso.

- **Task 27 — Correnti individuali e totale** → La corrente totale può essere
  un target, ma le componenti individuali restano target ausiliari e probe.

- **Task 28 — Blocco ionico completo teacher-forced** → Tutti i gate e le
  correnti locali con voltaggio ancora fornito dal teacher.

### ✅ Gate D — Componibilità ionica

Non si passa al closed loop se:

- i moduli isolati non sono validi;
- il modello congiunto degrada correnti importanti;
- il risparmio di compute non è materiale;
- non è possibile attribuire l'errore ai singoli meccanismi.

---

# 🔁 VI. Compartimento attivo closed-loop

Soltanto qui il modello comincia a generare il proprio voltaggio.

- **Task 29 — Ionico + passivo con clamp esterno**.
- **Task 30 — Voltage update autonomo**.
- **Task 31 — Scheduled sampling diagnostico**.
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
6. Spettri indipendenti: passivo / calcio / sinapsi / assiale
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
