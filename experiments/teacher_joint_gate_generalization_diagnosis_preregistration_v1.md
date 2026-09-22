# 🧪 GIADA Task 3d — diagnosi causale della generalizzazione m+h

## 🎯 Obiettivo

Separare in un solo run le cause ancora plausibili del fallimento Task 3c: densità dei dati, bilanciamento, estensione in tensione, identificabilità temporale di `h`, prior di forma, topologia condivisa e budget.

La Task 3d è **development-only**. Non legge né usa il fresh Task 3c e non apre un nuovo test sealed.

## 🧩 Matrice sperimentale

Otto bracci shared vengono addestrati in parallelo per tre seed, per un totale di 24 candidati nello stesso tensore GPU:

1. `baseline_current`;
2. `same_support_dense`;
3. `balanced_support`;
4. `expanded_support`;
5. `multihorizon`;
6. `shape_constrained`;
7. `expanded_multihorizon`;
8. `full_repair`.

Tre controlli `full_repair_independent`, uno per seed, vengono addestrati come secondo ensemble vettorializzato. Tre controlli `full_repair_wide` con larghezza 46 formano un terzo ensemble. Separiamo così topologia e capacità senza confonderle con un singolo seed.

## ⚡ Uso della GPU

- candidati shared vettorializzati su asse `arm × seed`;
- candidati independent vettorializzati sull'asse seed;
- stream appaiati e momenti Adam indipendenti;
- `torch.compile` richiesto con fallback sicuro;
- target multi-orizzonte ottenuti mediante `dt` efficace sotto voltage clamp, evitando 100–1.000 forward ricorsivi per minibatch;
- checkpoint a `0/1k/3k/10k/30k/50k` lungo la stessa traiettoria;
- output di progresso compatto, senza stampa di array.

## 🔬 Contrasti causali

| Ipotesi | Contrasto |
|---|---|
| Quantità di dati | baseline vs same-support dense |
| Distribuzione del supporto | dense vs balanced |
| Estensione in tensione | balanced vs expanded |
| Identificabilità temporale | balanced vs multihorizon |
| Prior strutturali | balanced vs shape-constrained |
| Interazione informazione × orizzonte | expanded vs expanded-multihorizon |
| Riparazione combinata | baseline vs full-repair |
| Interferenza shared | full-repair shared vs independent |
| Limite di capacità | full-repair width 23 vs width 46 |
| Limite di budget | curve ai sei checkpoint |

Una causa viene considerata materialmente supportata con almeno il 20% di riduzione dello score development di stress nel contrasto registrato.

## 🧠 Prior di forma

Il braccio vincolato penalizza soltanto proprietà giustificate per `Ca_HVA`:

- `m∞(V)` non decrescente;
- `h∞(V)` non crescente.

Non imponiamo monotonicità arbitraria a `τm` o `τh`.

## 🔒 Decisione

La Task 3d non può autorizzare la Task 4. Identifica la riparazione minima supportata. Tale riparazione dovrà essere congelata prima di una nuova conferma su un nuovo set sealed indipendente.
