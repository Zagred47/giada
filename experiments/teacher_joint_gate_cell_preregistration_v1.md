# 🧬 GIADA Task 3 — cella congiunta `m+h`

La Task 3 compone per la prima volta i due gate `Ca_HVA` sotto voltage clamp costante. Confronta due trunk privati con una rappresentazione di voltaggio condivisa, mantenendo quattro teste distinte e interpretabili: `m_inf`, `tau_m`, `h_inf`, `tau_h`.

## 🧪 Contrasto

| Famiglia | Rappresentazione di V | Ruolo |
|---|---|---|
| `independent` | due trunk width 16 | controllo senza condivisione |
| `shared_compact` | un trunk width 16 | test di compressione/condivisione |
| `shared_matched` | un trunk width 23 | controllo della capacità parametrica |

Il gate `m` conserva l'obiettivo endpoint selezionato dalla Task 1b; `h` conserva la rate supervision selezionata dalla Task 2b. Seed, minibatch, learning rate e checkpoint sono appaiati.

La diagnostica compositiva `m²h` è calcolata analiticamente. Non è ancora la corrente completa: non include conduttanza massima, driving force o feedback sul voltaggio.

## ✅ Gate

Il miglior braccio condiviso deve restare entro il 10% del controllo indipendente sul development e superare sul fresh: `m≤2,5e-3`, `h≤1e-3`, `m²h≤2,5e-3`, rollout-1000 `m≤2,5e-3`, rollout-1000 `h≤5e-3`, zero violazioni fisiche.

Il risultato non autorizza ancora affermazioni su voltage path variabile, corrente ionica completa o teacher embedded.
