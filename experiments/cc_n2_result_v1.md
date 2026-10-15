# 🧠 GIADA — CC-N2 / IE-01: esito e riconciliazione

Run Kaggle `alessandrobelli/giada-cc-n2-unified-cell-and-memory`, versione 1,
codice `469661b570ba56825309f917fdf25a4a886a83f8`. Il run è tecnicamente
completo; nessun candidato è promosso nei limiti CC. Nessuno speedup misurato.

## 🔎 Verifica indipendente degli artefatti

`scripts/audit_cc_n2_result.py` verifica hash del freeze, checkpoint, protocollo
e dataset, riproduce i 27 minimi di selezione development e ricalcola da
traiettorie salvate V/Ca pooled e worst-episode per 27 modelli × 2 ruoli ×
2 orizzonti = 108 valutazioni. Scarto relativo massimo 8,67e-7. Nessun nuovo
training, selezione o replay GPU dei checkpoint è effettuato dall'audit.
Indice SHA-256 e dimensioni in `results/cc_n2_kaggle_469661b/result_audit.json`.

## 📊 Risultati diagnostici a 8 ms

Media RMSE pooled sui tre seed; casi peggiori conservati nel JSON e nel DB.

| Braccio | V medio (mV) | Peggior episodio V (mV) | Seed promossi |
|---|---:|---:|---:|
| explicit18 | 0,2515 | 0,9675 | 0/3 |
| current_z0 | 1,3191 | 4,0494 | 0/3 |
| current_z4 | 1,0647 | 3,8315 | 0/3 |
| current_z8 | 1,1108 | 2,8918 | 0/3 |
| current_z16 | 1,1609 | 3,8841 | 0/3 |
| conductance_z8 | 0,6209 | 3,7683 | 0/3 |
| direct_z8 | 1,4100 | 4,5193 | 0/3 |
| current_z8_standard | 1,2213 | 3,8643 | 0/3 |
| current_z8_warmup | 1,2179 | 4,7467 | 0/3 |

Soglie CC: V pooled ≤0,05 mV; worst episodio ≤0,1 mV;
worst Ca ≤5e-6 mM. Nel diagnostico a 8 ms il massimo RMSE pooled V dei
riferimenti formula/independent/shared_heads è rispettivamente
0,007784/0,008099/0,007906 mV. Gli orizzonti a 30 ms non promuovono candidati.
Il run usa core ionico senza sinapsi e stati iniziali preparati con interventi:
nessuna nuova conferma Task35 o generalizzazione OOD.

## 🧪 Informazione acquisita e limiti

- Il controllo esplicito è migliore dei latenti testati ma non passa: la
  compressione dello stato non è una spiegazione sufficiente di tutti gli errori.
- La testa a conduttanze z8 migliora la media rispetto alle alternative
  latenti testate; non soddisfa la fedeltà e non è una soluzione promossa.
- Più dimensioni migliorano il probe lineare degli stati iniziali, senza
  miglioramento monotono del rollout. Il probe vede l'inizializzazione:
  non prova che lo stato mantenga quella informazione durante l'evoluzione.
- 24/27 checkpoint scelti sono a 1500 update, due a 150, uno a 500.
  Questo rende ragionevole studiare il budget; non dimostra che sia la causa.
- Reset/permutazioni e curve complete sono conservati per seed/ruolo.
  Un effetto sulla memoria in un modello inaccurato non certifica memoria
  sufficiente. Encoder, warmup e stato standard hanno informazioni differenti.
- Il grafo vettorizzato alloca lo stesso numero di pesi per membro; non è un
  confronto di inferenza ottimizzata né un isolamento puro della capacità.

## 🧭 Decisione

CC-05…08 concluse come matrice diagnostica; CC-B non superato. N3…N6 sospesi
secondo la nuova direzione approvata. Il prossimo passo è IE-02…05: profilo
dei costi e contratto del blocco ionico, prima di nuovi training. Nessuna
ipotesi universale di impossibilità e nessuna riapertura automatica Task36.
