# 🧠 GIADA — risultato CC-01…CC-04

**N1 completato.** Report finale Kaggle v4 valido; codice
`dc175f9d47d8a7f974f4c6a19a0468f89a22f6a6`. Report SHA-256:
`9c25536a7ef1ae5bf9f997a8c6ae67c61a370556667b7c22c73d58ebff30a59d`.

Nessun training e nessuna selezione di nuovi modelli. I target nativi v3 sono
riutilizzati nella v4 con verifica SHA-256; la v4 ripete accuratezza e costi
CUDA dopo la correzione dei metadati del grafo. Il riuso non è una conferma
scientifica indipendente.

## 📋 CC-01 — Contratto

Contratto versionato per stato/ingressi, inizializzazione, unità, domini e
criteri di fedeltà/costo. Il passo iniziale resta 0,1 ms. Per il futuro candidato
sono fissati V pooled ≤0,05 mV, peggior episodio ≤0,1 mV e Ca ≤5e-6 mM, oltre
a margini appaiati sulla baseline. Metriche comportamentali richiederanno il
supporto dedicato nelle task successive; N1 non certifica un nuovo modello.

## 🧪 CC-03/04 — La memoria ha un effetto osservabile

Sono state eseguite **288 continuazioni direttamente in NEURON**, da stati
ionici preparati sotto storie controllate di V/Ca clamp. Stessi V e Ca al
rilascio, parametri e corrente futura appaiati; soltanto la memoria dei gate
cambia. Tutti gli stati iniziali sono stati importati esattamente e i controlli
di replica e trapianto dello stato hanno errore massimo zero.

| Vista V/Ca identica | Gruppi confrontati | Gruppi rilevabili | Massima separazione V degli endpoint |
|---|---:|---:|---:|
| Istantanea | 96 | 60 | 3,26397 mV |
| Ultimi 0,5 ms | 80 | 44 | 2,98540 mV |
| Ultimi 2 ms | 64 | 28 | 2,89394 mV |
| Ultimi 10 ms | 48 | 16 | 1,94618 mV |
| Ultimi 40 ms | 32 | 8 | 1,61407 mV |
| Ultimi 200 ms | 16 | 4 | 1,47787 mV |

Ogni gruppo contiene 12 storie e fissa coda, pannello, corrente futura e
orizzonte. Le massime possono appartenere a condizioni differenti. I gruppi
sono diagnostici correlati, non repliche statisticamente indipendenti.

Se un predittore deterministico vede la stessa informazione per due futuri
separati di D, almeno uno dei suoi errori di endpoint è ≥D/2. Queste collisioni
sono quindi un argomento informativo esplicito: non derivano dall'insuccesso
di un particolare decoder.

**Interpretazione:** serve conservare memoria persistente oppure ricavarla da
un'interfaccia di inizializzazione/storia sufficientemente informativa. Non
abbiamo identificato la dimensione latente minima né dimostrato la learnability
di una particolare rete. La vista appaiata comprende V/Ca, non la storia della
corrente di clamp: non è una prova contro tutti i possibili ingressi storici.
Le preparazioni sono interventi controllati, non frequenze della fisiologia naturale.

## ⚙️ CC-02 — Accuratezza e costo della baseline

Le otto nuove prove del compartimento con sinapsi hanno superato i propri gate
nativi/formula. Il core GPU residente conserva l'accuratezza dei vecchi rollout:
massimo scarto V <0,000289 mV e Ca <5,22e-9 mM su tutti i 32 casi storici Task35
e tutti i sei checkpoint. Compilazione ed eager concordano alle tolleranze
preregistrate; nessun fallimento di compilazione nella v4.

Tesla T4, FP32, Inductor simmetrico; **batch 642 per seed e tre seed vettorizzati**:

| Core | Pesi per seed | Mediana wall-time per passo dell'ensemble | ns per compartimento-passo |
|---|---:|---:|---:|
| Formule | 0 | 0,423835 ms | 220,06 |
| Ibrido independent | 6.260 | 0,807482 ms | 419,25 |
| Ibrido shared_heads | 644 | 0,904102 ms | 469,42 |

Tutti mantengono V, Ca e 18 gate: **20 scalari di stato del core**, oltre allo
stato sinaptico esplicito. Il costo misurato include gate, correnti ioniche,
Ca/SK e solver con NMDA. Esclude la generazione sinaptica di eventi/plasticità/RNG,
riportata separatamente come lavoro host. Non confrontare questi numeri con
il tempo CPU di un singolo episodio osservato per dedurre uno speedup globale.

Gli ibridi compilati sono circa 1,91× e 2,13× più lenti delle formule nel
workload misurato. Il profiler registra 18/22/23 eventi CUDA per i tre bracci:
diagnostica di supporto, non un'attribuzione causale completa. La compressione
dei pesi del vecchio modello non ha compresso lo stato o migliorato questa
latenza. Questo motiva una diversa dinamica aggregata; non prova che una futura
cella ricorrente sarà più veloce.

## 🔎 Limite numerico scoperto e conservato

La mappa analitica non supera il floor Ca su alcuni stati fortemente perturbati.
La flag `analytic_memory_floor_passed=false` resta nel report finale. Il controllo
`h.fcurrent()` non ha risolto quel limite. Le conclusioni sulle collisioni usano
perciò le continuazioni native dirette, secondo l'emendamento prospettico.

Il compilatore iniziale trattava un indice NumPy come dinamico. La correzione
usa costanti Python e ha superato sia il test di cattura fullgraph locale sia
la verifica CUDA v4. Soglie, equazioni e checkpoint non sono stati modificati.

## ➡️ Decisione

CC-01…CC-04 sono concluse nel perimetro dichiarato. È giustificata la
**preparazione di N2 (CC-05…CC-08)**: controllo unificato a stato esplicito,
scala della memoria latente, bias fisici e inizializzazione. Nei regimi in cui
il floor analitico non passa, usare target nativi e non attribuire quel residuo
alla rete. Nessuna architettura nuova, accelerazione end-to-end o Task36 è promossa.

Artefatti completi:
[v3 nativa e diagnostica](results/cc_n1_v3_kaggle_9b6a581/),
[v4 finale](results/cc_n1_v4_kaggle_dc175f9/).
