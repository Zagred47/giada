# GIADA roadmap Task 15 — confronto decomposizione / corrente diretta (preregistrazione v1)

Stato: **preparato, non ancora eseguito su Kaggle**. È la Task 15 originale della roadmap, dopo Task 14. Non è la Task 11c, che resta differita. Notebook: `notebooks/15_roadmap_current_architecture_comparison.ipynb`. Prerequisito di provenienza: report esatto della Task 14, SHA-256 `0e1aed13dd73a78084c4009311b071b2445f5f9d8d823d4fed5201accb187a2e`; non viene usato per scegliere i pesi.

## Domanda e ipotesi

Su un playground sintetico Ca-HVA+pas a **un compartimento**, apprendere direttamente la corrente è più facile che apprendere il percorso di V e i gate m/h e poi calcolare la corrente con la formula fisica? La supervisione ausiliaria di V e gate rende più apprendibile la corrente diretta? I miglioramenti assoluti sopravvivono a interventi appaiati su gbar, ECa, gate iniziali e maschera del meccanismo?

Bracci freschi, addestrati nel medesimo notebook:

1. **A — decomposed_formula**: MLP → quattro valori del percorso V e gate finali m/h appresi, sigmoid per i gate; corrente `gbar_effective·m1²·h1·(V1−ECa)` calcolata analiticamente. Loss su V e gate, non direttamente su corrente. L'integrazione analitica dei gate lungo il percorso V è soltanto un probe diagnostico.
2. **B — direct_current**: MLP → corrente finale diretta, loss sulla corrente.
3. **C — auxiliary_current**: MLP → corrente finale diretta più percorso V e gate m/h; loss di corrente + 0,5 × loss di stato. Il suo head di corrente non determina la transizione di V nel rollout.

Tutti ricevono lo **stesso tensore numerico causale** di dieci campi: V0, m0, h0, gbar nominale, maschera esplicita, ECa e quattro correnti programmate nei quarti del prossimo millisecondo. Nessun endpoint del teacher è input. Il target di corrente è quello analitico all'endpoint; `ica` nativa sfasata non è target. Una proiezione con maschera nota per B/C è solo sidecar diagnostico, non quarto braccio. Le tre supervisioni differiscono intrinsecamente: questo è un confronto fra *pipeline architetturali e obiettivi*, non un isolamento puro dell'effetto della topologia.

## Allineamento e selezione

- Train: 192 episodi × 16 ms, seed 15011. Development: 48 × 16 ms, seed 15029. Sealed: 48 × 16 ms, seed 15059. Controfattuali appaiati: 48 × 16 ms, seed 15159.
- Variabilità episode-wise per train/development/sealed: ECa uniforme 100–140 mV; gbar ×0,5/1/1,5; shift iniziali m/h in −0,1/0/+0,1; maschera off per 1/8 degli episodi.
- Seed di training 17/29/43. Stessi indici dei minibatch per i tre bracci entro ogni seed, batch 256, Adam 0,001, 400 step. Checkpoint valutativi 50/100/200/400 sulla stessa traiettoria; nessun nuovo fit per checkpoint. MLP SiLU a due strati hidden: A 62 unità/6 output, B 64/1, C 62/7. Scarto dei parametri addestrabili ≤6%.
- Il seed selezionato per ciascun braccio minimizza **solo** RMSE di corrente development allo step 400. Freeze con hash di configurazione e checkpoint prima di generare sealed e controfattuali. I checkpoint intermedi sono mini scaling law, non candidate per la selezione.

## Metriche e criteri registrati

One-step sealed: RMSE/MAE della corrente complessiva, RMSE del quartile più attivo, RMSE con meccanismo off; errori V/m/h solo per A/C; conteggi di corrente entrante predetta con segno opposto, violazioni di occupanza gate e identità corrente propria. Probe per A/C: differenza fra gate appresi e gate ottenuti con integrazione analitica del percorso predetto, senza usare quel risultato per scegliere il modello.

Rollout ricorsivo di 16 ms: V, m e corrente solo per A/C. B non definisce una transizione di stato, quindi il rollout è **non disponibile**, non zero. Per C la corrente predetta è misurata ma non alimenta V; nessuna pretesa di accoppiamento chiuso. Controfattuali appaiati: delta-I per tutti i bracci e delta-V per A/C, su gbar dimezzato/aumentato, ECa 100/140, m iniziale aumentato, h diminuito, meccanismo off visibile. Escluso il controllo Task14 con maschera intenzionalmente nascosta perché violerebbe l'allineamento informativo qui scelto.

Regola interpretativa fissata prima del test: vantaggio B su A se RMSE I sealed totale ≤0,8×A **e** RMSE attivo ≤A. Beneficio degli ausiliari C su B se RMSE I sealed totale ≤0,9×B **e** mediana dell'errore delta-I appaiato ≤B. In ogni caso riportare anche le metriche divergenti, il costo parametrico e la mini scaling law; nessun singolo rapporto autorizza una conclusione generale. Un successo di corrente one-step non promuove al Gate C, al teacher multicompartimentale né alla fedeltà biologica completa.

Il run finale non è stato eseguito localmente; richiede la GPU CUDA di Kaggle. Il notebook emette un report e uno ZIP via Blob/base64 secondo il contratto operativo del progetto.
