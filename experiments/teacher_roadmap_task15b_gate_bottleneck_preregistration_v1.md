# GIADA supplemento Task 15b — identificare il collo di bottiglia dei gate

Stato: **preregistrato, run Kaggle non ancora eseguito**. È un supplemento della Task 15, non la Task 16 originale. Notebook `notebooks/15b_roadmap_gate_bottleneck_confirmation.ipynb`. Prerequisito: report originale Task 15 SHA-256 `6c4144caf95785576bdc9a786eeecd43e4da7151147999287d7a65614f3c7bc3`, codice `c4a91c79f853225840fe548e1a4aab170539dab3`. L'artefatto precedente è usato per provenienza e confronto storico, **non** per training, selezione o riapertura del suo sealed.

## Ipotesi e contrasti fattoriali

La Task 15 ha favorito V+gate appresi con formula analitica della corrente, ma m/h appresi erano molto meno precisi del probe che integra i gate con le equazioni note lungo V predetto. Separiamo tre possibilità senza una sequenza di run indipendenti: (H1) il budget di ottimizzazione è insufficiente; (H2) la loss di gate non pesa abbastanza; (H3) un gradiente di corrente fisica aiuta i gate nella regione importante. Una interazione H2×H3 è misurabile nella matrice 2×2.

| Braccio | Peso MSE gate | Peso MSE corrente analitica |
|---|---:|---:|
| baseline | 4 | 0 |
| gate_weighted | 16 | 0 |
| current_supervised | 4 | 1 |
| joint | 16 | 1 |

Ogni loss contiene anche MSE dei quattro nodi V predetti, normalizzato per 20 mV. Il termine corrente è normalizzato per `1e-4 mA/cm²`; la corrente deriva **differenziabilmente** da gbar efficace, m/h appresi, V finale predetto ed ECa. Non usa `ica` nativa sfasata. Ogni braccio ha lo stesso MLP SiLU 10→62→62→6 (4.966 parametri), stessa inizializzazione per seed, stessi 10 input causali della Task 15 e gli stessi minibatch. L'integrazione analitica dei gate resta un **probe oracle**: non è un quinto braccio né un candidato alla selezione. Aumentare peso della loss e aggiungere la corrente sono interventi di ottimizzazione, non nuove informazioni in input.

## Ruoli, selezione e budget

Nuovi seed di dati: train 15211 (192 episodi ×16 ms), development 15229 (48×16), sealed 15259 (48×16), controfattuali appaiati 15359 (48×16). Stessa distribuzione Task15: ECa uniforme 100–140 mV, gbar ×0,5/1/1,5, perturbazione iniziale dei gate, maschera off esplicita. I seed di inizializzazione sono 17/29/43, con minibatch 256 e Adam 0,001. Ogni traiettoria arriva a 800 step; checkpoint a 400 e 800 **lungo la stessa traiettoria**. La selezione del seed per ciascun braccio minimizza esclusivamente `m_rmse` development allo step 800; il modello dello step 400 del medesimo seed serve per il contrasto di budget, non per scegliere il seed. Tutti i checkpoint sono salvati con hash prima di generare sealed e controfattuali.

## Misure e decisioni fissate prima del test

Sealed: RMSE m/h, V e corrente one-step, corrente nel quartile attivo e con meccanismo off, violazioni di occupanza, rollout ricorsivo 16 ms V/m/I, probe gate analitico separato. Controfattuali appaiati: RMSE delta-I e delta-V per le sette condizioni con maschera osservabile della Task15. Riportare le metriche continue e la variabilità fra seed development, non soltanto i booleani.

- **H1 budget** passa se la baseline a 800 step riduce RMSE m sealed almeno del 25% rispetto al proprio checkpoint a 400 step e la corrente sealed non peggiora oltre il 10%.
- **Intervento di loss** passa solo se il braccio a 800 step dimezza RMSE m sealed rispetto alla baseline a 800, non peggiora RMSE corrente one-step e RMSE V rollout oltre il 10%, ha rollout finito e non peggiora la mediana RMSE delta-I appaiata. In caso contrario riportare quali componenti migliorano o peggiorano; il gate composito è severo per evitare una falsa soluzione che sacrifica corrente o causalità.
- Confrontare H2, H3 e H2×H3 anche come contrasti quantitativi development/sealed. Un beneficio della loss su corrente non prova che i gate siano identificabili in generale. Se nessun intervento passa, non rilassare soglie dopo aver visto il sealed; formulare l'ipotesi successiva in uno studio separato.

Questo è ancora un riferimento sintetico Ca-HVA+pas a un compartimento. Nessun risultato qui autorizza la Task 16 embedded, Gate C, il teacher multicompartimentale o affermazioni su calcio dinamico/CVode. Il costo GPU è 4 bracci ×3 seed ×800 step, ma tutte le ipotesi condividono un solo notebook, uno stesso input esterno piccolo e un unico download ZIP Blob/base64.
