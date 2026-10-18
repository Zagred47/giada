# 🔬 Task 39d — risultati verificati e prossime ipotesi

## Esito e perimetro

Colab, account principale dell'utente; Tesla T4, PyTorch 2.11.0+cu130.
Revisione scientifica `57a7b252c66c92e16d953f8a7be735cbb4ddf499`.
Kaggle v3 cancellata quando ancora in coda, senza output; nessun doppio training.
Training 1399.11 s (23.32 min), fase training+valutazione 1778.72 s;
36 membri, 12 bracci, 3 seed, 120k update. Preflight numerici/ottimizzatore
validi; freeze e checkpoint SHA verificati localmente; 1872 valutazioni,
20592 misure di canale, tutti finiti e zero violazioni di occupazione.

**NO_GO_TASK40_BLOCKED.** Nessuna conferma indipendente aperta, soglie invariate.
18 stati limitati, u/z appresi, readout economico invariato. V e Ca imposti:
non è ancora una prova di compartimento autonomo, multicompartimento o biforcazioni.
Il presente è un audit descrittivo dei risultati salvati, non un nuovo replay CUDA.

## Confronto con 39c

72 casi development per braccio (12 famiglie × 2 casi × 3 seed), stessi ID.
I seed/casi condivisi non sono esperimenti indipendenti. 39c girava su Kaggle,
PyTorch 2.11.0+cu128; 39d su Colab/cu130, con diverso protocollo di selezione:
il confronto fra task è descrittivo, non un'ablazione causale appaiata.
La baseline congelata mantiene gli stessi 53/72 e nessun caso cambia pass/fail.
Le riproduzioni congelate39c in39d mantengono 41/72 e44/72.

| Candidato | Development superati | Severità mediana* | Severità peggiore* |
| --- | ---: | ---: | ---: |
| 39c wide_i1_s1_d1, migliore conteggio nuovo | 44/72 | 0.731 | 247.92 |
| 39d base000 | 43/72 | 0.580 | 251.14 |
| 39d base011 | 43/72 | 0.664 | 231.75 |
| 39d base101 (scala locale, reset16, ribilanciamento) | 39/72 | 0.553 | 49.46 |
| 39d base110 (scala locale, storia64, pesi fissi) | 42/72 | 0.515 | 35.53 |
| 39d event000 | 43/72 | 0.661 | 52.76 |
| 39d event111 | 25/72 | 1.012 | 13.19 |

*Massimo dei rapporti errore/soglia per gate/corrente/carica; NON include timing.
Severità<1 quindi NON implica passaggio completo. Questi sono punti del profilo
qualità, non una nuova selezione di vincitore su stress.
Tutti i bracci nuovi variano25–43/72; la39d non supera44/72 di39c.
base000 guadagna due casi rispetto a39c-wide e ne perde tre: niente miglioramento
monotono nascosto dal conteggio. Il confronto completo è in result_audit.json.

## Cosa ha funzionato e cosa no

- La scala locale riduce il worst-case in tutti e4 i contesti fattoriali,
  nei3 seed. Nel contrasto preregistrato base001→base101, rapporto delle
  severità mediane 0.782/0.916/0.785; worst-case0.238/0.087/0.179.
  È l'unico contrasto che soddisfa l'intera regola (1/16).
  Il conteggio però scende41→39: guadagno sulle code, non riparazione completa.
- Per esempio base110 porta il worst-case a35.53 da218.17 del suo controllo
  base010. È importante ma ancora35 volte la soglia numerica più critica.
- La storia mirata+64 tick non soddisfa il criterio in nessuno dei4 contesti.
  È un pacchetto copertura+BPTT, non una prova contro ogni uso della memoria;
  64tick sono solo6.4ms, insufficienti a testare un'ipotesi generale sui tempi lenti.
- Il ribilanciamento, da solo, non dà contrasto positivo; la loss eventi non
  ripara il timing. event111 riduce l'estremo ma peggiora il centro e passa25/72
  contro41/72 del suo controllo. Ha3 fallimenti gate development.
- Il bias lineare a pari1220parametri non migliora robustamente e costa di più.
- Il budget resta rilevante: da60k a120k migliorano34/36 membri sullo score
  comune (riduzione mediana48.1%); da90k a120k29/36 (-32.8% mediana).
  29/36 freeze selezionano120k,6 selezionano90k,1 seleziona60k.
  Ciò suggerisce ottimizzazione ancora incompleta, NON garantisce che basti
  aumentare compute o che tutte le metriche migliorino con la stessa loss.

## Attribuzione causale congelata: cosa restringe il campo

Questi oracoli si applicano ai due parent39c, NON ai nuovi checkpoint39d.

| Intervento diagnostico | Parent base | Parent wide |
| --- | ---: | ---: |
| Appreso invariato | 41/72 | 44/72 |
| Solo u corretto | 57/72 | 58/72 |
| Solo z corretto | 44/72 | 48/72 |
| Entrambi corretti | 72/72 | 72/72 |

u è l'equilibrio verso cui tende lo stato; z è la frazione di rilassamento per
passo. L'errore di u spiega molta severità estrema; z contribuisce al timing e
alla memoria. Correggerne uno solo non basta. Entrambi esatti raggiungono72/72:
il contratto è raggiungibile nel modello diagnostico, non c'è qui evidenza di
un'impossibilità del livello di astrazione. Gli oracoli non sono candidati deployabili.

FP64 mantiene gli stessi passaggi di FP32. Le perturbazioni diagnostiche alla
scala epsilon non cambiano gli eventi né il pass/fail nei4 gruppi esaminati:
705 canali eleggibili ciascuno,15/14 timing falliti per parent in entrambe
le precisioni. Non è una misura universale del floor numerico, ma rende
meno plausibile una spiegazione basata sul solo arrotondamento FP32.
Swap gate9 (Nap_Et2.h) riduce timing del parent base15→10, wide14→10; gate16
(SK_E2) aumenta il conteggio wide44→50. Sono effetti condizionali, non cause uniche.

## Problemi rimasti

base000 passa72/72 gate ma solo43/72 qualità completa:20 casi con RMSE corrente,
17 con massimo corrente,14 con carica e14 con timing;9 falliscono solo timing.
Le categorie si sovrappongono. Le famiglie più difficili includono ramps,
same_v_history, slow_recovery, activation_boundary e reversal.
La sola RMSE gate media è un indicatore insufficiente.
base101 riduce carica14→9 rispetto a base000 ma aumenta i casi RMSE20→23;
non vanno confuse riduzione della magnitudine e numero di soglie superate.

La selezione rimane su score globale privo di timing, mentre il contratto finale
è locale e include eventi. La normalizzazione locale per finestra usa inoltre
una scala diversa da quella del valutatore sull'intera traccia. Sono limiti
noti del protocollo, non motivi per cambiare a posteriori il freeze o le soglie.

## Velocità

Il microbenchmark residente batch642/80tick misura circa8.0× rispetto alla
formula vettorizzata per i modelli base,6.55× per il bias lineare, su questa T4.
Include gate e readout per tick, stato resettato, backend CUDA Graph per entrambi,
30 coppie per seed. Non è speedup end-to-end, non confronta una LUT ottimizzata,
non qualifica qualità e non autorizza Task40. Il totale23min contro105min39c
non è un confronto architetturale allineato (implementazione/coorti differenti).

## Memoria storica interrogata

Query effettiva read-only: `SELECT * FROM findings WHERE "Codice stabile"=?`.
Codici/righe e gradienti sono salvati in residual_diagnosis.json.
Task2b: supervisione delle rate risolse identificabilità di h; Task3d:
supporto+prior condizionale risolsero dove aumentare width no; Task26:
gradienti in conflitto osservati, senza causalità stabilita; IE-P6:
copertura mirata migliorò code ma non conteggi, loss temporale peggiorò.
L'analogia utile è distinguere informazione, copertura e obiettivo; non è prova
che la vecchia soluzione si trasferisca automaticamente.
Nel probe FIT39d finale i rapporti norme ausiliarie/traiettoria sono perlopiù
vicini all'unità, ma event111 è3.08/5.13/4.48. I coseni hanno segni misti:
il semplice ribilanciamento non ha eliminato ogni possibile interferenza.

## Proposta successiva: Task39e, non ancora autorizzata/lanciata

Rimanere allo stesso livello di astrazione, mantenere soglie e readout.
Non ricadere in uno screening di architetture indiscriminato.

1. **Precisione delle code di u e allineamento dell'obiettivo.** Confrontare
   loss esistente con una loss ibrida assoluta/relativa bounded o logit-stabile
   per u, più correnti multi-scala, evitando il dominio esclusivo delle correnti
   deboli. Misurare errori u per intervallo e loro sensibilità nelle correnti.
2. **Copertura informativa.** A parità di numero di label e update, anchor
   empirici contro campionamento stratificato V/logCa e stati/bordi; generare
   solo FIT/FIT_EXTRA, non minare errori development. Se l'effetto è nullo con
   copertura verificata, indebolire questa versione dell'ipotesi dati.
3. **Interferenza/parametrizzazione.** Teste u/z separate a budget controllato
   contro trunk condiviso; nessuna formula canonica in inferenza. Probe per
   testa e canale, confronto appaiato costo/qualità. È un'ipotesi, non una
   conclusione dal solo coseno.

Piccolo fattoriale2×2×2,3seed (24membri), più2sidecar ai vertici000/111
per esposizione variabile più lunga (es.512tick) a parità di target-tick:
30membri totali. Il sidecar distingue il limite del pacchetto6.4ms, senza
moltiplicare inutilmente tutta la matrice. Stessa traiettoria con checkpoint
crescenti per mini scaling; budget e costo vanno preflightati prima del run.
Selezione prospettica allineata a peggiori canali/regimi e timing, senza
ritoccare l'evaluator; conservare anche l'antico score per confrontabilità.
Per isolare la sola selezione serviranno metriche a checkpoint preregistrati
su stessi pesi: non chiamare causale un confronto fra checkpoint disponibili
solo per un braccio. Tutti i confronti di nuova selezione vanno dichiarati prima.

Prima del training: oracoli u/z/singoli gate sui NUOVI39d congelati; misurare
curve u/z/errore corrente per regione V/Ca e intervalli di recovery. Questo
aggiorna l'attribuzione senza consumare una conferma indipendente.
Valutare invarianti, blocchi, carica, eventi mancanti, onset/peak/offset,
memoria e worst-case insieme, non soltanto medie. Stress già aperto solo
diagnostico; una vera conferma nuova rimane successiva al freeze.

Questa è una proposta informata da development: implementazione, loss precise,
numeri di campioni, contrasti, budget e soglie interpretative devono ancora
essere preregistrati. Nessuna promessa di riparazione definitiva o scaling illimitato.
