# ⚡ GIADA — IE-P2: esito e priorità computazionali

**Completato nella versione Kaggle 1**, codice
`b7c2f3cd141b009ef7dd1eb31818503cd378db05`, Tesla T4/FP32.
[Notebook eseguito](https://www.kaggle.com/code/alessandrobelli/giada-ie-p2-residual-block-and-execution-cost).
54 celle temporali, 72 preflight numerici superati, nessun errore CUDA Graph.
Tre test locali precedenti al run; nessun training o candidato promosso.

## 📊 Batch principale: 642 per seed, tre copie vettorizzate

Mediane wall-time per chiamata dell'ensemble, in millisecondi.

| Variante | Compilato ordinario | CUDA Graph |
|---|---:|---:|
| Full, ABI strumentato | 0,433645 | 0,102611 |
| Gate in cache | 0,375497 | 0,084677 |
| Readout conduttanze in cache | 0,267143 | 0,033613 |
| Intero prefisso in cache | 0,134182 | 0,010939 |
| Voltaggio finale in cache | 0,416867 | 0,069981 |
| Prefisso e voltaggio in cache | 0,025112 | 0,003883 |
| Vecchio ABI a tre output | 0,421304 | 0,101382 |

Risparmi mediani **appaiati** sul full nella stessa modalità:

| Intervento | Compilato ordinario | CUDA Graph |
|---|---:|---:|
| Gate in cache | 13,80% | 17,42% |
| Readout in cache | 39,34% | 67,15% |
| Prefisso in cache | 69,08% | 89,30% |
| Voltaggio in cache | 3,70% | 31,74% |

Il replay del full riduce il tempo del 76,34% (bootstrap entro-run
75,96–76,73%), circa 4,2× nel microbenchmark. Il criterio H_dispatch è
superato anche a batch1/4096, con riduzioni 83,86%/70,21%.
L'ABI strumentato differisce dal vecchio solo di circa 2,01% (ordinary)
e 1,07% (graph) nel confronto appaiato, sotto il limite prospettico 10%.

## 🔎 Che cosa sostengono i contrasti

1. **L'esecuzione conta molto:** un margine importante emerge senza cambiare
   le equazioni. Un futuro modello deve essere confrontato con questa baseline
   di esecuzione competitiva, oltre che con il vecchio ibrido.
2. **Il readout merita una verifica dedicata:** l'intervento conserva gate e
   Ca/SK, sostituendo soltanto le somme finali delle conduttanze e gE. Il
   risparmio resta grande anche con replay. Non significa che moltiplicazioni
   e somme siano intrinsecamente costose: possono pesare organizzazione dei
   tensori, materializzazioni e codice generato. Un controllo algebraico
   vettorizzato è necessario prima di attribuire quel margine al deep learning.
3. **Il prefisso ionico congiunto è un bersaglio plausibile:** supera il
   criterio diagnostico del 30% in entrambe le modalità. Comprende gate,
   Ca/SK e readout; il contrasto non separa ancora tutti questi contributi.
4. **Il solver non è escluso:** il suo contrasto supera il 30% solo nel
   replay al batch principale e a4096. La priorità dipende dall'esecuzione;
   non c'è una prova che sia sempre il collo di bottiglia dominante.

Le frazioni non sono una torta additiva: nel replay prefisso+solver superano
il 100% se sommati. Fusione e dipendenze cambiano con l'intervento; non è
lecito trasformare questi controfattuali in percentuali fisiche disgiunte.

## 🧪 Qualità numerica e limiti

Massimi scarti compilato/eager e graph/eager sugli output completi:
V 1,526e-5 mV; Ca 1,164e-10 mM; gate 1,192e-7;
g_sum 1,455e-11 S/cm²; gE_sum 6,985e-10 mA/cm².
Il confronto preserva l'ordine Ca→SK. Sono confronti one-step su stati
storici aperti, non una nuova conferma indipendente dei comportamenti neurali.

CUDA Graph riproduce qui operazioni con input fissi e buffer residenti.
Il guadagno include costi di invocazione/allocazione eliminati: non certifica
già un rollout operativo con feedback, ingressi aggiornati, eventi e RNG.
Le cache sono oracle non causali e nessuna rete ha realizzato quei risparmi.
Tutti i campioni e gli scarti sono in `results/ie_p2_kaggle_v1/`.
Le mediane appaiate di cache e replay sono ricalcolate indipendentemente
prima della registrazione nel mirror SQLite.

## 🧭 Prossima priorità proposta

Il confronto successivo deve mantenere solver e sinapsi e rendere più
competitiva la **porta ionica fino alle conduttanze**, con due controlli:
stesse equazioni con readout algebraico vettorizzato e futuro operatore
appreso con stato/interfaccia sufficienti. Applicare la stessa modalità
di esecuzione a tutti i bracci; non confrontare una rete ottimizzata con
formule eseguite peggio.

Prima di quel training, completare IE-03 con un budget di latenza sulla
baseline competitiva e IE-04/05 con stato, ordine causale e supporto del
playground isolato. Il limite del vecchio readout è un'opportunità da
verificare, non un motivo per approssimare formule semplici a tutti i costi.
IE-02 globale resta aperta per eventi/RNG host, trasferimenti e simulatore
operativo. Nessun kernel custom, RunPod/OpenEvolve o Task36 avviato.
