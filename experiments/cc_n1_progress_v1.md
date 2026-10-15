# GIADA CC-N1 — registro al lancio della versione 4

Data: 15 ottobre 2026. Questo documento registra lo stato **al lancio v4**;
gli eventuali report successivi hanno precedenza sullo stato di avanzamento.

## Contratto e provenienza

CC-01–CC-04 sono il primo notebook del dropshot CC. Protocollo iniziale:
`cc_n1_contract_memory_preregistration_v1.json`; nessun training. Codice e
notebook congelati per ogni versione; teacher `074c4666300a8ad246601dab179a97a6942f0f29`.
Gli artefatti Task35 già noti sono baseline diagnostiche, non nuovi test sealed.

## Verifiche native e interpretazione

Le versioni 1/2 hanno riprodotto esattamente gli stati iniziali ma la mappa
analitica non ha superato il floor Ca in due condizioni fortemente perturbate
(1,156e-6 e 1,541e-6 mM contro 1e-6 mM). La ripetizione con `h.fcurrent()` non
ha cambiato il risultato: l'ipotesi del solo campionamento non è supportata.
I due esiti sono conservati e non vengono promossi retroattivamente.

La v3 ha eseguito **tutte le 288 continuazioni direttamente in NEURON**, usando
gli endpoint nativi come riferimento delle collisioni. I controlli di replica
e trapianto dello stato sono passati; la nuova baseline con sinapsi ha superato
il proprio floor nativo/formula. I limiti numerici sono rimasti invariati.

Nel dominio di interventi testato:

- 60/96 gruppi coda × pannello × corrente futura × orizzonte superano la
  soglia di rilevabilità preregistrata;
- la massima separazione degli endpoint V è **3,26397 mV** a vista iniziale
  V/Ca identica;
- anche dopo una coda comune di V/Ca lunga **200 ms** esiste una separazione
  di **1,47787 mV** nel pannello sodium_x4 con corrente attiva;
- un predittore deterministico che riceve soltanto quella vista identica ha
  errore massimo inevitabile almeno pari a metà della separazione.

Questo dimostra l'insufficienza delle **specifiche viste V/Ca testate**.
Le storie della corrente di clamp non sono incluse nella vista appaiata:
non è una dimostrazione contro ogni possibile storia degli ingressi.
Non identifica una dimensione latente minima e non dimostra ancora che una
particolare rete riesca ad apprendere la memoria richiesta. Le preparazioni
sono interventi controllati sui gate, non stime della loro frequenza fisiologica.

Per N2 la candidatura operativa è una memoria persistente inizializzata una
volta da stato fisico o da un warmup causale da validare separatamente.
Le sinapsi restano inizialmente esplicite. I target nativi hanno priorità sulle
approssimazioni analitiche nei regimi in cui il relativo floor è fallito.

## Accuratezza GPU e benchmark

La v3 ha verificato che il core residente eager concorda con i rollout legacy
su tutti i 32 casi Task35 e tutti i sei checkpoint: massimo scarto V inferiore
a **0,000289 mV**, scarto Ca inferiore a **5,22e-9 mM**.

La compilazione dei nove casi arm × batch è fallita su un indice NumPy
interpretato da Dynamo come simbolico. La v4 sostituisce quel metadato con
costanti Python, usa un test di cattura fullgraph e riusa la matrice nativa
v3 verificata con SHA-256. Riesegue soltanto accuratezza e benchmark CUDA.
Non cambia formule, input, checkpoint o tolleranze.

Al lancio v4: CC-01, CC-03 e CC-04 hanno risultati; **CC-02 resta aperta fino
alla verifica del benchmark compilato**. Nessun modello nuovo è stato
addestrato o promosso; nessuna accelerazione end-to-end è stata dimostrata.

Report v3: `results/cc_n1_v3_kaggle_9b6a581/`.
Lancio v4: `cc_n1_kaggle_launch_v4.json`.

Kaggle: https://www.kaggle.com/code/alessandrobelli/giada-cc-n1-contract-cost-and-memory
