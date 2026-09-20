# GIADA Task 0.4 — emendamento della strumentazione

La prima esecuzione su Kaggle si e fermata **prima di produrre un risultato
scientifico**: il runtime NEURON/NMODL installato non esporta il blocco privato
`DERIVATIVE states` come funzione HOC pubblica `states_Ca_HVA`.

Questo non e un mismatch tra formule e teacher. E un'assunzione errata sul modo
in cui interrogare il meccanismo compilato.

## Correzione

Il secondo oracle ora usa il percorso runtime supportato da NEURON:

1. crea una sezione con il solo `Ca_HVA`;
2. imposta `gCa_HVAbar = 0`, eliminando qualunque corrente di membrana prodotta
   dal canale;
3. inizializza e ripristina il voltaggio richiesto;
4. assegna lo stato iniziale del gate;
5. esegue un singolo `h.fadvance()` con il `dt` del caso.

In questo modo NEURON esegue realmente `DERIVATIVE states` con `cnexp`, mentre
il voltaggio non possiede dinamica elettrica. L'unica variazione ammessa e
quella intenzionale del teacher a `-27 mV`, dove `rates()` aggiunge
`0.0001 mV` per evitare la singolarita.

## Cosa non cambia

Non sono stati modificati griglia, precisione, tolleranze, regola di
accettazione o numero di fallimenti ammesso. I due percorsi restano
indipendenti e l'oracle NEURON non richiama mai la formula Python.
