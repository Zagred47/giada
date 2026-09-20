# GIADA Task 0.4 — preregistrazione del doppio oracle

## Stato

Il contratto è **preregistrato**, ma l'oracle NEURON autentico non è ancora stato eseguito. Il runtime Python locale di Codex non dispone di una wheel NEURON compatibile; il notebook Linux/Kaggle è quindi il percorso canonico di esecuzione.

## I due percorsi indipendenti

1. **Formula estratta dal `.mod`**: interpreta in ordine le assegnazioni di `PROCEDURE rates` e applica l'update esponenziale esatto.
2. **NEURON isolato**: compila `Ca_HVA.mod` e invoca `states_Ca_HVA` sul meccanismo autentico, senza integrare la membrana e senza richiamare la formula Python.

## Griglia e soglia

- gate: `m`, `h`;
- voltaggi: ogni intero da −120 a 60 mV, più −27.0001, −27 e −26.9999 mV;
- stati iniziali: 0, 0.1, 0.5, 0.9, 1;
- passi: 0.025, 0.1, 1 e 2 ms;
- precisione: float64;
- accettazione per ogni caso: `|errore| <= 1e-10 + 1e-10 * |riferimento|`;
- fallimenti ammessi: zero.

Il punto esatto `−27 mV` è incluso deliberatamente perché il teacher applica una perturbazione di `+0.0001 mV` per evitare la singolarità numerica di `mAlpha`.

## Regola di arresto

Qualunque mismatch blocca l'accettazione del dataset atomico e la Task 1. Non è consentito allargare retroattivamente la tolleranza: prima si devono controllare parsing della formula, semantica NMODL, singolarità, invocazione del metodo `cnexp` e precisione.
