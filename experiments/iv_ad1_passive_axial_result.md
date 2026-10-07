# GIADA IV-A1/A2 → IV-D1: risultato del prerequisito passivo

La verifica indipendente dell'interfaccia a due compartimenti passivi è **passata** nella v3 (`c895e38`). Questo apre l'ingresso alla Task 36, che riguarda invece due compartimenti **attivi** e resta un esperimento separato.

| Revisione | Esito | Lezione |
| --- | --- | --- |
| v1 | NO-GO IV-A1/A2; IV-D1 non aperto | Il criterio a 0,1 ms confondeva lo scarto atteso dell'Euler implicito rispetto alla soluzione continua con una discrepanza NEURON/solver, che era nell'ordine di 1e-12 mV. |
| v2 | IV-A1/A2 passati; NO-GO IV-D1 | Dodici identità NEURON/solver passate, ma in due geometrie asimmetriche il controllo detto “simmetrico” applicava la stessa corrente assoluta a superfici diverse. Lo scarto di 0,0308–0,0368 mV era atteso per quella scelta di input. |
| v3 | IV-A1/A2 e IV-D1 passati | Correnti simmetriche a pari densità, su tre geometrie nuove e dodici condizioni. Tutti i gate numerici e fisici passati. |

Nel pannello v3, il massimo errore di tensione fra NEURON e solver indipendente è 1,279e-12 mV. Il controllo simmetrico a pari densità ha differenza fra i siti pari a zero nelle tre geometrie. Il controllo disconnesso, i bilanci di corrente, l'equilibrio e la convergenza temporale passano. L'errore dell'equilibrio nativo è 7,67e-9 mV.

I tre report sono versionati in `experiments/results/iv_ad1_v1_final_report.json`, `iv_ad1_v2_final_report.json` e `iv_ad1_v3_final_report.json`. I NO-GO precedenti non sono stati sovrascritti o reinterpretati come successi. La v3 ha una preregistrazione distinta e usa geometrie nuove. Non sono stati testati canali attivi, calcio, sinapsi, modelli appresi, alberi più grandi, cellula intera o speedup.
