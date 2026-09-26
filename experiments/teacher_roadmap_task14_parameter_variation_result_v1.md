# GIADA roadmap Task 14 — risultato e interpretazione

Artefatto `giada_roadmap_task14_parameter_variation.zip`, SHA-256 `94408bc2234d1634ee0e1456950d4256c212f3413405446b89a89fb51e307353`, revisione `3372465ff37dd8e10b4eaed5f12e73f92dc0f036`. Matrice preregistrata di 9 condizioni, 48 episodi seed 14059, 16 ms, modelli Task11 congelati. `valid=true`, riferimento ECa=120 identico a Task11 (`max_abs_error=0`), nessun training né riselezione. Tutti i gate prestazionali registrati per gbar_half, initial_m_plus, initial_h_minus e mechanism_off_visible passano.

## Accuratezza assoluta

`path_full` baseline: V one-step RMSE 0.0960 mV, m 0.000298, corrente analitica 3.98e-7 mA/cm²; V ricorsivo 16 ms 0.4024 mV. Sotto gbar_half: V 0.1069 mV, corrente 2.51e-7; sotto gbar×1.5 (potenzialmente OOD): V 0.1051 mV, corrente 8.52e-7. Con gate iniziali perturbati: V 0.1150 (m+0.1) e 0.1063 mV (h−0.1). Con canale off visibile: V 0.1354 mV e corrente 0 **per vincolo analitico**, non perché la rete abbia appreso lo zero. `effect_full` ha RMSE V baseline simile (0.1047 mV), ma m 0.01882 e corrente 2.52e-5 mA/cm²: la differenza nella corrente è ancora prevalentemente un problema di gate.

## Fedeltà delle risposte agli interventi

L'errore assoluto stabile non equivale a ricostruire l'effetto causale. Sullo stesso episodio, il teacher cambia V di 0.0951 mV RMS dimezzando gbar; l'errore di `path_full` sul **delta V** è 0.0483 mV (51% dell'ampiezza dell'effetto). Per m0+0.1: delta teacher 0.0621 mV, errore del delta 0.0515 mV (83%). Per h0−0.1: 0.0370 contro 0.0332 mV (90%). Con meccanismo off visibile: 0.1814 contro 0.0991 mV (55%). Questi rapporti sono descrittivi, non gate retroattivi: mostrano dove la risposta differenziale è meno fedele, senza negare i gate preregistrati superati. La metrica del delta è RMSE, non un coefficiente di “percentuale di effetto appreso”.

## Informazione e controlli negativi

I twin one-step ECa100/140 partono dallo stesso stato con identico input numerico per i checkpoint Task11, ma i target V differiscono di 0.01466 mV RMS. Ogni predittore deterministico con quell'input ha un lower bound di almeno 0.00733 mV per il peggior ramo. Il limite d'informazione è reale, ma piccolo rispetto all'errore V baseline 0.0960 mV; non è dimostrato che sia la causa dominante. Le traiettorie complete ECa non hanno input identici dopo il primo passo. Maschera nascosta: controllo negativo deliberato, non fallimento imputabile al modello; la corrente `path_full` erroneamente non nulla (RMSE 5.77e-5 mA/cm²) quando il teacher è off conferma l'importanza di comunicare la presenza/conduttanza efficace al front-end.

## Limiti e conseguenza

Solo Ca-HVA+pas a un compartimento, riferimento sintetico generalizzato per ECa e ancorato a Task11/Task7b; nessun teacher multicompartimentale, calcio dinamico o CVode. Il Gate C della roadmap non è raggiunto. La prossima voce è Task15: confronto informativamente allineato tra gate+formula, corrente diretta e target ausiliari. Includere gli errori sui **delta appaiati** emersi qui, oltre a RMSE assoluto. Task15 non va presentata in anticipo come fix della dinamica di V; il supplemento 11c rimane separato.
