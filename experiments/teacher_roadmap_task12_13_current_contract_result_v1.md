# GIADA roadmap Tasks 12/13 — risultato

Artefatto: `giada_roadmap_task12_13_current_contract.zip`, SHA-256 `61a262345b4f70d353ef0f5087bedde15970d72c0c55bd909493b3599c209cd9`; codice `48116b1e3edc4c0360413585abb21f9ae236aaa1`. Entrambi i report validi; Task 11c non eseguita.

## Task 12

48 episodi seed 12059, 768 finestre, modelli Task11 congelati. Corrente analitica da V/m/h predetti, non rete di corrente. RMSE corrente totale `path_full` 1.288694e-6 mA/cm²; `effect_full` 6.153402e-5 mA/cm² (47.75×). Nel quartile attivo: 2.577283e-6 contro 1.230597e-4. RMSE V quasi uguale, 0.13809 contro 0.13843 mV; RMSE m 0.0006845 contro 0.03510. Gli ibridi diagnostici con soli gate predetti indicano che la differenza dipende soprattutto dai gate. Non significa che il full neuron o il rollout siano validati.

## Task 13

16 episodi autentici Task7b con gbar non nullo; 12.800 confronti nativi a dt 0.025ms. La corrente campionata al nuovo istante coincide con V/m/h pre-step: massimo errore 1.30e-18 mA/cm². Tutti post-step: RMSE 4.14e-6 mA/cm² globale e 3.20e-6 sul quartile attivo. La conduttanza non era registrata: G osservata è inferita da I/(V_pre-E_Ca). Il contratto temporale riguarda il fixed-step Ca-HVA+pas, non CVode o neurone completo. Non richiede di cambiare l'interfaccia esterna di 1ms.

## Prossima domanda

Roadmap Task14: variazioni appaiate di gbar, E_Ca, stato iniziale e presenza/maschera. E_Ca non è un input dei checkpoint Task11; il suo controfattuale va trattato come diagnosi di sufficienza dell'ingresso, non come semplice test OOD del medesimo modello.
