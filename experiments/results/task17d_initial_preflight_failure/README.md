# Task 17d — primo run Kaggle, arresto tecnico

Archivio ricevuto: `giada_task17d_native_solver_54a6aed_d66dd637.zip`; SHA-256 `5606e296ce66e92a003181175ce219efa0eaf1ea60cab103e143a4f291dec0d3445`. Revisione eseguita: `54a6aed3672c9cec4fa5b6c695d5f0d378919531`.

L'archivio contiene `failure_report.json`, `process.log`, `process_status.json` e `last_phase.json`, ma nessun report scientifico finale. Il sottoprocesso termina con codice 1 dopo i trial `default`, `tight` e `ultra`: il preflight di `calcium_scaled` confrontava `atolscale=0.0001` con il valore riletto da NEURON `0.00009999999747378752` usando una tolleranza relativa di `1e-10`. La discrepanza è la normale rappresentazione float32 del parametro, non una modifica biologica né una mancata applicazione della policy.

Decisione: run tecnicamente incompleto e scientificamente non interpretabile. Le ipotesi preregistrate restano aperte; nessun GO/NO-GO sul meccanismo del solver. La correzione usa una tolleranza di lettura `1e-6` per le sole scale CVode e verifica tutte le policy prima dei trial lunghi. Il confronto completo va rieseguito con codice, notebook e configurazione della medesima revisione.
