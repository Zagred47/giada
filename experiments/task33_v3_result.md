# Task 33 v3 — feedback sinaptico osservabile confermato

La v3 prospettica è stata eseguita su [Kaggle](https://www.kaggle.com/code/alessandrobelli/giada-task33-synaptic-feedback) alla revisione GIADA `21e3c35e883403618a34f17c86a7257f8c0241d2`, con teacher canonico `074c4666300a8ad246601dab179a97a6942f0f29` e NEURON 8.2.7. Il [report integrale](results/task33_v3_final_report.json) è valido e dichiara `task33_passed=true`.

La calibrazione ha superato i gate prima dell'apertura dei 16 casi nuovi. Nei casi confermativi: 88 rilasci realizzati su 256 eventi prescritti; differenza massima del voltaggio causata dalle sinapsi 15,994 mV; shadow di rilascio, plasticità, RNG, stati recettoriali e corrente superato; quattro controlli negativi superati. Il floor formula–NEURON è ammissibile: RMSE V aggregato 0,000366 mV, massimo per caso 0,001193 mV, peggior RMSE del calcio 7,703×10⁻⁹ mM. La modifica di fase sinaptica preregistrata è dunque confermata su seed e schedule diversi da quelli della diagnosi v2.

Solo dopo il floor sono stati giudicati i sei checkpoint di Task 32, senza nuovo training. Tutti passano le soglie congelate su V e calcio per i 40 ms nel singolo compartimento attivo con quattro sinapsi interamente osservate. RMSE V aggregato per seed 17/29/43: famiglia `independent` 0,001056 / 0,001030 / 0,000885 mV; `shared_heads` 0,002237 / 0,001230 / 0,002224 mV. Non è stato usato come input il rilascio futuro del teacher.

**Decisione:** Task 33 confermata soltanto nel dominio preregistrato. Questo risultato non dimostra ancora generalizzazione a ingressi sinaptici non osservati, più compartimenti, accoppiamento assiale, cellula completa o speedup. Task 34, sui probe privilegiati, resta un esperimento distinto.
