# GIADA roadmap Task 15 — risultato del confronto architetturale

Artefatto utente: `giada_roadmap_task15_current_architecture_comparison.zip`, SHA-256 `86eae6b99dec2653db7a95a833bbc3412916c763baff25a958306bfa18205aa4`; report interno SHA-256 `6c4144caf95785576bdc9a786eeecd43e4da7151147999287d7a65614f3c7bc3`. Revisione codice `c4a91c79f853225840fe548e1a4aab170539dab3`, coerente con la preregistrazione v1. `valid=true`; hash del freeze e di tutti i nove checkpoint verificati; Task14 prerequisito verificato. Report: `sealed_or_counterfactual_used_for_selection=false`. Tre bracci con stessi 10 ingressi, ruoli e indici minibatch per seed; scarto parametri selezionati 2,03%. I seed scelti su development allo step 400 sono A=29, B=43, C=43. Sealed: 48 episodi ×16 ms; controfattuali: 48 ×16 ms.

| Misura sealed | A: V+gate appresi, formula I | B: I diretta | C: I diretta + ausiliari |
|---|---:|---:|---:|
| RMSE I one-step, mA/cm² | 5,44e-5 | 1,32e-4 | 1,33e-4 |
| RMSE I nel quartile attivo, mA/cm² | 1,21e-4 | 2,48e-4 | 2,53e-4 |
| RMSE I, meccanismo off, mA/cm² | 0 | 1,52e-4 | 1,49e-4 |
| RMSE V one-step, mV | 0,329 | N/D | 1,388 |
| RMSE gate m / h one-step | 0,0330 / 0,0146 | N/D | 0,1106 / 0,1245 |
| RMSE V ricorsivo 16 ms, mV | 0,854 | N/D | 3,743 |
| RMSE I ricorsiva 16 ms, mA/cm² | 1,07e-4 | N/D | 2,42e-4 |
| Mediana RMSE delta-I appaiato, mA/cm² | 2,37e-5 | 2,47e-5 | 2,59e-5 |

Il braccio A ha errore di corrente sealed circa 2,42× inferiore a B e 2,44× inferiore a C. I criteri preregistrati di vantaggio B su A e di beneficio C su B sono **entrambi falsi**. A ha il minore errore delta-I in 6 dei 7 interventi, ma questa è una lettura descrittiva: per `initial_m_plus` l'errore delta-I A (6,47e-6) è vicino all'ampiezza RMS del delta del teacher (7,61e-6); per `mechanism_off_visible` è 5,76e-5 contro 1,05e-4. Non equivale a dimostrare fedeltà causale piena. La condizione `gbar_one_half` mostra inoltre RMSE di corrente *modificata* migliore per B/C che per A, benché il delta-I sia migliore per A.

La corrente zero di A con maschera off è garantita dalla formula e da `gbar_effective=0`, **non** una regola appresa. B/C violano spesso quel vincolo senza proiezione esplicita; la proiezione diagnostica riduce ma non annulla il loro errore totale. Il probe interno è importante: integrando analiticamente i gate lungo il V predetto di A, il confronto col teacher dà RMSE m=0,00127 e h=5,35e-6, contro gli errori dei gate **appresi** 0,0330 e 0,0146. Ciò suggerisce che l'apprendimento diretto dei gate è ancora un collo di bottiglia nel braccio A; il probe usa però le equazioni note del teacher e non è un quarto candidato selezionato.

Le mini scaling laws development indicano A in forte miglioramento fra step 200 e 400 (seed29: RMSE I 1,14e-4 → 1,49e-5 mA/cm²), mentre B/C migliorano meno (seed43: B 1,13e-4 → 9,41e-5; C 1,17e-4 → 9,47e-5). Il sealed di A (5,44e-5) è peggiore del suo development (1,49e-5): segnalare questo gap, non inferire saturazione o generalizzazione robusta da un solo split. Servono replica/split aggiuntivi prima di una conclusione quantitativa generale.

**Interpretazione:** nel playground definito, la decomposizione con prior fisico è più apprendibile/accurata per la corrente di una testa diretta a budget simile. Non si può attribuire il vantaggio alla sola architettura: le loss e le informazioni strutturali nella formula differiscono, pur essendo identici gli input numerici. Il successo non prova il teacher multicompartimentale, calcio dinamico, CVode o Gate C. La Task16 della roadmap è conferma embedded congelata; prima di chiamare A un candidato solido occorre giudicare esplicitamente l'errore dei gate e dei delta contro requisiti biologici, o progettare un intervento mirato sui gate senza cambiare retroattivamente la Task15. Il supplemento 11c resta distinto.
