# GIADA roadmap Tasks 12 + 13 — protocollo v1

Stato: preregistrato, **risultati Kaggle non ancora acquisiti**. Le due task condividono `notebooks/12_13_roadmap_current_and_timing.ipynb`, ma hanno report e criteri separati. Il supplemento 11c resta futuro, non è un braccio implicito.

## Task 12 — corrente analitica

- Congelare i checkpoint `path_full` ed `effect_full` del risultato Task 11 verificato: hash del freeze e di tutti i checkpoint, nessun training o riselezione.
- Generare 48 episodi indipendenti del riferimento Ca-HVA+pas con seed `12059`, orizzonte 16 ms, ingresso pianificato identico per i due bracci. Il riferimento numerico è quello della Task 11, ancorato ai 24 episodi autentici della Task 7b; non è un nuovo test NEURON multicompartimentale.
- Calcolare `G_Ca = gbar·m²·h` e `I_Ca = G_Ca·(V-E_Ca)` con `E_Ca=120 mV`. La corrente è una grandezza analitica, non un layer appreso; segno negativo = corrente entrante.
- Per ogni modello confrontare: tutto predetto; soli gate predetti con V teacher; solo V predetto con gate teacher. Reportare RMSE/MAE corrente in mA/cm², errori di V/m/h, su tutte le finestre e sul quartile superiore di `|I_teacher|` non nullo. I due ibridi sono **oracle diagnostici**, non input legittimi per rollout o selezione.
- L'endpoint target è corrente analitica calcolata dallo stato teacher alla *stessa* frontiera del millisecondo. Non confrontarla ingenuamente con `ica` campionata nella stessa riga NEURON: l'allineamento è oggetto della Task 13.
- Gate: integrità artifact, finitezza, nessuna selezione su nuovi episodi. Non preregistriamo una soglia arbitraria di RMSE come prova di sufficienza fisiologica; la decisione è comparativa e limitata al playground.

## Task 13 — corrente e conduttanza nella cronologia NEURON

- Verificare gli hash degli episodi nativi Task 7b. Usare solo episodi con `gbar>0` per identificare il timing; i `gbar=0` sono controlli nulli.
- Per ogni step nativo di 0.025 ms confrontare `ica(t+dt)` con quattro calcoli: `(V,m,h)` pre; tutti post; gate pre/V post; gate post/V pre. Reportare RMSE e massimo errore assoluto sull'intera traccia e nel quartile superiore di `|ica|` per episodio.
- Inferire `G_obs = ica/(V_pre-E_Ca)` solo quando `|V_pre-E_Ca|>1 mV`, confrontandolo con `gbar·m_pre²·h_pre` e post. **G non è stato registrato direttamente**; non chiamarlo misura indipendente.
- Gate: il vincitore nell'insieme attivo deve essere pre-all e l'errore massimo pre-all sull'intero campione inferiore a `1e-10 mA/cm²`; in caso contrario fermarsi e investigare, senza reinterpretare il timing post hoc.
- L'affermazione è circoscritta a Ca-HVA+pas, fixed-step Task 7b `.025 ms`; non dimostra da sola la semantica di CVode, altri meccanismi o teacher completo.

## Provenienza, output e decisione

Input: `giada_roadmap_task11_causal_operator.zip` e `giada_cahva_active_closed_loop_microcanary.zip` (o cartelle Kaggle estratte equivalenti). Output: `task12_current_report.json`, `task13_timing_report.json`, `final_report.json`, ZIP con download Blob/base64 concordato. Nessun dataset HayFlow da gigabyte.

L'approfondimento **11c** su parametrizzazione, identificabilità dell'effetto integrato, curva budget e costo reale GPU è registrato come lavoro futuro. Non blocca 12/13 e non ne viene presentato il risultato come già acquisito.
