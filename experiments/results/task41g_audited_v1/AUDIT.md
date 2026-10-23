# Task41g — audit, memoria e proposta41h

Pannelli diagnostici gia aperti, non conferma indipendente. Tre seed condividono i protocolli. Soglie e riferimento ufficiale immutati; nessuna promozione post hoc o autorizzazione Task42. Report finale e costo precisione recuperati da membri completi CRC-validi; archivio intermedio SHA/CRC verificato con sette checkpoint finiti. Contenitore ZIP finale completo non recuperato; nessun riavvio o retraining. I probe numerici riguardano P110,Q0 e nominee41f, non dimostrano ancora la riparazione dei nuovi pesi41g.

## outcome

41g conclusa a60000update, worker_exit0, valid=True, PRESERVATION_REPAIR_NO_GO. Nessun braccio supera tutti i gate;0/20contrasti registrati soddisfatti. Nominee congelato M_constant_point_broad. Tutti i sette checkpoint salvati, inclusi Adam e freeze, sono finiti; hash freeze e checkpoint verificati.

## quality

Tutti12nuovi bracci passano development36/36, traiettorie528/528 in41/41c imposte/autonome, risposte imposte336/336, autonome41 336/336, storiche264/264traiettorie e168/168risposte, recurrence72/72development e72/72stress. Nominee41g:334/336autonome41c,36/36stress41,33/36stress41c. Nominee41f rimisurato:334,35,32. GenitoreQ0:334,36,33. Il nominee non migliora questi conteggi rispetto al proprio genitore.

## best_tail

M_anneal_sensitive_tail, migliore diagnostico non selezionato:334/336autonome41c,36/36stress41,35/36stress41c. Sette->tre fallimenti rispetto al nominee41f, ma cinque->tre rispetto al vero genitoreQ0. Il precedente miglior braccio41f aveva34/36stress41c: il progresso rispetto al precedente migliore e un caso, non tre. Nessuna promozione retroattiva.

## residuals

Nel migliore diagnostico restano Nap_Et2:caso122 seed17 NRMSE0.01401474792510271 ecaso108 seed43 0.010064242407679558, limite0.01; non sono due errori di timing. Stress slow_boundary_return caso1 seed29:Ca_HVA NRMSE5.676175333315245e-5 ma offset1.0ms, limite0.3ms. Rispetto a41f si risolveautonomo122seed29 ma compare108seed43, gia fallitoQ0: tenere identita e genitore oltreconteggi.

## precision

Fattoriale congelato2x2x2 su6risposte108/122: P110/Q0 passano4/6FP32,4/6readout64,5/6gate64,6/6V64; nominee41f4/6FP32/readout64/gate64 e6/6V64. Tutte combinazioniV64 passano6/6 contro lo stesso riferimento ufficialeFP32. Localizza una leva sufficiente nel percorsoV, non dimostra causa unica o GO generale. CostoV64 sul nominee41g:tempo+18.69%batch642,+8.78%4096 rispetto allo stesso modelloFP32; non soluzione gratis.

## preservation

Tutti8bracciM-only preservano36/36stress41; i4ALL scendono35,34,35,33. ALL_constant_sensitive_broad risolve336/336autonome41c ma perde stress:34/36 e32/36. Questo sostiene protezione selettiva nel dominio, non un impossibile compromesso universale. Ancora di conservazione comune: effetto individuale non isolato.

## projection

Probe least-squares non deployabile:linear64 a4096->16384FIT medianaMSE9.497e-7->5.920e-7;linear128 a4096 3.874e-7 ma worstmax0.01302 contro0.01227linear64. A16384linear128 mediana3.657e-7, worst0.01049. Quindi capacita/dati migliorano medie ma non monotonicamente estremi. smoothC2 non domina linear: a128/16384 mediana5.387e-7 eworst0.01230. Non giustifica uno sweep indiscriminato di reti o basi lisce.

## dose

Il codice applica il gradiente ausiliario corrente/crossing soltanto ogni256update, con norma target0.1 del point nella singola slot. A60k:3,840,000esempiatomici contro7488esempitail e936finestrefeedback. Il rapporto0.1/256 circa0.00039 descrive solo la dose nominale perupdate, NON il contributo effettivo Adam. Nei probe molte direzioni quasiortogonali o negative; a60k il cap1e6 riduce il rapporto minimo a0.003668 sulle slot. Riduzione del peso effettivo e un ipotesi da intervenire, non causa dimostrata.

## speed

Rimisura appaiata nello stesso run: formula/candidato integrated642 nominee41f1.111x -> nominee41g0.922x (circa20.5%piu tempo del41f);4096 .509x->.431x. Whole-loop642 nominee41g1.086x controstep.903x;4096whole.441x. Nessuno speedup qualificato. Nel sorgente41g si aggiunge un residuo aQ0 con nuova feature64 e bmmFP64; M-only calcola comunque tre output nella routineU64, due conpesi nulli. Q0 ha gia duecorrezioni con riduzioniFP64. Sono opportunita di eliminare lavoro duplicato, non attribuzione profilata del rallentamento.

## Query storiche riproducibili

SQL e risultati integrali in memory_queries.sql/json; join sulle catene claim→evidence→finding e findings→decisioni.

### numeric_feedback

Separare accuratezza del teacher/solver, precisione del feedback ed errore appreso prima di investire training. 17d riguardava tolleranze native, non identico problemaFP32;41e/f e41g sono un nesso sperimentale piu diretto.

`findings-giada-task17d-result-voltage-dominance-limited-v1`, `findings-giada-task17db-result-attribution-v1`, `findings-giada-task41e-result-reference-v1`, `findings-giada-task41f-result-numerical-v1`

### tail_not_memory

Sintomo temporale non identifica parametro:Ca_LVAst richiedeva z nelpacchetto40b,qui loracolo41f indica u_mCaHVA. Ripetere swapu/z/h sul nuovo checkpoint; non estendere automaticamente il vecchio oracolo.

`findings-giada-task40b-result-ablation-v1`, `lessons-giada-task41-audit-fast-gate-slow-tail-v1`, `findings-giada-task41f-result-equilibrium-v1`

### learnability

Piccola correzione supervisionata e padre congelato sbloccavano72/72; loss globale minore e modello piu libero non garantivano timing. Mantenere genitoreQ0 e gate gia corretti, misurare floor del fit e apprendimento effettivo; non dichiarare ripetuta la soluzione39j.

`findings-giada-task39j-result-sufficient_package-v1`, `findings-giada-task39j-result-freeze_contrast-v1`, `findings-giada-task39j-result-registered_contrasts-v1`

### objective_exposure

In39f il termine temporale dominava durante training; in41g la dose e invece rada. Va misurato peso effettivo nel tempo, non solo presenza della loss. Problemi analoghi di condizionamento, con direzioni opposte: non copiare i pesi di39f.

`lessons-giada-task39f-result-temporal-v1`, `findings-giada-ie-p6-outcome-v1`, `lessons-giada-task41d-result-gradient_balance-v1`

### execution

Stessa funzione e stessi pesi possono costare diversamente con riuso/fusione e granularita; pochi parametri non significano poche operazioni. Benchmarkpaired della composizione prima/dopo eliminazione duplicazioni,convalida numerica e nessun nuovo fallbackanalitico.

`lessons-giada-task41d-result-cost-v1`, `findings-giada-task41f-result-speed-v1`, `findings-giada-ie-p2-opportunity-v1`

### selection_gap

Aggregati dominati da canali invariati possono nascondere riparazioni; development perfetto non implica trasferimento. Nuovi development con code/piatti e risposte minuscole, mantenere gateglobali. Non scegliere il migliore guardando41c.

`lessons-giada-task41d-result-selection-v1`, `findings-giada-task40b-result-contrast-v1`, `findings-giada-task41f-result-selection-v1`

## Ipotesi da distinguere

### voltage_roundoff (alta)

I due residui autonomi hanno un contributo dominante dal percorso numerico del voltaggio, non dalla sola capacita ionica.

Test: CheckpointQ0,nominee41g,besttail41g congelati:VFP32 vsVFP64, rete/readout invariati, riferimento ufficiale identico; poi varianteFP32 numericamente stabile della stessaequazione solo se verificata controdiagnosticaFP64. Tutti i pannelli e costo,non solo6casi.

Falsificazione: V64 non ripara i nuovi checkpoint oppure introduce altri fallimenti; varianteFP32 resta oltrelimite.

### crossing_floor (alta)

Il residuoCaHVA e compatibile con bias locale dellequilibrio amplificato dalla piccola pendenza al crossing.

Test: Swapu_m/z_m/h congelati sul nuovocandidato,erroreconsegno/slope/crossing; sweep continuo inV,storiainiziale,durata su dati nuovi; oracoli solo diagnostici. Non rimuovere eventi piccoli.

Falsificazione: u_m accurato non risolve e il residuo e attribuito adaltroblocco; oppure precisione aritmetica sola risolve.

### effective_objective_dose (medio-alta)

La loss sensibile ha troppo poca esposizione rispetto al point per controllare gli eventi tardivi.

Test: A pari dati,slotcompute,Adam,schedule egradientratio:aux ogni256 vs16update,calcolando slotcomuni ma disabilitando ilgradiente nelcontrollo. Loggarecoseni,norme,clipping,contributi cumulativi eprogressi ai checkpoint. Stessa loss e nessunsoglia cambiata.

Falsificazione: Dose verificabilmente aumentata non migliora crossing/holdout o causa regressioni; lacausa ealtrove.

### data_vs_capacity_vs_optimizer (media)

Il floor locale puo dipendere sia dal supporto raro sia dalla risoluzione, mentreSGD puo non raggiungere il floor rappresentabile.

Test: FattoriFIT4096/16384 ebase64/128,stesse distribuzioni e minibatchaccoppiati,stessoupdatebudget. Mini scaling sullastessatraiettoria; fitting diretto come sidecardiagnostico conbound/targetdichiarati pernonconfondere floor eottimizzazione.

Falsificazione: Nessunbeneficioaumentando dati/base eSGDgia ragggiungefloor; oppure migliorano solo medie ma non gli estremi rilevanti.

### composed_execution_cost (alta)

Una quota del rallentamento deriva da feature/riduzioni/teste duplicate nella catena di correttori, non dal livello di astrazione.

Test: Stessipesi legacy vs esecuzionecompatta:riuso feature,solo testeattive,evitare duplicazionepadre,step/wholeloop;profilare tempo/memoria perblocco a1/642/4096. Conservare tanh separate:non sommare ingenuamente pesi dietrononlinearita. MisurareFP64accumulation separatamente.

Falsificazione: Duplicazioni eliminate conparita verificata ma nessunrisparmio; bottleneck osservato altrove.

### selection_coverage (media)

Il nuovo development/score di selezione non discrimina abbastanza i regimi che decidono il pass globale.

Test: Nuovi protocolli e seed,code fino1200ms,risposte minuscole con margine/slope registrati;score e guardie globali congelati prima deltraining. Testare correlazionerank con erroriperregime come diagnostica,non reranking sulvecchiotest.

Falsificazione: Nuovosupporto copre regimi ma selezionenontrasferisce; sensibilita/floor/varianza spiegano divergenza.

## Proposta prossima task

Otto bracci × tre seed: base64/128 × FIT4096/16384 × dose ausiliaria ogni256/16update, con padreQ0 congelato e sola correzione mCaHVA. Sidecar precisione e fusione sugli stessi pesi. Nessun lancio effettuato; preregistrazione esecutiva da finalizzare e approvare.
