# Task41f — audit finale e ipotesi discriminanti

Audit post hoc su pannelli aperti; non conferma indipendente. Freeze UZ64_broad_point e NO-GO invariati. Nessuna soglia o riferimento ufficiale modificato. I tre seed condividono protocolli; non sono repliche biologiche indipendenti. Oracoli, precisioni alternative e ranking diagnostici non autorizzano deployment o Task42.

## outcome

41f completata a60000 aggiornamenti, valid=True, PRECISION_RELAXATION_NO_GO; nessun braccio supera tutti i gate. Backup finale byte-identico al remoto, tutti i checkpoint recuperati finiti. Nessun riaddestramento durante il recupero.

## quality

Nominee UZ64_broad_point: development36/36, risposte autonome41c334/336, stress41c32/36 e vecchi stress41 35/36. Q0 della41e nello stesso retest:334/336,33/36,36/36. Il nominee peggiora quindi rispetto aQ0 su due stress, pur migliorando alcuni errori locali. U128_rare_point arriva336/336 risposte41c ma32/36stress41c e35/36stress41; UZ64_rare_response335/336,34/36,34/36. Nessun vincitore globale.

## regressions

Tutti i nuovi bracci:528/528traiettorie su41 e41c in entrambe modalita;336/336risposte imposte;recurrence72/72development+72/72stress. U128_broad_point introduce1risposta autonoma41fallita; tutti4U64 introducono1risposta storica40bfallita(167/168). Gli altri restano168/168. Identita risolte,introdotte e persistenti sono salvate perogni modello controP110,Q0,H10, non solo i conteggi.

## lineage

Tutti i nuovi41f partono daP110 congelata, non daQ0/H10: una mancata conservazione del progresso41e non equivale automaticamente a dimenticanza durante il fine-tuning diQ0. Esistono comunque regressioni reali rispetto aP110, separate nel confronto percaso.

## numerical

Nei6confronti causali108/122, tutti6controlli riproducono esattamente i fallimentiFP32 originali(4/6o5/6). Mixed-state64 eFP64 portano ciascuno a6/6 anche contro il riferimento ufficialeFP32. Ridurre il batch non cambia il pass/fail. ClampV risolve6/6, clampCa da solo no. Evidenza forte di sensibilita numerica nel feedback diV; precisione diV,gate e readout non ancora isolate. Non generalizzare automaticamente ai nuovi41f o a tutta la suite.

## equilibrium

Sui18replay diagnostici slow_boundary_return: P11013/18,Q0/H1015/18. Sostituire solo u_m(Ca_HVA) con oracolo porta tutti18/18; z_m,u_h,z_h e coefficientiNap non risolvono; u_all o u_z_all portano18/18. Quindi la necessita di z/h per questi fallimenti e contraddetta nel dominio del probe. Oracolo non utilizzabile come candidato: identifica la funzione appresa da migliorare.

## crossing

Il caso lento con offset di riferimento162.2ms ha pendenza locale circa-4.303917e-9mA/cm2/ms. InP110seed17 un errore corrente circa-1.635713e-8mA/cm2 accompagna3.8msdi ritardo; inQ0 restano0.8/0.7/0.8ms. L errore di corrente e minuscolo ma su coda piatta sposta molto il crossing. La relazione locale delta_t circa delta_I/pendenza e diagnostica, non sostituisce la metrica ufficiale. UZ64_rare_response mantiene un errore1.4ms conNRMSEcirca7.05e-5: media bassa non protegge il timing.

## contrasts

4/20contrasti preregistrati superati, tutti U128->UZ64 a pari384pesi attivi, riduzione della severita target nel nuovo development. Nessun contrasto uniforme per sola capacita, supporto rare o loss response. Questo non prova che z sia la causa dei residui lenti: l oracolo indica u_m e il beneficio locale non si trasferisce al GO globale.

## selection

Predicati ranking/valutatore coerenti in tutti i checkpoint. Solo3/36membri scelgono step0, contro25/36in41e; non e un effetto causale isolato del fix, perche cambiano anche dati e architetture. Tutti36/36nuovi development non equivalgono a generalizzazione: i vecchi pannelli rivelano regressioni. Il ranking non ha accesso ai pannelli aperti per scegliere pesi.

## objective

I probe a0 e60000 misurano separatamente effetti finiti delle direzioni point/response sulFIT e holdout, con pesi invariati dopo il probe. Conteggi e delta perognimembro sono nel dettaglio audit. Finestre feedback40tick=4ms; i crossing lenti diagnostici sono intorno157-162ms. E plausibile un disallineamento di orizzonte/obiettivo, non provato soltanto da questa differenza.

## optimization

A60000 tutti36membri hanno loss puntuale di selezione maggiore che a0, con normalizzazione iniziale fissa: il contributo medio u_h sale1.045->215.624, mentre u_m scende2.659->1.071 e u_Nap5.030->1.187. Anche nei checkpoint realmente congelati33/36 peggiorano sia la loss puntuale complessiva sia u_h; mediau_h143.833. Non sono errori assoluti: e un forte indizio di interferenza/precisione di ottimizzazione, non prova causale. Adam usa lr costante0.001. Nei probe finali un passo lungo il gradiente point migliora point36/36 ma peggiora rispostaFIT17/36; un passo response miglioraFIT36/36 ma peggiora holdout10/36. Le direzioni esistono, il trasferimento non e uniforme.

## representation

Le correzioni41f sono basi triangolari locali lineari inV:64nodi o128nodi tra-100e60mV (passo circa2.54o1.26mV). Non sono funzioni globali arbitrarie. Aumentare dati o64->128non garantisce il controllo di piccole differenze/interpolazione/pendenza tra nodi; projectionprobe separa fitting affine daSGD ma non e una prestazione deployabile.

## speed

Confronti stesso hardware/pesi: nominee integrato batch642 circa1.097x nel cost benchmark; execution matrix circa1.098xstep e1.186xwhole-loop. Albatch4096 resta circa0.51x, cioe piu lento della formula. Parita step/whole zero per tutte24celle. Fusione aiuta in alcuni regimi ma non recupera8x nel compartimento. Il vecchio8x resta nel suo scope isolato; nessuna accelerazione qualificata con tutti i gate superati.

## Proposta41g — non avviata

{
  "name": "Task41g proposed: precision attribution and preservation-aware equilibrium repair",
  "launched": false,
  "task42_authorized": false,
  "thresholds_changed": false,
  "frozen_block": "FactorialV/gate/readoutprecision, same official reference; no training; learnedequilibrium and canonical oracles distinctly labelled. Per-head41frollbackablationagainstactualP110parent; no retrospectivepromotion.",
  "training_block": "Core8pairedarms: u_monly/protectedh vs threeequilibriumheads x originalAdam/precision-orientedanneal x ordinarypoint/sensitivity-awarecurrent-crossingobjective. Same18states,fixed0.1ms,noanalyticfallback; zfrozen; startQ0withzeroresidualparity. Fouradditionalpairedsupportarms ontheprotectedheadhalf contrast broad/newFITcrossing-tailcoverage atsamecount/compute. Total12arms,36members. Preservationanchors andno-regressionselectionguard common toall; notclaimedasanisolatedfactor.",
  "representation_block": "Cheap data/capacity projections64/128/C2 onnewFIT+holdout; diagnostic lower-bound only, notselectedonoldtests.",
  "budget": "3pairedseeds,samedrawswithincontrasts,matchedexamples+walltimeandgradientexposure;checkpointsalongonetrainingtrajectory. Count additional preservation examples explicitly and provide matched-compute replaycontrol.",
  "guards": "NewFIT/selectionprotocols;freezeandno-regressionrulebeforeopenedpanels;originalthresholdsunchanged;noaggregate-onlyGO. Compareidentityofresolved/newfailures,allstress/historical/recurrenceandtimingmargins. Precisionpolicychangesareexplicitcandidates,notrescoringthe41f.",
  "speed": "Same optimized execution forformula/model; measure cost separately forprecisionpolicy and correction. Qualitygatesmustallpassbeforequalifiedspeedup.",
  "stop": "If targetlossfailsFIT or nolearnedrepairwithoutregressions, useprobesandminiscalingtoattributeoptimization/capacity/support; noautomaticpromotion orblindbudgetincrease."
}

### precision_optimization_interference

Correggere anche h, gia molto accurato, con passo costante e normalizzazione pertesta introduce rumore/interferenza mentre m migliora.

Test: Ablazioni congelate pertesta dei checkpoint41f contro il vero genitoreP110; poi stessoFIT/seed: u_msolo contro le tretesteu, e Adam0.001contro schedule a precisione crescente. Verificare rawMSE,gradienti,ampiezza degliupdate eparita grafo/eager; baselineh congelata, non formula.

Falsificazione: Ripristinareh non riduce regressioni e il passo piufine non migliora ilFIT/holdout a computeequo; oppure il peggioramento e soltanto la normalizzazione e non influisce suoutput.

### state_roundoff_localization

Le risposte autonome residue sono in parte dominate da accumulo/arrotondamento nel feedback, non da capacita insufficiente della rete.

Test: Probe congelato:2x2x2 precisioneV/gate/readout FP32oFP64, stessi pesi,batch,protocolli e riferimento ufficiale; includereP110,Q0,nominee41f e un controllo canonico. Misurare differenze con segno, errore di risposta e tempi. FP64globale eNumPy solo diagnostici.

Falsificazione: Il residuo resta oltre soglia con aritmetica convergente; oppure il miglioramento deriva soltanto dal cambio di riferimento.

### equilibrium_crossing_precision

La precisione locale dell equilibrio m su code a bassa pendenza e il bersaglio mancante; le correzioni z migliorano medie ma possono compensare il bias sbagliato.

Test: Correggere u_m appreso con loss corrente/crossing su nuove codeFIT, mantenendoz e altri gate congelati. Probecomplementare u-onlyvsu+z su stessi dati; oracolo solo perattribuzione. Errori assoluti con segno e sensibilita della coda oltreNRMSE.

Falsificazione: Una correzioneu_m sufficientemente accurata non migliora il timing, oppure serve z anche con u_m accurato su supporti nuovi.

### functional_preservation

L obiettivo locale non impone conservazione funzionale sui regimi gia corretti e permette regressioni quando si riparano le code.

Test: Ablazioni congelate pertesta; neltraining confrontare testa m sola con h/Nap congelate contro tutte tretesteu. Ancore di conservazione e regola di non regressione comuni, non attribuire loro un effetto causale isolato. Partire daQ0 con residuo zero e testare parita iniziale; P110resta controllo separato. Nessun esempio dai pannelli aperti nelFIT.

Falsificazione: A pari riparazione e dati, proteggere le altre teste non riduce regressioni; oppure impedisce ogni miglioramento per conflitto reale di rappresentazione.

### tail_coverage_horizon

I criteri rare precedenti e le finestre4ms non coprono abbastanza le code che determinano il timing finale.

Test: Supporto ampio contro arricchimento da pendenza/crossing dei soli nuoviFIT, stesso numeroesempi e compute; replay multiorizzonte breve/medio/lungo comune. Mini scaling dati1024/4096/16384inproiezione, budget crescenti dalla medesima traiettoria; separare mancanza dati da ottimizzazione.

Falsificazione: Copertura effettiva e orizzonte aumentano ma errori e curve non migliorano; il limite resta in funzione/aritmetica o obiettivo.

### interpolation_bias

La base lineare impone un errore tra nodi o di pendenza che il training medio non elimina abbastanza nei regimi sensibili.

Test: Piccolo probe su soliFIT/holdout indipendenti: basi lineari64/128 e cubiche a capacitaallineata, fitting diretto degli stessi target appresi; mappa errore tra nodi e crossing. Non promuovere una variante sul vecchio test; usare il probe per decidere se ha senso aumentare la capacita neltraining.

Falsificazione: Il floor affine e gia inferiore al livello necessario eSGDlo raggiunge; oppure la base piuliscia non riduce errore/crossing a parita di budget.
