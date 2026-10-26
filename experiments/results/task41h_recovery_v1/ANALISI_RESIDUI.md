# Task 41h — risultati, sei fallimenti e proposta successiva

Analisi post hoc, non conferma indipendente. I pannelli 41/41c già aperti e il development corrente sono ora diagnostici. I tre seed condividono protocolli: sei fallimenti sono sei valutazioni caso/seed, non sei meccanismi indipendenti. Soglie, riferimento, candidato congelato 41h e risultati primari sono invariati. Task 42 non autorizzata. I probe oracle CPU non qualificano il modello su CUDA e non sono un fallback analitico selezionabile. DB e backup restano locali; nessun nuovo esperimento avviato.

## Bilancio

Valutazione 41h completa e tecnicamente valida; PATH_EXPOSURE_REPAIR_NO_GO. Candidato H0_D256_N4096_B128. Sugli stessi pannelli correnti: 41g ufficiale 8 fallimenti, 41h 6; il migliore controllo 41g post hoc ne ha anch'esso 6. Sui soli pannelli precedentemente aperti: 5→3 rispetto al controllo ufficiale, ma anche il migliore 41g ne ha 3, con un diverso compromesso. Tre fallimenti del nuovo development sono condivisi dai due controlli G. Non è un miglioramento completo e non autorizza Task 42.

## Qualità completa

Candidato: development nuovo 33/36; traiettorie aperte 41 e 41c 528/528 sia imposte sia autonome; risposte 41 336/336, risposte 41c imposte 336/336 e autonome 335/336; stress 41 36/36 e stress 41c 34/36. Regressioni storiche: 264/264 traiettorie e 168/168 risposte in entrambe le modalità. Ricorrenza: 72/72 in ciascun ruolo. Checkpoint originali finiti e invariati; soglie invariate.

## Raggiungibilità

Codice e probe sui checkpoint: è addestrabile soltanto il residuo dell'equilibrio Ca_HVA.m. Gli altri 17 coefficienti u e tutti gli z sono identici al parent congelato sugli input development. I tre fallimenti del caso 8 a V/Ca imposti riguardano Nap_Et2: modificare Ca_HVA non può cambiarli. La protezione delle componenti riuscite va conservata, ma non può rendere irraggiungibile una componente che fallisce.

## Nuovo probe Nap

Il replay offline CPU/NumPy riproduce i fallimenti GPU del caso 8: offset 2,1/1,0/1,4 ms. Sostituire solo l'equilibrio Nap.m con il valore canonico porta tutti a 0,1 ms e fa passare il canale Nap, anche quantizzando i coefficienti e svolgendo la ricorrenza in FP32. Sostituire solo Nap.h, solo z, oppure usare stato FP64 con gli stessi coefficienti appresi FP32 lascia gli offset a 2,1/1,0/1,4 ms. Sostituire u e z porta a zero. Non emerge la necessità di FP64 per questo probe imposto; ciò non qualifica l'intero sistema accoppiato. Sono controlli diagnostici su dati aperti: il verdetto GPU originale resta NO-GO.

## Contrasti diagnostici

Tutti i 14 contrasti registrati hanno rapporto 1,0. Il massimo della severità Ca_HVA/Nap è dominato dagli offset di Nap congelato: 7,0/3,3333/4,6667 per seed. Questa statistica non può distinguere i miglioramenti Ca in questo pannello. I contrasti post hoc limitati a Ca sono piccoli e misti: non dimostrano una riparazione universale nascosta. La selezione originale usa anche criteri ausiliari successivi e non viene modificata. Il mancato contrasto globale non prova che storia, dati o dose abbiano effetto nullo.

## Due casi Ca

Restano due fallimenti slow_boundary_return: offset Ca_HVA di 0,4 ms per seed 17/29 contro limite 0,3 ms; NRMSE 5,57845e-5/4,85992e-5. L'oracle GPU u_m esatto porta lo stress a 36/36; z_m e h restano a 34/36. I due bracci B128 raggiungono 34/36 contro 33/36 degli otto B64: beneficio condizionato, non prova fattoriale completa della capacità. Nel probe locale [-50,-35] mV il residuo necessario massimo è 0,001462, contro bound 0,1, senza saturazione. Aumentare quel bound non è supportato da questi dati.

## Loss e osservabile

La loss ausiliaria della coda usa un crossing al 10% rispetto alla corrente stazionaria asintotica dopo il prefisso. Il valutatore ufficiale usa il 10% della massima deviazione dal primo campione di riferimento sull'intero percorso. Sono osservabili differenti: aumentare la dose della prima loss non garantisce il miglioramento del secondo crossing. Occorrono label e margini del crossing ufficiale su nuovi percorsi FIT, mantenendo invariato il valutatore.

## Risposta autonoma

Resta una risposta autonoma: caso 108, seed 43, Nap dopo blocco K_Pst durante irregular_train, NRMSE 0,0100642424 > 0,01. Ricalcolare differenza e riduzione dalle tracce salvate in FP64 dà 0,0100642449: l'arrotondamento della sola metrica non basta a spiegare il fallimento. Su GPU stato V64 + solver64 insieme chiudono 336/336 risposte; stato64 solo 335, solver64 solo 334, FP32 ricentrato 335. Il percorso numerico congiunto è una leva sufficiente su questo pannello, non prova di causa unica né qualifica su tutti i pannelli.

## Costo

Nel benchmark standard appaiato a 642 siti per seed (1926 effettivi), H richiede circa 15,1% di tempo ionico e 8,7% di tempo integrato in più del controllo G ufficiale. Nel ciclo accoppiato completo a 642 il rapporto formula/H è circa 1,0178: quasi parità, non il vecchio speedup 8×. Le righe provengono da Colab/Kaggle T4: usare confronti appaiati entro la stessa riga, non tempi assoluti tra VM. La modalità compatta modifica solo l'ultimo correttore; gli antenati costruiscono ancora feature, riduzioni FP64 e teste nulle. A 642 il risparmio compatto è circa 0,5%; a 4096 è circa 25,5%, ma resta circa 1,98 volte il tempo delle formule. Serve profilare l'intera catena.

## Che cosa insegnano le query

Le query storiche producono analogie e ipotesi, non riparazioni garantite. Task 39j e 40b avvertivano già che una severità aggregata può essere dominata da canali invariati. La 41h ha ripetuto questa limitazione diagnostica. Il prossimo preflight deve verificare che ogni errore bersaglio sia raggiungibile dai parametri addestrabili e che ogni contrasto diagnostico risponda a controlli positivi e negativi. La qualificazione globale resta invariata.

## Proposta: un solo notebook Task 41i

Riprodurre gli oracle correnti su GPU con H/G congelati. Misurare errori u_m e margini Ca/Nap. Verificare maschere addestrabili e visibilità dei controlli positivi/negativi nei contrasti per componente.

Matrice 2×2×2, otto bracci e tre seed:

- Correzione Nap.m congelata/addestrabile.
- Loss ausiliaria attuale/margine del crossing ufficiale.
- Base uniforme/adattata su FIT con 128 coefficienti.

24 membri vettorizzati; stessa funzione parent, percorsi FIT, minibatch, budget e inizializzazione appaiati. Fuori da Ca.m/Nap.m tutti gli u e tutti gli z sono protetti.

Congelare il budget tecnico prima dell'accesso ai risultati di qualità. Checkpoint crescenti lungo un training: aggiornamenti, tick target, tempo, norme dei gradienti, clipping e floor FIT/holdout. Non assumere scaling illimitato.

Politiche numeriche congelate su H e finalisti prospettici, senza altro training.

Testa attiva/catena con feature condivise contro legacy sugli stessi pesi: profiler, qualità completa e timing appaiato. Distillazione rinviata se l'esecuzione esatta non basta.

Richiedere riparazione congiunta Ca/Nap e controllare l'interazione con la politica numerica: singoli oracle positivi non garantiscono un candidato combinato positivo.

Conservare tutte le soglie per caso/seed e la congiunzione globale. Metriche aggiuntive solo per identificare effetti, non dispensare fallimenti. Nuove famiglie FIT/development; niente fitting delle vecchie tracce fallite. Conferma indipendente successiva dopo freeze.

### Nap: equilibrio di attivazione congelato, non necessariamente memoria insufficiente

I tre offset a percorso imposto possono essere riparati migliorando l'equilibrio appreso Nap.m, conservando il suo rilassamento e gli altri gate protetti.

Test: Su nuovi FIT/development ampi e sensibili: Ca-only contro Ca+Nap.m, con residui di equilibrio appresi, stessi input causali, 18 stati e 0,1 ms. Probe per testa e congiunti, ancore di preservazione FIT. Tutti i pannelli globali restano obbligatori.

Falsificazione: Migliorare davvero Nap.m su holdout non riduce gli offset, oppure introduce regressioni globali: il candidato non è una riparazione anche se l'oracle locale funzionava.

Limiti: L'oracle è sufficiente sui tre casi aperti, non prova generalizzazione, learnability o soluzione della risposta autonoma 108.

### Ca: allineare la supervisione al crossing che valutiamo

La differenza tra crossing della coda ausiliaria e crossing transiente ufficiale può lasciare un bias locale che più dati generici o maggiore dose non correggono in modo affidabile.

Test: Stessi percorsi FIT, parametri e budget: loss attuale contro loss di margine/errore firmato basata sul primo campione di riferimento e sulla soglia del percorso completo. Conservare loss e controlli di preservazione. Misurare pendenza e margine del crossing senza modificare il valutatore.

Falsificazione: La nuova loss cambia misurabilmente i gradienti della testa bersaglio ma non migliora margini/offset holdout a compute allineato, oppure la loss precedente ottiene lo stesso risultato.

Limiti: Il disallineamento delle definizioni è osservato; il suo ruolo causale resta da misurare. Non eliminare eventi deboli o piatti né ricentrare a posteriori le predizioni.

### Ca/Nap: risoluzione locale o limite di ottimizzazione

Il residuo può essere un limite della base uniforme oppure dell'ottimizzazione/pesatura. Non emerge la necessità di un bound di ampiezza maggiore.

Test: Base uniforme contro base non uniforme scelta solo su FIT, sempre 128 coefficienti attivi per testa e stessa funzione iniziale. Piccolo fitting vincolato FIT-only come diagnostica e checkpoint a budget crescenti lungo un solo training Adam. Confrontare errori su coefficienti, margini, derivate e massimi, non soltanto medie.

Falsificazione: Le due basi raggiungono lo stesso floor holdout e né fitting diagnostico né più budget cambiano il bias locale: dare priorità a obiettivo o numerica.

Limiti: Allocazione della base, costo e controlli vanno misurati. Il fitting diagnostico non è automaticamente selezionabile. Non addestrare sulle vecchie tracce stress fallite.

### Risposta autonoma: precisione all'interfaccia stato V/solver

L'arrotondamento accumulato attraverso stato di voltaggio e solver contribuisce abbastanza all'errore della debole risposta Nap da impedirne la qualificazione.

Test: Pesi congelati: FP32 originale, stato64+solver64 già efficace, percorso V in FP64 selettivo e implementazione compensata/in coordinate incrementali FP32 della stessa equazione con 12 iterazioni. Registrare incrementi e residui Newton; stesso riferimento e tutti i pannelli di qualità.

Falsificazione: Il beneficio congiunto non si riproduce su nuovi holdout, è annullato da regressioni oppure una versione con minore roundoff non migliora la risposta.

Limiti: Stato64 seguito da solver32 viene nuovamente quantizzato: il risultato nullo non esclude accumulo nello stato. FP32 ricentrato da solo è già fallito e non va presentato come nuovo. Nessun cambiamento di soglia.

### Velocità: costo cumulativo dei correttori, non necessariamente dell'astrazione

I correttori successivi duplicano feature, teste nulle, riduzioni FP64 e chiamate: ottimizzare solo l'ultimo può lasciare intatto il costo principale.

Test: Stesse funzioni congelate: legacy, singola testa attiva densa, ultimo correttore compatto già esistente e intera catena con feature condivise/teste attive. Profilare blocchi, allocazioni e chiamate a 1/642/4096 siti per seed; tempi wall e CUDA appaiati e intervalli di confidenza. Validare tutta la qualità per ogni implementazione numerica.

Falsificazione: Rimuovere il lavoro duplicato verificando parità di funzione non migliora il costo end-to-end e il profiler colloca altrove il collo di bottiglia.

Limiti: Non sommare pesi attraverso tanh distinte: non sarebbe equivalente. Nessuna pretesa di speedup da FLOP o numero di parametri. Distillazione è un'approssimazione separata, non un refactor esatto; nessun fallback canonico automatico.

## Riproducibilità

Archivio completo di 2.265.661.589 byte, SHA-256 13b91caa49c92e89b8fc2666c44f280fe5a648dc98acbcb861cc5c7336fa93cf verificato e CRC validi. Checkpoint originali byte-identici. Valutazioni mancanti recuperate su Kaggle T4 con Torch 2.11 cu130 corrispondente. Training originale: 60000 aggiornamenti; recupero: zero training e nessuna nuova selezione. Le tracce rigenerate conservano le metriche cache; non si afferma identità binaria dei contenitori NPZ perduti.

- Query: `memory_queries.sql`; risultati: `memory_queries.json`.
- Probe: `scripts/probe_task41h_residuals.py`; output: `residual_probe.json`.
- Ipotesi, evidenze e precedenti: `residual_hypotheses.json`.
- Mirror SQLite e backup locali; nessuna sincronizzazione Airtable.
