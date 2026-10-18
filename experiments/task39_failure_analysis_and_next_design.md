# Task 39 — esito e proposta diagnostica successiva

## Evidenza verificata

Kaggle v1, sorgente a6fbfa02b5514becaaa18497406d800fbae1d372, 30 membri,
10 bracci x 3 seed, 60k aggiornamenti, T4, compilazione comune verificata.
1872 righe uniche (13 modelli x 2 ruoli x 12 famiglie x 2 casi x 3 seed),
20592 metriche canale; tutte finite, zero violazioni di occupazione.
Freeze verificato; nessun test indipendente aperto. Esito NO_GO_TASK40_BLOCKED.
I 72 casi per modello/ruolo non sono 72 repliche indipendenti.

Il braccio factorized_expanded_fr supera 24/72 casi development e 26/72 stress.
Development: 14 fallimenti gate, 35 RMSE corrente, 28 massimo corrente,
25 carica, 22 timing. Conteggi sovrapposti; 10 casi falliscono solo sul timing
fra queste categorie. Ca_LVAst fallisce in 28 casi, Im in 25, Ca_HVA in 16.
Non basta quindi rimuovere correnti piccole o modificare un criterio temporale.

Famiglie development: steps e paired_recovery 6/6; same_v_history,
slow_recovery, ramps, reversal, activation_boundary e seams 0/6.
I tre peggiori gate RMSE sono same_v_history: 0.00984, 0.01017, 0.00978.
Slow_recovery fallisce anche se il peggior RMSE gate e' solo 0.00129:
un gate medio accurato non garantisce correnti e tempi accurati.

Contrasti preregistrati supportati: esposizione ricorrente nella famiglia
fattorizzata con entrambe le coperture; loss relativa delle correnti nel
confronto joint_fit_fr. La combinazione delle due leve NON e' stata testata.
La copertura migliora alcuni aggregati ma non supera il vincolo sui peggiori
casi. La fattorizzazione elimina per costruzione le dipendenze spurie testate,
non dimostra da sola sufficiente accuratezza.
9/10 bracci migliorano secondo la regola sul budget 30k->60k, riferita alla
loss breve, non alla ricorrenza lunga. 29/30 checkpoint selezionati sono 60k.

## Ipotesi aperte, non diagnosi definitive

1. **Orizzonte troppo corto:** 8 tick = 0.8 ms non vincolano sufficientemente
   i tempi lenti e l'accumulo del bias. Prova: contrasto orizzonte 8/64 tick,
   stessi target-tick totali e valutazione di costo anche in secondi GPU.
2. **Campo dinamico non identificato fuori dalle traiettorie:** poche storie
   possono lasciare ambigui attrattore e fattore di rilassamento. Prova:
   collocazione controllata di stati iniziali indipendenti a V/Ca fissati,
   con teacher offline. Due stati distinti identificano pendenza/intercetta
   dell'update affine nei bracci fattorizzati. Non fornire formule in inferenza.
3. **Obiettivi disallineati:** medie quadratiche sottopesano correnti deboli,
   code e recuperi. Prova: loss relativa per canale, senza rimuovere la base
   gate/corrente e senza cambiare le soglie. Monitorare regressioni attive.
4. **Saturazione e ottimizzazione dei tempi lenti:** la parametrizzazione
   sigmoid puo' rendere difficili frazioni d'aggiornamento molto piccole.
   Prima di cambiare architettura, misurare fattori appresi, equilibrio,
   pendenza per gate, errori sui piccoli incrementi e norme dei gradienti.
5. **Interferenza tra gate nel trunk:** possibile, non dimostrata dai dati
   aggregati. Probe per gate e gradienti tra task; separazione delle teste
   solo se i probe la giustificano, non aggiungere subito un altro sweep.
6. **Timing mal appreso o misura ambigua:** 10 fallimenti esclusivamente
   temporali richiedono ispezione di tracce, picchi multipli/piatti e crossing.
   Conservare la metrica registrata; aggiungere diagnostica, non sostituirla.

## Proposta: Task 39b, non promozione alla Task 40

Un unico notebook con matrice 2x2x2 sul modello fattorizzato e dati ampliati:
orizzonte 8/64 tick x collocazione off/on x loss relativa off/on = 8 bracci,
3 seed appaiati. Il controllo 8/off/off riproduce factorized_expanded_fr.
La loss relativa puo' interagire con la fattorizzazione: non si presume additiva.

Controllare stato iniziale, stream di famiglie, numero di target supervisionati,
inizializzazione e normalizzatori fit-only. Per collocazione usare slot
supervisionati sostitutivi preregistrati, non target extra non contabilizzati;
gli slot controllo usano stati da traiettoria a parita' di V/Ca. Riportare
sia target-tick sia GPU-secondi: orizzonti diversi non hanno costo identico.
Mini scaling con checkpoint lungo la stessa traiettoria; budget finale fissato
prima del lancio dopo un benchmark tecnico senza guardare i risultati scientifici.

Nel medesimo run: probe della mappa affine per ogni gate, equilibrio e
rilassamento, spettro/Jacobiano locale, errori per orizzonte e storia,
metriche per canale/famiglia/seed, correnti deboli e attive, timing con tracce
dei casi ambigui. Recuperi fino a 10s rimangono valutazione obbligatoria.
Controllo formula e baseline congelata; compilazione ed equivalenza numerica
comuni, Adam/clipping indipendenti. Nessuna promessa di speedup.

Selezione development preregistrata con orizzonti brevi E lunghi; non scegliere
solo con una loss di 0.8ms. Stress resta diagnostico gia' aperto: non chiamarlo
conferma indipendente. Test nuovo sigillato solo dopo il freeze del candidato.
Soglie Task37 invariate, guardrail su peggiori casi e tutti i seed.
Nessun rilascio o reinserimento causale senza audit e criteri originali superati.
Questa e' una proposta, non un esperimento gia' preregistrato o avviato.
