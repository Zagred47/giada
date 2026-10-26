# GIADA — Task 41j: cambiare lo spazio di apprendimento, non le soglie

## Motivazione verificata

La41i ha concluso tecnicamente il recupero senza altro training o selezione.
Sul medesimo pannello il controllo H ha4 fallimenti, il candidato I010 ne ha6.
Due risposte autonome si risolvono con una politica numerica congelata; non
si risolvono i restanti errori di corrente. Il candidato primario richiede
circa22,7% di tempo ionico e14,5% di tempo integrato in più rispetto a H,
a642 siti per seed. Non confrontare questi4 casi H con i6 del vecchio
pannello development: i protocolli sono diversi.

Il fallimento della riparazione vincolata non dimostra l'impossibilità del
livello di astrazione. I precedenti del DB motivano ipotesi, non ricette
causalmente garantite. Si interrompe l'aggiunta sequenziale di correttori.

## Confronti

| Braccio | Rappresentazione | Parametri per seed | Modificabile |
|---|---|---:|---|
| H_nominee | Catena precedente congelata |4004|Controllo non selezionabile|
| J_chain_joint | Stessa catena |4004|Tutti i parametri|
| J_compact64 | Un MLP V→34 output + testa Ca→SK |2404|Tutti i parametri|
| J_compact128 | Stessa struttura, tronco più ampio |4708|Tutti i parametri|
| J_restricted_control | H + vecchia riparazione limitata |4260|Solo due equilibri; controllo non selezionabile|

Tre seed17/29/43, stessi dati e minibatch, stessa supervisione. Il controllo
limitato distingue lo spazio di apprendimento dalla nuova loss/copertura:
non sostituisce i tre candidati e non è nominabile. La capacità
non è pareggiata:64/128 è una mini legge di scala a budget fisso. Nessuna
conclusione di superiorità architetturale parameter-matched.

## Invarianti

18 occupazioni apprese, stessi input causali V/Ca, passo0,1ms e transizione
convessa x'=x+sigmoid(z)*(sigmoid(u)-x).17 gate dipendono da V; SK da Ca.
Readout, aggiornamento del calcio, solver e sinapsi invariati. Nessun fallback
alle cinetiche canoniche in inferenza. Le formule sono soltanto label FIT e
controlli diagnostici. Tutte le soglie fisiche, correnti, timing, interventi,
regressioni e costo restano quelle precedenti.

## Apprendimento e identificazione

4096 slot di distillazione del parent; la catena parte esattamente da H e usa
learning rate nullo nel warmup (identità già nota), i compatti apprendono
partendo soltanto dal suo tronco. Questo costo di inizializzazione diverso è
dichiarato, non nascosto come parità di compute. Il gap iniziale
e quello dopo warmup sono misurati su ancore indipendenti, non presunti nulli.
Nuovo Adam per60000 aggiornamenti di raffinamento canonico con checkpoint
0/512/2048/8192/30000/60000. Tutti i36 coefficienti apprendibili.

Supervisione di equilibrio e frazione temporale; censura dei logit a1e-12
come scelta di training esplicita, non soglia di valutazione. Correnti su
storie complete di1200ms, margine dell'offset ufficiale per tutti gli11 canali
e risposta accoppiata a perturbazioni per40 passi. Il prefisso imposto si
calcola con composizione affine parallela, verificata anche nei gradienti:
è un acceleratore del training, non una modifica dell'inferenza.

Per distinguere assenza di trasferimento da approssimazione, salvare gap di
distillazione, errori atomici, gradienti e curve lungo lo stesso training.
La precisione del portatore di V viene auditata a pesi congelati e non viene
usata per scegliere il candidato. Non chiamare "errore della rete" un errore
che scompare cambiando soltanto la rappresentazione numerica.

## Selezione, valutazione e stop

Nuovi FIT/selection/development separati e seed preregistrati. Selezione per
seed prima dei pannelli aperti secondo (fallimenti globali development,
regressioni introdotte contro H, severità peggiore, loss ausiliaria, atomica).
Un unico nominee per il confronto prospettico; gli altri bracci restano
informativi ma non sostituiscono retroattivamente il nominee sui test aperti.

Tutti i pannelli originali, stress, interventi e ricorrenza dopo freeze.
Costo dell'intero operatore e sistema integrato contro formula/LUT/H agli
stessi batch,30 ripetizioni appaiate, tempi CUDA e wall. Non attribuire
automaticamente il rallentamento alla capacità: il costo della composizione
è un fattore distinto. Nessun GO alla42 senza nuova conferma indipendente.

## Esecuzione e durabilità

Un notebook, fasi Kaggle512/8192/30000/60000 poi valutazione congelata.
Ogni fase conserva pesi, optimizer, RNG, migliori checkpoint e selezione.
SHA256/CRC verificati su copia locale esterna al runtime prima della fase
successiva. Un errore tecnico arresta la continuazione, non diventa NO-GO
scientifico. DB e relativo backup restano soltanto locali.
