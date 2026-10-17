# 🧪 GIADA — Task37: playground e valutatori ionici

Task37 implementa il contratto della **nuova Task36**. Il protocollo è
[`task37_ionic_playground_preregistration_v1.json`](task37_ionic_playground_preregistration_v1.json),
con [precisazione delle coppie storiche v2](task37_history_matching_amendment_v2.json).
Nessun modello viene addestrato o selezionato.

## 🔌 Target e causalità

Si evolve lo stato dei 18 gate su V imposto, con un ponte del calcio imposto
separatamente. La corrente Ca è calcolata dopo l'update dei gate Ca; SK usa
il calcio nella fase successiva dichiarata. I riferimenti sono le cinetiche
canoniche e il readout firmato `gbar*a*(V-E)`. Un preflight indipendente con
il prefisso PyTorch esistente verifica update, Ca, SK, correnti e somme su
512 casi casuali/di confine, in FP64 e FP32. Non è una nuova simulazione NEURON.

La generazione dei target è ricorrente, senza reset; V e Ca rimangono imposti.
In closed loop CaDynamics dovrà generare il ponte causalmente dalla sorgente
predetta: questi dati non autorizzano a usare calcio futuro del teacher.

## 🗂️ Copertura, split e budget

Dodici famiglie: step, rampe, chirp, treni di V rapidi, impulsi appaiati con
recupero, stesso V dopo storie diverse, recupero lento, cambi di calcio,
blocchi, inversioni, confini di attivazione e punti numerici delicati.
Le durate vanno da80ms a10s; dt0,1ms. V in[-100,60]mV, Ca in[1e-7,1e-2]mM;
stress Ca separato a1e-8/2e-2mM. Non sono frequenze fisiologiche stimate.

| Partizione | Traiettorie/famiglia | Uso |
|---|---:|---|
| calibration |2|Verifica del riferimento e degli strumenti, non selezione fresh|
| fit |4|Prima copertura di training futura|
| fit_extra |4|Nuove storie per il contrasto diversità × budget|
| development |2|Scelta futura di modello/checkpoint|
| stress_diagnostic |2|Stress aperti, sempre separati dalla congiunzione in-domain|

Ogni traiettoria ha firma dei propri ingressi/stato/parametri. Nessuna
divisione per punti adiacenti; le coppie causali rimangono nello stesso ruolo.
La conferma Task41 **non è generata**. I suoi nuovi casi e il freeze saranno
registrati separatamente. I semi pubblici non costituiscono una protezione
crittografica del test.

Il contrasto futuro conserva numero di update e logging di esposizione per
famiglia: fit contro fit+fit_extra misura diversità; checkpoint lungo la
stessa traiettoria misurano budget. Trunk width32/64, loss gate oppure
gate+correnti individuali, tre seed e ladder0/1k/3k/10k/30k sono una bozza
per Task38–39, non un training già autorizzato o svolto.

## 📏 Gate e controlli

Per ogni canale e traiettoria: NRMSE≤1%, massimo normalizzato≤5%, errori
di carica firmata/assoluta≤2%; RMSE gate≤0,005, massimo≤0,03. La scala è
`max(picco assoluto del riferimento,1e-7 mA/cm²)`; la carica si normalizza
con durata×scala, non con carica firmata prossima a zero. Blocchi esatti,
non-finiti e occupazioni non ammissibili fanno fallire il gate.

Timing del transiente≤0,2ms e onset/offset≤0,3ms soltanto con supporto
sufficiente; picchi piatti hanno una banda dichiarata dello0,1% per non
trasformare l'arrotondamento FP32 in falso ritardo. Si riportano errori per
orizzonte e primo superamento della tolleranza. Le finestre transitorie qui
non sono un nuovo algoritmo completo di matching dei burst/spike.

Otto controlli di output: attenuazione50%, ritardo1ms, compensazione tra
correnti opposte, deriva, proxy di reset, NaN, evento mancante e falso evento.
Si aggiunge un **vero reset dello stato ricorrente** in un fixture indipendente.
La corrente totale può apparire perfetta nel controllo di compensazione:
il valutatore deve comunque rifiutarlo per gli errori individuali.

Il reference corretto e la sua conversione FP32 devono passare. Queste soglie
sono fissate prima di qualunque training: non sono state adattate a un modello.
Il loro significato biologico complessivo richiederà il confronto embedded.

## 👁️ Qualitativa e limiti

Pannelli separati per V, Ca e correnti, unità dichiarate e correnti entranti
negative. Sono conservati un controllo di drift e una coppia storica scelta
con regola riproducibile. La revisione visuale integra i numeri; non può
promuovere un modello che fallisce i gate.

I treni di V non sono spike emergenti. I confini di attivazione non sono
biforcazioni dinamiche del neurone. Recupero/risposte sotto ingressi imposti
si possono valutare qui; V/Ca autonomi, burst/ISI, vicinanza alle biforcazioni
e propagazione richiedono i rispettivi protocolli Task40–41 e spaziali.

Il workload di costo primario è80tick ionici causali a batch642, FP32 e
compilazione simmetrica, tutte11correnti e sorgente Ca osservabili. Obiettivo
prospettico≥10% di risparmio appaiato con limite inferiore bootstrap positivo.
Task37 non misura la GPU né promette speedup; core/operativo restano distinti.

Esecuzione: `python scripts/run_task37_playground.py --output <nuova-cartella>`;
audit dei dati v2: `python scripts/audit_task37_playground.py`.
NumPy, PyTorch CPU e matplotlib sono necessari; NEURON e CUDA non vengono
caricati insieme. Le librerie grafiche locali non sono versionate come codice.
