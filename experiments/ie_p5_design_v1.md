# ⚡ IE-P5 — accuratezza, attribuzione e costo di rappresentazione

## 🎯 Tre ipotesi, non un altro aumento indiscriminato di capacità

IE-P4 ha separato un forte beneficio della supervisione interna dal suo costo
d'inferenza. Qui la larghezza rimane64 e lo stato resta fisico18gate. Si studia
IE-08/12 senza rinumerare le task originali, comprimere memoria, cambiare
dt0,1ms o modificare il solver. SK rimane canonico nella fase nuovoCa→SK.

1. **Dove nasce l'errore?** Per tutti3seed congelati IE-P4 internal64,
   sostituiamo u o z esatti, oppure entrambi per gruppiCa/Na/K/Ih-Im.
   Sono18oracle diagnostici sulle tracce Task35 già aperte. Non sono candidati
   causali, non entrano nei timing e non selezionano il training.
2. **Quale training serve?** Due refit appaiati×3seed, stessa larghezza64,
   propri pesi iniziali IE-P4, stesso resetAdam e stessi minibatch nuovi:
   perdita interna standard oppure standard più termini per code, orizzonte
   costante40ms e gate peggiore. Checkpoint crescenti nella stessa traiettoria.
3. **È il calcolo d'inferenza a costare?** Ogni MLP congelata viene eseguita
   direttamente e convertita in due tabelle interpolate delle **sue** uscite,
   con passo0,25/1mV. Non inseriamo valori analitici per migliorare la tabella
   della rete. Formula e LUT analitica restano controlli competitivi separati.

La tabulazione è una rappresentazione numerica di una rete appresa, **non**
una nuova architettura neurale né apprendimento dei nodi di una spline.
Può diminuire il costo, non correggere da sola il bias della rete. Si misura
separatamente l'errore di conversione; entrambe le risoluzioni sono riportate,
anche se falliscono. Non si seleziona una griglia dopo il fresh.

## 🧪 Matrice appaiata

23mappe misurate: formula/LUT,3MLP congelate,6MLP refit e12tabulazioni.
I18oracle sono soltanto diagnostici. Sei refit simultanei sulla GPU, Adam e
clipping indipendenti; preflight confronta pesi iniziali/predizioni e singolo
aggiornamento separato. Le statisticheAdam precedenti non sono disponibili:
il reset è comune ai due bracci, ma il confronto con IE-P4 non isola il solo
budget, perché cambiano anche dati e ottimizzatore reinizializzato.

Fit/development nuovi nel dominio V[-100,60]mV, gate[0,1], targetformula.
Il controllo standard usa esattamente la perdita IE-P4; il bilanciato aggiunge
logit-u supervisionato (targetlimitato per rendere finito il logit), endpoint
costante400step da x0=0/1 e massimo tra le perdite medie dei17gate.
I pesi dei termini sono preregistrati, non adattati con il fresh.

La selezione **comune** guarda il peggiore fra RMSEtransizione/0,001,
maxtransizione/0,01 e maxendpoint40ms/0,03. Questo criterio è nuovo e
prospettico, non modifica retroattivamente IE-P4. Per ogni membro si salva il
migliore dei checkpoint0/1000/3000/10000/30000; poi freeze conSHA256.

Solo dopo: fresh atomico,24nuovi path imposti in6famiglie e la diagnostica
Task35 già aperta. Metriche per famiglia/gate/corrente/carica, occupazioni e
uscite dal dominio. Limiti qualità identici a IE-P4. Nessuna nuova conferma
nativa o autorizzazione operativa. Il probe40ms costante non equivale a tutti
i possibili path: i path variabili restano un test distinto.

## ⏱️ Esecuzione e interpretazione

Inductor/CUDA Graph condivisi, FP32/TF32off, batch1/642/4096,30blocchi×20call;
compile/capture esclusi. Il target rimane almeno10% di risparmio a642 conCI
appaiato positivo contro le formule ottimizzate **oltre** a tutti i gate di
qualità. Il confronto MLP/tabella è sullo stesso modello e sui medesimi input.

Una griglia densa65537punti controlla la differenza massima della mappa a
un passo fra tabella e rete agli estremi x0=0/1 (budget5e-5). È un limite
diagnostico sulla griglia, non una dimostrazione su tutto il continuo.
Si riportano anche byte dei pesi e delle tabelle: un guadagno di tempo può
costare più memoria. Tabelle fuori dominio mantengono il bordo; nessuna
copertura dell'estrapolazione è dichiarata.

Un successo tabulato significherebbe "funzione appresa, esecuzione numerica
economica", non "MLP diretta più veloce". Se il training migliora ma il costo
rimane alto, le due evidenze restano separate. Nessuno speedup dell'intero
simulatore con eventi/RNG/trasferimenti è implicito. Niente kernel custom o
OpenEvolve; fallimenti tecnici separati da NO-GO scientifici.
