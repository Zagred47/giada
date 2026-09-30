# GIADA Task 17c — precisione dei rate m × tolleranza CVode

Stato: esperimento diagnostico successivo alla Task17b. I due casi critici
sono già noti; non sono un test sealed né dati ammessi per selezionare un
candidato finale. Il NO-GO della Task17a, la soglia gate 0,01 e il blocco di
Gate C rimangono invariati. Nessuna rete viene addestrata.

## Ipotesi e contrasti

1. **Quantizzazione della tabella**: se il formato float32 dei 513 nodi è
   responsabile, la stessa griglia in float64 riduce nettamente l'errore.
2. **Interpolazione spaziale**: se domina la distanza fra nodi, 1025 e 2049
   nodi float64 dovrebbero ridurre progressivamente l'errore, a parità di
   solver. Non si assume monotonicità esatta dei massimi closed-loop.
3. **Sensibilità del solver**: se l'errore è dominato dall'integrazione
   numerica, stringere `CVode.atol/rtol` modifica molto il confronto LUT–nativo
   anche senza cambiare la tabella. Va riportato anche quanto si sposta il
   nativo fra le due tolleranze.
4. **Interazione**: i benefici della griglia possono dipendere dalla
   tolleranza. Tutti i confronti fra approssimazioni vanno fatti rispetto al
   nativo con la *stessa* impostazione del solver.

## Matrice appaiata

- Teacher nativo a 642 segmenti e protocollo NMDA della Task17b, 40 ms,
  campione 0,025 ms, siti 0/387/460/469, stessa gestione causale degli
  input sinaptici e nuovi SaveState dopo ogni cambio struttura.
- Condizioni: seed 170029 × gbar 1,0 (fallimento noto al sito 460); seed
  170083 × gbar 1,5 (fallimento noto al soma); seed 170029 × gbar 0,5
  (controllo debole). Sono tre condizioni, non tre repliche indipendenti.
- Solver: impostazione CVode originale letta dal teacher; impostazione
  stretta `atol=1e-5`, `rtol=1e-6`. Nessuna modifica a dt di campionamento.
- Sette bracci per solver e condizione: native, clone formula, LUT congelata
  Task15c/17a (513 nodi float32, tutti e quattro i rate), `mInf+mTau` a
  513 nodi float32, `mInf+mTau` a 513 nodi float64, 1025 nodi float64 e
  2049 nodi float64. I rate h negli ultimi quattro bracci restano formula.
- Totale: 42 traiettorie più preflight. Griglie su [-135,75] mV, valori
  estratti dalla stessa formula canonica, interpolazione lineare NMODL.

Controlli stop: hash del Task15c verificato; compilazione NMODL; 642
segmenti; replay nativo ripetuto per entrambe le impostazioni del solver;
clone formula con RMSE V ≤0,05 mV e massimo errore gate ≤0,002; nessun
NaN/Inf o stato gate fuori [0,1]. Prima dei trial, probe delle funzioni
compilate contro l'interpolazione attesa in punti di griglia e intermedi.

Metriche: per sito e condizione, massimo errore di m/h, RMSE e massimo errore
di V, RMSE corrente Ca_HVA. Per interpretare la fonte dell'errore si
riportano anche errori statici dei rate su un fitto asse di voltaggio e il
divario fra le traiettorie native dei due solver. La soglia 0,01 compare
soltanto come riferimento storico della Task17a; passare su queste
condizioni note non autorizza una promozione.

## Emendamento diagnostico dopo la prima matrice locale

La prima matrice ha mostrato una discrepanza **del nativo stesso** fra
solver originale e solver stretto che raggiunge molti mV nei due casi
critici. La tolleranza originale effettivamente letta dal teacher è
`atol=0.001`, `rtol=0`; quella stretta è `atol=1e-5`, `rtol=1e-6`.
Prima di interpretare la soluzione stretta come riferimento più accurato,
si aggiunge un ladder **native-only**, senza scegliere né riaddestrare
candidati: `atol=1e-6, rtol=1e-7` e `atol=1e-7, rtol=1e-8`, sulle medesime
tre condizioni. Si confronteranno default→tight, tight→stricter e
stricter→ultra, con replay ripetuto per ogni nuovo livello. Se anche gli
ultimi due livelli divergono molto, il confronto di accuratezza della LUT
resta scopiato al solver, non alla soluzione biologica convergente.
