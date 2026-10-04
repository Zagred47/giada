# GIADA Task 17f — supporto eventi e conferma causale Ca_HVA

## Domanda e stato

Preparazione prospettica, successiva al risultato 17e. È una sottotask della
Task 17 originale; la Task 18 resta bloccata fino al passaggio di Gate C.
Il candidato resta `m_pair_2049_f64`, LUT dei rate m a 2049 nodi, h analitico,
SHA-256 `e15f712faccc68dafbc7d861a14e23262388b9715ae5b564c51c06f261251765`.
Nessun training, nessuna nuova architettura e nessuna ricerca di soglie favorevoli.

La 17e ha passato i contrasti numerici nel supporto visitato ma non ha coperto
somatic spike, calcium spike e NMDA spike. Un plateau NMDA era presente. Il
probe NMDA era inoltre fisso al nexus 437, anziché al centro del cluster
richiesto dalla schedule. Il sito 460 è tuft, non nexus. Questi fatti motivano
due interventi distinti: correggere la localizzazione e prequalificare gli stimoli.

## Ipotesi e contrasti nello stesso notebook

| Ipotesi | Contrasto osservabile | Esito che la limita |
|---|---|---|
| H1: il probe fisso nasconde eventi locali NMDA | Detector legacy al nexus e detector al centro effettivo, sulla stessa traccia nativa | Nessuna differenza nei conteggi; entrambi i risultati vengono salvati |
| H2: le schedule precedenti hanno supporto insufficiente | Ladder nativo di eventi sinaptici a pesi canonici e corrente somatica esplicita | Nessuna schedule produce positivi non censurati su tutti i seed pilot |
| H3: il solver ultra è affidabile anche durante eventi attivi | Nativo ultra vs super-ultra su ogni condizione confermativa, più ripetizione attiva | Uno dei controlli di V, gate, corrente, eventi o rilascio fallisce |
| H4: la LUT congelata preserva gli eventi nel sistema accoppiato | Nativo/formula/LUT, stessi input, seed, gbar, siti e solver | Almeno una soglia preregistrata del candidato fallisce dopo i controlli |

H1/H2 sono diagnostiche. La loro conferma non è necessaria per promuovere il
candidato, ma il supporto effettivo degli eventi e i controlli H3 sono necessari.
L'assenza di eventi non è una prova contro la LUT. Le etichette sono quelle del
detector diagnostico versionato, non una dimostrazione autonoma del meccanismo
biofisico che causa ciascun evento.

## Fase A: pilot sul solo nativo

Seed `171501,171519,171543`, disgiunti dai seed confermativi. Tutti gli episodi
durano 60 ms, osservazione ogni 0,025 ms; gbar = 1; CVode ultra atol=1e-7,
rtol=1e-8, nessuna scala selettiva. La partenza è l'equilibrio canonico.

Ladder fissati e selezione deterministica del **primo** stimolo che genera
almeno un evento richiesto non censurato in ciascuno dei tre seed:

1. Soma: due impulsi a 2,05 e 3,05 ms, lunghi 0,9 ms, ampiezza
   0,5 / 1 / 2 / 4 / 6 nA, in quest'ordine.
2. Calcio: 12 sinapsi hot-zone, 3 burst a 3/4/5 ms, finestra 0,8 ms;
   1 / 2 / 4 eventi per sinapsi per burst; prima senza corrente somatica,
   poi gli stessi tre livelli con l'ampiezza selezionata nel pilot soma.
3. NMDA: 10 sinapsi nel cluster tuft, 2 burst a 3/4 ms, finestra 0,4 ms;
   1 / 2 / 4 eventi per sinapsi per burst, senza assistenza somatica.

Pesi NetCon e parametri delle sinapsi immutati. Si varia il numero di eventi,
non la conduttanza sinaptica. Massimo 42 episodi pilot; stop della ladder dopo
il primo candidato qualificato, senza consultare il surrogato. Se una classe
non è coperta, report `PILOT_SUPPORT_INSUFFICIENT`; nessuna apertura dei seed
confermativi. `--pilot-only` serve a verifiche di sviluppo e non autorizza Gate C.

Si salvano tracce dense e detector legacy/corretto di ogni trial. Il file
`protocol_freeze.json` congela input espliciti, sinapsi, segmenti del probe,
definizioni complete, seed e hash LUT **prima** della conferma.

## Fase B: conferma congelata

Seed `171601,171619,171643`. Quiete e tre schedule qualificate; gbar 0,5/1/1,5.
36 condizioni per braccio; 144 episodi fra nativo ultra, nativo super-ultra
(atol=1e-8, rtol=1e-9), clone formula ultra e LUT ultra, più una replica nativa
attiva. Si richiede copertura di ogni classe su **tutti e tre i seed a gbar=1**,
con positivi non censurati. Si contano tutti gli eventi anche agli altri gbar,
senza richiedere che le perturbazioni preservino il regime.

Prima si esegue e valida il nativo; se il supporto manca, si produce
`CONFIRMATION_SUPPORT_INSUFFICIENT` senza cambiare schedule o seed. Tutti i
contrasti nativo ultra/super-ultra devono passare prima di interpretare formula
o candidato. Le sostituzioni NMODL sono reali, in ogni compartimento Ca_HVA,
con corrente nel sistema accoppiato e nuovi SaveState a ogni cambio di struttura.

## Misure e criteri congelati

Siti m/h/corrente: 0,387,460,469. V ed eventi anche a soma0, AIS640, trunk361,
hot-zone387, nexus437, tuft460 e al centro effettivo del cluster. Il detector
NMDA usa quest'ultimo solo per lo stimolo cluster; soglie/durate restano quelle
di `diagnostic-v0.2.0`. Si salva il mapping in ogni trial. Nessun allineamento
post hoc delle tracce.

- Replica nativa attiva: max errore V nei quattro siti <=1e-5 mV e tutti i
  controlli di confronto validi.
- Nativo super-ultra vs ultra e formula vs nativo: RMSE V <=0,05 mV; max
  errore gate <=0,002; RMSE corrente Ca_HVA <=1e-4 mA/cm².
- LUT vs nativo: RMSE V <=2 mV; max errore gate <=0,01; RMSE corrente
  Ca_HVA <=1e-4 mA/cm², in ogni condizione/sito.
- Eventi: stessi conteggi e censura, onset <=0,2 ms soma/AIS e <=0,5 ms
  dendriti. Falsi positivi in quiete fanno fallire il confronto.
- Rilasci: identità, draw, successo identici; quantità realizzata entro 1e-10.
- Contrasto gbar 1,5−0,5: almeno un effetto identificabile >=0,05 mV;
  errore relativo <=20% per ogni contrasto identificabile.
- Finitezza, occupanza [0,1], morfologia, pesi canonici, hash della LUT e
  schedule condivisa sono precondizioni, non metriche ottimizzabili.

Il run salva i risultati dopo ogni episodio e distingue insufficienza di
supporto, riferimento instabile, clone formula invalido, candidato inaccurato
e errore tecnico. Solo tutti i criteri superati autorizzano
`GATE_C_PASS_BOUNDED_CAHVA` e Task 18, entro questo ambito. Il tempo CPU non
dimostra un vantaggio hardware; questa task non richiede GPU o dataset esterni.

## Riproducibilità e consegna

Teacher canonico commit `074c4666300a8ad246601dab179a97a6942f0f29`, NEURON 8.2.7
(anche stringa runtime 8.2.7+). Notebook, runner e preregistrazione della stessa
revisione GIADA. Il notebook salva automaticamente uno ZIP anche dopo NO-GO;
il download via JavaScript resta disponibile per il flusso manuale, mentre
l'output ZIP è recuperabile via MCP. La selezione pilot non consulta gli
outcome di conferma e non viene rifatta dopo averli aperti.

Il controllo tecnico `--development-smoke` ripete il pilot e prova i quattro
bracci soltanto sul primo seed pilot a gbar1 (16 episodi più replica attiva).
Non apre i seed confermativi e non può autorizzare Gate C, anche se passa.
Il suo risultato è evidenza di sviluppo da registrare separatamente.
