# ✅ GIADA — Task37: playground e valutatori isolati

**GO alla preparazione/implementazione della Task38**, subordinata ai propri
preflight di architettura, ottimizzatore e GPU. Nessun nuovo training eseguito.

Materializzazione completa in `results/task37_playground_v2/`:

- **168 traiettorie, 12 famiglie, 5 partizioni**, 60 file NPZ (~57,6MB).
- **23.685.200 target scalari di corrente**; 126 traiettorie nei regimi marcati
  rari e 215.320ms biologici complessivi, fino a10s per traiettoria.
- Nessun duplicato esatto di storia tra i ruoli; ogni array target è stato
  rigenerato e confrontato con il file salvato. Questo non equivale a una
  prova di indipendenza statistica di tutte le famiglie.
- **7 coppie causali** con stato iniziale/Ca/parametri uguali e solo storia
  di V diversa; corrente futura diversa nel tratto a V comune.
- **9 test**, **8 controlli difettosi** e **reset reale della memoria** passati.
- Riferimenti FP64/FP32 sul prefisso esistente: massimo scarto gate FP64
  3,33e-16, FP32 1,88e-7; scarto corrente FP32 6,73e-10mA/cm².
- Conversione storage di tutti i casi ammessa; massimo scarto assoluto
  di corrente in storage2,33e-10mA/cm². Tutti i dati binari finali sono conservati.

Le osservazioni sono **misure dei riferimenti e dei valutatori**, non errori
di una nuova rete. Nessun test finale di conferma è generato o aperto.
Le soglie specifiche di corrente, carica e transitori sono prospettiche
rispetto al training; non certificano da sole la fedeltà biologica closed-loop.

## 🔧 Iterazione di implementazione conservata

V1 si è fermata alla figura per matplotlib mancante, dopo la generazione:
incidente tecnico, non NO-GO scientifico. Le dipendenze grafiche sono state
installate in una directory locale isolata. Prima di v2 sono state rese
appaiate stato iniziale, Ca e parametri nelle coppie storiche: il contrasto
isola ora davvero la storia. Soglie invariate, nessun modello giudicato.
Il codice iniziale è recuperabile al commit9a39b6a; JSON v1 conservati, NPZ
intermedi locali rigenerabili. I binari completi versionati sono quelli v2.

## 🚧 Perimetro della conclusione

Passa il playground ionico con V/Ca imposti e i suoi controlli di sensibilità.
Restano da verificare con i candidati: V/Ca autonomi, comportamento emergente,
treni di spike, biforcazioni dinamiche e costo GPU. Sono obblighi dei run
successivi, non marcati come superati dalla Task37. Lo stato latente non è
ancora introdotto; la Task38 conserva i18gate espliciti.

Audit: `results/task37_playground_v2/result_audit.json`.
Figura: `results/task37_playground_v2/matched_history_review.png`.
Registrazione DB e integrità nella ricevuta `memory_receipt.json` dello stesso
archivio. Airtable resta congelato.
