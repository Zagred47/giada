# 🧬 Task18b — il potassio non annulla i successi Ca/Na

## 🔎 Diagnosi sul candidato già congelato (non nuova conferma)

Task18 valida, codicea094fa9, training114.39s: Ca_HVA e NaTa_t3/3 positivi;
K_Pst0/3. Audit nativo, equivalenza, CRC, freeze, provenance e checkpoint validi.
La LUT2049 passa per tutti: la funzione si approssima bene numericamente;
non è ancora stata appresa abbastanza bene dalle reti al budget15k.

Nel nuovo probe dei checkpoint K rate-supervised, i peggiori errori sono:

| Seed | V peggiore(mV) | dt(ms) | Errore massimo gate | SSE entro5mV da−60mV |
|---|---:|---:|---:|---:|
|17|−133.5208|25|0.02875|5.08%|
|29|−134.6975|100|0.02554|15.12%|
|43|−134.6975|100|0.01521|10.37%|

L'errore RMSE fuori dal vicinato del cambio di ramo resta0.00116–0.00145.
Le tau_m ai due lati di−60mV sono16.577441/16.577312ms: il piccolo salto
canonico non spiega automaticamente gli errori dominanti osservati nella coda.
Non aggiungere head piecewise come primo intervento senza questa evidenza.
La ladder K migliora da5k a15k: budget è plausibile, non già dimostrato sufficiente.

## ⚗️ Matrice parallela2×2×2

- Dati: uniformi oppure25% delleV sostituite con campioni in[-135,−115]mV.
- Supervisione: pesoMSE(logtau)0.01 oppure0.1; infweight0.1 invariato.
- Capacità: width32 oppure64; stessa cella fisicamente vincolata dellaTask18.

Tre seed per braccio:24modelli in due ensemble. Stessi stati/dt/indici e
inizializzazioni appaiati. Adam/clipping indipendenti, verificati anche contro
la copia singola. Checkpoint0/15k/30k/60k sulla medesima traiettoria: nessun
riaddestramento duplicato per la curva di budget. Si riparte da inizializzazioni
controllate per tutti i bracci, non da checkpoint15k scelti diversamente per seed.

Campionamento arricchito testa la copertura, non un maggior numero di tuple.
Peso tau testa enfasi della loss, non disponibilità di nuove informazioni:
tutti i bracci hanno le stesse etichette di rate. Width testa capacità a
stessi dati e budget. Le interazioni fra fattori sono visibili nella ladder.

## 🔒 Freeze e decisione

Una sola combinazione(braccio,width,step) per i tre seed viene scelta sul
peggiore score development, considerando separatamente uniforme e coda.
Si congelano gli hash prima di materializzare nuovi seedfresh181401/2/3.
Vecchi testTask18 restano consumati, utilizzabili per diagnosi non come
nuova conferma indipendente. I checkpoint non includono momentiAdam: non
vengono presentati come artefatti sufficienti a riprendere identicamente training.

Soglie originali: gateRMSE≤0.001; maxgate≤0.01; infRMSE≤0.002;
logtauRMSE≤0.02; openRMSE≤0.002; zero violazioni, tutto finito.
Tutte devono passare su entrambi i nuovi strati interni per ciascun seed.
OOD[-155,−135]mV e griglia fine di coda restano diagnostici, senza selezione
post-test. Un eventuale successo autorizzaTask19 solo nel contratto atomico;
non dimostra sostituzione embedded Na/K, universalità o speedup hardware.

No dataset montati. Teacher canonico074c466; auditNEURON isolato dal workerCUDA.
ZIP automatico anche in caso di failure del worker; output compatto, log su disco.
