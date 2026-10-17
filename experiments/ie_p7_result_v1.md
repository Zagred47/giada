# IE-P7 v4 — chiusura verificata per la Task36

Run `alessandrobelli/giada-ie-p7-native-and-operational-confirmation`, v4;
revisione scientifica `328dee00b62d85af0f49d8edaf51b6b20fadc03d`, T4.
Artefatti in `results/ie_p7_kaggle_v4/`; audit riproducibile con
`scripts/audit_ie_p7_result.py`. Nessun nuovo training o test aperto nell'audit.

Verificati protocollo/checkpoint/freeze, 56 celle e preflight del core,
56 celle e preflight operativi, 14 classificazioni di qualità e decisioni
dei sei candidati locali. Ricomputati i confronti temporali dai campioni.
Replay CPU di formula, LUT e sei mappe sulle 32 tracce native: massimo
scarto di RMSE V CPU/GPU 1,025e-5mV. L'audit non riesegue NEURON e non
ricronometra la GPU; ricontrolla misure e replay sugli artefatti salvati.

| Candidato | Risparmio core graph642 | Risparmio operativo graph642 |
|---|---:|---:|
| local81 seed17 |22,86%|−0,20%|
| local81 seed29 |22,87%|3,35%|
| local81 seed43 |22,83%|3,26%|
| local161 seed17 |22,77%|3,43%|
| local161 seed29 |22,27%|1,20%|
| local161 seed43 |22,00%|4,57%|

Tutti sei passano qualità e criterio di costo del core, nessuno il criterio
operativo del 10%. Il core comprende membrana/Ca/readout/solver: non è una
latenza ionica isolata. Il perimetro operativo comprende produzione host
indipendente di eventi/plasticità/recettori, IPC, trasferimenti e 80 passi.
Nessun candidato promosso. Le mappe sono tabelle locali apprese con 18 stati
espliciti, non reti a memoria compressa; SK resta canonico.

Il supporto comportamentale nativo richiesto non è raggiunto: niente claim
generale su burst, adattamento o recupero lungo da questi 40ms. La fase
host/RPC domina questa specifica orchestrazione; non è un limite intrinseco
né un motivo per sostituire il bersaglio ionico del nuovo dropshot.
Le soglie Task35/CC/IE restano diverse e non sono reinterpretate.
Le interruzioni v1/v2/v3 restano incidenti tecnici archiviati, non fallimenti
scientifici dei candidati. La v4 chiude il confronto prospettico originale.
