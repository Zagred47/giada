# GIADA Task 17d-b — conferma rilascio e riferimento numerico

Stato: protocollo fissato dopo la Task 17d e prima del nuovo run. I tre casi sono già noti e servono per diagnosi. Gate C resta chiuso; le soglie della Task 17a non cambiano.

## Matrice unica

Teacher canonico a 642 segmenti, stesso protocollo NMDA della 17d, seed/gbar `(170029, 1,0)`, `(170083, 1,5)`, `(170029, 0,5)`, durata 60 ms, campioni ogni 0,025 ms. Per ogni condizione, quattro bracci nativi: `default` (atol 1e-3, rtol 0), `voltage_scaled` (default con scala di v 1e-2), `ultra` (1e-7, 1e-8), `super_ultra` (1e-8, 1e-9). La coppia `ultra`–`super_ultra` verifica se il riferimento è stabile entro 0,05 mV RMSE e 0,002 errore massimo dei gate nei quattro siti. Queste soglie sono diagnostiche e non dimostrano convergenza matematica globale.

Alla stessa policy `ultra`, tre sostituzioni Ca_HVA: clone formula, LUT a 513 nodi float32 ricostruita dalla formula canonica con la procedura della Task 17a, e sola coppia di rate m con LUT a 2049 nodi float64 della Task 17c. Tutte mantengono corrente e `USEION` autentici; ogni braccio riparte da snapshot e Random123 canonici. La tabella a 513 nodi è ricostruita: questo run non pretende che sia byte per byte il checkpoint della Task 15c, ma la procedura di interpolazione compilata è verificata prima del teacher completo.

Per ogni episodio registrare separatamente ID evento, synapse ID, timestamp schedulato, posizione e draw Random123, successo di rilascio; salvare anche probabilità e quantità realizzata per diagnosticare differenze continue. L'identità discreta richiede uguaglianza esatta dei campi discreti e del draw, ma una differenza in probabilità/quantità è riportata anche quando il successo coincide. Verificare la frontiera di ogni millisecondo.

## Decisioni interpretative

- Se `ultra` e `super_ultra` concordano entro le soglie registrate, usare `ultra` come riferimento operativo limitato a questi casi. Se non concordano, ogni contrasto col candidato resta diagnostico e serve un'altra analisi del solver.
- Se decisioni/draw di rilascio divergono fra policy, il contrasto dei voltaggi non isola il solo integratore di membrana. Se coincidono ma probabilità o quantità differiscono, riportare la differenza come componente dell'input realizzato.
- Il clone formula deve restare entro 0,05 mV RMSE e 0,002 errore massimo gate rispetto al nativo `ultra` in ciascun caso per rendere interpretabile il contrasto delle LUT. Una sua violazione ferma qualsiasi claim sulla LUT.
- Per le due LUT riportare RMSE raw e massimi dei gate/voltaggi, timing e stati locali. Nessuna selezione su questi casi, nessuna modifica del candidato congelato, nessun claim su nuovi seed o performance hardware.

Interrompere su teacher errato, meccanismi compilati non verificati, policy non applicata, SaveState invalido, outcome di rilascio non verificati, NaN/Inf o errore del sottoprocesso. L'artefatto deve restare scaricabile anche in caso di stop e i risultati vanno nel solo mirror SQLite dopo l'ispezione.
