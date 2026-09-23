# 🧠 Task 9 — risultato diagnostico

ZIP SHA-256 `6c28d4a1b0019a309a3ec65a4f62eae232f014b2bbb1b7680b82b1b16823ac44`.
Il report è valido sul dataset targeted canonico e usa soltanto `train`:
350 percorsi da 115 traiettorie. Nessun riaddestramento o selezione del seed.

La LUT-513 congelata approssima la formula composta sui 40 campioni con RMSE
di `m` dell'ordine `1e-6`–`2.3e-5`. Usando solo il voltaggio iniziale, durante
lo spike somatico l'RMSE di `m` verso la formula fine è `0.649`; con otto
campioni è `0.028`. L'informazione temporale è quindi essenziale.

Il **reference floor formula–teacher** è però oltre la soglia `0.005` in tutti
i gruppi spike: RMSE di `m` rispettivamente circa `0.0075`, `0.0088`, `0.0069`
e `0.0085` ai segmenti 0/387/460/469. Perciò il report dichiara correttamente
non interpretabile l'errore candidato–teacher negli spike. Il supporto spike
dendritico non raggiunge 24 per gruppo (14/9/15).

La Task 9b confronta metodi di integrazione sulla stessa microtraccia per
valutare se lo scarto sia coerente con il campionamento a 0,025 ms. È una
rianalisi forense dei dati aperti, non una conferma indipendente.
