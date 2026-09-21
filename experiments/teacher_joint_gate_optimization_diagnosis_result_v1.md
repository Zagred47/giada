# GIADA Task 3b — risultato

L’esperimento è valido, non ha aperto dati fresh e ha localizzato due leve
causali: **supervisione simmetrica dei rate** e **budget di training**.

| Braccio | Score development |
|---|---:|
| baseline 50k | 0,2650 |
| rate simmetrici 50k | 0,1022 |
| baseline 100k | 0,0978 |
| PCGrad 50k | 0,4845 |
| rate simmetrici + PCGrad | 0,3849 |
| pretraining rate-only | 0,1990 |

La supervisione di `m_inf` e `log(tau_m)`, allineata a quella già presente per
`h`, migliora lo score del **61,5%** a 50k e riduce fortemente la dispersione fra
seed: deviazione standard `0,0083`, contro `0,1963` della baseline 50k. La
baseline prolungata a 100k migliora del **63,1%** e raggiunge uno score medio
leggermente migliore (`0,0978`), ma con il doppio del budget e stabilità inferiore
al braccio simmetrico 50k (`0,0246` contro `0,0083`).

La conclusione precisa non è che l’endpoint di `m` sia privo dell’informazione
necessaria: con abbastanza step la baseline arriva nello stesso regime. La loss
simmetrica fornisce invece una **scorciatoia identificante e ben condizionata**,
accelerando la convergenza e rendendola riproducibile fra seed.

Il 66,7% dei minibatch diagnostici iniziali mostra coseno negativo fra i
gradienti, ma PCGrad peggiora lo score dell’82,8%. Il conflitto geometrico grezzo
non è quindi una causa operativa sufficiente e PCGrad non va mantenuto. Anche il
warm-start rate-only peggiora rispetto al training congiunto simmetrico.

La riparazione minima raccomandata è la supervisione congiunta e simmetrica di
`m_inf`, `tau_m`, `h_inf` e `tau_h`, mantenendo solver esponenziale, trunk
condiviso e budget 50k. La Task 4 resta non autorizzata: il prossimo passo è una
Task 3c preregistrata, con freeze su development e una nuova conferma fresh
disgiunta aperta una sola volta.
