# 🌊 Task 8 — risultato verificato

ZIP SHA-256 `728cd58f5d19f1fecb3014df9ac0ed76efd17556434e5e74a1d570894fdf7f18`.
Freeze e hash del report coerenti; pilot NEURON autentico su 48 casi con errore
massimo dei gate `1.55e-15`. Sealed: 1.024 percorsi, nessun overlap col
development, nessun riaddestramento o scelta del seed sul sealed.

| Famiglia | Formula solo `V_t` | Formula 8 campioni | LUT 40 campioni |
|---|---:|---:|---:|
| Rampa lenta | 45,71 | 4,48 | 0,00311 |
| Rampa rapida | 48,91 | 4,50 | 0,00357 |
| Chirp basso | 56,20 | 3,36 | 0,00321 |
| Chirp alto | 63,04 | 20,39 | 0,00336 |

Sono **score normalizzati, non mV**. Il chirp alto dimostra che persino la
formula esatta perde molto con otto campioni. Con il path completo la LUT
congelata è precisa e supera tutti e tre i seed physical-τ in ciascuna
famiglia. Nessuna violazione d'occupazione.

Questo è un test con voltaggio futuro esogeno noto. La Task 9 trasferisce il
confronto ai percorsi registrati dal teacher completo, ma rimane teacher-forced:
non concede il futuro a un rollout autonomo.
