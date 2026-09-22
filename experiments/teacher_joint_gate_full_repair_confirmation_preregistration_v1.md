# 🔒 GIADA Task 3e — conferma sealed del full repair

## Candidato immutabile

La Task 3e non addestra e non seleziona modelli. Riusa gli esatti checkpoint Task 3d:

- `full_repair`;
- trunk shared;
- width 23;
- step 50.000;
- seed 17, 29 e 43.

Prima dell'apertura sealed vengono verificati SHA-256, origine del checkpoint e riproduzione development entro `1e-10`.

## Nuovo sealed set

Il set viene materializzato soltanto durante la valutazione successiva al freeze:

- 4.096 tuple centrali;
- 2.048 tuple nelle code di tensione;
- 2.048 tuple a orizzonte held-voltage lungo.

Le tuple devono avere overlap zero con fit e development Task 3d. Il fresh Task 3c non viene letto o riutilizzato.

## Gate preregistrati

- score sealed medio ≤ `1,0`;
- peggior seed ≤ `1,0`;
- RMSE massimo `m∞` e `h∞` ≤ `0,01`;
- RMSE massimo di `log τ` ≤ `0,15`;
- massima inversione monotona ≤ `0,001`;
- zero violazioni di occupazione.

Tutti i gate devono passare. Solo in quel caso viene autorizzata la Task 4.

## Limite epistemico

Un esito positivo conferma la cella congiunta `Ca_HVA m+h` sotto voltage clamp, comprese code e orizzonti lunghi. Non conferma ancora il coupling con un voltaggio dinamico, la corrente ionica completa o il neurone embedded.
