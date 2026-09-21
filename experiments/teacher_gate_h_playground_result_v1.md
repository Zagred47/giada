# Task 2 — Risultato del gate `h` di `Ca_HVA`

## Esito sintetico

L'artefatto e valido, gli hash coincidono, il sealed test e stato aperto una sola volta e non ha partecipato alla selezione. Il gate `h` e **fortemente apprendibile**, ma l'ipotesi preregistrata di trasferire direttamente la cella physical-τ da `m` a `h` non supera il requisito di rollout.

| Braccio | RMSE sealed one-step | RMSE rollout 1.000 passi |
|---|---:|---:|
| Direct MLP | `0.0031526` | `0.1412682` |
| Physical-τ | `0.0003198` | `0.0212920` |
| Direct-z | **`0.0000690`** | **`0.0017334`** |
| LUT lineare delle rate | `0.0001830` | non valutato |
| Persistenza | `0.0021462` | non valutato |

`direct-z` supera con largo margine la soglia one-step `1e-3`, resta sotto `5e-3` a 1.000 passi e non viola mai `[0,1]`. Quindi la dinamica atomica di `h` non e un mistero irrisolto: abbiamo gia un controllo positivo molto forte.

Il gate scientifico complessivo della Task 2 resta tuttavia **NO-GO**, perche richiedeva esplicitamente anche la riuscita del physical-τ nel rollout. Non possiamo cambiare questa regola dopo aver osservato il risultato.

## Perche physical-τ fallisce nel rollout pur essendo accurato one-step

Physical-τ ottiene errori one-step molto piccoli in interpolazione e persino OOD:

- OOD-voltage: `0.0003378`;
- OOD-`dt`: `0.0001391`;
- stress fattoriale: `0.0020377`.

Ma nella condizione di stress le sue variabili interne hanno errori molto maggiori:

- RMSE di `h_inf`: `0.02740`;
- RMSE di `log(tau_h)`: `0.54379`.

Con un gate lento, molti accoppiamenti diversi di `h_inf` e `tau_h` possono produrre quasi lo stesso piccolo spostamento in un singolo intervallo. La loss endpoint one-step vede il risultato corretto, ma non obbliga la rete a identificare separatamente equilibrio e costante temporale. Gli errori possono compensarsi a un passo e poi accumularsi quando l'update viene ripetuto centinaia di volte.

`direct-z`, invece, apprende direttamente quanto stato aggiornare per `(V,dt)`. Non deve separare due quantità debolmente identificabili dalla sola transizione breve e ottiene:

- OOD-voltage `0.0000816`;
- OOD-`dt` `0.0000190`;
- stress fattoriale `0.0004900`;
- rollout a 1.000 passi `0.0017334`.

Questo non dimostra che direct-z sara necessariamente la scelta finale. Dimostra che informazione e capacita sono sufficienti e localizza il problema di physical-τ nell'identificabilita dell'obiettivo interno.

## Scaling law

Tutti i bracci migliorano ancora fra 10.000 e 50.000 aggiornamenti:

- Direct MLP: `60,6%`;
- Physical-τ: `88,5%`;
- Direct-z: `96,7%`.

La decisione di estendere subito il budget e stata quindi utile. Tuttavia, per physical-τ il solo scaling dell'endpoint non risolve il rollout: a 50.000 step la development one-step e eccellente, mentre la dinamica interna resta non identificata abbastanza bene.

## Decisione

Non passiamo ancora direttamente alla cella congiunta `m+h`, perche il protocollo aveva legato l'autorizzazione alla stabilita di physical-τ. Il prossimo esperimento deve essere piccolo e causale:

1. congelare direct-z come controllo positivo;
2. confrontare physical-τ endpoint-only con supervisione delle rate e con loss multi-horizon;
3. usare gli stessi seed, batch e budget;
4. verificare se migliorano `h_inf`, `tau_h` e rollout senza sacrificare one-step;
5. autorizzare Task 3 soltanto dopo questa diagnosi, oppure scegliere esplicitamente direct-z con una nuova decisione architetturale.

La conclusione resta confinata al gate `h` di `Ca_HVA` sotto voltage clamp costante.
