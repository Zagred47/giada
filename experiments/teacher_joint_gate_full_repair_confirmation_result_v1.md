# ✅ GIADA Task 3e — conferma sealed del full repair

## Decisione

**PASS confermativo. La Task 4 è autorizzata.**

I tre checkpoint `full_repair`, shared width 23 al passo 50.000, sono stati
recuperati senza riaddestramento. La riproduzione development ha errore massimo
esattamente `0.0`; soltanto dopo il freeze valido è stato aperto una volta un
nuovo set sealed di 8.192 esempi, senza sovrapposizioni con la Task 3d.

## Risultati principali

| Controllo | Risultato | Limite | Esito |
|---|---:|---:|---|
| Score sealed medio | 0,3981 | ≤ 1,0 | PASS |
| Peggior seed | 0,5678 | ≤ 1,0 | PASS |
| Massimo RMSE `m_inf` | 0,000941 | ≤ 0,01 | PASS |
| Massimo RMSE `h_inf` | 0,001306 | ≤ 0,01 | PASS |
| Massimo RMSE `log(tau)` | 0,012757 | ≤ 0,15 | PASS |
| Reversal monotono massimo | 0 | ≤ 0,001 | PASS |
| Violazioni di occupazione | 0 | 0 | PASS |

Gli score per seed sono `0,3487`, `0,2777` e `0,5678`. Tutti i gate sono
superati con margine; non esiste un singolo seed borderline.

## Informazione scientifica acquisita

La riparazione identificata nella Task 3d non era un artefatto del development:
l'estensione del supporto in tensione, combinata con il prior monotono
condizionale e la supervisione multi-orizzonte, generalizza su dati nuovi.
La topologia shared compatta è quindi sufficiente nel dominio held-voltage e
non serve ripiegare su trunk indipendenti o aumentare la larghezza.

Il claim resta circoscritto a `Ca_HVA m+h` isolato sotto voltage clamp. Non
dimostra ancora sufficienza con un percorso di tensione variabile, correttezza
della corrente ionica completa o stabilità dentro un compartimento accoppiato.

## Prossimo passo

Eseguire la **Task 4 — matrice di primitive appaiate**, confrontando nello
stesso run formula originale, LUT/interpolazione, polinomio/Chebyshev, MLP,
GRU, physical-τ e direct-z con dati, seed, budget e metriche allineati.
