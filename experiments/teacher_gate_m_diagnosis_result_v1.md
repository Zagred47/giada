# Task 1b — Risultato della diagnosi del gate `m`

## Esito sintetico

L'artefatto `giada_gate_m_data_budget_capacity_diagnosis.zip` e valido e rispetta il protocollo preregistrato. La diagnosi causale e **budget-limited**: aumentare il numero di aggiornamenti migliora in modo consistente i candidati, mentre densita dei dati, larghezza e supervisione ausiliaria dei rate non mostrano un effetto principale robusto alla soglia preregistrata del 20%.

Il candidato scelto esclusivamente sul development, `base-w16-endpoint_only` a 50.000 step, ottiene RMSE fresh macro `0.0022578625`. Non supera il gate scientifico `1e-3`, ma supera il gate ingegneristico `2.5e-3`, con zero violazioni di occupazione e rollout stabile fino a 1.000 passi. E quindi metodologicamente lecito procedere al gate `h`, mantenendo l'accuratezza OOD-voltage di `m` come debito scientifico esplicito.

## Integrita e separazione della selezione

- SHA-256 ZIP: `3ed551398b0ce5d12ef44ad3509ae2e64871305358a3cdfd745a90319d723c8f`.
- Hash del checkpoint, selection freeze e final report verificati.
- La fresh confirmation e stata aperta una sola volta e non e stata usata per la selezione.
- Le righe sealed della Task 1 non sono state riutilizzate.
- Il vincitore e stato selezionato solo sul development, su tre seed (`17`, `29`, `43`).

Una variante non selezionata, `base-w32-endpoint_only`, ha ottenuto sul fresh test un RMSE macro migliore (`0.0018670684`). Questo dato e descrittivo ma **non autorizza** a sostituire retrospettivamente il vincitore: farlo costituirebbe selezione sul test.

## Prestazioni del vincitore legittimo

| Regime | RMSE `m` |
|---|---:|
| Fresh in-support | `0.0002378306` |
| Fresh OOD-`dt` | `0.0001411297` |
| Fresh OOD-voltage | `0.0063946271` |
| Fresh macro | `0.0022578625` |

Nel rollout ricorsivo l'RMSE e `0.0004963250` a 10 passi, `0.0003602370` a 100 e `0.0003801586` a 1.000, senza violazioni dell'intervallo fisico `[0,1]`. L'errore non esplode: la cella esponenziale vincolata e ricorsivamente stabile.

Il residuo e fortemente localizzato nell'extrapolazione in tensione. Sullo strato OOD-voltage, l'errore di `m_inf` e soltanto `3.40e-5`, mentre l'RMSE di `log(tau)` sale a `0.04597`. Quindi il pezzo ancora difficile non e la ricorrenza ne il valore asintotico: e soprattutto l'extrapolazione della costante temporale fuori dal supporto di tensione.

## Cosa ha escluso il fattoriale

Al checkpoint comune di 10.000 step, gli effetti mediani appaiati sono:

- piu dati: miglioramento `9.35%`, sotto la soglia preregistrata del `20%`;
- maggiore larghezza: `-6.53%`, quindi nessun beneficio principale robusto;
- loss ausiliaria su `m_inf` e `tau`: miglioramento `6.58%`, sotto soglia e non consistente.

Questi risultati non significano che dati o capacita siano sempre irrilevanti: esistono interazioni tra singoli bracci. Significano che nessuno dei tre fattori spiega in modo robusto il limite residuo come effetto principale.

Il budget, invece, produce miglioramenti forti lungo le stesse traiettorie: `38.5%` per `base-w16-endpoint_only`, `21.5%` per `dense-w16-endpoint_only`, `62.8%` per `base-w32-endpoint_only`, `38.9%` per `base-w16` con loss ausiliaria e `36.4%` per `dense-w32` con loss ausiliaria. Alcuni bracci mostrano un lieve ritorno dopo 30.000 step, confermando che la selezione del checkpoint rimane necessaria.

## Decisione

1. La soglia scientifica della Task 1 resta formalmente non superata.
2. La mappa held-voltage in-support e OOD-`dt` e appresa con errore molto inferiore a `1e-3` e stabilita ricorsiva forte.
3. Si puo procedere alla Task 2 sul gate `h` senza fingere che il debito scientifico sia risolto.
4. L'extrapolazione OOD-voltage di `tau_m` andra affrontata con un intervento mirato, soprattutto se il dominio operativo coupled includera quelle tensioni o se l'errore verra amplificato nell'accoppiamento `m+h+V`.

La conclusione resta confinata al gate `m` di `Ca_HVA` sotto voltage clamp costante; non viene estesa a traiettorie di tensione variabile, canale completo o neurone multicompartimentale.
