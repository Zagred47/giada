# GIADA Task 3c v2 — conferma fresh della cella congiunta m+h

## Esito

**NO-GO confermativo circoscritto.** L'artefatto è valido e il modello valutato è esattamente il candidato `symmetric_rates` congelato nella Task 3b, senza nuovo training e senza usare il fresh per la selezione. Il modello supera nuovamente il gate development, ma non supera tutti i limiti fresh preregistrati; la Task 4 resta quindi bloccata.

## Integrità sperimentale

- archivio: `giada_joint_m_h_symmetric_confirmation_v2.zip`;
- SHA-256 archivio: `ce62136ff6b6407f86e22146ecad8a7f2bb5e3b509147c49cce457431a2ad5f5`;
- revisione: `e4cf8df671e8e0beac5510e53646643c552a02ff`;
- origine checkpoint: esatti checkpoint Task 3b `symmetric_rates`, step 50.000, seed 17/29/43;
- riproduzione development: errore massimo `0.0`;
- retraining: assente;
- fresh usato per selezione: no;
- fresh disgiunto: 2.000 tuple, overlap con Task 3/development pari a zero;
- violazioni di occupazione `[0,1]`: zero.

## Risultati registrati

| Metrica | Risultato | Limite | Esito |
|---|---:|---:|---|
| Development, score medio | 0,102165 | ≤ 0,15 | PASS |
| Development, peggior seed | 0,113004 | ≤ 0,20 | PASS |
| Fresh, RMSE `m` | 0,009892 | ≤ 0,0025 | **FAIL** |
| Fresh, RMSE `h` | 0,000060 | ≤ 0,001 | PASS |
| Fresh, RMSE `m²h` | 0,008650 | ≤ 0,0025 | **FAIL** |
| Rollout 1.000, RMSE `m` | 0,000446 | ≤ 0,0025 | PASS |
| Rollout 1.000, RMSE `h` | 0,009667 | ≤ 0,005 | **FAIL** |

## Che cosa abbiamo imparato

La conclusione positiva della Task 3b sull'ottimizzazione resta vera: la supervisione simmetrica delle rate produce un candidato riproducibile e molto accurato nel supporto osservato. La conclusione più forte — generalizzazione nel dominio fresh preregistrato — è invece falsificata.

Il difetto è localizzato soprattutto nell'estrapolazione in tensione. Nel fresh in-support gli errori one-step sono dell'ordine di `3e-4` per `m`, `2e-6` per `h` e `2e-4` per `m²h`; anche l'OOD sul passo temporale resta contenuto. Nel fresh OOD-voltage, invece, il seed 29 raggiunge RMSE one-step `m=0,05956` e `m²h=0,05345`. Nel rollout OOD-voltage a 1.000 passi l'errore di `h` arriva a `0,04270` per lo stesso seed. Questo spiega contemporaneamente il fallimento delle metriche aggregate fresh `m`/`m²h` e del rollout `h`.

Non è un problema di vincoli fisici grossolani: non compaiono violazioni di occupazione. È un problema di supporto/identificabilità e robustezza delle funzioni di rate fuori dall'intervallo di tensione di sviluppo, con forte sensibilità al seed.

## Decisione

Non autorizzare la Task 4. Non fare tuning sul fresh appena aperto. Il prossimo intervento deve usare soltanto training/development per separare almeno:

1. insufficiente copertura di tensione;
2. parametrizzazione delle code di `m_inf`, `h_inf`, `tau_m` e `tau_h`;
3. varianza tra seed e vincoli monotoni/asintotici;
4. accumulo ricorsivo specifico di `h`.

Una futura conferma richiederà un nuovo insieme sealed indipendente.

## Limiti della conclusione

Il test riguarda la dinamica congiunta dei gate `m+h` di `Ca_HVA` sotto voltage clamp. Non autorizza conclusioni su voltaggio variabile accoppiato, corrente ionica completa o integrazione nel neurone multicompartmentale.
