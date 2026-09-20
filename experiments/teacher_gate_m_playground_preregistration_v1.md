# GIADA Task 1 — playground atomico del gate `m`

La Task 1 studia esclusivamente la mappa causale sotto voltage clamp costante

\[
(V_t,m_t,\Delta t)\longmapsto m_{t+1}.
\]

Il target è prodotto in `float64` dalla formula di `Ca_HVA.mod`, già accettata dal doppio oracle della Task 0.4. Le reti ricevono soltanto `V_t`, `m_t` e `dt`: `m_inf` e `tau_m` sono diagnostiche privilegiate e non supervisionano il training.

## Bracci appaiati

- `direct_mlp`: controllo espressivo non strutturato;
- `physical_tau`: apprende `m_inf(V)` e `tau(V)>0`, poi usa l'update esponenziale;
- `direct_z`: apprende `m_inf(V)` e un update gate limitato in `[0,1]`;
- formula esatta, LUT lineare delle rate e persistenza come riferimenti non addestrati.

I bracci appresi condividono seed, minibatch, learning-rate grid, checkpoint e metriche definiti dalla Task 0.6. La selezione usa soltanto gli strati development e sceglie learning rate e checkpoint sulla media dei tre seed: non è consentito selezionare il seed più favorevole. Tutte e tre le repliche entrano nel risultato finale. I pesi e la scelta vengono congelati e identificati da hash prima di aprire una sola volta gli strati sealed-test.

## Gate preregistrato

La fase è positiva se almeno un modello appreso ottiene RMSE sealed aggregato non superiore a `1e-3`, il physical-τ soddisfa la stessa soglia e resta sotto `5e-3` nel rollout held-voltage a 1.000 passi. I bracci strutturati non possono produrre occupazioni fuori `[0,1]`.

Il confronto con la LUT è informativo e non viene trasformato retroattivamente in una soglia. Nessun risultato autorizza ancora affermazioni sul voltaggio variabile, sul neurone accoppiato o sulla sostituzione embedded del teacher.
