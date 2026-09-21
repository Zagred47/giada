# 🧪 GIADA Task 2 — playground atomico del gate `h`

La Task 2 studia esclusivamente la mappa causale held-voltage

\[
(V_t,h_t,\Delta t)\longmapsto h_{t+1}.
\]

Il target `float64` proviene dalla formula `Ca_HVA.mod` gia validata dal doppio oracle. Le reti ricevono soltanto `V_t`, `h_t` e `dt`; `h_inf` e `tau_h` rimangono diagnostiche privilegiate e non diventano input o target ausiliari.

## 🧠 Cosa ereditiamo dalla Task 1b

La Task 1b ha mostrato che, per `m`, il fattore robusto era il budget e non una griglia piu larga di capacita, densita o obiettivi. Per `h` non ripetiamo quindi dodici bracci ridondanti:

- learning rate fissato a `0,003`;
- tre seed appaiati `17`, `29`, `43`;
- checkpoint `0`, `1k`, `3k`, `10k`, `30k`, `50k`;
- selezione del checkpoint sulla media development dei tre seed;
- apertura unica del sealed test soltanto dopo il freeze.

Questo produce nello stesso run una mini scaling law e permette di distinguere errore di rappresentazione da budget incompleto.

## ⚖️ Bracci appaiati

- `direct_mlp`: controllo non strutturato che predice direttamente `h_t+1`;
- `physical_tau`: apprende `h_inf(V)` e `tau_h(V)>0`, poi applica il solver esponenziale;
- `direct_z`: apprende `h_inf(V)` e un gate `z(V,dt)` limitato;
- formula, LUT lineare delle rate e persistenza come riferimenti non addestrati.

Per `h`, il fattore di conduttanza e lineare (`m²h`): la metrica di conduttanza usa quindi potenza uno per `h`, non `h²`.

## ✅ Decisioni preregistrate

Il gate scientifico richiede RMSE sealed aggregato non superiore a `1e-3` per almeno un modello e per physical-τ, rollout physical-τ a 1.000 passi non superiore a `5e-3`, metriche finite e zero violazioni di `[0,1]`.

Separatamente, il gate ingegneristico permette di procedere alla cella congiunta `m+h` con RMSE physical-τ non superiore a `2,5e-3`, mantenendo l'eventuale debito di accuratezza esplicito. Non e consentito presentare il secondo come superamento del primo.

I risultati saranno separati per in-support, OOD-voltage e OOD-`dt`, includendo errori su `h_inf` e `log(tau_h)`. Nessun esito autorizza ancora affermazioni sul voltaggio variabile, sul canale completo o sul neurone accoppiato.
