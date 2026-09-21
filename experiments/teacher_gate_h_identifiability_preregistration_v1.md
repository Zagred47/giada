# 🔬 GIADA Task 2b — identificabilità del gate `h`

La Task 2b verifica causalmente se il fallimento rollout di physical-τ dipende dalla debole identificazione di `h_inf` e `tau_h` con la sola loss endpoint.

## Matrice 2×2

| Braccio | Rate supervision | Loss multi-orizzonte |
|---|---:|---:|
| `endpoint_only` | no | no |
| `rate_supervision` | sì | no |
| `multi_horizon` | no | sì, 1/10/100 passi |
| `rate_plus_multi_horizon` | sì | sì |

Seed, minibatch, capacità, learning rate e budget sono appaiati. La selezione usa esclusivamente development e minimizza l'errore medio agli orizzonti 1, 10 e 100. Il direct-z congelato della Task 2 è soltanto un controllo positivo e non viene riaddestrato.

Una nuova conferma disgiunta viene aperta una sola volta dopo il freeze. Per autorizzare la Task 3, il vincitore physical-τ deve ottenere RMSE fresh `≤1e-3`, rollout a 1.000 passi `≤5e-3`, RMSE `h_inf≤0,01`, RMSE `log(tau_h)≤0,1` e zero violazioni fisiche.
