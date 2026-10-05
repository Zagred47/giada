# IV-B1/B2: primo run nativo e correzione semantica

Run Kaggle `b40e07a`: [notebook](https://www.kaggle.com/code/alessandrobelli/giada-iv-b1-b2-calcium-prerequisites-b40e07a). Artefatto integrale, report e tracce sono in `results/iv_b_kaggle_b40e07a/`.

- IV-B1 supera 48/48 condizioni con `CaDynamics_E2` canonico: il massimo RMSE del calcio è 4,80×10⁻¹⁴ mM, contro il limite registrato di 10⁻⁶ mM.
- IV-B2 **non supera** la v1. Il calcio rimane accurato, il clamp è entro contratto e l'impulso modifica chiaramente calcio e SK; lo scarto del gate (fino a RMSE 0,00347) proviene dal riferimento che leggeva `cai[t]`.
- Audit post-hoc dei byte nativi: `SK_E2` in NEURON usa il calcio già aggiornato nello stesso passo. Con `cai[t+1]` il massimo RMSE *one-step* del gate è 5,51×10⁻¹⁷. È un'attribuzione, **non** una promozione retroattiva della v1.

La v2 è preregistrata in `iv_b1_b2_calcium_prerequisite_v2.json`: aggiorna la semantica causale interna al passo, lascia invariate tutte le soglie numeriche, conserva i vecchi protocolli e aggiunge `late_single` e `triplet` come conferma indipendente. Fino all'esito v2, il feedback elettrico della Task 32 non è autorizzato. Nessun training o speedup è stato dimostrato.
