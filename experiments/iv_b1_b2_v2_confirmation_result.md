# IV-B1/B2: conferma prospettica v2

Notebook Kaggle: [GIADA IV-B1/B2 calcium confirmation](https://www.kaggle.com/code/alessandrobelli/giada-iv-b1-b2-calcium-confirmation-8b3a4e5). Codice `8b3a4e5`, teacher canonico `074c466`, NEURON 8.2.7. L'artefatto integrale, il report e l'audit indipendente sono in `results/iv_b_v2_kaggle_8b3a4e5/`.

La v2 preregistrata supera **72/72 IV-B1** e **12/12 IV-B2** con le soglie numeriche immutate dalla v1. Peggiore RMSE del calcio IV-B1: **4,80×10⁻¹⁴ mM** (limite 10⁻⁶). Peggiore RMSE del gate SK ricomposto: **2,20×10⁻¹¹** (limite 10⁻⁴); peggiori RMSE della corrente SK **4,40×10⁻¹⁶ mA/cm²** e errore del clamp **1,57×10⁻⁵ mV**. Anche i protocolli nuovi `late_single` e `triplet` passano; tutti i contrasti impulso/riposo sono informativi.

Questo conferma prospetticamente che, nel passo fisso del teacher, `CaDynamics_E2` aggiorna `cai` e `SK_E2` usa quel valore aggiornato nello stesso passo. L'errore IV-B2 della v1 resta registrato come fallimento rispetto al suo vecchio riferimento; non è stato promosso retroattivamente.

Decisione: è superato il prerequisito per **preparare** la Task32 con feedback elettrico. Non sono ancora stati provati voltaggio autonomo, accoppiamento morfologico, sinapsi o accelerazione: in IV-B2 `ica` e V erano imposti. Non c'è stato training né accesso ad Airtable.
