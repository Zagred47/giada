# Task 30b — conferma con esposizione attiva

L'audit indipendente dell'artefatto Kaggle `ddbffb0` è valido. Il primo run della Task 30 resta non decision-grade: nessun campione di corrente iniettata precedeva l'orizzonte primario di 8 ms. La Task 30b corregge prospetticamente il protocollo, senza cambiare checkpoint congelati né soglie numeriche.

Nel nuovo run, tutti i 24 episodi non-rest hanno corrente iniettata prima di 8 ms (1.440 campioni); sei traiettorie di riferimento attraversano -20 mV entro l'orizzonte primario. Il controllo passivo NEURON concorda con il solver semi-implicito. Per ogni seed e famiglia nel braccio `both`, RMSE pooled a 8 ms < 0,003 mV e peggior episodio < 0,010 mV, contro limiti preregistrati rispettivamente di 1 e 2 mV. Le 1.920 righe, i checksum, l'esposizione e le soglie sono verificati in `results/task30b_kaggle_ddbffb0/result_audit.json`.

**Decisione scientifica: GO circoscritto.** I gate ionici congelati mantengono un errore di voltaggio molto basso in questo sistema autonomo a singolo compartimento, con stimoli attivi anticipati. Non è una convalida di un neurone multicompartimentale né del comportamento attivo completo di NEURON: il riferimento attivo è la mappa interna a formule esatte, mentre NEURON è stato usato per calibrare soltanto il caso passivo. Il calcio e la corrente iniettata sono imposti; restano fuori CaDynamics, accoppiamento assiale, sinapsi e morfologia. Nessuno speedup è dimostrato; Gate D prestazionale resta NO-GO. Task 31 non è autorizzata automaticamente: prima occorre una decisione separata sulla validazione nativa attiva e sul suo scopo.

Notebook Kaggle: https://www.kaggle.com/code/alessandrobelli/giada-task30b-active-exposure-ddbffb0
