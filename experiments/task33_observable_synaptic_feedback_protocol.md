# GIADA Task 33 — sinapsi osservabili nel compartimento attivo

Stato: preregistrata prima del run. IV-C3 e Task 32 v3 sono prerequisiti verificati per hash. Non è ancora un risultato.

Il singolo compartimento canonico della Task 32 mantiene 11 canali, CaDynamics, calcio/voltaggio autonomi e gli stessi sei checkpoint congelati (due famiglie × tre seed). Quattro sinapsi EMS stocastiche usano solo eventi presinaptici prescritti e stream Random123 espliciti. Lo shadow ricostruisce rilasci, plasticità, stati A/B e correnti; l'uscita NMDA usa il **voltaggio proprio** di ciascun rollout, non quello futuro del teacher.

Una cella di calibrazione separata verifica API, fase eventi e conversione da µS a densità di corrente; 16 celle confermative non sono usate per selezione. I controlli comprendono pesi nulli/RNG immobile, mancata equivalenza evento-rilascio, effetto misurabile delle sinapsi su V e divieto di usare correnti future native nella selezione. Il primo gate è il floor formula–NEURON: un suo fallimento non giudica i checkpoint. Solo dopo un floor ammissibile si valutano i sei modelli senza retraining, alle soglie V/cai congelate e a 40 ms. Gli errori del modello, se presenti dopo floor valido, sono un NO-GO scientifico di Task 33 nel perimetro registrato.

La [preregistrazione](task33_observable_synaptic_feedback_preregistration.json) contiene seed, protocolli, pesi, budget, soglie e limiti. Nessuna conclusione multicompartimentale o di accelerazione hardware è autorizzata da questo esperimento.

La prima calibrazione v1 (`34c96d5`) è stata arrestata prima di qualunque conferma: plasticità, RNG e rilascio combaciavano, ma gli eventi a metà tick producevano una differenza di fase A/B e corrente. Il [protocollo prospettico v2](task33_observable_synaptic_feedback_v2.json), vincolato agli hash del primo report, allinea gli eventi alla griglia fissa da 0,1 ms. Seed, pesi, soglie, checkpoint e numero di celle sono immutati. Il risultato v1 non è stato promosso.
