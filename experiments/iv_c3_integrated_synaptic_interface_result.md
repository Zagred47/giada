# GIADA IV-C3 — risultato dell'interfaccia sinaptica integrata

La revisione `acdac68` ha eseguito su Kaggle il protocollo preregistrato con teacher canonico NEURON 8.2.7. [Calibrazione](results/iv_c3_kaggle_acdac68/calibration_report.json) superata prima della matrice; [report finale](results/iv_c3_kaggle_acdac68/final_report.json) valido, processo exit 0, 36/36 celle confermative superate.

Nel dominio di quattro sinapsi EMS canoniche e un compartimento a voltaggio imposto, lo shadow causale online ha prodotto zero discrepanze nello stato plastico, nella sequenza Random123 e nel conteggio dei rilasci. Il massimo errore A/B sui recettori è `9,03×10⁻¹³`; sulla conduttanza `7,83×10⁻¹⁶ µS`, sulla corrente `6,37×10⁻¹⁴ nA`, sulla carica integrata `3,78×10⁻¹⁴ nA·ms`. Il restart nativo dell'intero sistema dopo il checkpoint ha errore zero su stato, corrente e RNG nei 36 casi.

Il supporto realizzato è presente per tutti i recettori: AMPA 87, NMDA 87, GABAA 153, GABAB 153 rilasci conteggiati nelle celle. I tre controlli negativi passano: pesi tutti nulli, impulsi programmati distinti dai rilasci, e segni delle correnti/blocco NMDA coerenti con il voltaggio.

Decisione: **IV-C3 supera il suo contratto isolato e autorizza la preparazione di Task33**. Non dimostra ancora che il compartimento con voltaggio autonomo o il neurone multicompartimentale funzionino: quelle sono le prove della composizione successiva. I risultati non usano rilasci futuri osservati dal teacher come input dello shadow.
