# Registro di corrispondenza tra roadmap e studi

La numerazione vincolante è quella di `giada_modular_surrogate_roadmap.md`.
L'esistenza di un notebook con lo stesso numero **non** dimostra il
completamento di una task. Verificare sempre domanda, protocollo e gate.

| Voce roadmap | Studi effettivi | Stato rispetto alla roadmap |
|---|---|---|
| Task 1–6 | Notebook 01–06, con approfondimenti 1b, 2b, 3b–3e | Temi principali allineati; gli approfondimenti non sono nuove task principali. |
| Task 7: sequenze di voltage step esogene | Task 6 copre step singoli e bifasici; `07r_roadmap_voltage_step_sequences.ipynb` copre i contrasti temporali mancanti. I notebook storici 07/07b/07c riguardano Ca-HVA closed-loop. | **Conclusa nel dominio esogeno preregistrato**: ZIP 07r verificato, pilot NEURON e gate superati. Non attribuire i risultati closed-loop a questa voce. |
| Task 8: rampe e chirp | Notebook 08 | Tema allineato. |
| Task 9: percorsi fisiologici teacher-forced | Notebook 09, approfondimenti 09b/09c | Tema allineato. |
| Task 10: sufficienza dell'ingresso | `10_roadmap_input_sufficiency.ipynb`: otto viste del path Ca-HVA × cinque granularità d'integrazione sui percorsi Task 9. | **Conclusa nel perimetro diagnostico.** Prima vista campionata testata al gate: 21 campioni oracle/40 passi; non minimo universale. Contratto causale operativo ancora aperto. TG-01 non la sostituisce. |
| TG-01: sottopassi interni full-state | Notebook con nome file storico `10_temporal_granularity_decision_matrix.ipynb` | Studio supplementare fuori roadmap; nessuna promozione implicita a Task 10. |
| Task 11: operatore path-aware causale | `11_roadmap_causal_operator.ipynb`, fasi 11a/11b nello stesso run. | **Conclusa nel playground Ca-HVA+pas a un compartimento**: ancora NEURON e integrità superate; path causale a quattro nodi appreso, effetto integrato oracle esatto ma appreso meno accurato. Nessuna prova sugli spike o sul teacher multicompartimentale. |

I nomi di file e i codici stabili SQLite precedenti restano alias storici per
non rompere hash, collegamenti o provenienza. Le etichette visibili e questo
registro distinguono l'identità scientifica degli studi. Prima di segnare una
task come completata, confrontare domanda, input, controlli, criteri e output
con la voce della roadmap; se non coincidono, assegnare un codice
supplementare senza riutilizzare il numero principale.
