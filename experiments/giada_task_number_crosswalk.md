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
| Task 10: sufficienza dell'ingresso | Nessuno studio dedicato alla matrice completa degli ingressi | **Aperta.** TG-01 non la sostituisce. |
| TG-01: sottopassi interni full-state | Notebook con nome file storico `10_temporal_granularity_decision_matrix.ipynb` | Studio supplementare fuori roadmap; nessuna promozione implicita a Task 10. |
| Task 11 e successive | Nessun risultato attribuito da questo registro | Da seguire secondo i gate della roadmap. |

I nomi di file e i codici stabili SQLite precedenti restano alias storici per
non rompere hash, collegamenti o provenienza. Le etichette visibili e questo
registro distinguono l'identità scientifica degli studi. Prima di segnare una
task come completata, confrontare domanda, input, controlli, criteri e output
con la voce della roadmap; se non coincidono, assegnare un codice
supplementare senza riutilizzare il numero principale.
