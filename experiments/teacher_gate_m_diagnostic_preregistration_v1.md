# GIADA Task 1b — diagnosi dati, budget, capacità e obiettivo

La Task 1b non modifica il verdetto della Task 1. Usa una matrice fattoriale `2×3×2` della sola famiglia physical-τ per distinguere quattro spiegazioni concorrenti dell'errore residuo:

- densità insufficiente dei punti fit;
- capacità insufficiente (`width 8/16/32`);
- obiettivo endpoint poco identificabile rispetto alla supervisione ausiliaria di `m_inf` e `log τ`;
- budget insufficiente, con checkpoint fino a 50.000 step per cinque varianti preregistrate.

La supervisione delle rate è soltanto un target diagnostico e non diventa mai un input del modello. Tutti i confronti usano tre seed e stream appaiati entro ciascuna densità.

I contrasti causali su densità, width e obiettivo vengono calcolati al checkpoint comune fisso di 10.000 step. Le estensioni a 30.000–50.000 step entrano soltanto nel contrasto sul budget e nella scelta finale del candidato: non possono contaminare gli altri effetti fattoriali.

## Nuova conferma

Il sealed set della Task 1 è ormai aperto e non può selezionare nuove varianti. La Task 1b congela selezione, pesi e hash usando soltanto development, poi apre una nuova griglia disgiunta di punti half-step, stati e `dt`, includendo nuovi punti OOD.

## Due decisioni non intercambiabili

1. **Gate scientifico:** RMSE fresh ≤ `1e-3`, rollout 1.000-step ≤ `5e-3`, zero violazioni fisiche.
2. **Gate di prosecuzione ingegneristica:** RMSE fresh ≤ `2,5e-3`, stesso gate rollout e zero violazioni. Permette di iniziare il gate `h`, ma conserva esplicitamente il debito di accuratezza di `m` e non può essere descritto come superamento del gate scientifico.
