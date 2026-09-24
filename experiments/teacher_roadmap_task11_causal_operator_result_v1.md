# GIADA — roadmap Task 11: operatore causale Ca-HVA

## Integrità e perimetro

- Artefatto: `giada_roadmap_task11_causal_operator.zip`, SHA-256 `f64f6339ebe4a7927fdf050c6fbc4db6f812b3e1382a4d17609ed255adcdec52`.
- Revisione del run: `c2ffeb54133ccfce284518561b83e59a05f6dee9`, identica alla revisione pubblicata per il notebook.
- `valid=true`; hash freeze e dei nove checkpoint coerenti; sealed non usato per selezione; nessun futuro teacher come input modello.
- Ancora NEURON: 24 episodi verificati, massimo errore V `2.82e-12 mV`, massimo gate `1.24e-13`.
- Ruoli: 192/48/48 episodi train/development/sealed, 16 ms ciascuno. Sealed: 768 transizioni da 1 ms; 157 quiete, 484 moderati, 127 attivi secondo il cambiamento di V.

## Contrasti preregistrati

| Braccio | RMSE V one-step (mV) | RMSE m one-step | RMSE V ricorsivo 16 ms (mV) | RMSE m ricorsivo |
| --- | ---: | ---: | ---: | ---: |
| Effetto integrato, ingressi completi | 0.10048 | 0.01997 | 0.47194 | 0.06945 |
| Effetto integrato, correnti successive mascherate | 3.30236 | 0.02262 | 6.69580 | 0.06951 |
| Path a 4 nodi predetto, poi formula dei gate | 0.09672 | 0.000327 | 0.36937 | 0.005894 |

Il confronto reduced/full è appaiato, con capacità numerica e stream uguali: i tre quarti successivi della corrente sono soltanto mascherati nel braccio ridotto. Un controfattuale fisico con lo stesso input visibile al braccio ridotto ma impulso ritardato diverso cambia il voltaggio finale di almeno 6.12 mV e m fino a 0.0397. Il peggioramento reduced riguarda soprattutto V: nel gruppo attivo l'RMSE V one-step è 6.255 mV contro 0.157 mV del braccio completo. L'esito dimostra l'insufficienza di *questa* interfaccia ridotta, non di ogni rappresentazione locale.

Il path oracle a soli quattro nodi ha RMSE gate m `0.000178` e h `0.00000166` contro il riferimento denso; il path appreso ha RMSE dei nodi V `0.0773 mV` e ottiene RMSE m `0.000327`. Il path appreso viene poi integrato con 40 update analitici dei gate: è un risultato di accuratezza e apprendibilità, **non ancora una dimostrazione di accelerazione**.

I coefficienti integrati oracle `A,B` ricostruiscono il gate finale a precisione numerica (`m RMSE 2.83e-17`). Il modello che deve apprenderli mostra invece `A RMSE 0.0576`, `B RMSE 0.0178` e `m RMSE 0.01997` one-step. Il divario oracle-appreso localizza una difficoltà di apprendimento/parametrizzazione con questo budget; non smentisce la compressione matematica. Le curve development continuano a migliorare fra 200 e 400 step, quindi il limite di budget non è escluso.

Per H1, gli 8 stadi sequenziali danno RMSE V `0.0200 mV` e m `0.000661`; il predittore/correttore a 8 stadi ha V `0.0201 mV` e m `0.000229`. A 4 stadi, rispettivamente V `0.0448/0.0449 mV`, m `0.00152/0.000340`. Il predictor/corrector migliora i gate ma non il voltaggio in questo sistema. Il riferimento a 40 stadi coincide per costruzione. Le misure di tempo sono CPU su array piccoli, non benchmark GPU omogeneo.

Per H4, l'adattivo usa 8 stadi nel 48.6% delle transizioni e ottiene V RMSE `0.0366 mV` e m `0.00114`, fra i bracci fissi a 4 e 8 stadi. Il tempo CPU riportato è `0.00351 s` per il batch contro `0.00193 s` degli 8 stadi fissi: il controllo 1-vs-2 e l'overhead annullano il risparmio in questa implementazione. Nessuno speedup rivendicabile.

## Decisione scopiata

Il path causale compatto è il candidato più convincente per accuratezza dei gate in questo playground, ma il suo decoder impiega 40 aggiornamenti analitici. La prossima domanda riguarda una compressione *apprendibile e veloce* dell'effetto integrato o un solver accoppiato GPU a pochi stadi; prima va misurato il costo hardware a parità di batch. Non basta il valore RMSE one-step: il modello integrated-effect accumula errore su 16 ms.

Il teacher sealed non attraversa 0 mV: `teacher_threshold_crossings=0` per tutti i rollout. Perciò la F1 degli eventi è assente e **non abbiamo testato gli spike**. Il sistema contiene solo Ca-HVA+pas, con ECa fisso; niente calcio dinamico, sinapsi, altri canali, coupling assiale o teacher a 642 segmenti. Non promuovere questo esito a validazione del neurone completo.
