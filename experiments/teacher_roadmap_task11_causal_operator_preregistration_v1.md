# 🧭 GIADA — roadmap Task 11, operatore causale Ca-HVA

Questa è la **Task 11 originale**, con due fasi nello stesso notebook. La fase 11a controlla la sufficienza degli ingressi; la fase 11b confronta operatori numerici e appresi. Non estende il risultato al teacher multicompartimentale.

## Sistema e ancoraggio

Sistema minimo: un compartimento `Ca_HVA + pas`, capacità 1 µF/cm², `E_Ca=120 mV` fissato, `g_pas=0.0001 S/cm²`, `E_pas=-76 mV`. Lo stato all'inizio di ogni millisecondo è `(V,m,h)`. Conduttanza Ca-HVA nota; ingresso causale: quattro valori di corrente già pianificati per i quattro quarti successivi di millisecondo. Non ci sono release aleatori in questo sistema. L'interfaccia esterna resta 1 ms.

Prima di generare dati, la formula vettorizzata e il solver devono riprodurre i **24 episodi autentici NEURON 7b** verificati per hash: massimo errore V e gate `<=1e-8`. Il riferimento per i nuovi episodi usa la stessa semantica numerica a 0.025 ms. Il dataset nuovo è sintetico e piccolo, non un nuovo run NEURON né un test sul neurone completo.

## Ruoli e budget congelati

- Train: 192 episodi, seed generatore 11011; development: 48, seed 11029; sealed: 48, seed 11059. Durata 16 ms, 16 transizioni da 1 ms per episodio. Nessuna finestra attraversa ruoli. Il seed 11043 è stato usato durante la verifica tecnica preliminare del codice e non entra in questa prova sealed.
- Corrente fissata prima dell'episodio a quarti di millisecondo, con valori e impulsi ritardati; gbar e condizioni iniziali variano fra episodi. Sono disponibili anche quarti con corrente zero.
- Modelli GPU: MLP SiLU 2×64, AdamW 0.001, batch 256, 400 step, checkpoint diagnostici 50/100/200/400, seed 17/29/43. Medesimo stream per i bracci dello stesso seed. Scelta del seed solo su development all'ultimo checkpoint mediante `(RMSE V / 20 mV)+RMSE m+RMSE h+RMSE m²h`. Se nessun braccio raggiunge un risultato utile, il test sealed va comunque eseguito **una sola volta dopo freeze** per documentare il fallimento, senza ri-selezionare.

## Contrasti nello stesso run

| Ipotesi | Confronto preregistrato | Sonda decisiva |
| --- | --- | --- |
| H0 informazione | `effect_reduced`: `(V,m,h,gbar,corrente primo quarto)` contro `effect_full`: stessi più altri tre quarti | Coppie con identico input ridotto e impulso ritardato diverso; differenza nei target. Differenza di performance delle reti interpretata solo assieme al controllo di apprendimento. |
| H1 coupling numerico | Integratore sequenziale Ca-HVA+pas a 1/2/4/8/40 stadi e predittore/correttore con conduttanza di midpoint a 1/2/4/8 stadi, con correnti pianificate | RMSE V, m, h, m²h contro il riferimento a 40 stadi e tempo di calcolo. Solo il sequenziale a 40 stadi coincide col riferimento per costruzione. |
| H2 path compatto | Quattro nodi a 0.25/0.5/0.75/1 ms, prima oracle poi predetti da `path_full`, integrati con 40 update gate | Scarto oracle contro riferimento; scarto predetto contro oracle. Futuro teacher solo target, mai input del modello. |
| H3 effetto integrato | Coefficienti oracle `A,B` contro `effect_full` appreso con `x1=A*x0+B`, implementato con `A` e `q` vincolati in `[0,1]`, `B=(1-A)q` | Verificare identità oracle, errori gate endpoint, probe A/B e rollout ricorsivo. |
| H4 sforzo adattivo | Due stadi normalmente, otto se stima locale 1-vs-2 o variazione programmata della corrente segnala difficoltà | RMSE e costo reale rispetto a 2 e 8 fissi; costo della sonda incluso. |

Per ogni modello riportare transizione a stato teacher e rollout ricorsivo di 16 ms, occupazione fisica dei gate, RMSE di V/m/h/m²h, attraversamenti 0 mV e F1 di intervallo solo con supporto positivo e negativo. Riportare la composizione di regime e le distribuzioni degli errori, non solo la media. Misurare latenza CPU degli integratori come diagnostica comparativa, non asserire speedup GPU da essa. La metrica di velocità finale richiederà benchmark hardware omogeneo successivo.

## Decisione e limiti

La Task 11 è un playground causale d'identificabilità e apprendimento. Un braccio che fallisce a 400 step non prova impossibilità architetturale; checkpoint e seed indicano ottimizzazione e variabilità. Un oracle accurato con predittore scarso localizza il problema di apprendimento. L'errore native anchor invalida l'intero esperimento. Modelli e integratori ricevono soltanto stato iniziale e correnti pianificate; niente V futuro o gate teacher a inferenza. Il sealed non decide nuove soglie o architetture.

La scelta scientifica resta scopiata: niente calcio dinamico, altre conduttanze, sinapsi, assi dendritici, 642 segmenti o generalizzazione full neuron. Il passaggio a un sistema più complesso richiederà un esperimento distinto.
