# GIADA roadmap Task 14 — variazione appaiata dei parametri Ca-HVA

Stato: preregistrato, run Kaggle non ancora eseguito. Notebook `14_roadmap_parameter_variation.ipynb`. Prerequisiti immutabili Task11 e Task12/13 verificati via SHA-256. Nessun training, tuning o selezione di checkpoint. Il supplemento 11c non viene eseguito qui.

## Domande causali e matrice

48 episodi nuovi seed 14059, 16 ms, stessi stato iniziale di base e input di corrente pianificato per tutte le condizioni. Reference Ca-HVA+pas a un compartimento, 40 sottopassi da 0.025 ms per intervallo esterno da 1 ms. La ricorrenza con ECa=120 mV deve essere identica bit-a-bit al riferimento Task11; gate di validità: max differenza <=1e-11. Confronti uno-fattore-alla-volta:

| Condizione | Intervento | Ruolo interpretativo |
|---|---|---|
| baseline | gbar originale, ECa=120, gate iniziali originali | controllo |
| gbar_half | gbar ×0.5 | variazione nel supporto originale |
| gbar_one_half | gbar ×1.5 | possibile extrapolazione, descrittiva |
| eca_100 / eca_140 | ECa=100/140 mV | controllo di sufficienza dell'ingresso: ECa **non** è input del checkpoint |
| initial_m_plus / initial_h_minus | m0+0.1 / h0−0.1, clipping [0,1] | stato iniziale osservabile al modello |
| mechanism_off_visible | maschera=0, gbar efficace=0 passato al modello | assenza osservabile |
| mechanism_off_hidden | teacher off, modello vede gbar nominale | controllo negativo, non fallimento del modello |

Per il controllo di sufficienza dell'ingresso si aggiungono **twin controfattuali one-step**: ogni finestra da 1 ms parte dallo stesso stato di frontiera della baseline e riceve la stessa corrente e gbar, ma ECa=100 oppure 140. Questi twin hanno lo **stesso tensore numerico di input** e endpoint teacher differenti; fissano un lower bound informativo per qualunque predittore deterministico con quegli input. Le traiettorie complete ECa100/140 sono test separati: dopo il primo passo i loro stati e dunque gli input osservati divergono, quindi **non** sono presentate come coppie a input identico. La formula della corrente riceve l'ECa reale noto al front-end, ma i checkpoint Task11 non lo ricevono per prevedere l'evoluzione dello stato.

## Misure e contratti

Usare i modelli congelati `path_full` ed `effect_full` per predizione one-step di V/m/h, corrente analitica all'endpoint, rollout ricorsivo 16 ms, e errore del **cambiamento appaiato** di V e corrente rispetto alla baseline. Reportare per condizione RMSE V/m/h/I e validità del rollout. Ogni intervento usa le stesse schedule; la perturbazione è applicata prima della simulazione di riferimento. Nessun valore futuro teacher entra nel modello durante rollout.

Gate prestazionali registrati, separati dalla validità tecnica, per `gbar_half`, `initial_m_plus`, `initial_h_minus` e `mechanism_off_visible`: RMSE V e m one-step <=2× baseline dello stesso seed/ruolo, RMSE corrente <=3× baseline. `gbar_one_half` è OOD descrittivo; `eca_100/140` sono test di informazione assente; `mechanism_off_hidden` è controllo negativo. Il fallimento di uno di questi gate prestazionali è un risultato, non motivo per riscrivere soglie.

Gate tecnici: artifact SHA validi; identità baseline; tensore d'ingresso dei **twin one-step** ECa100/140 esattamente uguale e teacher V diverso; finitezza delle metriche. Fermarsi per corruzione input, divergenza dell'identità o stato non finito. Non si inferisce generalizzazione al teacher 642 segmenti, al calcio dinamico, CVode o ad altri meccanismi.

Output: `task14_parameter_matrix_report.json` e ZIP con metodo Blob/base64 concordato. La Task15 sarà un confronto architetturale separato; non è parte della Task14.
