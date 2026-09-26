# GIADA supplemento Task 15c — ponte d'interfaccia embedded

Stato: **preregistrato, run Kaggle non ancora eseguito**. Questo studio precede ma **non sostituisce** la Task 16 originale (conferma embedded congelata, ancora teacher-forced). Notebook: `notebooks/15c_roadmap_embedded_interface_bridge.ipynb`.

## Perché serve

La MLP Task 15b riceve `V0,m0,h0,gbar_nominal,mask,E_Ca` e quattro valori di corrente **locale pianificata**. Il dataset autentico del teacher multicompartimentale non registra quattro valori equivalenti per ciascun sito: la traccia `somatic_current_na` è soltanto l'IClamp somatico, non la corrente locale totale. Sostituire i quattro ingressi con zeri, proxy o valori futuri invaliderebbe l'interfaccia e il confronto. Questo studio non attribuisce dunque una prestazione embedded al checkpoint 15b.

## Contrasti e perimetro

Fonte teacher: dataset targeted v1.1 base esatto, 29.240 transizioni, HDF5 verificato per SHA-256 e commit teacher. Solo split `validation`; nessun dato di test o held-out per selezione. Quattro siti preregistrati: segmenti 0, 387, 460, 469. Quattro regimi determinati dal solo path V: quiet, rising, spike, falling. Seed campionamento 15471; massimo 16 e minimo 8 percorsi per ciascuna delle 16 celle sito×regime. Griglia della microtraccia: 41 campioni in 1 ms (40 intervalli da 0,025 ms).

Confronto **congelato**, nessun training: formula Ca_HVA analitica, LUT lineare 513 e `physical_tau` width 32 nei tre seed Task 5 (17, 29, 43), caricati da freeze verificato. Per ciascuno degli stessi percorsi/inizializzazioni si valutano tre viste di V:

| Vista | Tensione di ogni sottopasso | Interpretazione |
|---|---|---|
| `left_available` | campione sinistro | Disponibile quando il sottopasso inizia, **non** tutta la sequenza al confine iniziale di 1 ms. |
| `midpoint_oracle` | media sinistro/destro | Ricostruzione teacher-forced; usa il campione destro del sottopasso. |
| `right_oracle` | campione destro | Oracle diagnostico, non input causale iniziale. |

Endpoint m/h autentici, RMSE m/h e di apertura `m²h`, violazioni di occupanza, errore candidato–formula e formula–teacher per ogni gruppo sono salvati. Il floor numerico della formula è esplicitamente distinto dall'errore del candidato. Non viene sostituita la corrente né la membrana del teacher.

## Decisione registrata prima dei risultati

Il confronto candidato–teacher è interpretabile solo se ciascuno dei 16 gruppi ha almeno 8 percorsi e, nella vista midpoint, sia RMSE m sia RMSE h della formula verso il teacher sono ≤0,005. Se fallisce, esito **NO-GO diagnostico**; non rilassare soglia né promuovere candidati. Se passa, selezionare per la futura verifica teacher-forced Task 16 il candidato con minimo della media non pesata, sui 16 gruppi, di `max(RMSE_m, RMSE_h)` nella vista midpoint. La vista left è il contrasto di causalità e la right il controllo oracle. Conservare anche le metriche continue e i singoli gruppi; nessuna conclusione su spike, corrente, membrane rollout o Gate C deriva da questa selezione.

La Task 16 dovrà usare un protocollo separato su test indipendente e dichiarare l'interfaccia effettivamente disponibile nel teacher. Se la LUT o un `physical_tau` viene scelto, è un **candidato diverso dalla MLP Task 15b**: non si deve retroattribuire a 15b una conferma embedded. La Task 17 resta l'eventuale sostituzione causale del meccanismo nel teacher.
