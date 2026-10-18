# Task 39c: risultati, errori residui e proposta Task 39d

Analisi del 18 ottobre 2026. Proposta esplorativa; non è ancora una preregistrazione eseguibile e non autorizza Task 40.

## Risultato verificato

Kaggle v1 completato, revisione scientifica 33ae2ba96d04c15ed75bc6c8bef8a4edb3ed918f. Freeze e checkpoint verificati; 1.872 casi ricorrenti e 20.592 valutazioni di canale, tutti finiti, nessuna violazione di occupazione. Le 36 esecuzioni addestrate sono complete. Nessun braccio supera tutti i criteri: NO_GO_TASK40_BLOCKED, soglie immutate, conferma indipendente non aperta.

Il migliore nuovo modello passa da 16/72 in 39b a 44/72 in 39c. Sono 24 traiettorie per tre seed, non 72 repliche indipendenti. È un progresso descrittivo fra esperimenti con training, budget e selezione differenti; non identifica un singolo intervento causale. Il riferimento storico local161 con SK canonico supera 53/72 e resta una baseline parzialmente analitica.

Entro 39c, tre dei sedici contrasti preregistrati sono supportati: due aggiunte di supervisione dei coefficienti e un'aggiunta di durata ai target held. Nessun contrasto architetturale soddisfa congiuntamente tutte le condizioni. Fra 60k e 120k migliorano 32/36 score di selezione, riduzione mediana 24,27%; fra 90k e 120k migliorano 20/36, riduzione mediana 2,58%. Questo non prova che basti aumentare indefinitamente il budget.

Il benchmark residente T4, 642 siti indipendenti per 80 passi, mostra circa 76,28 ms per la formula vettorizzata e 9,53 ms per la rete base: circa 8x sul blocco ionico, stesso backend CUDA Graph. Mancano equivalenza qualitativa, confronto LUT e migliore kernel analitico, costo operativo e compartimento completo. Nessuna promozione o accelerazione end-to-end dichiarata.

## Dove falliamo

| Modello | Casi completi | Gate falliti | RMSE corrente | Massimo corrente | Carica | Timing | Solo timing |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_i1_s0_d1 | 41/72 | 0 | 22 | 16 | 17 | 15 | 9 |
| wide_i1_s1_d1 | 44/72 | 1 | 20 | 16 | 16 | 14 | 7 |

Le colonne di fallimento si sovrappongono. Nel base: quattro onset, otto offset, tre picchi di canale falliscono; nel wide: quattro onset, sei offset, quattro picchi. Sono dettagli degli eventi estratti dal valutatore, non una nuova ricostruzione delle tracce.

Le famiglie problematiche includono activation_boundary, ramps, same_v_history, slow_recovery e rapid trains. Esempio wide: NaTa_t vicino alla soglia ha RMSE assoluto 2,693e-7 mA/cm² e scala di valutazione 1,086e-7; la scala globale di training è circa 361 volte maggiore. Il quadrato normalizzato contribuisce quindi circa 130mila volte meno rispetto alla normalizzazione locale, prima di considerare altre medie e pesi. Nel base, un caso Ca_LVAst ha rapporto delle scale circa 923. Non sono ragioni per esentare correnti deboli: sono indizi di disallineamento dell'obiettivo.

Il superamento delle soglie gate non garantisce le soglie correnti: il readout usa prodotti e potenze delle occupazioni; piccole differenze assolute possono diventare grandi errori relativi quando l'apertura è piccola. Le soglie gate aggregate non impongono da sole un limite alla corrente di ogni canale.

Il timing non è semplicemente un ritardo uniforme: same_v_history/Im mostra anche errori di onset circa 1190 ms, mentre molti altri errori sono 0,4–22,5 ms. Il valutatore usa il primo/ultimo attraversamento del 10% del transiente rispetto al valore iniziale del teacher e un intervallo di plateau per il picco. Un piccolo errore vicino a soglia può spostare l'attraversamento a una fase successiva; serve ispezione delle tracce per distinguere questa discontinuità da una dinamica davvero fuori fase. I file timing NPZ attuali appartengono agli oracoli dei parent 39b, non ai candidati 39c: non li confondiamo.

La loss attuale ha MSE di gate/correnti su finestre variabili di 32 passi (3,2 ms) e target held anche di 10 s, più supervisione u/log(z). Non contiene un termine esplicito di carica o timing. Un endpoint a V costante per 10 s non sostituisce una lunga storia a V variabile. I pesi ausiliari, calibrati all'inizializzazione, sono circa 520 e 563; il clipping è frequente. Tuttavia nel probe finale del base migliore le norme ausiliarie pesate e quelle di traiettoria sono circa comparabili: non abbiamo prova di dominanza universale o conflitto. Vanno misurati coseni, aggiornamenti effettivi e contributi per gate.

Gli oracoli 39c riguardano pesi parent 39b: sostituire u migliora molto le code peggiori; u+z esatti superano 72/72; il solo FP64 non ripara quei parent. Non dimostrano quale coefficiente causi gli errori dei nuovi pesi 39c.

## Ipotesi e falsificazione

1. **Obiettivo correnti poco sensibile ai casi deboli** (priorità alta). Testare normalizzazione per traiettoria/canale con floor fisso, contro globale, stessi campioni e pesi iniziali. Falsificazione locale: nessun miglioramento consistente nei casi deboli a esposizione/budget appaiati, senza peggiorare i forti.
2. **Esposizione insufficiente alle storie e alle code** (alta). Testare campionamento fit-only bilanciato fra transiente, soglia, recupero e corrente debole, con finestre causali più lunghe/burn-in. Riportare sia confronti a pari aggiornamenti sia a pari target-tick/tempo: non confondere più dati/calcolo con miglior campionamento. Mini scaling lungo la stessa traiettoria. Falsificazione: guadagno assente sulle categorie mirate dopo verifica della copertura.
3. **Condizionamento e interferenza degli obiettivi** (media). Contro pesi fissi iniziali, calibrazione prospettica per fasi esclusivamente fit, con pesi congelati dentro ogni fase; tutti i termini restano presenti. Stessa soglia di clipping. Misurare norme, coseni, frequenza clipping e update/parametro; il clipping frequente da solo non prova un problema. Falsificazione: calibrazione cambia realmente i gradienti ma non migliora l'esito appaiato.
4. **Loss puntuale non controlla carica e timing** (medio-alta). Aggiungere un termine di carica a finestre e un surrogato differenziabile degli attraversamenti, calcolati solo sul fit; conservare la metrica originale per decidere. Falsificazione: nessuna riduzione appaiata degli errori temporali/carica a pari esposizione. Non correggere l'onset sottraendo a posteriori il baseline del candidato.
5. **Bias architetturale inadeguato nelle code** (aperta, dopo le precedenti). Testare una piccola componente lineare apprendibile nell'output logit, oltre al ramo nonlineare, per rappresentare meglio piccole aperture e rilassamenti. Stesso u/z appreso, 18 stati e readout, nessuna formula cinetica in inferenza. Controllo a budget di parametri comparabile, costo misurato. Un miglioramento su un regime con regressione altrove non basta.
6. **Precisione o sensibilità dell'estrattore degli eventi** (diagnostica obbligatoria). Sui nuovi checkpoint congelati: FP32/FP64 stessi pesi, swap u/z/per-gate esatti solo come oracoli non selezionabili, controllo teacher completo, tracce complete degli eventi, slope/margine alla soglia, confronto shape e carica. Se il pass/fail cambia sotto perturbazioni entro il floor numerico del teacher, documentare sensibilità; nessuna riclassificazione retroattiva o modifica delle soglie.

## Un solo notebook proposto, Task 39d

Prima dei training, sidecar congelati sui due parent 39c (base 41/72 e wide 44/72): attribuzione u/z e gate/canale, FP64, ricostruzione timing. Un controllo teacher non conforme ferma il training per errore tecnico.

Matrice principale 2x2x2: normalizzazione corrente globale/locale × esposizione ordinaria/mirata alle storie × pesi loss iniziali/calibrati per fase. Otto bracci. Due bracci aggiungono il termine carica/timing, rispettivamente alla baseline e alla combinazione completa; altri due aggiungono il bias logit apprendibile agli stessi due riferimenti. Totale massimo dodici bracci per tre seed, vettorizzati in coorti; stesso livello di astrazione. L'effetto del pacchetto carica/timing va dichiarato come pacchetto: non attribuirlo separatamente a ciascun termine.

Usare inizializzazioni appaiate da zero nel confronto principale e conservare i due parent congelati come riferimenti; evitare una continuation selettiva del solo vincitore. Determinare lunghezze, pesi, budget e costo con preflight fit-only prima della preregistrazione definitiva. Checkpoint crescenti lungo la stessa traiettoria, common development selection preregistrata, stress solo post-freeze. I casi development già osservati sono diagnostici e non possono diventare una nuova conferma indipendente. Nuove traiettorie di conferma solo dopo freeze e quando il gate previsto dalla roadmap lo autorizza.

Salvare pass/fail per canale/famiglia/seed, errori assoluti e relativi, carica, onset/picco/offset, peggior caso, code lunghe, blocchi e inversione; separare effetti informativi, costo di training e latenza di inferenza. I sidecar si eseguono nello stesso run ma non diventano candidati promossi. Task 40 resta bloccata.

## Precedenti interrogati nel mirror

Query esatta parametrizzata su findings e codici riportati in residual_diagnosis.json:
- Task 2b: supervisione dei rate/multihorizon ripara identificabilità endpoint, nel suo dominio.
- Task 3d: copertura in V e prior condizionale aiutano; allargare o separare non basta.
- Task 24b: clipping inattivo; spiegazione via clipping non supportata in quel run.
- Task 26: coseni negativi dei gradienti osservati, senza attribuzione causale automatica.
- IE-P6: squilibrio di gradienti osservato, e loss temporale testata peggiorava; aggiungere termini non garantisce un guadagno.

Questi precedenti motivano controlli, non trasferiscono automaticamente la loro conclusione alla 39c.
