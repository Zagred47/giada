# GIADA supplemento Task 15c — esito del ponte d'interfaccia

Artefatto utente `giada_roadmap_task15c_embedded_interface_bridge.zip`, SHA-256 `8a0aa1f85dbc1d62e639f3fcf1db4e0e4ecd28e6e5abfccc1a17cfead140a25a`; report interno SHA-256 `ffde98ea986cc6dc293688b0e9335efb6af60639fe95281440df17523c4b2fc9`. Codice `fb8a55cdb038628af6dd35cac9b576dfb4b3ff99`, coincidente con la revisione preregistrata. Report valido; Task 5, Task 15b e HDF5 teacher identificati dagli hash attesi. Nessun retraining, nessuna apertura degli split teacher di test, nessuna valutazione della MLP 15b con input fittizi.

## Gate preregistrato

Sono stati verificati 5.280 indici `validation`; selezionati 248 percorsi sito×transizione da 144 transizioni e 45 traiettorie. Tutti i 16 gruppi sito×regime hanno almeno 8 percorsi: 14 gruppi con 16, `387:spike` con 13, `460:spike` con 11. I 248 record dell'indice allegato sono tutti `validation`, senza coppie sito×transizione duplicate. Il peggior RMSE m/h formula–teacher per gruppo nella vista midpoint è 0,00026183 (`460:spike`), contro limite preregistrato 0,005. **Floor e supporto passano.**

| Vista V | Formula–teacher RMSE m globale | LUT–teacher RMSE m globale | Lettura |
|---|---:|---:|---|
| left_available | 0,00357153 | 0,00357958 | Il floor dovuto al campionamento sinistro domina il piccolo scarto LUT–formula. I campioni V sono disponibili soltanto progressivamente nei sottopassi, non tutti all'inizio del millisecondo. |
| midpoint_oracle | 0,00015542 | 0,00015818 | Floor molto più basso, ma il punto medio usa anche il campione V futuro del sottopasso: teacher-forced, non input causale iniziale. |
| right_oracle | 0,00344623 | 0,00343834 | Controllo oracle, meno fedele del midpoint sul medesimo insieme. |

Nella vista midpoint, gli score macro preregistrati (media sui 16 gruppi di `max(RMSE_m,RMSE_h)`) sono: LUT513 `0,00013356`; `physical_tau` seed43 `0,00021748`, seed29 `0,00028056`, seed17 `0,00035771`. Ricalcolo indipendente dal report conferma l'ordinamento. Nessuna violazione di occupanza nelle tre viste per i quattro candidati. La LUT513 è quindi il **candidato congelato selezionato per una Task 16 teacher-forced**, come previsto dal protocollo; non è la MLP Task15b.

## Interpretazione e limiti

La Task 15c dimostra che l'interfaccia a path V del primitivo LUT può essere confrontata con i gate Ca_HVA del teacher su transizioni vere, con un floor midpoint sufficientemente piccolo. Non prova che una rete m/h appresa abbia battuto la LUT, né che l'intero neurone sia accurato in rollout. L'errore left è circa 23 volte quello midpoint già per la formula: non attribuire quel divario alla LUT. I percorsi campionati sono correlati perché provengono da 45 traiettorie; non trattare 248 righe come repliche biologiche indipendenti.

La Task 16 originale resta una conferma embedded separata, su test indipendente, con V teacher-forced e una definizione esplicita del momento in cui il path V diventa disponibile. La Task 17 e Gate C non sono autorizzati da questo report. Il checkpoint MLP 15b rimane non testabile in modo equo nel teacher attuale senza prima costruire un'interfaccia di corrente locale equivalente; non usare corrente somatica IClamp o zeri come surrogato.
