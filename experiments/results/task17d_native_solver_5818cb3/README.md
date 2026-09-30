# GIADA Task 17d — sensibilità del solver nativo

Fonte: archivio Kaggle `giada_task17d_native_solver_5818cb3_618e6dba.zip`, SHA-256 `c5a31059964c5f355d0230906fa159983ad1cfda6e52f52ab064c80df38f128e`, eseguito alla revisione `5818cb3287ac44d31e313deef92a03d1e72167b7`. Contiene `final_report.json`, 17 confronti in `paired_metrics.json`, tracciati nativi, hash dei rilasci e tempi. Il processo termina con exit 0; report `valid=true`, teacher a 642 segmenti, controlli di ripetibilità identici, sei policy CVode applicate come previsto. Il report è diagnostico: `gate_c_authorized=false` e il precedente `CAUSAL_MICROCANARY_NO_GO` resta valido.

## Contrasti preregistrati

Il massimo RMSE di voltaggio sui quattro siti monitorati, rispetto a `ultra`, è:

| Condizione nota | default | tight | calcium_scaled | voltage_scaled | ultra_dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 170029, gbar 1,0 | 43,2270 mV | 0,6681 mV | 43,2327 mV | 0,1197 mV | 0 mV alla precisione riportata |
| seed 170083, gbar 1,5 | 5,6381 mV | 0,2667 mV | 5,6372 mV | 0,0041 mV | 0 mV alla precisione riportata |
| seed 170029, gbar 0,5 | 0,0429 mV | 0,0086 mV | 0,0429 mV | 0,0013 mV | non eseguito |

Nel sito 387 del primo caso, `default` attraversa 0 mV circa a 45,089 ms mentre `ultra` e `voltage_scaled` non lo attraversano; `calcium_scaled` resta simile a `default`. Nel secondo caso, il primo attraversamento del sito 387 è circa 0,200 ms in anticipo sotto `default` rispetto a `ultra`, ma quasi coincidente sotto `voltage_scaled`. Sono descrizioni di traiettorie su casi di sviluppo già noti, non stime di generalizzazione.

## Interpretazione e limiti

La sensibilità alla precisione del voltaggio è molto più forte di quella alla sola precisione di `cai` nei casi esaminati. Questa è evidenza a favore del voltaggio come leva numerica prossimale, ma non identifica da sola quale termine biofisico a valle amplifichi l'errore. La coincidenza `ultra_dense`–`ultra` mostra che, a tolleranza ultra, aumentare la frequenza delle chiamate `solve` da 0,025 a 0,005 ms non cambia i tracciati ai numeri riportati per due casi; non è una dimostrazione generale sulla sufficienza del passo di modellazione da 1 ms.

`all_release_hashes_match_ultra=false` richiede cautela. L'hash è costruito dall'intero `CausalReleaseOutcome.to_dict()`: include `pre_synapse_state`, `post_synapse_state`, probabilità e altre grandezze continue oltre a `release_success` e alla sequenza RNG. Una differenza di hash **non dimostra** che la decisione discreta di rilascio sia cambiata. L'archivio conserva soltanto gli hash, non le righe degli outcome, quindi non permette di separare ex post le due possibilità. L'audit di bordo risulta valido e le repliche per policy sono identiche; tuttavia non si può rivendicare un contrasto con tutti gli input realizzati appaiati finché i successi di rilascio non vengono confrontati esplicitamente.

Conclusione: risultato diagnostico utile, non GO per sostituire il meccanismo né prova definitiva di convergenza biologica del riferimento `ultra`. Il prossimo controllo mirato dovrebbe salvare un fingerprint dei soli eventi discreti/RNG e confrontare gli outcome completi fra policy, senza riaprire la selezione del candidato congelato.
