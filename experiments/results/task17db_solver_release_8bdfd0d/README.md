# GIADA Task 17d-b — rilascio e riferimento numerico

Fonte: `giada_task17db_solver_release_8bdfd0d_4d019be5.zip`, SHA-256 `bad9cb4603906a703cee38d406b69caa148b2e3daa188fc7afc56df3f31c2b522`, revisione eseguita `8bdfd0d1453e0b7169de9cfc25804ecd3ebfec16`. L'archivio contiene report finale, nove confronti nativi, nove confronti con i candidati, decisioni di rilascio per 21 episodi, log e stato del processo. Exit 0, `valid=true`, teacher 642 segmenti e policy applicate. Gate C resta falso.

Per ogni condizione ci sono 20 eventi programmati. I 21 episodi riproducono gli stessi eventi, draw Random123, successi/fallimenti e quantità rilasciate: 14 successi per seed 170029/gbar 1,0; 13 per seed 170083/gbar 1,5; 14 per seed 170029/gbar 0,5. Anche le differenze massime di probabilità e quantità registrate nei 18 confronti sono zero. Gli hash completi della 17d differivano perché comprendevano stati sinaptici continui non conservati in quell'archivio; la 17d-b risolve il dubbio *per queste condizioni e questi campi salvati*.

## Solver nativo

Massimo RMSE di voltaggio tra i quattro siti, rispetto a `ultra` (mV):

| Seed/gbar | default | voltage_scaled | super_ultra |
| --- | ---: | ---: | ---: |
| 170029/1,0 | 43,226966 | 0,119684 | 0,006729 |
| 170083/1,5 | 5,638085 | 0,004086 | 0,003001 |
| 170029/0,5 | 0,042933 | 0,001267 | 0,000097 |

Il massimo errore gate `ultra`–`super_ultra` è rispettivamente 0,000391, 0,000263 e circa 0,000010. Tutte le condizioni passano le soglie diagnostiche preregistrate di 0,05 mV RMSE e 0,002 errore massimo gate. `ultra` è quindi un riferimento operativo stabile *nei casi esaminati*. Il contrasto forte del primo caso fra default e ultra non è causato da un diverso rilascio sinaptico registrato.

## Canale Ca_HVA alla stessa policy ultra

Massimo sui quattro siti per ciascuna condizione:

| Braccio | 170029/1,0: V RMSE; gate | 170083/1,5: V RMSE; gate | 170029/0,5: V RMSE; gate |
| --- | --- | --- | --- |
| Clone formula | 0 mV; 0 | 0 mV; 0 | 0 mV; 0 |
| LUT 513 float32, quattro rate | 0,177680 mV; 0,010317 | 0,133836 mV; 0,011229 | 0,000745 mV; 0,000083 |
| LUT 2049 float64, soli rate m | 0,010962 mV; 0,000637 | 0,008718 mV; 0,000731 | 0,000046 mV; 0,000006 |

Il controllo formula passa esattamente ai numeri riportati. La LUT 513 supera di poco il limite gate 0,01 della Task 17a nei due casi critici, pur avendo RMSE V inferiore al limite storico di 2 mV. La LUT 2049 rientra in entrambi i limiti su questi casi, ma è un braccio diagnostico nuovo, non il candidato congelato e non un nuovo GO causale. Il suo errore di voltaggio è dello stesso ordine del piccolo scarto `ultra`–`super_ultra`; serve conferma indipendente con protocollo e seed non già usati per guidare la progettazione.

Questa esecuzione chiarisce che, nei casi noti, la sensibilità numerica del teacher è dominata dall'impostazione del voltaggio e che aumentare la risoluzione dei rate m riduce nettamente l'errore del surrogato. Non dimostra ancora validità generale, vantaggio computazionale né autorizza una sostituzione nella simulazione completa. La tabella 513 è ricostruita dalla formula canonica con la procedura della Task 17a; l'identità byte per byte con un artefatto storico esterno non è rivendicata.
