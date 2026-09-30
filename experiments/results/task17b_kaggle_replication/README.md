# Task 17b — replica tecnica Kaggle

Lo ZIP fornito dall'utente `giada_task17b_rate_attribution_4ee6b57_5e2fac30.zip`
(SHA-256 `d509f51b507a72c44bf00ef21fce759f0c4cc7abe90bd26c5a1bcded6dc3ef77`)
contiene una matrice Task17b completata sul commit GIADA `4ee6b57`.
`process_status.json` riporta exit code 0; `final_report.json` riporta
`valid=true`, `decision=DIAGNOSTIC_ONLY`, `gate_c_authorized=false`, 642
segmenti. Il replay nativo ripetuto e il clone-formula hanno errore zero.

La replica conserva i due casi con errore gate della LUT >0,01:

| Seed | gbar | Sito | Errore massimo m, LUT |
| --- | ---: | ---: | ---: |
| 170029 | 1,0 | 460 | 0,01835139 |
| 170083 | 1,5 | 0 (soma) | 0,01719944 |

La coppia `mInf+mTau` riproduce quasi la LUT completa; il massimo divario
tra i rispettivi errori gate è circa `3,024×10⁻⁵` sui 36 confronti
condizione–sito. Il massimo scarto assoluto di una metrica numerica tra run
locale e Kaggle è `6,2063×10⁻⁶` (mV, `voltage_max_error_mv`): differenze
minime di calcolo, nessun cambio della diagnosi.

Questa è una **replica tecnica con gli stessi seed**, non un nuovo test
indipendente né una stima della frequenza di fallimento. Il NO-GO Task17a
resta valido e Gate C non è autorizzato. Qui sono versionati report, metriche
e stato del processo; il tracciato completo del caso difficile rimane nello
ZIP originale dell'utente e in `artifacts/task17b_kaggle_import/` sul
workspace locale, non nel repository.
