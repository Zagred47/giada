# GIADA Task 17c — esito diagnostico locale

Esecuzione nativa Ubuntu/NEURON 8.2.7 completata (`process_status.returncode=0`)
nel teacher a 642 segmenti. Il clone-formula e i replay nativi ripetuti
coincidono esattamente a tutte le tolleranze testate. Le funzioni m
interpolate nei quattro NMODL compilati superano il probe dei valori attesi.
Il risultato resta `DIAGNOSTIC_ONLY`; la Task17a resta NO-GO e Gate C falso.

## Griglia versus precisione del valore

Massimo errore gate sulle tre condizioni e quattro siti, sempre rispetto al
nativo eseguito con la *stessa* tolleranza CVode:

| Braccio | Solver originale (`atol=0,001`, `rtol=0`) | Solver stretto (`atol=1e-5`, `rtol=1e-6`) |
| --- | ---: | ---: |
| LUT Task15c congelata, 513 float32, tutti i rate | 0,018351 | 0,010754 |
| Solo m, 513 float32 | 0,018382 | 0,010804 |
| Solo m, 513 float64 | 0,018381 | 0,010804 |
| Solo m, 1025 float64 | 0,003657 | 0,002772 |
| Solo m, 2049 float64 | 0,001760 | 0,000673 |

Il passaggio float32→float64 **senza aggiungere nodi** ha effetto minimo;
raddoppiare e quadruplicare gli intervalli riduce nettamente l'errore in
entrambi i solver. Anche l'errore statico massimo di `mInf` sulla banda
[-80,55] mV scende da circa 1,154×10⁻⁴ (513) a 2,886×10⁻⁵ (1025) e
7,630×10⁻⁶ (2049). Ciò localizza la sorgente principale nella
interpolazione dei rate m, non nella sola quantizzazione float32. I bracci
1025/2049 sono **sonde diagnostiche**; il superamento delle condizioni note
non costituisce selezione o validazione indipendente.

## Sensibilità numerica del teacher

Il nativo stesso cambia quando si stringe CVode. Il massimo RMSE V sui
quattro siti, nei due casi critici, è:

| Confronto tolleranze | seed 170029, gbar 1,0 | seed 170083, gbar 1,5 |
| --- | ---: | ---: |
| originale → stretto | 13,526 mV | 5,937 mV |
| stretto → più stretto (`1e-6`, `1e-7`) | 0,726 mV | 0,196 mV |
| più stretto → ultra (`1e-7`, `1e-8`) | 0,0183 mV | 0,1039 mV |

Il controllo debole (seed 170029, gbar 0,5) resta sotto 0,047 mV anche nel
primo salto. Nel caso seed 170083, gbar 1,5 rimane invece uno scarto non
trascurabile anche tra gli ultimi due livelli: **non dichiarare ancora
convergente** la traiettoria nativa critica. La differenza nasce dentro
l'oracolo numerico sotto feedback, non dimostra da sola un errore biologico
del meccanismo o una diversa realizzazione delle sinapsi.

## Provenienza

Il JSON registra `code_revision=3bb2754` perché il run è stato eseguito
prima del commit Task17c: quel campo identifica il commit di base, non
pretende che contenesse già questo codice. SHA-256 dei sorgenti eseguiti nel
working tree:

- `src/giada_teacher/roadmap_task17c_m_precision_solver.py`:
  `e7e8c4675b67db90eeae48117aaf0aadb64a915c12242c46993dbd0ad71eab1c`
- `src/giada_teacher/roadmap_causal_cahva_replacement.py`:
  `33bdd609cccdde4e0c88e7aff6b99155622d6f9f157b6788c92ac9c4d62de2c8`
- `scripts/run_roadmap_task17c.py`:
  `7f9c37c0f1543d851b7bc7495734fcb33f73d7f8c00a45a14d75769ec21328bd`

I tracciati a 0,025 ms dei due casi critici restano nell'output di lavoro
`artifacts/giada_task17c_local_with_ladder/diagnostic_traces.json`; qui sono
versionati report, metriche appaiate e ladder nativo compatti.
