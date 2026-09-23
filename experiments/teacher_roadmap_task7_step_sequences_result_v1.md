# ✅ Roadmap Task 7 — storia dei voltage step

ZIP `giada_roadmap_task7_report.zip`, SHA-256
`1704f9460bfd9a6bebdaafd643b3a8cd9e4b6731f9516a6fa5da02b461cff201`.
Revisione `3f0a17839c07d95a2230c9e88e04dd4d6c24356b`.

Il run è valido e supera il gate preregistrato. Pilot NEURON: 96 casi,
errore gate massimo `4.44e-16`; nessun overlap development/sealed, nessun
retraining o selezione sul sealed.

| Famiglia | Mediana differenza massima gate entro coppia | Score formula solo V iniziale | Score LUT path completo |
|---|---:|---:|---:|
| Salita precoce/tardiva | 0.28114 | 149.87 | 0.00510 |
| Impulso precoce/tardivo | 0.04361 | 72.65 | 0.00301 |
| Impulso breve/lungo | 0.18935 | 86.78 | 0.00333 |
| Ordine invertito | 0.00335 | 83.09 | 0.00317 |

Gli score sono normalizzati, **non mV**. Le coppie condividono V iniziale e
finale e m/h iniziali: le differenze finali sono quindi attribuibili alla
storia del voltaggio. La LUT frozen path-aware resta entro la soglia 0.02
registrata in tutte le famiglie, senza violazioni d'occupazione. Il candidato
physical-tau frozen è meno accurato della LUT in tutte le famiglie e seed.

Il path futuro è noto perché esogeno. Non è un input causale disponibile in
un neurone autonomo. La Task 10 della roadmap deve distinguere sufficienza
informativa da disponibilità causale degli input prima di un nuovo esperimento
di granularità interna.
