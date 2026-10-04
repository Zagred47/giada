# Task17f — Gate C superato su Kaggle

Notebook privato `alessandrobelli/giada-task17f-event-supported-confirmation`,
versione1, kernel137021889, sessione355133447. Codice fissato al commit
`1b87bdd083f550f655c043d0fcbc731de972c9cd`; teacher canonico
`074c4666300a8ad246601dab179a97a6942f0f29`. Esecuzione CPU via MCP.

## Esito e interpretazione

`valid=true`, `decision=GATE_C_PASS_BOUNDED_CAHVA`,
`gate_c_authorized=true`, `task18_authorized=true`. Tutti i dieci gate passano.
27 episodi pilot, 144 episodi confermativi e una replica nativa attiva;
108 confronti appaiati. Il processo termina con exit0, in circa17 minuti
inclusi setup, compilazione ed esportazione secondo il log Kaggle.

Gli stimoli scelti sul solo pilot producono somatic_spike, calcium_spike e
nmda_spike non censurati in ciascuno dei tre seed nuovi171601/171619/171643
a gbar1. La quiete resta silente. Le perturbazioni gbar0.5/1/1.5 sono
valutate senza selezionare nuove schedule dopo l'apertura della conferma.

| Braccio vs nativo | Peggior RMSE V (mV) | Peggior errore gate | Peggior RMSE ica (mA/cm²) |
|---|---:|---:|---:|
| Nativo super-ultra | 0.0073127591 | 0.0005093485 | 7.60034415e-7 |
| Clone formula | 3.00356427e-9 | 1.26189670e-10 | 8.00878992e-13 |
| LUT m2049 f64 | 0.0138580152 | 0.0010223992 | 1.52633462e-6 |

V include i siti evento e il centro del cluster NMDA; gate/corrente i quattro
siti Ca_HVA registrati. Ogni condizione passa; questi massimi non sostituiscono
i confronti individuali. Conteggi e censura coincidono; la differenza negli
onset è0ms alla risoluzione0.025ms per tutti i114 confronti evento della LUT.
Questo non dimostra identità temporale al di sotto del campionamento;
draw/successo/identità dei rilasci sono identici e la differenza nelle quantità
è zero. Tutti i36 contrasti gbar identificabili passano: peggior errore relativo
0.000645678689, cioè0.0645679%, contro il limite20%.

La sostituzione causale nel teacher multicompartimentale preserva la dinamica
nei protocolli attivi osservati: il risultato favorevole non dipende più da
una copertura evento mancante. Il candidato è la LUT dei rate del gate m;
il gate h e l'equazione della corrente restano analitici. Non si dimostra
un vantaggio hardware né l'universalità su altri canali, morfologie o stimoli.
La prossima task del piano è18, generalizzazione alla famiglia HH standard.

## Recupero e verifica indipendente

Archivio completo locale, fuori dal set di file versionati:
`artifacts/giada_task17f_event_confirmation_1b87bdd_3a7e4bfc.zip`.
113739112 byte, SHA-256
`e409a154aab0680ef037050ff2e244564114fb1312c01391c5a9556d529a14f2`.

`scripts/audit_task17f_archive.py` verifica CRC, identità dei sorgenti rispetto
al commit, hash del freeze, disgiunzione dei seed, completezza della matrice,
gate dei108 confronti, copertura e tutti i171 file di tracce dense.
7908 array sono finiti e tutte le occupanze dense m/h restano in[0,1].
I report originali sono conservati accanto a questo documento. Il manifest
di lancio conserva lo statoRUNNING osservato allora; l'esito finale è nel
final_report e nell'archive_audit, senza riscrivere la storia del lancio.
