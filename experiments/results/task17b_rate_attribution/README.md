# Task 17b — esito della matrice nativa locale

Stato: esecuzione Linux locale completata, 72 traiettorie (`8 bracci × 3
seed × 3 gbar`) con teacher originale a 642 segmenti. Verifica NMODL e
clone-formula superate. Sono 9 condizioni seed–gbar, non 72 replicazioni
indipendenti. La Task 17a rimane `CAUSAL_MICROCANARY_NO_GO` e Gate C non è
autorizzato.

Sul caso che ha fallito in 17a (NMDA, seed 170029, gbar 1, sito 460), massimo
errore di m: LUT completa **0,018351**, solo `mInf` LUT **0,011058**, solo
`mTau` LUT **0,005484**, coppia `mInf+mTau` LUT **0,018382**. Il clone-formula
ha errore zero. Il divario fra il massimo errore gate della LUT completa e
quello della coppia m è ≤ **3,024×10⁻⁵** su tutti i 36 confronti
condizione–sito; il divario nei rispettivi RMSE V è ≤ **0,000945 mV**.

Il nuovo seed 170083 produce un secondo superamento della soglia gate 0,01,
al soma con gbar 1,5: LUT completa **0,017199**, coppia m **0,017211**.
Questo non era il seed che aveva fallito in 17a. Il seed 170097 non supera
0,01 nelle nove condizioni di sito–gbar considerate, a conferma che l'errore
non è uniforme. I bracci `hInf` e `hTau` isolati restano molto più piccoli,
ma non dimostrano che h sia irrilevante in ogni regime.

Conclusione scopiata: nel protocollo NMDA testato l'errore della LUT completa
è quasi interamente riprodotto dall'approssimazione combinata delle due
funzioni di attivazione m. Il feedback closed-loop amplifica una piccola
perturbazione dei rate in specifici regimi; i massimi dei bracci singoli non
sono additivi. Questo **non** stabilisce ancora se tolleranze CVode,
risoluzione della griglia o altra scelta di implementazione possano evitare
l'amplificazione. Nessuna rete è stata addestrata e nessun test sealed è stato
utilizzato per scegliere un nuovo candidato.

Il primo screening con 170059/170071 ha motivato l'emendamento della coppia
m; quei seed non sono trattati come verifica nuova. Report e metriche della
matrice emendata sono nei due JSON qui accanto, protetti da SHA-256 nel
registratore SQLite. L'output locale originale include anche i tracciati a
0,025 ms del caso difficile in
`artifacts/giada_task17b_local_final/diagnostic_traces.json` (non versionato).
La seconda esecuzione include il controllo native-repeat: differenza di
voltaggio esattamente zero nei quattro siti. Le metriche della prima matrice
emendata e della ripetizione finale sono byte-identiche; qui è versionato il
report della seconda.

## Provenienza del codice locale

Il campo `code_revision` nel JSON è il **commit di base** `982ad58` al
momento del run: il codice Task17b era ancora nel working tree. Non va letto
come se quel commit contenesse già Task17b. Gli SHA-256 dei sorgenti
effettivamente eseguiti (byte del working tree prima del commit) sono:

- `src/giada_teacher/roadmap_task17b_rate_attribution.py`:
  `8fe7edbde061be718d616af3c39ca7a25d34330ae262bee0b2d6e727ae6f8dd1`
- `src/giada_teacher/roadmap_causal_cahva_replacement.py`:
  `33bdd609cccdde4e0c88e7aff6b99155622d6f9f157b6788c92ac9c4d62de2c8`
- `scripts/run_roadmap_task17b.py`:
  `ef5a4827eeda23a37df4148cd7900e3200f83ee11270ed059606172fe960d430`
