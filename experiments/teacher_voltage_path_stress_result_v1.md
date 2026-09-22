# ⚡ Task 6 — esito verificato

Lo ZIP fornito dall'utente ha SHA-256
`a4795710d8ddf49c678bcef62e8a84e77dfb1ca47531cbcfc2e7bf03553a851e`.
Freeze e report hanno hash coerenti. Il pilot con NEURON autentico copre 144
casi e ha errore massimo dei gate `4,44e-16`; il sealed ha 3.072 casi,
senza overlap con development e senza selezione sul sealed.

La distinzione causale è netta: se usiamo soltanto il voltaggio iniziale per
tutto il millisecondo, perfino la formula esatta arriva a score normalizzato
355,32 nel caso degli estremi. La LUT a 513 punti nella stessa condizione
fallisce allo stesso modo. Il problema qui è l'informazione temporale mancante,
non la forma della LUT.

Con il percorso di voltaggio esogeno noto, la LUT congelata ha score massimo
fra famiglie `0,0111` (intorno a -27 mV), zero violazioni d'occupazione e
score `0,00042` sugli estremi. La rete physical-τ congelata è meno accurata
e mostra dispersione fra seed (score peggiore per seed circa
`0,709/0,454/0,479`). I valori sono normalizzati, **non mV**.

**Limite:** il percorso futuro è imposto dal voltage clamp. Nel neurone
endogeno non è noto in anticipo. Il prossimo test deve risolvere voltaggio e
gate congiuntamente o sottopassare causalmente usando il voltaggio predetto;
la Task 6 non è una validazione del neurone accoppiato.
