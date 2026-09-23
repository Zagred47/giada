# ✅ Task 9b — interpolazione e reference floor

Lo ZIP `giada_physiological_path_floor_forensic.zip` (SHA-256
`bec67549d8f772d7885df1332214f29e0b5c02cf4ba5db091edda058fc7d8f86`)
è valido. La formula con voltaggio destro riproduce esattamente il floor Task 9;
cinque sottopassi allo stesso voltaggio differiscono solo di `3,22e-15`.

L'interpolazione lineare porta l'RMSE di `m` globale da `0,003682` a
`0,0002065`. Nei quattro gruppi spike, le riduzioni sono circa 97–98% e tutti
i percorsi selezionati migliorano. Il gate preregistrato passa.

Questo isola una forte sensibilità al trattamento del voltaggio *tra* i 40
campioni, ma non verifica il path effettivo non osservato. La Task 9c propone
un piccolo replay autentico dagli snapshot originali a grana temporale più
fitta, con riproduzione obbligatoria della traccia a 40 punti prima di qualunque
attribuzione.
