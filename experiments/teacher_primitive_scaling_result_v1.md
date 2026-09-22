# 📈 Task 5 — risultato verificato

Lo ZIP fornito dall'utente ha SHA-256
`ec6e691fe8a3c7fa30ec70db967f508f358f05b6bbaec58f3b7bcf0efed383f3`.
Freeze, checkpoint e report finale hanno hash coerenti. La selezione è stata
fatta sul development; il nuovo sealed di 8.192 esempi è disgiunto.

La rete physical-τ di larghezza 32 è la migliore fra le tre reti studiate,
ma lo score normalizzato sealed è 0,461 e la dispersione fra seed rimane alta.
Lo score della LUT lineare scende a circa 0,03 con 257 punti e 0,01 con 513
punti; quest'ultima richiede 8.208 byte in float32. I valori non sono mV.
L'aumento di capacità o budget non produce un miglioramento appreso stabile.

**Portata:** Ca_HVA isolato in voltage clamp costante. Non è prova che la LUT
funzioni a voltaggio variabile o nel neurone accoppiato. Il prossimo contrasto
deve congelare i candidati e misurarli su percorsi di voltaggio noti prima di
un'eventuale sostituzione nel teacher.
