# ⏱️ GIADA Task 10 — granularità temporale, stesso run

La domanda operativa non è se dobbiamo *mostrare* uno stato ogni `0,125 ms`:
un core può restituire (S_{t+1\,ms}) dopo 1, 2, 4 oppure 8 aggiornamenti
interni. Questi quattro bracci sono composti sullo **stesso orizzonte esterno
di 1 ms** e ricevono la stessa sequenza temporizzata `U_realized`: i token
conservano offset continuo, ID sinaptico, segmento, quantità e segno. Il run
si blocca se superiamo 64 eventi realizzati in un contenitore da `0,125 ms`.

Il notebook riapre una volta il teacher autentico, riproduce transizioni train
già congelate nella Task 9 e acquisisce lo **stato canonico completo** ogni
`0,125 ms`. Stato finale, voltaggi microcampionati originali e RNG devono
coincidere con il dataset entro `1e-5`:
ogni mismatch blocca il training. I gruppi train/development/test diagnostico
sono disgiunti per traiettoria, tutti ricavati dallo split train originale.

Quattro modelli full-state piccoli, con identico numero di parametri e due seed
appaiati, differiscono solo per la durata di ogni aggiornamento (`1`, `0,5`,
`0,25`, `0,125 ms`). Salviamo metriche ai budget 100/300/600 step lungo le
stesse traiettorie. La scelta dei checkpoint usa solo development; poi si
apre il sottoinsieme diagnostico separato per traiettoria. Misuriamo errore a
1 ms, rollout chiuso a 8 ms, proxy spike somatico ai confini di 1 ms, tempo
GPU per ms simulato e non-finiti. Il proxy **non sostituisce** le etichette
biologiche microtemporali.

Regola preregistrata: almeno 8 finestre di rollout e 2 positive; il miglior
braccio deve ridurre l'RMSE 8 ms di almeno 20% rispetto alla persistenza e
avere F1 proxy ≥0,5. Scegliamo il numero **minimo** di sottopassi che resta
entro il 10% dell'RMSE migliore e 0,05 di F1, senza non-finiti né voltaggi
fuori `[-120,80] mV`. Se i modelli non apprendono,
il risultato è **inconclusivo**, non «1 ms impossibile».

⚠️ Questa è una matrice informativa per una piccola famiglia MLP full-state,
non una sentenza su tutte le architetture possibili. Nessuna transizione dello
split test originale viene aperta. La scala temporale interna minima utile si
sceglie qui solo se il canary supera le soglie biologiche e di learnability.
