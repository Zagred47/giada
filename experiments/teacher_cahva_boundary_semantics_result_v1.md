# 🧭 Task 7c — esito del riallineamento temporale

Lo ZIP fornito dall'utente ha SHA-256 `f9cc32a6894718d73b10ec95417aad1b53a28d6f4ef3bb97559b4c22cd81160f`.
Il report dichiara `valid=true` su 24 episodi; non ha rigenerato il teacher,
riaddestrato modelli né aperto un nuovo test indipendente.

La formula corretta riproduce i boundary state NEURON con errore massimo
`2.82e-12 mV` sul voltaggio, RMSE massimo `6.89e-14` sui gate ed errore
massimo `2.68e-16 mA/cm²` sulla corrente registrata. Il precedente mismatch
dei gate e della corrente era quindi dovuto al confronto fra sottostati
temporalmente diversi, non a una dinamica Ca_HVA inesatta.

Sui gruppi da otto episodi, a conduttanza `1x` la LUT-513 congelata ha RMSE
medio di voltaggio `0.000269 mV`, contro `0.001376`, `0.001500` e `0.000739 mV`
dei seed physical-τ 17/29/43. A `4x`: `0.000332 mV` contro `0.005090`,
`0.005648` e `0.002752 mV`. A `gbar=0` i candidati coincidono nel voltaggio.
Nessun seed è selezionato a posteriori.

⚠️ Conclusione limitata a Ca_HVA+pas in un compartimento, `E_Ca` fisso e
tracce 7b già aperte. Non prova il neurone completo, il calcio dinamico o
l'updater di voltaggio appreso. La Task 8 studia velocità e frequenza di
percorsi di voltaggio esogeni; non ripete la semplice rampa a otto campioni
già valutata nella Task 6.
