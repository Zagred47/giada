# TG-01 — esito inconclusivo della matrice temporale

ZIP `giada_temporal_granularity_matrix_report.zip`, SHA-256
`f38a7513a675b49d817c80dcdab10480c6fca24ca1188c4d349abab5c7caa68d`.
Revisione codice `06decb8dd298a451b83de18d339e6a7e9e6b57e9`.

L'acquisizione autentica è valida su 195 transizioni train originarie,
ripartite per traiettoria in 115 fit, 51 development e 29 test diagnostico.
Stato finale e RNG sono esatti; errore microvoltaggio massimo
`3.81468e-6 mV`.

Il braccio 1 aggiornamento/ms è il migliore nel canary: RMSE 1 ms
`2.54–2.71 mV`; RMSE rollout 8 ms `16.55–16.57 mV` contro persistenza
`21.30 mV`. Ma F1 del proxy spike è `0`, ci sono `9–33` violazioni del
voltaggio in rollout e nessun braccio soddisfa i gate. I bracci 2/4/8
sottopassi sono peggiori a 8 ms. Il report conclude correttamente
`INCONCLUSIVE_INSUFFICIENT_LEARNABILITY_OR_SUPPORT`.

Il confronto non stabilisce il passo temporale minimo. I modelli hanno
5,304,770 parametri e solo 115 transizioni fit; le 28 finestre rollout
contengono tre positive proxy e possono sovrapporsi. Nessun test sigillato
esterno è stato aperto. La Task 10 originale sulla sufficienza degli input
resta aperta; il refit di TG-01 è posticipato fino a quel controllo.
