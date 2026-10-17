# 🧪 Task38 — Update ionico congiunto a stato esplicito

## 🎯 Cosa cambia

Si impara la transizione discreta dei 18 gate, non soltanto i rate V→inf/tau.
Il trunk riceve V e i 17 stati precedenti dipendenti da V; produce candidate
e frazioni di aggiornamento limitate. SK ha una testa separata oldSK/logCa,
eseguita nella fase successiva al ponte del calcio. Una combinazione convessa
mantiene le occupazioni in [0,1]; non garantisce da sola la correttezza dinamica.
Nessuna formula canonica dei rate compare nel forward del candidato.

Restano analitici aperture, gbar, I=g(V−E), sorgente Ca e solver esterno.
Qui V e il ponte Ca sono imposti: il reinserimento autonomo è Task40.

## ⚖️ Confronti appaiati

Width32/64 × seed17/29/43: sei membri indipendenti, vettorizzati per gruppi
di capacità sulla stessa GPU. Stessi campioni, stream dati e budget; prefisso
di inizializzazione condiviso, non una falsa uguaglianza tra reti di dimensioni
diverse. Adam e clipping indipendenti verificati contro update separati.
Checkpoint0/1k/3k/10k/30k lungo la stessa traiettoria. Loss gate normalizzata
per incremento e correnti individuali normalizzate da sole statistiche fit.
Campionamento bilanciato per famiglia, non per durata del protocollo.

Formula canonica come controllo numerico; frozen local161 IE-P6 come baseline
a coefficienti discreti con SK canonico, esplicitamente sostituzione parziale.
Task39 mantiene i confronti loss e copertura fit_extra: non sono già conclusi.

## 🔍 Valutazione senza scorciatoie

Selezione per membro solo sul punteggio one-step development preregistrato.
Dopo il freeze: ricorrenze complete senza reset su tutte le traiettorie
development e stress, fino a10s. Nessuna riselezione retrospettiva sulla
base dei rollout finali. Tutte le soglie Task37 restano identiche.
Errori individuali, carica, eventi valutabili, orizzonti, recuperi e storie
diverse a V uguale restano separati: RMSE media bassa non equivale a GO.

I CUDA graph avanzano32 tick per blocco e conservano lo stato; un preflight
di65 tick attraversa due confini e confronta l'intera ricorrenza con eager.
Le accelerazioni del training non costituiscono uno speedup di inferenza.
Il costo completo e i comportamenti autonomi richiedono Task40–41.

## 🗃️ Registrazione e riproducibilità

Protocollo: `task38_joint_transition_preregistration_v1.json`.
Mirror SQLite esclusivo; ipotesi aperte, bracci e metriche prospettiche,
nessuna osservazione di risultato inventata. Kaggle usa un commit completo
pinnato dopo il push; non usa codice remoto in movimento.
Errore tecnico → failure_report; fallimento del modello → final_report NO-GO.
