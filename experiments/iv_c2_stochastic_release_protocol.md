# GIADA IV-C2 — plasticità breve e rilascio stocastico

Stato: preregistrato, in attesa del run nativo. IV-C1 v3 è il prerequisito confermato; IV-C3 e Task33 restano chiuse.

Il teacher canonico conserva due stati plastici distinti: `u` si aggiorna a **ogni impulso programmato**, anche senza rilascio; `Rstate` e `tsyn` governano la disponibilità. Quando il terminale è depresso, `NET_RECEIVE` estrae prima per il recupero e, soltanto se disponibile, estrae di nuovo per il rilascio. Il generatore canonico è `Random123` configurato con `negexp(1.0)`, non con una uniforme. Perciò la probabilità di un'estrazione inferiore a `u` è `1-exp(-u)`; il nome `Use` non autorizza da solo a identificarla con la probabilità marginale di rilascio.

Il [contratto numerico](iv_c2_stochastic_release_preregistration.json) fissa, prima del run, meccanismi, bracci, seed, stream, orari, soglie e controlli. Il notebook esegue sul teacher fissato un confronto evento-per-evento contro uno shadow che usa stato esplicito e uno stream indipendente ma appaiato. Verifica anche il ripristino di stato nativo più sequenza RNG a metà burst, la distribuzione delle estrazioni e che una connessione a peso zero non consumi RNG. Le tracce registrano separatamente impulsi programmati e rilasci realizzati.

Un errore di API/ambiente produce `failure_report.json` e non è interpretabile come fallimento biologico. Anche un eventuale pass IV-C2 non autorizza Task33: rimane IV-C3, l'interfaccia sinaptica integrata.
