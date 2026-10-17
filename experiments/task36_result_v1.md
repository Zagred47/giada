# ✅ GIADA — nuova Task36: contratto e riconciliazione

**Esito: GO alla preparazione della Task37**, non al nuovo training.
L'exTask36 «due compartimenti attivi» è Task72; la numerazione corrente è
quella del dropshot aggiornato, con mapping preservato nei record di piano.

Eseguito `scripts/run_task36_contract_audit.py`: 22 sorgenti/documenti e
checkpoint indicizzati SHA-256; inventario confermato a 11 canali e 18 gate,
17 V-dipendenti e SK in posizione16. Ordine della sorgente Ca, CaDynamics e
update SK verificato nel codice. Il readout deve fornire conduttanze per
canale, corrente Ca di sorgente e sum_g/sum_gE senza cambiare il solver.

La riconciliazione comprende Task35, CC-N1/N2 e IE-P1…P7. Per IE-P7 è stato
completato l'archivio dei 21 file scientifici di primo livello e l'audit
indipendente; esito in [`ie_p7_result_v1.md`](ie_p7_result_v1.md).
I test già aperti restano storici, non fresh della nuova serie.

Il [contratto](task36_ionic_abstraction_contract_v1.json) definisce sette
ipotesi aperte, otto gruppi di valutazione e tre livelli di astrazione.
Il livello1 sostituisce l'aggiornamento conservando stato fisico e readout;
il livello2 astrae per famiglie, il livello3 condivide la memoria ionica.
Ciascun livello ha una verifica locale e una su sistemi realmente accoppiati.

## 📋 Cosa deve produrre la Task37

1. Supporto stratificato per storie, regimi rari, recupero, treni rapidi e
   interventi; numerosità e diversità effettive, con split per traiettoria.
2. Valutatori ionici eseguibili, unità/fasi verificate, soglie numeriche
   prospettiche e controlli difettosi che devono essere riconosciuti.
3. Gate per regime e revisione qualitativa strutturata; supporto insufficiente
   significa non valutabile. Transizioni dinamiche confrontate con un teacher
   convergente e con sensibilità misurata alle perturbazioni.
4. Contrasti copertura × budget e ladder appaiata per distinguere training
   insufficiente da rappresentazione o memoria insufficienti.
5. Workload ionico principale e obiettivo di risparmio preregistrati;
   costi del core e operativi riportati separatamente.

La Task36 è una verifica documentale e delle interfacce. Non aggiunge dati,
non dimostra nuovi comportamenti emergenti e non promuove un modello.
Contratto, evidenze, ipotesi, specifiche e dipendenze entrano nel mirror;
la ricevuta locale riporta l'esito della verifica DB.
