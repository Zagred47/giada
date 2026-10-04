# GIADA Task 17e — conferma indipendente Ca_HVA

Archivio utente: `giada_task17e_causal_confirmation_4800545_d698f1ab.zip`, SHA-256 `ab29bebd85da9434e3b387b8aba3d4308c455dc5ee2082273d66ec00f93f57f7`. Revisione eseguita `480054532290c96800cb4312228335dd295bb6e8`. Il report nell'archivio è `valid=true`, il processo termina con exit 0, la tabella congelata corrisponde alla SHA-256 preregistrata e il replay nativo ripetuto è esatto ai siti osservati. Sono stati eseguiti 135 episodi, 90 confronti e 15 coppie di perturbazione.

## Esito

`GATE_C_NO_GO`, ma **non per regressione numerica del candidato**. Passano controllo formula, voltaggio, gate, corrente, eventi *quando presenti*, rilascio e contrasto gbar. Fallisce soltanto `event_support`: nei 45 episodi nativi non è stato rilevato alcun somatic spike, calcium spike o NMDA spike secondo il detector preregistrato. È invece presente un `nmda_plateau` (protocollo 1, seed 171001, gbar 1,5), riprodotto da formula e candidato con identico onset campionato. Il confronto eventi non è quindi interamente vacuo, ma manca la copertura delle tre classi richieste. Gate C e Task 18 restano non autorizzati.

Il peggior RMSE V del candidato sui quattro siti è `0.002002896` mV contro la soglia `2` mV; peggior errore gate `0.000115659` contro `0.01`; peggior RMSE `ica_hva` `3.577629e-06` mA/cm² contro `1e-4`; peggior errore relativo del contrasto gbar identificabile `8.677551e-05` contro `0.2` (sette contrasti identificabili). Questi risultati sono limitati al supporto effettivamente visitato, prevalentemente sub-soglia.

Ispezione delle tracce native diagnostiche a 1 ms: massimo soma circa `−79.39` mV, massimo hot-zone circa `−58.76` mV; il sito 460 sale fino a circa `−28.16` mV nel protocollo NMDA. **Rettifica della prima interpretazione:** 460 è il tuft; il nexus canonico è 437. Non è giustificata l'affermazione precedente che lo spike NMDA mancasse per durata insufficiente. Il plateau salvato al sito 460 dura 30,175 ms (onset 6,65 ms, offset 36,825 ms).

L'audit del codice identifica un disallineamento del probe: `candidate_protocols()` richiede per NMDA `event_probe_mode=cluster_center`, ma `_recording_trial()` in Task17e usa sempre `default_event_definitions(representatives)`, che colloca `nmda_spike` al nexus. Il calibratore canonico invece sposta le definizioni richieste sul centro del cluster selezionato. Applicare esplorativamente la soglia −40/reset −50/durata minima 1 ms agli estratti a 1 ms del sito 460 rileva sei eventi nelle nove condizioni native del protocollo NMDA, con durate 4–30 ms. Questo controllo post hoc non determina il centro del cluster eseguito e non vale come conferma alle risoluzioni preregistrate; dimostra però che la precedente spiegazione per durata non è sostenuta.

L'assenza di spike somatici/calcio rimane coerente con le tracce disponibili. Nessuna delle schedule era stata prequalificata sul teacher ultra per coprire tutte le classi richieste. Va separato il difetto di localizzazione del probe dalla mancanza di stimoli sufficientemente attivi. Le metriche numeriche già calcolate restano valide per i siti e le condizioni salvati.

## Decisione successiva

Conservare la 17e come conferma forte dei contrasti sub-soglia e di gbar, senza riaprire né ritoccare seed o soglie. Per completare Task 17 serve un **nuovo protocollo preregistrato di supporto eventi**, che selezioni schedule dal patrimonio già validato o da un pilot nativo separato e poi congeli casi indipendenti per la conferma causale. Non riutilizzare i 45 casi 17e come test indipendente e non chiamare questo run un Gate C positivo.
