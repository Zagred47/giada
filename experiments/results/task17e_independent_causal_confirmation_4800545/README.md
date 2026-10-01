# GIADA Task 17e — conferma indipendente Ca_HVA

Archivio utente: `giada_task17e_causal_confirmation_4800545_d698f1ab.zip`, SHA-256 `ab29bebd85da9434e3b387b8aba3d4308c455dc5ee2082273d66ec00f93f57f7`. Revisione eseguita `480054532290c96800cb4312228335dd295bb6e8`. Il report nell'archivio è `valid=true`, il processo termina con exit 0, la tabella congelata corrisponde alla SHA-256 preregistrata e il replay nativo ripetuto è esatto ai siti osservati. Sono stati eseguiti 135 episodi, 90 confronti e 15 coppie di perturbazione.

## Esito

`GATE_C_NO_GO`, ma **non per regressione numerica del candidato**. Passano controllo formula, voltaggio, gate, corrente, eventi *quando presenti*, rilascio e contrasto gbar. Fallisce soltanto `event_support`: nei 45 episodi nativi non è stato rilevato alcun somatic spike, calcium spike o NMDA spike secondo il detector preregistrato. Quindi il controllo eventi è vacuo e non dimostra che la LUT preservi gli eventi rigenerativi. Gate C e Task 18 restano non autorizzati; questo è un esito non informativo sulla dimensione eventi, non prova che la LUT sia errata.

Il peggior RMSE V del candidato sui quattro siti è `0.002002896` mV contro la soglia `2` mV; peggior errore gate `0.000115659` contro `0.01`; peggior RMSE `ica_hva` `3.577629e-06` mA/cm² contro `1e-4`; peggior errore relativo del contrasto gbar identificabile `8.677551e-05` contro `0.2` (sette contrasti identificabili). Questi risultati sono limitati al supporto effettivamente visitato, prevalentemente sub-soglia.

Ispezione delle tracce native diagnostiche a 1 ms: massimo soma circa `−79.39` mV, massimo hot-zone circa `−58.76` mV; il nexus sale fino a circa `−28.16` mV nel protocollo NMDA, ma nessun evento soddisfa insieme soglia, reset e durata minima del detector sui tracciati completi. Non attribuire l'assenza di eventi a un errore del candidato. I protocolli della 17e non hanno fornito il supporto rigenerativo necessario per chiudere Gate C.

## Decisione successiva

Conservare la 17e come conferma forte dei contrasti sub-soglia e di gbar, senza riaprire né ritoccare seed o soglie. Per completare Task 17 serve un **nuovo protocollo preregistrato di supporto eventi**, che selezioni schedule dal patrimonio già validato o da un pilot nativo separato e poi congeli casi indipendenti per la conferma causale. Non riutilizzare i 45 casi 17e come test indipendente e non chiamare questo run un Gate C positivo.
