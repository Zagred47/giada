# Memoria di ricerca condivisa

La base `Scientific Modeling & Experimentation Lab` (`app9dZ7ghujkFJIo5`) ha un
mirror SQLite versionato in `research_memory/data/research_memory.sqlite`.
Leggere `research_memory/README.md` prima di modificarne schema o record.

Per i pattern di interrogazione consultare `research_memory/queries/QUERY_REVIEW.md`:
contiene 34 query commentate, parametri e limiti interpretativi. SQL e catalogo
JSON eseguibile sono nella stessa cartella. Le associazioni estratte non sono
automaticamente prove causali; seguire le istruzioni di binding dei parametri.

## Doppia scrittura obbligatoria

Quando l'utente autorizza la creazione o modifica di record scientifici, scrivere
sia su Airtable sia nel mirror SQLite. Non inserire righe dimostrative o ipotesi
presentate come fatti. Usare la outbox (`stage`), preflight remoto, upsert con
codice stabile e readback completo (`import-snapshot` o `sync`). Una risposta
positiva alla sola chiamata Airtable non dimostra la sincronizzazione.

Se il remoto è indisponibile o un risultato è incerto, lasciare l'operazione
pendente e comunicarlo; non presentare una modifica locale come sincronizzata.
Non recuperare credenziali del connettore da file interni. Per automazione REST
usare solo un token esplicitamente fornito via ambiente; altrimenti usare i tool
Airtable autorizzati con il workflow documentato.

Le query SQL sono lenti sulla memoria: usare `python -m research_memory query`
(oppure `research_memory/memory.ps1` su questa macchina). Non modificare i record
con SQL ad hoc bypassando la doppia scrittura. Non modificare lo schema su un
solo lato. Dopo ogni aggiornamento verificare il mirror ed esportare lo snapshot
JSON; committare DB e snapshot insieme quando richiesto. Nessun sync in background
è implicito. Le altre istruzioni di progetto nella directory padre restano valide.
