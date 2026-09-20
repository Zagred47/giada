# Memoria di ricerca condivisa

La base `Scientific Modeling & Experimentation Lab` (`app9dZ7ghujkFJIo5`) ha un
mirror SQLite versionato in `research_memory/data/research_memory.sqlite`.
Leggere `research_memory/README.md` prima di modificarne schema o record.

Per i pattern di interrogazione consultare `research_memory/queries/QUERY_REVIEW.md`:
contiene 34 query commentate, parametri e limiti interpretativi. SQL e catalogo
JSON eseguibile sono nella stessa cartella. Le associazioni estratte non sono
automaticamente prove causali; seguire le istruzioni di binding dei parametri.

## Mirror SQLite esclusivo fino alla futura sincronizzazione

Per decisione esplicita dell'utente, fino a nuova richiesta interfacciarsi solo
con `research_memory/data/research_memory.sqlite`. Non leggere, scrivere o fare
readback su Airtable. Airtable resta congelato allo stato precedente e non deve
essere dichiarato sincronizzato con le modifiche locali.

Non inserire righe dimostrative o ipotesi presentate come fatti. La futura
sincronizzazione con Airtable sara un'operazione separata ed esplicita: dovra
riconciliare i record per codice stabile, rimappare e verificare le relazioni e
fare un readback completo. Non usare una sovrascrittura automatica
last-write-wins.

Le query SQL sono lenti sulla memoria: usare `python -m research_memory query`
(oppure `research_memory/memory.ps1` su questa macchina). Non modificare i record
con SQL ad hoc: usare l'interfaccia versionata del mirror e conservarne gli
identificatori stabili. Dopo ogni aggiornamento verificare integrita e query
logiche del mirror ed esportare lo snapshot locale coerente; committare DB e
snapshot insieme quando richiesto. Nessun sync remoto o in background e
implicito. Le altre istruzioni di progetto nella directory padre restano valide.
