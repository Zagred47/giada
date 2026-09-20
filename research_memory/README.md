# Scientific Modeling & Experimentation Lab — SQLite mirror

Mirror della [base Airtable](https://airtable.com/app9dZ7ghujkFJIo5), nel workspace **My First Workspace**.

## Già utilizzabile

Il file `data/research_memory.sqlite` è il database reale, incluso in Git. Non è uno stub.
Richiede soltanto Python >= 3.10 con SQLite/JSON1, senza pip, server o servizi persistenti.
Su questa macchina il runtime Python di Codex è disponibile; il launcher PowerShell lo risolve automaticamente.

```powershell
.\research_memory\memory.ps1 status
.\research_memory\memory.ps1 verify
.\research_memory\memory.ps1 query 'SELECT * FROM v_evidence_chains'
```

Su Linux/Kaggle o con Python nel PATH, dalla root della repository:

```sh
python -m research_memory status
python -m research_memory query 'SELECT * FROM "06 · Specifiche dei modelli"'
python -m unittest discover -s tests -p test_research_memory.py -v
```

La base rimane vuota: nessun esperimento o esempio è stato inventato/importato. I test usano database temporanei separati e non scrivono su Airtable.

## Identità logica, rappresentazione relazionale

- 78 tabelle semantiche, con chiavi brevi stabili (`sources`, `models`, `claims`, …).
- 513 campi scalari originali, con nome e tipo compatibile conservati.
- 241 tabelle associative `rel_fld…`, con chiavi esterne verso gli ID Airtable: ogni relazione è memorizzata una volta, non duplicata nei due versi.
- 78 viste con i **nomi esatti delle tabelle Airtable**, esponendo tutti i 995 campi: 513 scalari e 482 collegamenti diretti/inversi. I collegamenti sono array JSON di record ID, anche nelle viste inverse; l'ordine di entrambi i lati è conservato.
- `_record_id` e `_created_time` mantengono identità e provenienza Airtable.
- `_mirror_*`, `_remote_records` e `_outbox` sono tabelle tecniche locali, non nuove entità scientifiche da creare su Airtable.

Non si pretende identità delle funzionalità UI di Airtable, né una replica del suo motore. L'identità è quella di schema logico, valori e relazioni. `verify` controlla integrità SQLite, foreign key e roundtrip tra viste SQL e snapshot remoto conservato.

I contratti originali, gli ID effettivi e le configurazioni dei campi sono in `schema/`. Cambiamenti di schema richiedono una migrazione esplicita: il comando non ricrea e non sovrascrive un database incompatibile.

## Lenti di query

- `v_records`: catalogo uniforme dei record semantici.
- `v_links`: archi tipizzati, un solo verso canonico per relazione.
- `v_semantic_edges`: archi con nomi leggibili degli estremi.
- `v_evidence_chains`: affermazione → valutazione → risultato scientifico.
- `queries/`: pattern SQL riutilizzabili, senza cambiare classificazioni sorgenti.

Il [catalogo commentato Q01–Q34](queries/QUERY_REVIEW.md) è il punto di ingresso
per utenti e agenti: descrive domande, parametri, SQL e limiti interpretativi.
Sono disponibili anche il [SQL completo](queries/query_catalog_review.sql) e il
[catalogo JSON](queries/query_catalog_review.json). Sono template da consultare,
non nuove viste installate. Per le query parametrizzate seguire il binding
`sqlite3` documentato nel catalogo: il comando CLI `query` non accetta parametri.

Validazione dello schema e dei casi sintetici, senza modificare il database persistente:

```powershell
python research_memory/queries/validate_review_catalog.py
```

Il comando `query` è in sola lettura, limita risultati e lavoro computazionale, e rifiuta ATTACH, PRAGMA e DML. Usare le query per cambiare lente, non per trasformare automaticamente un'associazione in evidenza causale.

## Modalita operativa corrente: solo mirror SQLite

Per decisione dell'utente, Airtable e temporaneamente congelato. Fino a una
richiesta esplicita di sincronizzazione, agenti e script devono interrogare e
aggiornare soltanto il mirror SQLite versionato; non devono effettuare readback
o scritture remote e non devono descrivere Airtable come sincronizzato.

La successiva sincronizzazione sara una fase separata: riconciliazione per
`Codice stabile`, rimappatura degli ID e delle relazioni, readback completo e
verifica. Non applicare una politica automatica last-write-wins.

Le scritture locali passano dall'interfaccia versionata, non da SQL ad hoc:

```sh
python -m research_memory local-upsert findings fields.json
python -m research_memory local-upsert actions patch.json --record-id recXXXXXXXXXXXXXX
python -m research_memory verify
```

`local-upsert` genera per i nuovi record un ID locale deterministico, mantiene
automaticamente i collegamenti reciproci e aggiorna anche lo snapshot JSON
versionato. Questi ID sono provvisori: durante la futura sincronizzazione i
record verranno riconciliati per `Codice stabile` e le relazioni saranno
rimappate sugli ID Airtable effettivi.

La sezione seguente documenta il protocollo storico di doppia scrittura e resta
come specifica per quella futura riconciliazione; non e il flusso operativo
attivo.

## Protocollo di doppia scrittura sospeso

Le scritture non sono transazioni distribuite: Airtable e SQLite non possono eseguire un commit atomico condiviso. Il protocollo usa una **outbox persistente**, preflight contro modifiche concorrenti, upsert con codice stabile e readback finale.

1. Aggiornare la copia locale da uno snapshot completo corrente di Airtable.
2. Preparare un JSON di campi (nomi o ID). Per nuovi record, `Nome` e `Codice stabile` sono obbligatori; il codice è immutabile per il workflow. Non aggiungere record dimostrativi alla base reale.
3. `stage TABLE fields.json [--record-id rec…]`: salva l'intenzione nella outbox, **non pubblica un record scientifico speculativo in SQLite**.
4. Preflight sullo stato remoto corrente. Se il record è cambiato, fermarsi; niente last-write-wins silenzioso.
5. Upsert su Airtable tramite il codice stabile. Le modifiche a relazioni usano ID già verificati; creare prima i record referenziati.
6. Rileggere la base completa, includendo tutte le pagine e tutti i campi, e importarla. Solo se il valore remoto coincide con la patch la outbox passa ad `applied`.
7. Eseguire `verify`, esportare lo snapshot JSON per il diff Git e committare insieme DB, snapshot e modifiche di schema se presenti.

Letture e scritture devono avere **un solo autore attivo** durante la sincronizzazione. Il preflight non è una compare-and-swap remota: una modifica esterna nell'intervallo tra controllo e scrittura non può essere esclusa dall'API. Le relazioni inverse incoerenti fanno rifiutare lo snapshot anziché importare una vista temporaneamente spezzata.

### Con il connettore Airtable di Codex (senza PAT locale)

Il connettore già autorizzato può eseguire il payload prodotto da:

```sh
python -m research_memory stage sources fields.json
python -m research_memory payload OPERATION_ID
python -m research_memory preflight OPERATION_ID remote_snapshot.json
```

`payload` restituisce tool e argomenti, **non esegue la scrittura**. Se il preflight restituisce `already_applied`, non riscrivere: importare lo snapshot. Dopo un esito incerto, rileggere sempre prima del retry. La risposta del solo tool di scrittura non basta per chiudere l'operazione.

Per acquisire lo snapshot, chiamare `list_tables_for_base`, controllare schema/ID e poi `list_records_for_table` per ciascuna tabella, con tutti i `fieldIds`; seguire `nextCursor` fino all'ultima pagina. Formato di interscambio:

```json
{
  "format": "airtable-record-snapshot-v1",
  "base_id": "app9dZ7ghujkFJIo5",
  "complete": true,
  "tables": {"tbl…": [{"id": "rec…", "createdTime": "…", "cellValuesByFieldId": {"fld…": "valore"}}]}
}
```

Sono richieste tutte le 78 tabelle, anche quelle con array vuoto. `complete` si imposta solo dopo aver completato tutte le pagine. Il file reale iniziale è `data/airtable_snapshot.json`.

```sh
python -m research_memory import-snapshot remote_snapshot.json
python -m research_memory verify
python -m research_memory export-snapshot research_memory/data/airtable_snapshot.json
```

`import-snapshot` sostituisce transazionalmente il mirror verificato (incluse eventuali rimozioni già avvenute su Airtable); non elimina dati remoti e non scarta la outbox. Non usare snapshot storici al posto del readback corrente. Nessun comando di cancellazione remota è implementato.

### Trasporto REST opzionale

Con un PAT fornito dall'utente tramite variabile d'ambiente `AIRTABLE_TOKEN`, con accesso alla base e scope `schema.bases:read`, `data.records:read`, `data.records:write`:

```sh
python -m research_memory pull
python -m research_memory sync OPERATION_ID
```

Non serve un token per SQL, test, import/export o workflow via connettore. Le credenziali del connettore non vengono estratte, copiate o salvate. Il percorso REST è testato con trasporto simulato, non con scritture di prova nella base reale.

## Git e recupero

- Database SQLite e snapshot JSON vanno aggiornati nello stesso commit; il JSON fornisce diff ispezionabili.
- Non fare merge binari del DB: in caso di conflitto, riconciliare la outbox e rigenerare/importare lo snapshot corrente verificato, preservando gli intenti pendenti.
- Nessun polling, automazione o sync in background è stato attivato. Modifiche manuali su Airtable diventano locali al prossimo pull/import.
- Non dichiarare «scritto su entrambi» se `pending_writes` è diverso da zero.
- Le cardinalità singole documentate sono controllate sulle nuove patch dirette; l'import deve rispecchiare anche eventuali dati remoti non conformi. Obbligatorietà, aciclicità e vincoli epistemici richiedono ancora una validazione di dominio, non vanno confusi con le foreign key SQL.
