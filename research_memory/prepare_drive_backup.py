"""Consistent single-file SQLite export plus original-format documentation."""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime,timezone
from .mirror import ROOT,DEFAULT_DB


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%SZ')
    folder=ROOT/'drive_exports'/stamp;folder.mkdir(parents=True,exist_ok=False)
    path=folder/f'giada__research-memory__{stamp}.sqlite'
    with closing(sqlite3.connect(DEFAULT_DB.resolve().as_uri()+'?mode=ro',uri=True)) as source:
        with closing(sqlite3.connect(path)) as backup:source.backup(backup)
    with closing(sqlite3.connect(path)) as c:
        integrity=c.execute('pragma integrity_check').fetchone()[0]
        fk=c.execute('pragma foreign_key_check').fetchall()
        assert integrity=='ok' and not fk
        names=[r[0] for r in c.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name")]
        counts={n:c.execute('select count(*) from "'+n.replace('"','""')+'"').fetchone()[0] for n in names}
        schema='\n'.join(r[0]+';' for r in c.execute("select sql from sqlite_master where sql is not null order by type,name"))
        records=c.execute('select count(*) from v_records').fetchone()[0]
    (folder/'schema.sql').write_text(schema+'\n',encoding='utf-8')
    h=hashlib.sha256();md5=hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk);md5.update(chunk)
    manifest=dict(project='GIADA',database='research-memory',snapshot_utc=stamp,format='SQLite3',filename=path.name,
        bytes=path.stat().st_size,sha256=h.hexdigest(),md5=md5.hexdigest(),semantic_records=records,
        table_row_counts=counts,integrity_check=integrity,foreign_key_violations=fk,
        source=str(DEFAULT_DB),single_file=True,airtable_accessed=False,
        external_dependencies='Artifact paths and scientific source/checkpoint references require Git repository or Kaggle artifacts. Numeric case/channel metrics remain queryable inside SQLite.',
        post_upload_verification='Pending; see separate Drive receipt.')
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme=f'''# GIADA — research memory SQLite\n\nSnapshot `{path.name}`, UTC {stamp}.\n\nDatabase reale originale, singolo file, {path.stat().st_size} byte; {records} record semantici.\nSHA-256: `{h.hexdigest()}`. Backup SQLite API consistente; integrity_check=ok, zero violazioni FK.\n\nOrigine: mirror locale autorevole, nessun accesso Airtable. Include risultati Task39 e preregistrazione Task39b; lo stato del lancio e' quello registrato alla data dello snapshot.\n\nAprire inizialmente in sola lettura: `sqlite3.connect('file:PATH?mode=ro', uri=True)`.\nSchema completo in `schema.sql`, conteggi e checksum in `manifest.json`.\n78 tabelle semantiche e241 relazioni; `v_records`, `v_links`, `v_evidence_chains` per esplorazione.\nLe osservazioni contengono anche JSON dettagliato interrogabile con JSON1.\n\nPer aggiornare usare l'interfaccia versionata research_memory, poi creare una nuova versione; non sovrascrivere database concorrenti o snapshot precedenti.\nIl DB non contiene i file binari esterni dei checkpoint: i percorsi rimandano alla repository GitHub Zagred47/giada e agli artefatti Kaggle. Le metriche numeriche archiviate sono interne al DB.\n\nGoogle Drive e' archivio, non server SQLite. Nessuna condivisione pubblica o sincronizzazione Airtable effettuata.\n'''
    readme=readme.replace("Include risultati Task39 e preregistrazione Task39b; lo stato del lancio e' quello registrato alla data dello snapshot.",
        "Include la memoria locale allo stato della data indicata; i risultati e gli stati di esecuzione sono interrogabili nelle tabelle findings, runs e observations. Consultare la ricevuta della versione per il perimetro aggiornato.")
    (folder/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'drive_exports/latest_local.json').write_text(json.dumps(dict(folder=str(folder),**manifest),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(folder=str(folder),filename=path.name,bytes=path.stat().st_size,sha256=h.hexdigest(),records=records)),flush=True)

if __name__=='__main__':main()
