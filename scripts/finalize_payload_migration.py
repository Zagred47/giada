"""Finish validated cutover, verify query parity and two portable backups."""
import json,sqlite3,subprocess,sys,time,hashlib
from pathlib import Path
from research_memory.mirror import Mirror,DEFAULT_DB,ROOT,write_json,quote
from research_memory.payload_store import Store,active,install_size_guards,LIMIT
from research_memory.incremental_batch import IncrementalBatchMirror
from research_memory.payload_backup import verify_reports

def main():
    receipt=ROOT/'data/payload_migration_receipt.json';r=json.loads(receipt.read_text());source=DEFAULT_DB.resolve();rollback=source.with_name('research_memory.pre-payload-v1.sqlite')
    if r['state'] not in ('complete_original_rollback_retained','cutover_pending'):raise ValueError('Migration has not completed verified cutover; inspect first')
    m=IncrementalBatchMirror(source)
    with Mirror(rollback).connect() as old:
        for cell in r['changed_cells']:
            key=cell['table']
            if key=='_remote_records':
                row=old.execute('SELECT table_key,record_id FROM _remote_records WHERE rowid=?',(cell['rowid'],)).fetchone()
                cell['stable_record_identity']=[row[0],row[1]]
            else:
                row=old.execute('SELECT _record_id FROM '+quote(key)+' WHERE rowid=?',(cell['rowid'],)).fetchone()
                cell['stable_record_identity']=[key,row[0]]
    with m.connect() as c:
        if not active(c) or not rollback.is_file():raise ValueError('Verified source rollback required')
        if c.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()[0]!=r['original_snapshot_hash']:raise ValueError('Unexpected logical change')
        # Compact any large metadata documents (long link arrays) left by an
        # earlier running migration implementation, without re-encoding fields.
        store=Store(source.parent/'raw_reports');wrapped=0
        for key in m.contract.tables:
            table='_store_'+key
            for col in [r[1] for r in c.execute('PRAGMA table_info('+quote(table)+')') if r[2]=='TEXT']:
                for rowid,text in c.execute('SELECT rowid,'+quote(col)+' FROM '+quote(table)+' WHERE length(cast('+quote(col)+' as blob))>?',(LIMIT,)):
                    ref=store.put(text)
                    if store.read(ref)!=text:raise ValueError('Unicode text wrapper roundtrip failed')
                    c.execute('UPDATE '+quote(table)+' SET '+quote(col)+'=? WHERE rowid=?',(ref,rowid))
        for rowid,text in c.execute('SELECT rowid,fields_json FROM _store__remote_records WHERE length(cast(fields_json as blob))>?',(LIMIT,)):
            ref=store.put(text)
            if store.read(ref)!=text:raise ValueError('Metadata wrapper roundtrip failed')
            c.execute('UPDATE _store__remote_records SET fields_json=? WHERE rowid=?',(ref,rowid));wrapped+=1
        install_size_guards(c,m.contract)
    r['metadata_documents_wrapped']=wrapped
    catalog=json.loads((ROOT/'queries/query_catalog_review.json').read_text(encoding='utf-8'))
    for path in (rollback,source):
        with Mirror(path).connect() as c:
            for q in catalog:
                params={p:'rec00000000000000' for p in q['params']}
                if 'max_depth' in params:params['max_depth']=12
                if 'expected_strata_json' in params:params['expected_strata_json']='["riposo"]'
                if 'axis' in params:params['axis']='instances'
                c.execute('EXPLAIN '+q['full_sql'],params).fetchall()
    queries=[
        "SELECT Stato,count(*) FROM claims GROUP BY Stato ORDER BY Stato",
        "SELECT Stato,count(*) FROM runs GROUP BY Stato ORDER BY Stato",
        "SELECT table_key,count(*) FROM v_records GROUP BY table_key ORDER BY table_key",
        "SELECT json_extract(Descrizione,'$[2].rows[218].channels[4].nrmse'),json_extract(Descrizione,'$[2].rows[234].channels[4].nrmse') FROM observations WHERE [Codice stabile]='observations-giada-task41n-partial-raw-canonical_floors.json-v1'",
        "SELECT j.key,json_type(j.value) FROM observations o,json_each(o.Descrizione) j WHERE o.[Codice stabile]='observations-giada-task41n-partial-raw-canonical_floors.json-v1' ORDER BY j.key",
    ]
    parity=[]
    for sql in queries:
        with Mirror(rollback).connect() as a:before=[tuple(row) for row in a.execute(sql)]
        with m.connect() as b:after=[tuple(row) for row in b.execute(sql)]
        if before!=after:raise ValueError('Query regression: '+sql)
        parity.append(dict(sql=sql,rows=after,equal=True))
    r['query_parity']=parity;r['catalog_queries_explained']=len(catalog)
    with m.connect() as c:
        h=m._snapshot_hash(c)
        if h!=r['original_snapshot_hash']:raise ValueError('Post-guard logical snapshot mismatch')
    r['final_logical_hash']=h
    # Both the original untouched database and all old verified snapshots remain
    # available until each complete migrated backup passes the retention guard.
    r['state']='query_parity_verified_backups_pending';write_json(receipt,r)
    m.export_snapshot_to(source.parent/'airtable_snapshot.json')
    for generation in range(2):
        subprocess.run([sys.executable,'-u','-m','research_memory.prepare_drive_backup'],cwd=ROOT.parent,check=True)
        subprocess.run([sys.executable,'-u','-m','research_memory.record_local_backup','--raw-sqlite','--scope','Lossless payload-v2 migration:175308records,allIDs/relations/JSONmetrics unchanged; original canonical logical hash retained. Complete DB plus compressed raw_reports bundle; two verified local generations. No Airtable/Drive access.'],cwd=ROOT.parent,check=True)
        print('Complete portable backup generation '+str(generation+1)+' verified',flush=True)
    r['final_backup']=json.loads((ROOT/'data/local_backup_receipt.json').read_text())
    if r['final_backup']['single_file'] or not r['final_backup']['valid']:raise ValueError('Incomplete migrated backup')
    from research_memory.backup_retention import rotate
    check=rotate(apply=False);r['retained_backups']=check['retained']
    for row in check['retained']:
        folder=Path(row['folder']);info=json.loads((folder/'manifest.json').read_text())
        if not info.get('payload_manifest'):raise ValueError('Both retained generations must include raw reports')
        verify_reports(folder,info)
    # Windows SQLite handles in the completed migration may have delayed this
    # redundant staging-file cleanup. Scope checks precede every unlink.
    for p in (source.with_name('research_memory.payload-stage.sqlite'),rollback):
        if p.exists():
            if p.resolve().parent!=source.parent or p.is_symlink():raise ValueError('Unsafe migration cleanup path')
            p.unlink()
    r.update(state='complete_two_portable_backups_verified',final_database_bytes=source.stat().st_size,final_export_bytes=(source.parent/'airtable_snapshot.json').stat().st_size,final_payload_bytes=sum(p.stat().st_size for p in (source.parent/'raw_reports').rglob('*.gz')),original_rollback_removed_after_two_verified_backups=True)
    write_json(receipt,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('changed_cells','before_counts','relationship_counts','query_parity','verification')},indent=2),flush=True)
if __name__=='__main__':main()
