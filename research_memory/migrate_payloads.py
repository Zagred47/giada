"""Versioned lossless physical migration; never mutates the source database."""
import argparse,hashlib,json,os,shutil,sqlite3,time
from pathlib import Path
from .mirror import ROOT,DEFAULT_DB,Contract,Mirror,quote,write_json
from .payload_store import Store,register,install_views,install_size_guards,active,PROFILE

def convert(con,path,contract,progress=print):
    if active(con):raise ValueError('Already migrated')
    store=register(con,path);install_views(con,contract);changed=[]
    for key in [*contract.tables,'_remote_records']:
        physical='_store_'+key;columns=list(con.execute('PRAGMA table_info('+quote(physical)+')'))
        text=[r[1] for r in columns if r[2]=='TEXT' and r[1] not in ('_record_id','_created_time','record_id','table_key','created_time')]
        if not text:continue
        count=0
        for row in con.execute('SELECT rowid,'+','.join(map(quote,text))+' FROM '+quote(physical)):
            edits=[]
            for i,col in enumerate(text,1):
                old=row[i];new=store.fields_put(old) if key=='_remote_records' else store.put(old)
                if new!=old:
                    # Byte-for-byte roundtrip, not merely parsed JSON equality.
                    restored=store.fields_read(new) if key=='_remote_records' else store.read(new)
                    if restored!=old:raise ValueError('Lossless cell roundtrip failed: '+key+'/'+col)
                    edits.append((col,new))
                    changed.append(dict(table=key,rowid=row[0],column=col,logical_sha256=hashlib.sha256(old.encode()).hexdigest(),logical_bytes=len(old.encode())))
            if edits:
                con.execute('UPDATE '+quote(physical)+' SET '+','.join(quote(c)+'=?' for c,_ in edits)+' WHERE rowid=?',[*[v for _,v in edits],row[0]])
                count+=1
                if count%25==0:progress(f'Externalized {key}: {count} large records',flush=True)
        progress(f'Converted {key}: {count} records',flush=True)
    install_size_guards(con,contract)
    return changed

def relation_hash(con,contract):
    h=hashlib.sha256();counts={}
    for link in contract.links:
        table=link['sql_table'];count=0;h.update(table.encode())
        for row in con.execute('SELECT * FROM '+quote(table)+' ORDER BY source_id,target_id'):
            h.update(json.dumps(tuple(row),separators=(',',':')).encode());count+=1
        counts[table]=count
    return h.hexdigest(),counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    if not args.apply:raise SystemExit('Explicit --apply required; creates a candidate and validates before cutover')
    start=time.time();source=DEFAULT_DB.resolve();stage=source.with_name('research_memory.payload-stage.sqlite');compact=source.with_name('research_memory.payload-compact.sqlite')
    rollback=source.with_name('research_memory.pre-payload-v1.sqlite')
    with Mirror(source).connect() as probe:
        if active(probe):raise ValueError('Operational DB already migrated; completed receipt left untouched')
    if any(p.exists() for p in (stage,compact,rollback)):raise ValueError('Migration files already exist; inspect instead of overwriting')
    report=dict(state='copying_source',source_bytes=source.stat().st_size,profile=PROFILE,changed_cells=[])
    receipt=ROOT/'data/payload_migration_receipt.json'
    def save(state):report.update(state=state,elapsed_seconds=time.time()-start);write_json(receipt,report)
    save('copying_source');contract=Contract()
    with Mirror(source).connect() as original:
        if active(original):raise ValueError('Operational DB already migrated')
        report['original_snapshot_hash']=original.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()[0]
        report['before_counts']={k:original.execute('SELECT count(*) FROM '+quote(k)).fetchone()[0] for k in contract.tables}
        report['relationship_hash'],report['relationship_counts']=relation_hash(original,contract)
        with sqlite3.connect(stage) as target:original.backup(target)
        target.close()
    save('externalizing_candidate')
    with sqlite3.connect(stage) as target:
        target.execute('PRAGMA foreign_keys=OFF')
        report['changed_cells']=convert(target,stage,contract)
        target.commit()
        if target.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('Migrated FK failure')
        if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Migrated integrity failure')
        save('compacting_candidate')
        target.execute('VACUUM INTO ?', (str(compact),))
    target.close()
    save('verifying_candidate')
    candidate=Mirror(compact)
    verified=candidate.verify()
    if not verified['valid'] or verified['table_counts']!=report['before_counts']:raise ValueError('Full logical verification failed')
    with candidate.connect() as con:
        rh,rc=relation_hash(con,contract)
        if (rh,rc)!=(report['relationship_hash'],report['relationship_counts']):raise ValueError('Relationships/order changed')
        from .incremental_batch import IncrementalBatchMirror
        h=IncrementalBatchMirror(compact)._snapshot_hash(con)
        if h!=report['original_snapshot_hash']:raise ValueError('Canonical logical snapshot hash changed')
    report.update(verification=verified,logical_snapshot_hash_recomputed=h,candidate_bytes=compact.stat().st_size)
    save('cutover_pending')
    # Recheck source identity and exact paths before the only operational move.
    if source.parent!=ROOT.resolve()/'data' or compact.parent!=source.parent:raise ValueError('Unsafe migration path')
    with Mirror(source).connect() as con:
        if con.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()[0]!=h:raise ValueError('Source changed concurrently; no cutover')
    source.rename(rollback)
    try:compact.rename(source)
    except BaseException:rollback.rename(source);raise
    report['rollback']=str(rollback);report['payload_files']=len(list((source.parent/'raw_reports').rglob('*.gz')))
    report['payload_bytes']=sum(p.stat().st_size for p in (source.parent/'raw_reports').rglob('*.gz'))
    from .incremental_batch import IncrementalBatchMirror
    IncrementalBatchMirror(source).export_snapshot_to(source.parent/'airtable_snapshot.json')
    # Candidate staging copy is redundant; keep the untouched original rollback.
    if stage.resolve().parent!=source.parent:raise ValueError('Unsafe staging cleanup')
    stage.unlink();save('complete_original_rollback_retained')
    print(json.dumps({k:v for k,v in report.items() if k not in ('changed_cells','relationship_counts','before_counts')},indent=2),flush=True)

if __name__=='__main__':main()
