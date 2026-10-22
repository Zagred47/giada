"""Bounded-memory local upserts with reciprocal links and atomic readback.

Uses the same Contract, stable IDs and scalar/link tables as Mirror. No remote
transport. Only touched records are staged; untouched scientific JSON is never
loaded as a complete snapshot or rewritten. A streaming canonical hash retains
the existing snapshot_hash meaning. Optimistic before-images reject races.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import uuid
from .mirror import Mirror,dumps,quote


class IncrementalBatchMirror(Mirror):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rows={};self.before={};self.indices={}

    def _get(self,key,rid):
        pair=(key,rid)
        if pair not in self.rows:
            with self.connect() as con:
                row=con.execute('SELECT created_time,fields_json FROM _remote_records WHERE table_key=? AND record_id=?',pair).fetchone()
            if row is None:raise ValueError('Linked or updated record missing: '+rid)
            self.before[pair]=(row['created_time'],row['fields_json'])
            self.rows[pair]=dict(createdTime=row['created_time'],fields=self.contract.normalize(key,json.loads(row['fields_json'])))
        return self.rows[pair]

    def local_upsert(self,table,values,record_id=None):
        key=self.contract.key(table);patch=self.contract.normalize(key,values,partial=True)
        is_new=False
        code_fid=self.field_id(key,'Codice stabile');name_fid=self.field_id(key,'Nome')
        if key not in self.indices:
            with self.connect() as con:
                index={}
                for row in con.execute(f'SELECT _record_id,{quote("Codice stabile")} FROM {quote(key)}'):
                    index.setdefault(row[1],[]).append(row[0])
            self.indices[key]=index
        matches=self.indices[key].get(patch.get(code_fid),[])
        if record_id is not None:
            rid=record_id;current=self._get(key,rid)
            code=current['fields'][code_fid]
            if not code or code_fid in patch and patch[code_fid]!=code:raise ValueError('Immutable stable code')
            patch[code_fid]=code
        elif matches:
            if len(matches)!=1:raise ValueError('Duplicate stable code')
            rid=matches[0];current=self._get(key,rid)
        else:
            code=patch.get(code_fid)
            if not code or not patch.get(name_fid):raise ValueError('New records require Nome and Codice stabile')
            rid='rec'+hashlib.sha256((key+'\0'+code).encode()).hexdigest()[:14]
            with self.connect() as con:exists=con.execute('SELECT 1 FROM v_records WHERE record_id=? LIMIT 1',(rid,)).fetchone()
            if exists or any(rid==r for _,r in self.rows):raise ValueError('ID collision')
            current=dict(createdTime=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),fields=self.contract.normalize(key,{}))
            is_new=True
        original=dict(current['fields']);merged=dict(original);merged.update(patch)
        merged=self.contract.normalize(key,merged)
        # Validate all targets before changing staged reciprocal fields.
        for fid,targets in patch.items():
            if fid not in self.contract.link_fields:continue
            link,reverse=self.contract.link_fields[fid]
            if link['single'] and not reverse and len(targets)>1:raise ValueError('Single relationship has several targets')
            tk=link['from'] if reverse else link['to']
            for target in targets:
                if not (is_new and tk==key and target==rid):self._get(tk,target)
        if is_new:
            self.before[(key,rid)]=None;self.rows[(key,rid)]=current;self.indices[key][code]=[rid]
        current['fields']=merged
        for fid,targets in patch.items():
            if fid not in self.contract.link_fields:continue
            link,reverse=self.contract.link_fields[fid]
            tk=link['from'] if reverse else link['to'];inverse=link['fieldId'] if reverse else link['inverseFieldId']
            old,new=set(original[fid]),set(targets)
            for target in old|new:
                row=self._get(tk,target);items=list(row['fields'][inverse])
                if target in old-new and rid in items:items.remove(rid)
                if target in new and rid not in items:items.append(rid)
                row['fields'][inverse]=items
        return dict(record_id=rid,stable_code=merged[code_fid],table=key)

    def _snapshot_hash(self,con):
        h=hashlib.sha256();h.update(b'{')
        for ti,key in enumerate(sorted(self.contract.tables)):
            if ti:h.update(b',')
            h.update((dumps(key)+':{').encode())
            for ri,row in enumerate(con.execute('SELECT rowid,record_id,created_time FROM _remote_records WHERE table_key=? ORDER BY record_id',(key,))):
                if ri:h.update(b',')
                h.update((dumps(row['record_id'])+':{"createdTime":'+dumps(row['created_time'])+',"fields":').encode())
                # _remote_records is written exclusively with canonical dumps.
                # Stream its UTF-8 TEXT bytes through SQLite incremental IO;
                # several original scientific reports exceed400MB per row.
                with con.blobopen('_remote_records','fields_json',row['rowid'],readonly=True) as blob:
                    while chunk:=blob.read(1024*1024):h.update(chunk)
                h.update(b'}')
            h.update(b'}')
        h.update(b'}');return h.hexdigest()

    def commit_batch(self):
        dirty={pair:row for pair,row in self.rows.items() if self.before[pair] is None or
            (row['createdTime'],dumps(row['fields']))!=self.before[pair]}
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            for pair in self.rows:
                old=con.execute('SELECT created_time,fields_json FROM _remote_records WHERE table_key=? AND record_id=?',pair).fetchone()
                if (tuple(old) if old else None)!=self.before[pair]:raise ValueError('Concurrent record change; batch rolled back')
            def fields(key,rid):
                if (key,rid) in self.rows:return self.rows[(key,rid)]['fields']
                row=con.execute('SELECT fields_json FROM _remote_records WHERE table_key=? AND record_id=?',(key,rid)).fetchone()
                if row is None:raise ValueError('Dangling relationship')
                return json.loads(row[0])
            # Recheck reciprocity against the transaction's final logical view.
            for (key,rid),row in dirty.items():
                for fid in self.contract.link_fields:
                    if fid not in row['fields']:continue
                    link,reverse=self.contract.link_fields[fid]
                    tk=link['from'] if reverse else link['to'];inv=link['fieldId'] if reverse else link['inverseFieldId']
                    for target in row['fields'][fid]:
                        if rid not in fields(tk,target)[inv]:raise ValueError('Inconsistent reciprocal edge')
            for (key,rid),row in dirty.items():
                scalar=[f for f in self.contract.tables[key]['fields'] if f['type']!='multipleRecordLinks']
                cols=['_record_id','_created_time']+[f['name'] for f in scalar]
                values=[rid,row['createdTime']]+[row['fields'][f['id']] for f in scalar]
                con.execute(f'INSERT INTO {quote(key)} ({",".join(map(quote,cols))}) VALUES ({",".join("?" for _ in cols)}) ON CONFLICT(_record_id) DO UPDATE SET '+','.join(quote(k)+'=excluded.'+quote(k) for k in cols[1:]),values)
                con.execute('INSERT INTO _remote_records VALUES (?,?,?,?) ON CONFLICT(table_key,record_id) DO UPDATE SET created_time=excluded.created_time,fields_json=excluded.fields_json',(key,rid,row['createdTime'],dumps(row['fields'])))
            for link in self.contract.links:
                sources=set()
                for (key,rid),row in dirty.items():
                    if key==link['from']:sources.add(rid)
                    if key==link['to']:
                        sources.update(row['fields'][link['inverseFieldId']])
                        if self.before[(key,rid)]:sources.update(json.loads(self.before[(key,rid)][1])[link['inverseFieldId']])
                for source in sources:
                    targets=fields(link['from'],source)[link['fieldId']]
                    con.execute(f'DELETE FROM {quote(link["sql_table"])} WHERE source_id=?',(source,))
                    for pos,target in enumerate(targets):
                        inverse=fields(link['to'],target)[link['inverseFieldId']]
                        con.execute(f'INSERT INTO {quote(link["sql_table"])} VALUES (?,?,?,?)',(source,target,pos,inverse.index(source)))
            assert not con.execute('PRAGMA foreign_key_check').fetchall()
            snapshot_hash=self._snapshot_hash(con)
            con.execute("INSERT OR REPLACE INTO _mirror_meta VALUES ('snapshot_hash',?)",(snapshot_hash,))
            con.execute("INSERT OR REPLACE INTO _mirror_meta VALUES ('operating_mode','local_sqlite_authoritative_pending_airtable_reconciliation')")
        self.rows.clear();self.before.clear();self.indices.clear()
        return dict(changed_records=len(dirty),snapshot_hash=snapshot_hash)

    def export_snapshot_to(self,path):
        path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with self.connect() as con,temp.open('wb') as stream:
                con.execute('BEGIN')
                stream.write(('{"format":"airtable-record-snapshot-v1","base_id":'+dumps(self.contract.base_id)+',"complete":true,"tables":{').encode())
                for ti,(key,table) in enumerate(self.contract.tables.items()):
                    if ti:stream.write(b',')
                    stream.write((dumps(table['id'])+':[').encode())
                    for ri,row in enumerate(con.execute('SELECT rowid,record_id,created_time FROM _remote_records WHERE table_key=? ORDER BY record_id',(key,))):
                        if ri:stream.write(b',')
                        stream.write(('{"id":'+dumps(row[1])+',"createdTime":'+dumps(row[2])+',"cellValuesByFieldId":').encode())
                        with con.blobopen('_remote_records','fields_json',row[0],readonly=True) as blob:
                            while chunk:=blob.read(1024*1024):stream.write(chunk)
                        stream.write(b'}')
                    stream.write(b']')
                stream.write(b'}}\n')
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)
