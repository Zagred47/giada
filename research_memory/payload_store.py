"""Lossless external text payloads with transparent SQL/JSON compatibility.

Use Mirror.connect()/open_readonly(), not bare sqlite3, for logical report fields.
Missing/corrupt payloads raise: they never silently become NULL or summaries.
"""
import gzip,hashlib,json,os,re,sqlite3,uuid
from pathlib import Path

PROFILE='external-text-gzip-v2'
PREFIX='giada-payload-v2:'
REF=re.compile(r'giada-payload-v2:([0-9a-f]{64}):(\d+)\Z')
LIMIT=65536

class Store:
    def __init__(self,root):self.root=Path(root);self.verified=set()
    def path(self,h):return self.root/h[:2]/(h+'.txt.gz')
    def read(self,value):
        if not isinstance(value,str) or not value.startswith(PREFIX):return value
        match=REF.fullmatch(value)
        if not match:raise ValueError('Malformed external payload reference')
        h,size=match.groups();path=self.path(h)
        with gzip.open(path,'rb') as f:raw=f.read()
        if len(raw)!=int(size) or hashlib.sha256(raw).hexdigest()!=h:raise ValueError('External report checksum mismatch: '+h)
        return raw.decode('utf-8')
    def put(self,value):
        if not isinstance(value,str) or (len(value)<LIMIT//4 and not value.startswith(PREFIX)):return value
        raw=value.encode('utf-8')
        if len(raw)<LIMIT and not value.startswith(PREFIX):return value
        h=hashlib.sha256(raw).hexdigest();ref=PREFIX+h+':'+str(len(raw));path=self.path(h)
        if h not in self.verified:
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    with tmp.open('xb') as stream:
                        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0,compresslevel=3) as z:z.write(raw)
                    # Never overwrite another immutable payload.
                    try:
                        if os.name=='nt':tmp.rename(path) # Windows rename refuses overwrite.
                        else:os.link(tmp,path)
                    except FileExistsError:pass
                finally:tmp.unlink(missing_ok=True)
            if self.read(ref)!=value:raise ValueError('Payload readback mismatch')
            self.verified.add(h)
        return ref
    def fields_put(self,value):
        fields=json.loads(value)
        fields={k:self.put(v) for k,v in fields.items()}
        return self.put(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False))
    def fields_read(self,value):
        if value.startswith(PREFIX):value=self.read(value)
        if PREFIX not in value:return value
        fields=json.loads(value);fields={k:self.read(v) for k,v in fields.items()}
        return json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def active(con):
    row=con.execute("SELECT value FROM _mirror_meta WHERE key='storage_profile'").fetchone()
    if row is not None and row[0]!=PROFILE:raise ValueError('Unsupported storage profile')
    return row is not None

def observation_bytes(con,dbpath,rid):
    table=physical(con,'observations')
    if not active(con):
        rowid=con.execute('SELECT rowid FROM observations WHERE _record_id=?',(rid,)).fetchone()[0]
        with con.blobopen('observations','Descrizione',rowid,readonly=True) as f:
            while block:=f.read(1024*1024):yield block
    else:
        value=con.execute('SELECT Descrizione FROM "'+table+'" WHERE _record_id=?',(rid,)).fetchone()[0]
        match=REF.fullmatch(value) if isinstance(value,str) else None
        if match:
            h,size=match.groups();digest=hashlib.sha256();count=0
            with gzip.open(Store(Path(dbpath).parent/'raw_reports').path(h),'rb') as f:
                while block:=f.read(1024*1024):digest.update(block);count+=len(block);yield block
            if digest.hexdigest()!=h or count!=int(size):raise ValueError('Report byte stream checksum failed')
        else:yield value.encode('utf-8')

def register(con,dbpath,writable=True):
    store=Store(Path(dbpath).resolve().parent/'raw_reports')
    def denied(value):raise ValueError('Read-only payload connection')
    con.create_function('giada_text',1,store.read)
    con.create_function('giada_store_text',1,store.put if writable else denied)
    con.create_function('giada_fields',1,store.fields_read)
    con.create_function('giada_store_fields',1,store.fields_put if writable else denied)
    return store

def open_readonly(path):
    from .mirror import ClosingConnection
    path=Path(path).resolve();con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,factory=ClosingConnection)
    con.row_factory=sqlite3.Row;register(con,path,writable=False);con.execute('PRAGMA query_only=ON');return con

def physical(con,key):return '_store_'+key if active(con) else key

def values_for(con,dbpath,key,cols,values):
    if not active(con):return values
    store=Store(Path(dbpath).resolve().parent/'raw_reports')
    return [store.fields_put(v) if key=='_remote_records' and col=='fields_json' else store.put(v) for col,v in zip(cols,values)]

def remote_rows(con,key):
    table=physical(con,'_remote_records')
    return con.execute('SELECT rowid,record_id,created_time FROM "'+table+'" WHERE table_key=? ORDER BY record_id',(key,))

def remote_bytes(con,dbpath,rowid):
    table=physical(con,'_remote_records')
    if not active(con):
        with con.blobopen(table,'fields_json',rowid,readonly=True) as blob:
            while block:=blob.read(1024*1024):yield block
    else:
        value=con.execute('SELECT fields_json FROM "'+table+'" WHERE rowid=?',(rowid,)).fetchone()[0]
        text=Store(Path(dbpath).resolve().parent/'raw_reports').fields_read(value)
        for i in range(0,len(text),262144):yield text[i:i+262144].encode('utf-8')

def compact_export(mirror,path):
    """Small versioned export of references; raw reports remain lossless in CAS."""
    from .mirror import dumps
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    if path.resolve().parent!=mirror.path.resolve().parent:raise ValueError('Compact snapshot must remain beside raw_reports; use the portable backup command for another location')
    try:
        with mirror.connect() as con,tmp.open('w',encoding='utf-8',newline='\n') as f:
            f.write('{"format":"giada-payload-snapshot-v2","base_id":'+dumps(mirror.contract.base_id)+',"complete":true,"payload_store":"raw_reports","tables":{')
            for ti,(key,table) in enumerate(mirror.contract.tables.items()):
                if ti:f.write(',')
                f.write(dumps(table['id'])+':[')
                for ri,r in enumerate(con.execute('SELECT record_id,created_time,fields_json FROM _store__remote_records WHERE table_key=? ORDER BY record_id',(key,))):
                    if ri:f.write(',')
                    document=Store(mirror.path.parent/'raw_reports').read(r[2])
                    f.write('{"id":'+dumps(r[0])+',"createdTime":'+dumps(r[1])+',"cellValuesByFieldId":'+document+'}')
                f.write(']')
            f.write('}}\n')
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)

def install_views(con,contract):
    """Explicit physical migration, logical schema/IDs/relationships unchanged."""
    from .mirror import quote
    views=list(con.execute("SELECT name,sql FROM sqlite_master WHERE type='view'"))
    for name,_ in views:con.execute('DROP VIEW '+quote(name))
    keys=[*contract.tables,'_remote_records']
    for key in keys:con.execute('ALTER TABLE '+quote(key)+' RENAME TO '+quote('_store_'+key))
    for key in keys:
        table='_store_'+key;cols=list(con.execute('PRAGMA table_info('+quote(table)+')'));names=[r[1] for r in cols]
        def read(col):
            if key=='_remote_records' and col=='fields_json':return 'giada_fields('+quote(col)+')'
            kind=next(r[2] for r in cols if r[1]==col)
            return 'giada_text('+quote(col)+')' if kind=='TEXT' and col not in ('_record_id','_created_time','table_key','record_id','created_time') else quote(col)
        con.execute('CREATE VIEW '+quote(key)+' AS SELECT '+','.join(read(n)+' AS '+quote(n) for n in names)+' FROM '+quote(table))
        def write(n):
            if key=='_remote_records' and n=='fields_json':return 'giada_store_fields(NEW.'+quote(n)+')'
            kind=next(r[2] for r in cols if r[1]==n)
            return 'giada_store_text(NEW.'+quote(n)+')' if kind=='TEXT' and n not in ('_record_id','_created_time','table_key','record_id','created_time') else 'NEW.'+quote(n)
        pk=[r[1] for r in cols if r[5]]
        where=' AND '.join(quote(n)+'=OLD.'+quote(n) for n in pk)
        con.execute('CREATE TRIGGER '+quote('_payload_insert_'+key)+' INSTEAD OF INSERT ON '+quote(key)+' BEGIN INSERT INTO '+quote(table)+' ('+','.join(map(quote,names))+') VALUES ('+','.join(write(n) for n in names)+'); END')
        con.execute('CREATE TRIGGER '+quote('_payload_update_'+key)+' INSTEAD OF UPDATE ON '+quote(key)+' BEGIN UPDATE '+quote(table)+' SET '+','.join(quote(n)+'='+write(n) for n in names)+' WHERE '+where+'; END')
        con.execute('CREATE TRIGGER '+quote('_payload_delete_'+key)+' INSTEAD OF DELETE ON '+quote(key)+' BEGIN DELETE FROM '+quote(table)+' WHERE '+where+'; END')
    for _,sql in views:con.execute(sql)
    con.execute("INSERT INTO _mirror_meta VALUES ('storage_profile',?)",(PROFILE,))

def install_size_guards(con,contract):
    """Even direct physical SQL cannot silently reinsert giant inline text."""
    from .mirror import quote
    for key in [*contract.tables,'_remote_records']:
        table='_store_'+key
        cols=[r[1] for r in con.execute('PRAGMA table_info('+quote(table)+')') if r[2]=='TEXT']
        condition=' OR '.join('length(CAST(NEW.'+quote(col)+' AS BLOB))>'+str(LIMIT) for col in cols)
        for op in ('INSERT','UPDATE'):
            con.execute('CREATE TRIGGER IF NOT EXISTS '+quote('_payload_guard_'+op+'_'+key)+' BEFORE '+op+' ON '+quote(table)+' WHEN '+condition+" BEGIN SELECT RAISE(ABORT,'Large inline text forbidden: use the GIADA payload-aware writer'); END")
