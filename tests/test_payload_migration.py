import json,sqlite3,tempfile,unittest
from pathlib import Path
from research_memory.mirror import Mirror,read_json
from research_memory.incremental_batch import IncrementalBatchMirror
from research_memory.migrate_payloads import convert,relation_hash
from research_memory.payload_store import Store,register,open_readonly
ROOT=Path(__file__).resolve().parents[1]
class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'db.sqlite'
        self.m=Mirror(self.path);self.m.initialize()
        self.m.import_snapshot(dict(format='airtable-record-snapshot-v1',base_id=self.m.contract.base_id,complete=True,tables={v['id']:[] for v in self.m.contract.tables.values()}))
        b=IncrementalBatchMirror(self.path)
        art=b.local_upsert('artifacts',{'Nome':'source','Codice stabile':'source'})['record_id']
        self.text=json.dumps({'numbers':[1,2.5,False,None], 'unicode':'caffè😀','pad':'x'*100000},ensure_ascii=False)
        self.oid=b.local_upsert('observations',{'Nome':'raw','Codice stabile':'raw','Descrizione':self.text,'Artefatti dettagliati':[art]})['record_id'];b.commit_batch()
        self.before=b.export_snapshot();self.hash=b.status()['snapshot_hash']
        with b.connect() as con:self.relations=relation_hash(con,b.contract)
    def migrate(self):
        with self.m.connect() as con:convert(con,self.path,self.m.contract,progress=lambda *a,**k:None)
    def test_full_roundtrip_queries_and_new_writes(self):
        self.migrate();b=IncrementalBatchMirror(self.path)
        self.assertEqual(self.before,b.export_snapshot());self.assertTrue(b.verify()['valid'])
        with b.connect() as c:
            self.assertEqual(b._snapshot_hash(c),self.hash);self.assertEqual(relation_hash(c,b.contract),self.relations)
        rows=b.query("SELECT json_extract(Descrizione,'$.unicode') as u,json_extract(Descrizione,'$.numbers[1]') as n FROM observations")['rows']
        self.assertEqual(rows,[{'u':'caffè😀','n':2.5}])
        self.assertEqual(len(b.query("SELECT j.value FROM observations o,json_each(o.Descrizione,'$.numbers') j")['rows']),4)
        b.local_upsert('observations',{'Codice stabile':'raw','Descrizione':self.text+' '});b.commit_batch();self.assertTrue(b.verify()['valid'])
        self.m.local_upsert('observations',{'Codice stabile':'raw','Nome':'renamed'});self.assertTrue(b.verify()['valid'])
        with b.connect() as c:self.assertLess(len(c.execute('SELECT Descrizione FROM _store_observations').fetchone()[0]),200)
        with b.connect() as c:
            with self.assertRaises(sqlite3.IntegrityError):c.execute('UPDATE _store_observations SET Descrizione=?',('x'*70000,))
        with self.assertRaises(sqlite3.OperationalError):b.query("SELECT giada_store_text('not allowed')")
        with open_readonly(self.path) as c:self.assertEqual(c.execute("SELECT json_extract(Descrizione,'$.numbers[1]') FROM observations").fetchone()[0],2.5)
    def test_small_export_and_restore(self):
        self.migrate();b=IncrementalBatchMirror(self.path);out=self.path.parent/'snapshot.json';b.export_snapshot_to(out)
        self.assertLess(out.stat().st_size,20000)
        self.assertEqual(b.normalize_snapshot(read_json(out)),b.normalize_snapshot(self.before))
    def test_missing_report_fails_closed(self):
        self.migrate();p=next((self.path.parent/'raw_reports').rglob('*.gz'));p.unlink()
        with self.assertRaises(sqlite3.OperationalError):self.m.query('SELECT Descrizione FROM observations')
    def test_corruption_and_reserved_literal(self):
        store=Store(self.path.parent/'test_store');value='giada-payload-v2:'+'a'*64+':10';ref=store.put(value);self.assertEqual(store.read(ref),value)
        unicode='😀'*20000;self.assertEqual(store.read(store.put(unicode)),unicode);self.assertNotEqual(store.put(unicode),unicode)
        store.path(ref.split(':')[1]).write_bytes(b'broken')
        with self.assertRaises((OSError,ValueError)):store.read(ref)
    def test_portable_backup_includes_payloads(self):
        self.migrate()
        from research_memory.payload_backup import copy_reports,verify_reports
        folder=self.path.parent/'backup';folder.mkdir();target=folder/'db.sqlite'
        with self.m.connect() as source:
            dest=sqlite3.connect(target);source.backup(dest);dest.close()
        info=copy_reports(self.path,folder);verify_reports(folder,info)
        self.assertTrue(Mirror(target).verify()['valid'])
        self.assertEqual(Mirror(target).export_snapshot(),self.before)
        next((folder/'raw_reports').rglob('*.gz')).unlink()
        with self.assertRaises((ValueError,OSError)):verify_reports(folder,info)
    def test_no_legacy_export_inflation_or_incomplete_pack(self):
        self.migrate();db=self.path.parent/'research_memory.sqlite';self.path.rename(db)
        from research_memory.packed_snapshot import restore_snapshot_if_missing
        from research_memory.packed_database import pack
        out=db.parent/'snapshot.json';self.assertTrue(restore_snapshot_if_missing(out))
        self.assertEqual(read_json(out)['format'],'giada-payload-snapshot-v2');self.assertLess(out.stat().st_size,20000)
        with self.assertRaises(ValueError):pack(db)
    def test_legacy_outbox_cannot_duplicate_large_reports(self):
        self.migrate()
        with self.assertRaises(ValueError):self.m.stage('observations',{'Codice stabile':'raw','Nome':'renamed'},self.oid)
        self.assertEqual(self.m.status()['pending_writes'],0)
if __name__=='__main__':unittest.main()
