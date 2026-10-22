import json
from pathlib import Path
import tempfile
import unittest
from research_memory.mirror import Contract,Mirror,digest
from research_memory.incremental_batch import IncrementalBatchMirror

class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'memory.sqlite'
        self.m=Mirror(self.path);self.m.initialize()
        self.m.import_snapshot(dict(format='airtable-record-snapshot-v1',base_id=self.m.contract.base_id,complete=True,tables={t['id']:[] for t in self.m.contract.tables.values()}))
    def put(self,m,table,code,**values):return m.local_upsert(table,dict(Nome=code,**{'Codice stabile':code},**values))['record_id']
    def test_forward_inverse_order_update_and_stream_hash(self):
        b=IncrementalBatchMirror(self.path)
        a=self.put(b,'actors','authorA');bb=self.put(b,'actors','authorB')
        s=self.put(b,'sources','source',Autori=[bb,a]);b.commit_batch()
        self.assertTrue(b.verify()['valid'])
        b=IncrementalBatchMirror(self.path)
        self.put(b,'sources','source',Autori=[a,bb]);b.commit_batch()
        self.assertTrue(b.verify()['valid'])
        with b.connect() as con:
            fields=json.loads(con.execute('SELECT fields_json FROM _remote_records WHERE table_key=? AND record_id=?',('sources',s)).fetchone()[0])
        self.assertEqual(fields[b.field_id('sources','Autori')],[a,bb])
        snap=b.export_snapshot();expected=digest(b.normalize_snapshot(snap))
        self.assertEqual(expected,b.status()['snapshot_hash'])
        target=Path(self.temp.name)/'stream.json';b.export_snapshot_to(target)
        self.assertEqual(json.loads(target.read_text()),snap)
        b=IncrementalBatchMirror(self.path);self.put(b,'sources','source',Autori=[bb]);b.commit_batch()
        self.assertTrue(b.verify()['valid'])
        self.assertEqual(len(b.query('SELECT * FROM v_links')['rows']),1)
    def test_reverse_update_self_link_and_unchanged_blob(self):
        b=IncrementalBatchMirror(self.path)
        a=self.put(b,'artifacts','A');bb=self.put(b,'artifacts','B',**{'Artefatti di origine':[a]})
        self.put(b,'observations','raw',Descrizione=json.dumps({'text':'x'*100000,'nested':{'n':[1,2,3]}}));b.commit_batch()
        before=b.export_snapshot()
        b=IncrementalBatchMirror(self.path);self.put(b,'artifacts','B',**{'Artefatti di origine':[]});b.commit_batch()
        self.assertTrue(b.verify()['valid'])
        after=b.export_snapshot()
        key=b.contract.tables['observations']['id'];self.assertEqual(before['tables'][key],after['tables'][key])
    def test_missing_link_does_not_stage_invalid_record(self):
        b=IncrementalBatchMirror(self.path)
        with self.assertRaises(ValueError):self.put(b,'sources','bad',Autori=['rec99999999999999'])
        b.commit_batch();self.assertEqual(b.status()['records'],0)
    def test_concurrent_edit_fails_without_overwrite(self):
        b=IncrementalBatchMirror(self.path);self.put(b,'actors','A');b.commit_batch()
        one=IncrementalBatchMirror(self.path);two=IncrementalBatchMirror(self.path)
        one.local_upsert('actors',{'Codice stabile':'A','Nome':'first'})
        two.local_upsert('actors',{'Codice stabile':'A','Nome':'second'});two.commit_batch()
        with self.assertRaises(ValueError):one.commit_batch()
        self.assertEqual(two.query('SELECT Nome FROM actors')['rows'][0]['Nome'],'second')
    def test_reverse_link_reordering(self):
        b=IncrementalBatchMirror(self.path)
        a=self.put(b,'actors','A');s1=self.put(b,'sources','S1',Autori=[a]);s2=self.put(b,'sources','S2',Autori=[a]);b.commit_batch()
        link=next(l for l in b.contract.links if l['from']=='sources' and l['name']=='Autori')
        b=IncrementalBatchMirror(self.path);b.local_upsert('actors',{'Codice stabile':'A',link['inverseFieldId']:[s2,s1]});b.commit_batch()
        self.assertTrue(b.verify()['valid'])
        with b.connect() as con:
            value=json.loads(con.execute('SELECT fields_json FROM _remote_records WHERE table_key=? AND record_id=?',('actors',a)).fetchone()[0])
        self.assertEqual(value[link['inverseFieldId']],[s2,s1])

if __name__=='__main__':unittest.main()
