import tempfile
import shutil
import unittest
from pathlib import Path
from research_memory.validated_batch import ValidatedBatchMirror

class ValidatedBatchTests(unittest.TestCase):
    def test_id_cache_includes_pending_rows(self):
        m=ValidatedBatchMirror()
        self.assertFalse(m._validated_batch_id_exists(m.staged,'recPendingTest'))
        m._validated_batch_id_added('recPendingTest')
        self.assertTrue(m._validated_batch_id_exists(m.staged,'recPendingTest'))
        for rows in m.staged['tables'].values():
            if rows:
                self.assertTrue(m._validated_batch_id_exists(m.staged,rows[0]['id']))
                break

    def test_cached_row_matches_original_without_aliasing(self):
        m=ValidatedBatchMirror()
        for key in m.contract.tables:
            for row in m.staged['tables'][m.contract.tables[key]['id']][:10]:
                value=row['cellValuesByFieldId']
                expected=m._original_normalize(key,value)
                self.assertEqual(m.contract.normalize(key,value),expected)
                actual=m.contract.normalize(key,value)
                self.assertEqual(actual,expected)
                for k,v in actual.items():
                    if isinstance(v,list):self.assertIsNot(v,value[k])

    def test_replacement_invalid_input_and_commit(self):
        m=ValidatedBatchMirror()
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/'memory.sqlite';shutil.copyfile(m.path,copy);m.path=copy
            row=m.local_upsert('claims',{'Codice stabile':'claims-test-validated-batch','Nome':'batchtest','Tipo':'Ipotesi','Stato':'Aperta'})
            m.local_upsert('claims',{'Codice stabile':'claims-test-validated-batch','Nome':'changed'},record_id=row['record_id'])
            m.local_upsert('claims',{'Nome':'changed again'},record_id=row['record_id'])
            with self.assertRaises(ValueError):
                m.local_upsert('claims',{'Nome':'missing stable code'})
            with self.assertRaises(ValueError):
                m.local_upsert('evidence',{'Codice stabile':'evidence-test-invalid','Nome':'invalid','Esito':'not-a-choice'})
            m.commit_batch()
            self.assertTrue(m.verify()['valid'])
            self.assertEqual(m.contract.normalize,m._original_normalize)

if __name__=='__main__':unittest.main()
